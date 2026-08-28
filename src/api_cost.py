"""Persistent, conservative API cost accounting for paid model requests."""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CostBudgetExceeded(RuntimeError):
    """Raised before a request that could cross the configured shared cap."""


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    reserved_usd: float
    input_token_upper_bound: int
    output_token_upper_bound: int


def _nonnegative_int(value: object) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


class APICostGuard:
    """Append-only shared ledger with persistent in-flight reservations.

    The reservation uses UTF-8 request bytes as a conservative input-token
    upper bound and separately configured standard-tier rates. A terminated
    process leaves its reservation active, which is intentionally fail-safe.
    """

    def __init__(self, config: dict[str, Any], model: str):
        self.model = model
        self.ledger_path = Path(str(config["ledger_path"]))
        self.pricing_verified_date = str(config.get("pricing_verified_date", ""))
        self.max_usd = float(config["max_usd"])
        required_tier = config.get("required_service_tier", "flex")
        self.required_service_tier = (
            None if required_tier in (None, "") else str(required_tier)
        )
        self.max_short_context_tokens = int(
            config.get("max_short_context_tokens", 272_000)
        )
        self.pricing = self._rates(config.get("pricing"), "pricing")
        self.reserve_pricing = self._rates(
            config.get("reserve_pricing"), "reserve_pricing"
        )
        self.context = dict(config.get("context") or {})
        if self.max_usd <= 0:
            raise ValueError("cost_guard.max_usd must be positive")

    @staticmethod
    def _rates(value: object, name: str) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError(f"cost_guard.{name} must be a mapping")
        required = ("input", "cached_input", "cache_write", "output")
        rates = {key: float(value[key]) for key in required}
        if any(rate < 0 for rate in rates.values()):
            raise ValueError(f"cost_guard.{name} rates must be nonnegative")
        return rates

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _read_events(self, handle: Any) -> list[dict[str, Any]]:
        handle.seek(0)
        events: list[dict[str, Any]] = []
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid cost-ledger JSON in {self.ledger_path}"
                ) from exc
            if isinstance(row, dict):
                events.append(row)
        return events

    @staticmethod
    def committed_usd(events: list[dict[str, Any]]) -> float:
        active: dict[str, float] = {}
        settled = 0.0
        for event in events:
            reservation_id = str(event.get("reservation_id") or "")
            if event.get("event") == "reserve" and reservation_id:
                active[reservation_id] = float(event.get("reserved_usd", 0.0))
            elif event.get("event") == "settle" and reservation_id:
                active.pop(reservation_id, None)
                settled += float(event.get("cost_usd", 0.0))
        return settled + sum(active.values())

    def _append_locked(self, event: dict[str, Any], *, check_amount: float = 0.0) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            events = self._read_events(handle)
            committed = self.committed_usd(events)
            if check_amount and committed + check_amount > self.max_usd + 1e-12:
                raise CostBudgetExceeded(
                    f"Shared API cost cap would be exceeded: committed/reserved "
                    f"${committed:.6f} + request reservation ${check_amount:.6f} "
                    f"> cap ${self.max_usd:.2f}. Ledger: {self.ledger_path}"
                )
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def reserve(self, payload: dict[str, Any]) -> Reservation:
        request_bytes = len(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        input_upper = request_bytes + 1024
        if input_upper > self.max_short_context_tokens:
            raise CostBudgetExceeded(
                f"Request byte upper bound {input_upper} exceeds the pilot short-context "
                f"limit {self.max_short_context_tokens}; long-context pricing is disabled."
            )
        output_upper = _nonnegative_int(
            payload.get("max_completion_tokens", payload.get("max_tokens"))
        )
        if output_upper <= 0:
            raise ValueError(
                "Paid requests require max_completion_tokens or max_tokens for cost reservation"
            )
        # Reserve every possible input token at the most expensive configured
        # input-side rate, including the cache-write tariff.
        reserve_input_rate = max(
            self.reserve_pricing["input"],
            self.reserve_pricing["cached_input"],
            self.reserve_pricing["cache_write"],
        )
        reserved = (
            input_upper * reserve_input_rate
            + output_upper * self.reserve_pricing["output"]
        ) / 1_000_000
        reservation = Reservation(
            reservation_id=uuid.uuid4().hex,
            reserved_usd=reserved,
            input_token_upper_bound=input_upper,
            output_token_upper_bound=output_upper,
        )
        event = {
            "event": "reserve",
            "timestamp_utc": self._timestamp(),
            "reservation_id": reservation.reservation_id,
            "reserved_usd": reservation.reserved_usd,
            "model": self.model,
            "run_kind": os.environ.get("LLM_PGG_RUN_KIND", "unspecified"),
            "input_token_upper_bound": input_upper,
            "output_token_upper_bound": output_upper,
            **self.context,
        }
        self._append_locked(event, check_amount=reserved)
        return reservation

    def _cost_from_usage(self, usage: dict[str, Any]) -> tuple[float, dict[str, int]]:
        prompt_tokens = _nonnegative_int(usage.get("prompt_tokens"))
        output_tokens = _nonnegative_int(usage.get("completion_tokens"))
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        cached_tokens = _nonnegative_int(prompt_details.get("cached_tokens"))
        cache_write_tokens = _nonnegative_int(
            prompt_details.get("cache_write_tokens")
        )
        uncached_tokens = max(prompt_tokens - cached_tokens - cache_write_tokens, 0)
        reasoning_tokens = _nonnegative_int(completion_details.get("reasoning_tokens"))
        cost = (
            uncached_tokens * self.pricing["input"]
            + cached_tokens * self.pricing["cached_input"]
            + cache_write_tokens * self.pricing["cache_write"]
            + output_tokens * self.pricing["output"]
        ) / 1_000_000
        normalized = {
            "input_tokens": prompt_tokens,
            "uncached_input_tokens": uncached_tokens,
            "cached_input_tokens": cached_tokens,
            "cache_write_tokens": cache_write_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": _nonnegative_int(usage.get("total_tokens")),
        }
        return cost, normalized

    def settle_success(
        self,
        reservation: Reservation,
        response_data: dict[str, Any],
        *,
        api_retry_count: int = 0,
    ) -> dict[str, Any]:
        usage = response_data.get("usage")
        tier = str(response_data.get("service_tier") or "")
        if not isinstance(usage, dict):
            cost = reservation.reserved_usd
            normalized: dict[str, Any] = {}
            status = "missing_usage_conservative_reservation"
        else:
            cost, normalized = self._cost_from_usage(usage)
            status = "settled"
        if self.required_service_tier is not None and tier != self.required_service_tier:
            # An unexpected tier could carry a higher tariff. Retain the full
            # conservative reservation rather than undercounting it at Flex.
            cost = max(cost, reservation.reserved_usd)
            status = "unexpected_tier_conservative_reservation"

        record = {
            "event": "settle",
            "timestamp_utc": self._timestamp(),
            "reservation_id": reservation.reservation_id,
            "reserved_usd": reservation.reserved_usd,
            "cost_usd": cost,
            "api_retry_count": max(int(api_retry_count), 0),
            "billing_status": status,
            "input_token_upper_bound": reservation.input_token_upper_bound,
            "output_token_upper_bound": reservation.output_token_upper_bound,
            "pricing_verified_date": self.pricing_verified_date,
            "input_price_per_million_usd": self.pricing["input"],
            "cached_input_price_per_million_usd": self.pricing["cached_input"],
            "cache_write_price_per_million_usd": self.pricing["cache_write"],
            "output_price_per_million_usd": self.pricing["output"],
            "required_service_tier": self.required_service_tier,
            "max_usd": self.max_usd,
            "model": str(response_data.get("model") or self.model),
            "configured_model": self.model,
            "service_tier": tier,
            "response_id": response_data.get("id"),
            "system_fingerprint": response_data.get("system_fingerprint"),
            "run_kind": os.environ.get("LLM_PGG_RUN_KIND", "unspecified"),
            **self.context,
            **normalized,
        }
        self._append_locked(record)
        if status == "missing_usage_conservative_reservation":
            raise RuntimeError(
                "Paid API response omitted token usage; the full reservation was charged "
                "to the local ledger and execution stopped."
            )
        if self.required_service_tier is not None and tier != self.required_service_tier:
            raise RuntimeError(
                f"Expected service_tier={self.required_service_tier!r}, got {tier!r}; "
                "the response was recorded and execution stopped."
            )
        return {key: value for key, value in record.items() if key != "event"}

    def settle_error(
        self,
        reservation: Reservation,
        *,
        ambiguous_billing: bool,
        error: str,
        api_retry_count: int = 0,
    ) -> None:
        self._append_locked(
            {
                "event": "settle",
                "timestamp_utc": self._timestamp(),
                "reservation_id": reservation.reservation_id,
                "reserved_usd": reservation.reserved_usd,
                "cost_usd": reservation.reserved_usd if ambiguous_billing else 0.0,
                "billing_status": (
                    "uncertain_conservative_reservation"
                    if ambiguous_billing
                    else "http_error_not_charged"
                ),
                "api_retry_count": max(int(api_retry_count), 0),
                "error": error,
                "model": self.model,
                "configured_model": self.model,
                "run_kind": os.environ.get("LLM_PGG_RUN_KIND", "unspecified"),
                "input_token_upper_bound": reservation.input_token_upper_bound,
                "output_token_upper_bound": reservation.output_token_upper_bound,
                "pricing_verified_date": self.pricing_verified_date,
                "required_service_tier": self.required_service_tier,
                "max_usd": self.max_usd,
                **self.context,
            }
        )
