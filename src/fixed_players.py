from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional

from src.agents import Decision


POLICIES = {"always_free_ride", "always_cooperate", "random", "alternating"}
PLACEMENTS = {"lowest_player_ids", "rotating_player_ids", "randomized_player_ids"}
RO_GAMES = {"ro_public_good", "ro_bccm", "ro_ccm", "ro_ccf", "ro_sccm", "ro_vcm"}
MECHANISM_ALIASES = {
    "bccm": "bccm",
    "ccm": "ccm",
    "ccf": "ccf",
    "sccm": "sccm",
    "vcm": "vcm",
    "ro_bccm": "bccm",
    "ro_ccm": "ccm",
    "ro_ccf": "ccf",
    "ro_sccm": "sccm",
    "ro_vcm": "vcm",
}


@dataclass(frozen=True)
class FixedPlayerConfig:
    count: int = 0
    policy: Optional[str] = None
    placement: str = "lowest_player_ids"
    total_players: int = 5

    @property
    def enabled(self) -> bool:
        return self.count > 0

    @property
    def row_policy(self) -> str:
        return self.policy if self.enabled and self.policy is not None else "none"

    def player_ids(self, repetition: int = 1, chain: int = 1, seed: Optional[int] = None) -> set[int]:
        if not self.enabled:
            return set()
        ids = list(range(1, self.total_players + 1))
        if self.placement == "lowest_player_ids":
            return set(ids[: self.count])
        if self.placement == "rotating_player_ids":
            offset = _stable_rotation_offset(seed, repetition, chain, self.total_players)
            rotated = ids[offset:] + ids[:offset]
            return set(rotated[: self.count])
        if self.placement == "randomized_player_ids":
            return set(_stable_random_player_sample(ids, self.count, seed, repetition, chain))
        raise ValueError(f"Unsupported fixed player placement: {self.placement}")


def normalize_mechanism_name(mechanism_name: str) -> str:
    mechanism = MECHANISM_ALIASES.get(mechanism_name)
    if mechanism is None:
        raise ValueError(
            f"Fixed players are only supported for RO public-good mechanisms: "
            f"{sorted(MECHANISM_ALIASES)}"
        )
    return mechanism


def parse_fixed_player_config(
    raw: Optional[dict[str, Any]],
    *,
    total_players: int,
    game_name: str,
    mechanism_name: str,
) -> FixedPlayerConfig:
    if raw is None:
        return FixedPlayerConfig()
    if not isinstance(raw, dict):
        raise ValueError("fixed_players must be a mapping with count, policy, and placement fields.")

    count = int(raw.get("count", 0) or 0)
    if count < 0 or count > 4:
        raise ValueError("fixed_players.count must be an integer in 0..4.")
    if count > total_players:
        raise ValueError("fixed_players.count cannot exceed players_per_generation.")

    placement = raw.get("placement", "lowest_player_ids")
    if placement not in PLACEMENTS:
        raise ValueError(f"Unsupported fixed_players.placement: {placement}. Available: {sorted(PLACEMENTS)}")

    policy = raw.get("policy")
    if count == 0:
        return FixedPlayerConfig(count=0, policy=None, placement=placement, total_players=total_players)
    if game_name not in RO_GAMES:
        raise ValueError("fixed_players are only supported for RO public-good games.")
    normalize_mechanism_name(mechanism_name)
    if policy not in POLICIES:
        raise ValueError(f"Unsupported fixed_players.policy: {policy}. Available: {sorted(POLICIES)}")
    return FixedPlayerConfig(count=count, policy=policy, placement=placement, total_players=total_players)


def _stable_rotation_offset(seed: Optional[int], repetition: int, chain: int, total_players: int) -> int:
    payload = f"rotate|{seed}|{repetition}|{chain}|{total_players}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % total_players


def _stable_random_player_sample(
    player_ids: list[int],
    count: int,
    seed: Optional[int],
    repetition: int,
    chain: int,
) -> list[int]:
    ranked = []
    for player in player_ids:
        payload = f"sample|{seed}|{repetition}|{chain}|{player}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        ranked.append((digest, player))
    ranked.sort()
    return [player for _, player in ranked[:count]]


def decision_for_fixed_policy(
    *,
    policy: str,
    mechanism_name: str,
    generation: int,
    player: int,
    repetition: int,
    chain: int,
    seed: Optional[int],
) -> Decision:
    action = effective_fixed_action(
        policy=policy,
        mechanism_name=mechanism_name,
        generation=generation,
        player=player,
        repetition=repetition,
        chain=chain,
        seed=seed,
    )
    return _decision_for_action(action, normalize_mechanism_name(mechanism_name), policy)


def effective_fixed_action(
    *,
    policy: str,
    mechanism_name: str,
    generation: int,
    player: int,
    repetition: int,
    chain: int,
    seed: Optional[int],
) -> str:
    if policy == "always_free_ride":
        return "always_free_ride"
    if policy == "always_cooperate":
        return "always_cooperate"
    if policy == "alternating":
        return "always_cooperate" if generation % 2 == 1 else "always_free_ride"
    if policy == "random":
        bit = _stable_random_bit(seed, repetition, chain, generation, player, mechanism_name)
        return "always_cooperate" if bit else "always_free_ride"
    raise ValueError(f"Unsupported fixed policy: {policy}")


def _stable_random_bit(
    seed: Optional[int],
    repetition: int,
    chain: int,
    generation: int,
    player: int,
    mechanism_name: str,
) -> int:
    payload = f"{seed}|{repetition}|{chain}|{generation}|{player}|{mechanism_name}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % 2


def _decision_for_action(action: str, mechanism: str, policy: str) -> Decision:
    if mechanism == "vcm":
        choice = 10 if action == "always_cooperate" else 0
        payload = {"fixed_policy": policy, "effective_action": action, "choice": choice}
        return Decision(
            choice=choice,
            prediction=None,
            raw_response=json.dumps(payload, sort_keys=True),
            reason=f"fixed policy: {policy}",
            parser_mode="fixed_policy",
        )

    if mechanism == "bccm":
        condition = 0 if action == "always_cooperate" else 5
        payload = {"fixed_policy": policy, "effective_action": action, "condition": condition}
        return Decision(
            choice=condition,
            prediction=None,
            raw_response=json.dumps(payload, sort_keys=True),
            reason=f"fixed policy: {policy}",
            condition=condition,
            parser_mode="fixed_policy",
        )

    if mechanism == "sccm":
        contribution = 10 if action == "always_cooperate" else 0
        payload = {
            "fixed_policy": policy,
            "effective_action": action,
            "contribution_1": contribution,
            "threshold_1": 0,
        }
        return Decision(
            choice=0,
            prediction=None,
            raw_response=json.dumps(payload, sort_keys=True),
            reason=f"fixed policy: {policy}",
            contribution_1=contribution,
            threshold_1=0,
            parser_mode="fixed_policy",
        )

    if mechanism == "ccm":
        contribution = 10 if action == "always_cooperate" else 0
        payload = {
            "fixed_policy": policy,
            "effective_action": action,
            "contribution_1": contribution,
            "threshold_1": 0,
            "contribution_2": contribution,
            "threshold_2": 0,
        }
        return Decision(
            choice=0,
            prediction=None,
            raw_response=json.dumps(payload, sort_keys=True),
            reason=f"fixed policy: {policy}",
            contribution_1=contribution,
            threshold_1=0,
            contribution_2=contribution,
            threshold_2=0,
            parser_mode="fixed_policy",
        )

    if mechanism == "ccf":
        contribution = 10 if action == "always_cooperate" else 0
        source = f"def ccf(others_total):\n    return {contribution}\n"
        payload = {
            "fixed_policy": policy,
            "effective_action": action,
            "ccf_source": source,
        }
        return Decision(
            choice=0,
            prediction=None,
            raw_response=json.dumps(payload, sort_keys=True),
            reason=f"fixed policy: {policy}",
            ccf_source=source,
            parser_mode="fixed_policy",
        )

    raise ValueError(f"Unsupported fixed-player mechanism: {mechanism}")
