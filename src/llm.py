import json
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Protocol

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python < 3.11

import requests
from src.api_cost import APICostGuard

_DEFAULT = object()
_QWEN_THINKING_STOP = (
    "\n\nConsidering the limited time by the user, I have to give the solution "
    "based on the thinking directly now."
)
_ORS_PERIOD_BOUNDARY = re.compile(r"(?m)^[ \t]*Period \d+ out of \d+\.")


@lru_cache(maxsize=4)
def _load_tokenizer(name_or_path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name_or_path)


def _without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _strip_trailing_slash(value: str) -> str:
    return value.rstrip("/")


def _messages_with_explicit_prompt_cache(
    messages: list[dict[str, str]], extra_body: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Mark only the stable ORS prefix when explicit caching is configured."""
    cache_options = (extra_body or {}).get("prompt_cache_options")
    if not isinstance(cache_options, dict) or cache_options.get("mode") != "explicit":
        return messages

    for index, message in enumerate(messages):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, str):
            continue
        boundary = _ORS_PERIOD_BOUNDARY.search(content)
        if boundary is None or boundary.start() == 0:
            continue
        split_messages: list[dict[str, Any]] = [dict(item) for item in messages]
        split_messages[index]["content"] = [
            {
                "type": "text",
                "text": content[: boundary.start()],
                "prompt_cache_breakpoint": {"mode": "explicit"},
            },
            {"type": "text", "text": content[boundary.start() :]},
        ]
        return split_messages
    return messages


@dataclass(frozen=True)
class ChatResponse:
    content: str
    thinking: str | None = None
    api_usage: dict[str, Any] | None = None


class ChatClient(Protocol):
    def check_ready(self) -> None: ...

    def chat(
        self,
        messages: list[dict[str, str]],
        seed: int | None = None,
        response_format: str | None = None,
        include_thinking: bool = False,
        think: bool | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str | ChatResponse: ...


@lru_cache(maxsize=1)
def global_model_config() -> dict[str, Any]:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.exists():
        return {}

    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    model = data.get("tool", {}).get("llm-pgg", {}).get("model", {})
    return model if isinstance(model, dict) else {}


def _merged_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    return {**global_model_config(), **model_config}


def build_llm_client(
    model_config: dict[str, Any],
    timeout: int | None = None,
    temperature: Any = _DEFAULT,
    preflight: bool = False,
) -> ChatClient:
    """Build the configured chat client.

    Supported providers:
    - ollama: native Ollama /api/chat endpoint.
    - openai_compatible: /v1/chat/completions endpoint used by vLLM,
      SGLang, TGI, OpenAI-compatible gateways, and OpenAI itself.
    """
    model_config = _merged_model_config(model_config)
    provider = str(model_config.get("provider", "ollama")).lower()

    common_timeout = timeout if timeout is not None else model_config.get("timeout", 120)
    common_temperature = (
        model_config.get("temperature") if temperature is _DEFAULT else temperature
    )

    if provider in {"ollama", "native_ollama"}:
        client: ChatClient = OllamaClient(
            base_url=model_config.get("base_url", ""),
            model=model_config["name"],
            timeout=common_timeout,
            temperature=common_temperature,
            seed=model_config.get("seed"),
            think=model_config.get("think"),
            num_predict=model_config.get("num_predict"),
        )
    elif provider in {"openai", "openai_compatible", "vllm", "sglang", "tgi"}:
        api_key = model_config.get("api_key", "EMPTY")
        api_key_env = model_config.get("api_key_env")
        if api_key_env:
            api_key = os.environ.get(str(api_key_env), "")
            if not api_key:
                raise ValueError(
                    f"Required API key environment variable {api_key_env!r} is not set"
                )
        client = OpenAICompatibleClient(
            base_url=model_config.get("base_url", "http://127.0.0.1:8000/v1"),
            model=model_config["name"],
            api_key=api_key,
            timeout=common_timeout,
            temperature=common_temperature,
            seed=model_config.get("seed"),
            max_tokens=model_config.get("max_tokens", model_config.get("num_predict")),
            extra_body=model_config.get("extra_body"),
            json_response_format=model_config.get("json_response_format", True),
            endpoint_mode=model_config.get("endpoint_mode", "chat"),
            response_format_mode=model_config.get("response_format_mode", "json_object"),
            thinking_budget=model_config.get("thinking_budget"),
            tokenizer_name_or_path=model_config.get("tokenizer_name_or_path"),
            response_format_fallback=model_config.get(
                "response_format_fallback", True
            ),
            resource_unavailable_retries=int(
                model_config.get("resource_unavailable_retries", 0)
            ),
            resource_unavailable_initial_delay=float(
                model_config.get("resource_unavailable_initial_delay", 1.0)
            ),
            resource_unavailable_max_delay=float(
                model_config.get("resource_unavailable_max_delay", 60.0)
            ),
            resource_unavailable_jitter=float(
                model_config.get("resource_unavailable_jitter", 0.25)
            ),
            cost_guard_config=model_config.get("cost_guard"),
        )
    else:
        raise ValueError(
            f"Unsupported model provider {provider!r}. Use 'ollama' or 'openai_compatible'."
        )

    if preflight:
        client.check_ready()
    return client


def build_ollama_client(
    model_config: dict[str, Any],
    timeout: int | None = None,
    temperature: Any = _DEFAULT,
    preflight: bool = False,
) -> "OllamaClient":
    """Backward-compatible constructor for older code paths."""
    client = build_llm_client(
        {**model_config, "provider": "ollama"},
        timeout=timeout,
        temperature=temperature,
        preflight=preflight,
    )
    if not isinstance(client, OllamaClient):
        raise TypeError("Expected OllamaClient")
    return client


@dataclass
class OllamaClient:
    base_url: str
    model: str
    timeout: int = 120
    temperature: float | None = None
    seed: int | None = None
    think: bool | None = None
    num_predict: int | None = None

    def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> requests.Response:
        try:
            response = requests.request(
                method, f"{_strip_trailing_slash(self.base_url)}{path}", **kwargs
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Ollama error at {self.base_url}. Check model.base_url or pyproject.toml. Original error: {exc}"
            ) from exc

    def check_ready(self) -> None:
        self._request("get", "/api/tags", timeout=min(self.timeout, 10))

    @staticmethod
    def _parse_chat_json(data: dict[str, Any]) -> ChatResponse:
        try:
            message = data["message"]
        except KeyError as exc:
            raise RuntimeError("Unexpected Ollama response: missing message payload") from exc
        if not isinstance(message, dict):
            raise RuntimeError("Unexpected Ollama response: message payload is not an object")

        content = message.get("content", "")
        thinking = message.get("thinking")
        return ChatResponse(
            content="" if content is None else str(content),
            thinking=None if thinking in (None, "") else str(thinking),
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        seed: int | None = None,
        response_format: str | None = None,
        include_thinking: bool = False,
        think: bool | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str | ChatResponse:
        options = _without_none(
            {
                "temperature": self.temperature,
                "seed": self.seed if seed is None else seed,
                "num_predict": self.num_predict,
            }
        )
        payload = _without_none(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": options,
                "think": self.think if think is None else think,
                "format": response_format,
            }
        )
        response = self._request(
            "post", "/api/chat", json=payload, timeout=self.timeout
        )

        try:
            parsed = self._parse_chat_json(response.json())
            return parsed if include_thinking else parsed.content
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Unexpected Ollama response from {self.base_url}/api/chat"
            ) from exc


@dataclass
class OpenAICompatibleClient:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout: int = 120
    temperature: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    extra_body: dict[str, Any] | None = None
    json_response_format: bool = True
    endpoint_mode: str = "chat"
    response_format_mode: str = "json_object"
    thinking_budget: int | None = None
    tokenizer_name_or_path: str | None = None
    response_format_fallback: bool = True
    resource_unavailable_retries: int = 0
    resource_unavailable_initial_delay: float = 1.0
    resource_unavailable_max_delay: float = 60.0
    resource_unavailable_jitter: float = 0.25
    cost_guard_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self._cost_guard = (
            APICostGuard(self.cost_guard_config, self.model)
            if self.cost_guard_config
            else None
        )

    def _json_response_format_payload(
        self,
        json_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        mode = self.response_format_mode
        if json_schema and mode == "json_object":
            mode = "json_schema"
        if mode == "json_schema" and json_schema:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "decision",
                    "schema": json_schema,
                    "strict": True,
                },
            }
        return {"type": "json_object"}

    @staticmethod
    def _http_error_details(response: requests.Response | None) -> dict[str, Any]:
        if response is None:
            return {}
        details: dict[str, Any] = {
            "status": response.status_code,
            "request_id": response.headers.get("x-request-id"),
            "retry_after": response.headers.get("Retry-After"),
        }
        try:
            payload = response.json()
        except ValueError:
            payload = None
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            details.update(
                {
                    "error_code": error.get("code"),
                    "error_type": error.get("type"),
                    "error_message": str(error.get("message") or "")[:1000],
                }
            )
        else:
            details["response_body"] = str(response.text or "")[:1000]
        return {key: value for key, value in details.items() if value not in (None, "")}

    @classmethod
    def _retryable_http_error(cls, response: requests.Response) -> bool:
        status = response.status_code

        # Match the transient HTTP errors retried by the OpenAI SDK.
        if status in {408, 409} or status >= 500:
            return True

        if status != 429:
            return False

        details = cls._http_error_details(response)
        code = str(details.get("error_code") or "").lower()
        message = str(details.get("error_message") or "").lower()
        nonretryable = (
            "insufficient_quota",
            "billing_hard_limit_reached",
            "current quota",
            "billing",
        )
        return not any(marker in code or marker in message for marker in nonretryable)

    def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> requests.Response:
        headers = kwargs.pop("headers", {}) or {}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **headers,
        }
        payload = kwargs.get("json")
        reservation = (
            self._cost_guard.reserve(payload)
            if self._cost_guard is not None and isinstance(payload, dict)
            else None
        )
        attempts = max(int(self.resource_unavailable_retries), 0) + 1
        attempt = 0
        try:
            for attempt in range(attempts):
                response = requests.request(
                    method,
                    f"{_strip_trailing_slash(self.base_url)}{path}",
                    headers=headers,
                    **kwargs,
                )
                if self._retryable_http_error(response) and attempt + 1 < attempts:
                    retry_after = response.headers.get("Retry-After")
                    base_delay = self.resource_unavailable_initial_delay * (2**attempt)
                    try:
                        delay = float(retry_after) if retry_after else base_delay
                    except ValueError:
                        delay = base_delay
                    delay = min(max(delay, 0.0), self.resource_unavailable_max_delay)
                    if not retry_after and self.resource_unavailable_jitter > 0:
                        jitter = min(max(self.resource_unavailable_jitter, 0.0), 1.0)
                        delay *= random.uniform(1.0 - jitter, 1.0 + jitter)
                    details = json.dumps(
                        self._http_error_details(response), sort_keys=True
                    )
                    print(
                        f"Retryable OpenAI HTTP ({attempt + 1}/{attempts}); "
                        f"sleeping {delay:.1f}s; {details}",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                if reservation is not None and self._cost_guard is not None:
                    try:
                        response_data = response.json()
                    except ValueError as exc:
                        self._cost_guard.settle_error(
                            reservation,
                            ambiguous_billing=True,
                            error=f"invalid success JSON: {exc}",
                            api_retry_count=attempt,
                        )
                        raise RuntimeError(
                            "Paid API returned invalid JSON; reservation retained"
                        ) from exc
                    response._llm_pgg_api_usage = self._cost_guard.settle_success(  # type: ignore[attr-defined]
                        reservation, response_data, api_retry_count=attempt
                    )
                return response
            raise AssertionError("unreachable request retry loop")
        except requests.HTTPError as exc:
            details = json.dumps(
                self._http_error_details(exc.response), sort_keys=True
            )
            error = f"{exc}; response_details={details}"
            if reservation is not None and self._cost_guard is not None:
                self._cost_guard.settle_error(
                    reservation,
                    ambiguous_billing=False,
                    error=error,
                    api_retry_count=attempt,
                )
            raise RuntimeError(
                f"OpenAI-compatible API HTTP error at {self.base_url}. {error}"
            ) from exc
        except requests.RequestException as exc:
            if reservation is not None and self._cost_guard is not None:
                self._cost_guard.settle_error(
                    reservation,
                    ambiguous_billing=True,
                    error=str(exc),
                    api_retry_count=attempt,
                )
            raise RuntimeError(
                f"OpenAI-compatible API transport error at {self.base_url}. Original error: {exc}"
            ) from exc

    def check_ready(self) -> None:
        self._request("get", "/models", timeout=min(self.timeout, 10))
        if self.thinking_budget is not None:
            self._validate_thinking_budget()
            _load_tokenizer(str(self.tokenizer_name_or_path))

    def _validate_thinking_budget(self) -> None:
        if not self.tokenizer_name_or_path:
            raise ValueError("tokenizer_name_or_path is required with thinking_budget")
        if self.max_tokens is None or self.thinking_budget is None:
            raise ValueError("max_tokens and thinking_budget must both be configured")
        if self.thinking_budget <= 0 or self.max_tokens <= self.thinking_budget:
            raise ValueError(
                "max_tokens must be greater than a positive thinking_budget"
            )

    @staticmethod
    def _parse_chat_json(
        data: dict[str, Any], api_usage: dict[str, Any] | None = None
    ) -> ChatResponse:
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Unexpected OpenAI-compatible response: missing choices[0].message payload"
            ) from exc
        if not isinstance(message, dict):
            raise RuntimeError(
                "Unexpected OpenAI-compatible response: message payload is not an object"
            )

        content = message.get("content", "")
        thinking = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or message.get("thinking")
        )
        raw_usage = data.get("usage") or {}
        return ChatResponse(
            content="" if content is None else str(content),
            thinking=None if thinking in (None, "") else str(thinking),
            api_usage=api_usage or {
                "input_tokens": int(raw_usage.get("prompt_tokens") or 0),
                "output_tokens": int(raw_usage.get("completion_tokens") or 0),
                "total_tokens": int(raw_usage.get("total_tokens") or 0),
                "response_id": data.get("id"),
                "response_model": data.get("model"),
                "service_tier": data.get("service_tier"),
                "system_fingerprint": data.get("system_fingerprint"),
            },
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        seed: int | None = None,
        response_format: str | None = None,
        include_thinking: bool = False,
        think: bool | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str | ChatResponse:
        if self.thinking_budget is not None:
            return self._thinking_budget_chat(
                messages=messages,
                seed=seed,
                response_format=response_format,
                include_thinking=include_thinking,
                think=think,
                json_schema=json_schema,
            )
        if self.endpoint_mode == "completion":
            return self._completion_chat(
                messages=messages,
                seed=seed,
                response_format=response_format,
                include_thinking=include_thinking,
                json_schema=json_schema,
            )

        payload = _without_none(
            {
                "model": self.model,
                "messages": _messages_with_explicit_prompt_cache(
                    messages, self.extra_body
                ),
                "temperature": self.temperature,
                "seed": self.seed if seed is None else seed,
                "max_tokens": self.max_tokens,
            }
        )

        if response_format == "json" and self.json_response_format:
            payload["response_format"] = self._json_response_format_payload(json_schema)

        if think is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": think}

        if self.extra_body:
            payload.update(self.extra_body)
        try:
            response = self._request(
                "post", "/chat/completions", json=payload, timeout=self.timeout
            )
        except RuntimeError as exc:
            if (
                response_format == "json"
                and self.json_response_format
                and self.response_format_fallback
            ):
                fallback_payload = dict(payload)
                fallback_payload.pop("response_format", None)
                response = self._request(
                    "post", "/chat/completions", json=fallback_payload, timeout=self.timeout
                )
            else:
                raise exc

        try:
            parsed = self._parse_chat_json(
                response.json(), getattr(response, "_llm_pgg_api_usage", None)
            )
            return parsed if include_thinking else parsed.content
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Unexpected OpenAI-compatible response from {self.base_url}/chat/completions"
            ) from exc

    def _thinking_budget_chat(
        self,
        messages: list[dict[str, str]],
        seed: int | None,
        response_format: str | None,
        include_thinking: bool,
        think: bool | None,
        json_schema: dict[str, Any] | None,
    ) -> str | ChatResponse:
        self._validate_thinking_budget()
        payload = _without_none(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "seed": self.seed if seed is None else seed,
                "max_tokens": self.thinking_budget,
            }
        )
        if response_format == "json" and self.json_response_format:
            payload["response_format"] = self._json_response_format_payload(json_schema)
        if think is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": think}
        if self.extra_body:
            payload.update(self.extra_body)

        response = self._request(
            "post", "/chat/completions", json=payload, timeout=self.timeout
        )
        parsed = self._parse_chat_json(response.json())
        raw_reasoning = (parsed.thinking or "").strip("\n")
        reasoning = raw_reasoning
        if not parsed.content:
            reasoning += _QWEN_THINKING_STOP

        tokenizer = _load_tokenizer(str(self.tokenizer_name_or_path))
        reasoning_tokens = len(tokenizer.encode(reasoning, add_special_tokens=False))
        remaining_tokens = int(self.max_tokens) - reasoning_tokens
        if remaining_tokens <= 0:
            raise RuntimeError(
                "Qwen thinking continuation has no final-answer budget; increase "
                "max_tokens or lower thinking_budget"
            )

        continued_messages = [
            *messages,
            {
                "role": "assistant",
                "content": f"<think>\n{reasoning}\n</think>\n\n",
            },
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                continued_messages,
                tokenize=False,
                continue_final_message=True,
            )
        except ValueError as exc:
            if "continue_final_message is set" not in str(exc):
                raise
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if str(prompt).rstrip().endswith("<think>"):
                prompt = f"{prompt}{reasoning}\n</think>\n\n"
            else:
                prompt = f"{prompt}<think>\n{reasoning}\n</think>\n\n"
        completion_payload = _without_none(
            {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "seed": self.seed if seed is None else seed,
                "max_tokens": remaining_tokens,
            }
        )
        if response_format == "json" and self.json_response_format:
            completion_payload["response_format"] = self._json_response_format_payload(
                json_schema
            )
        if self.extra_body:
            completion_payload.update(
                {
                    key: value
                    for key, value in self.extra_body.items()
                    if key != "chat_template_kwargs"
                }
            )
        try:
            completion = self._request(
                "post", "/completions", json=completion_payload, timeout=self.timeout
            )
        except RuntimeError as exc:
            if response_format == "json" and self.json_response_format:
                fallback_payload = dict(completion_payload)
                fallback_payload.pop("response_format", None)
                completion = self._request(
                    "post", "/completions", json=fallback_payload, timeout=self.timeout
                )
            else:
                raise exc
        try:
            content = completion.json()["choices"][0].get("text", "")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Unexpected OpenAI-compatible response from {self.base_url}/completions"
            ) from exc
        if (
            response_format == "json"
            and content is not None
            and not str(content).lstrip().startswith("{")
        ):
            content = "{" + str(content)
        result = ChatResponse(
            content="" if content is None else str(content),
            thinking=raw_reasoning or None,
        )
        return result if include_thinking else result.content


    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]], json_prefix: bool = False) -> str:
        rendered: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).strip().upper()
            content = str(message.get("content", ""))
            rendered.append(f"{role}:\n{content}".rstrip())
        rendered.append("ASSISTANT:\n{" if json_prefix else "ASSISTANT:\n")
        return "\n\n".join(rendered)

    def _completion_chat(
        self,
        messages: list[dict[str, str]],
        seed: int | None,
        response_format: str | None,
        include_thinking: bool,
        json_schema: dict[str, Any] | None,
    ) -> str | ChatResponse:
        json_prefix = response_format == "json"
        prompt = self._messages_to_prompt(messages, json_prefix=json_prefix)
        payload = _without_none(
            {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "seed": self.seed if seed is None else seed,
                "max_tokens": self.max_tokens,
            }
        )
        if response_format == "json" and self.json_response_format:
            payload["response_format"] = self._json_response_format_payload(json_schema)
        if self.extra_body:
            payload.update(self.extra_body)
        try:
            response = self._request(
                "post", "/completions", json=payload, timeout=self.timeout
            )
        except RuntimeError as exc:
            if response_format == "json" and self.json_response_format:
                fallback_payload = dict(payload)
                fallback_payload.pop("response_format", None)
                response = self._request(
                    "post", "/completions", json=fallback_payload, timeout=self.timeout
                )
            else:
                raise exc
        try:
            text = response.json()["choices"][0].get("text", "")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Unexpected OpenAI-compatible response from {self.base_url}/completions"
            ) from exc

        if json_prefix and text is not None and not str(text).lstrip().startswith("{"):
            text = "{" + str(text)
        parsed = ChatResponse(content="" if text is None else str(text), thinking=None)
        return parsed if include_thinking else parsed.content
