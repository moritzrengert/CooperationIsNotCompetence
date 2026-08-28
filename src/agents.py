import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from src.llm import ChatClient


@dataclass
class Decision:
    choice: int
    prediction: Optional[int]
    raw_response: str
    thinking: Optional[str] = None
    reason: Optional[str] = None
    condition: Optional[int] = None
    contribution_1: Optional[int] = None
    threshold_1: Optional[int] = None
    contribution_2: Optional[int] = None
    threshold_2: Optional[int] = None
    ccf_source: Optional[str] = None
    fallback_used: bool = False
    fallback_error: Optional[str] = None
    parser_mode: Optional[str] = None
    retry_count: int = 0
    component_outputs: dict[str, object] = field(default_factory=dict)
    component_log: dict[str, object] = field(default_factory=dict)
    api_usage: dict[str, object] = field(default_factory=dict)


class LLMAgent:
    def __init__(self, agent_id: str, llm: ChatClient, components: Any = None):
        self.agent_id = agent_id
        self.llm = llm
        self.components = components

    def decide(self, prompt: str, observation: dict) -> Decision:
        schema = self._decision_json_schema(observation)
        if self.components is not None:
            schema = self.components.augment_schema(schema)
        response = self.llm.chat(
            [{"role": "user", "content": prompt}],
            seed=observation.get("effective_seed"),
            response_format="json",
            include_thinking=True,
            json_schema=schema,
        )
        if isinstance(response, str):
            raw = response
            thinking = None
            api_usage = {}
        else:
            raw = getattr(response, "content", None)
            thinking = getattr(response, "thinking", None)
            api_usage = getattr(response, "api_usage", None) or {}
            if raw is None:
                raise RuntimeError("Expected chat response with content text")

        if not raw.strip():
            if thinking:
                raise ValueError("Model returned reasoning without final content")
            raise ValueError("Empty response from model")

        parse_text = raw
        raw_response = raw
        errors: list[Exception] = []
        for candidate in self._decision_candidates(parse_text, raw_response, observation):
            try:
                return self._decision_from_parsed(
                    parsed=candidate,
                    raw_response=raw_response,
                    thinking=thinking,
                    api_usage=api_usage,
                    observation=observation,
                    components=self.components,
                )
            except ValueError as exc:
                errors.append(exc)

        if errors:
            snippet = raw_response[:1200].replace("\n", "\\n")
            thinking_snippet = (thinking or "")[:600].replace("\n", "\\n")
            raise ValueError(
                f"{errors[-1]} | raw_response={snippet!r} | thinking={thinking_snippet!r}"
            )
        raise ValueError("Failed to parse model response")

    @staticmethod
    def _optional_int(value):
        return None if value is None else int(value)

    @classmethod
    def _decision_candidates(cls, parse_text: str, raw_response: str, observation: dict) -> list[dict]:
        candidates: list[dict] = []
        try:
            candidates.append(cls._parse_json(parse_text))
        except ValueError:
            pass
        heuristic = cls._extract_decision_from_text(raw_response, observation)
        if heuristic:
            heuristic["_parser_mode"] = "heuristic_extraction"
            candidates.append(heuristic)
        return candidates

    @classmethod
    def _decision_from_parsed(
        cls,
        parsed: dict,
        raw_response: str,
        thinking: Optional[str],
        observation: dict,
        components: Any = None,
        api_usage: dict[str, object] | None = None,
    ) -> Decision:
        parsed = cls._normalize_ccm_payload(parsed)
        condition = parsed.get("condition")
        condition = None if condition is None else int(condition)
        choice = int(parsed.get("choice", condition if condition is not None else 0))
        prediction = parsed.get("prediction")
        prediction = None if prediction is None else int(prediction)
        if "prediction_options" not in observation:
            # Some chat models include an unsolicited prediction field even when
            # the active mechanism does not request or use it. Treat such fields
            # as extraneous rather than failing the whole run.
            prediction = None

        contribution_1 = cls._optional_int(parsed.get("contribution_1"))
        threshold_1 = cls._optional_int(parsed.get("threshold_1"))
        contribution_2 = cls._optional_int(parsed.get("contribution_2"))
        threshold_2 = cls._optional_int(parsed.get("threshold_2"))
        ccf_source = parsed.get("ccf_source")
        ccf_source = None if ccf_source is None else str(ccf_source)
        if "choice_options" in observation and choice not in observation["choice_options"]:
            raise ValueError(f"Invalid choice: {choice}")
        if prediction is not None and "prediction_options" in observation and prediction not in observation["prediction_options"]:
            raise ValueError(f"Invalid prediction: {prediction}")
        if "condition_options" in observation and condition not in observation["condition_options"]:
            raise ValueError(f"Invalid condition: {condition}")
        if "contribution_options" in observation:
            offer_count = int(observation.get("offer_count", 2))
            contribution_values = (contribution_1, contribution_2)[:offer_count]
            for value in contribution_values:
                if value not in observation["contribution_options"]:
                    raise ValueError(f"Invalid CCM contribution: {value}")
        if "threshold_options" in observation:
            offer_count = int(observation.get("offer_count", 2))
            threshold_values = (threshold_1, threshold_2)[:offer_count]
            for value in threshold_values:
                if value not in observation["threshold_options"]:
                    raise ValueError(f"Invalid CCM threshold: {value}")
        if observation.get("ccf_source_required") and not str(ccf_source or "").strip():
            raise ValueError("Missing CCF source.")

        return Decision(
            choice=choice,
            prediction=prediction,
            raw_response=raw_response,
            thinking=thinking,
            api_usage=api_usage or {},
            reason=parsed.get("reason"),
            condition=condition,
            contribution_1=contribution_1,
            threshold_1=threshold_1,
            contribution_2=contribution_2,
            threshold_2=threshold_2,
            ccf_source=ccf_source,
            parser_mode=str(parsed.get("_parser_mode") or "unknown"),
            component_outputs=components.outputs(parsed) if components is not None else {},
        )

    @staticmethod
    def _coerce_offer_dict(value: object) -> Optional[tuple[int, int]]:
        if not isinstance(value, dict):
            return None
        contribution_keys = ("contribution", "offer", "amount", "x")
        threshold_keys = ("threshold", "condition", "minimum_total", "y")

        contribution = None
        threshold = None
        for key in contribution_keys:
            if key in value and value[key] is not None:
                contribution = int(value[key])
                break
        for key in threshold_keys:
            if key in value and value[key] is not None:
                threshold = int(value[key])
                break
        if contribution is None or threshold is None:
            return None
        return contribution, threshold

    @classmethod
    def _normalize_ccm_payload(cls, parsed: dict) -> dict:
        if not isinstance(parsed, dict):
            return parsed
        keys = ("contribution_1", "threshold_1", "contribution_2", "threshold_2")
        if all(key in parsed for key in keys):
            return parsed

        def collect_offers(value: object, found: list[tuple[int, int]]) -> None:
            if len(found) >= 2:
                return
            offer = cls._coerce_offer_dict(value)
            if offer is not None:
                found.append(offer)
                return
            if isinstance(value, dict):
                contributions = value.get("contributions")
                thresholds = value.get("thresholds")
                if isinstance(contributions, list) and isinstance(thresholds, list):
                    for contribution, threshold in zip(contributions, thresholds):
                        if contribution is None or threshold is None:
                            continue
                        found.append((int(contribution), int(threshold)))
                        if len(found) >= 2:
                            return
                for child in value.values():
                    collect_offers(child, found)
                    if len(found) >= 2:
                        return
            elif isinstance(value, list):
                for child in value:
                    collect_offers(child, found)
                    if len(found) >= 2:
                        return

        offers: list[tuple[int, int]] = []
        collect_offers(parsed, offers)

        if offers:
            normalized = dict(parsed)
            normalized["contribution_1"], normalized["threshold_1"] = offers[0]
            if len(offers) >= 2:
                normalized["contribution_2"], normalized["threshold_2"] = offers[1]
            return normalized
        return parsed

    @staticmethod
    def _parse_json(text: str) -> dict:
        stripped = text.strip()
        candidates: list[tuple[str, str]] = [(stripped, "strict_json")]
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidate = match.group(0)
            if candidate != stripped:
                candidates.append((candidate, "repaired_json"))

        start = text.find("{")
        if start != -1:
            fragment = text[start:].strip()
            missing_closing_braces = fragment.count("{") - fragment.count("}")
            if missing_closing_braces > 0:
                candidates.append((fragment + ("}" * missing_closing_braces), "repaired_json"))

        errors = []
        for candidate, parser_mode in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    parsed["_parser_mode"] = parser_mode
                return parsed
            except json.JSONDecodeError as err:
                errors.append(err)

        partial = LLMAgent._parse_partial_decision(text)
        if partial:
            partial["_parser_mode"] = "partial_field_parse"
            return partial

        if errors:
            raise ValueError(f"No valid JSON found in response: {text}") from errors[-1]
        raise ValueError(f"No JSON found in response: {text}")

    @staticmethod
    def _parse_partial_decision(text: str) -> dict:
        parsed: dict[str, object] = {}
        for key in (
            "condition",
            "choice",
            "prediction",
            "contribution_1",
            "threshold_1",
            "contribution_2",
            "threshold_2",
        ):
            match = re.search(rf'(?:\"{key}\"|\b{key}\b)\s*[:=]\s*(-?\d+)', text)
            if match:
                parsed[key] = int(match.group(1))

        reason = LLMAgent._parse_string_field(text, "reason")
        if reason is not None:
            parsed["reason"] = reason
        ccf_source = LLMAgent._parse_string_field(text, "ccf_source")
        if ccf_source is not None:
            parsed["ccf_source"] = ccf_source

        return parsed

    @staticmethod
    def _parse_string_field(text: str, key: str) -> Optional[str]:
        strict_match = re.search(
            rf'"{key}"\s*:\s*("(?:\\.|[^"\\])*")', text, flags=re.DOTALL
        )
        if strict_match:
            try:
                return json.loads(strict_match.group(1))
            except json.JSONDecodeError:
                return LLMAgent._clean_loose_string(strict_match.group(1)[1:-1])

        start_match = re.search(rf'"{key}"\s*:\s*"', text)
        if not start_match:
            return None

        start = start_match.end()
        end = len(text)
        close_brace = text.rfind("}")
        if close_brace > start:
            close_quote = text.rfind('"', start, close_brace)
            end = close_quote if close_quote > start else close_brace

        value = text[start:end].strip()
        if not value:
            return None
        return LLMAgent._clean_loose_string(value)

    @staticmethod
    def _clean_loose_string(value: str) -> str:
        return (
            value.replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t")
            .replace('\\"', '"')
            .strip()
        )

    @staticmethod
    def _extract_decision_from_text(text: str, observation: dict) -> dict:
        parsed: dict[str, object] = {}
        normalized = (
            text.replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("\u00a0", " ")
        )
        compact = " ".join(normalized.split())
        lower = compact.lower()

        if "condition_options" in observation:
            condition = LLMAgent._extract_first_int(
                compact,
                [
                    r"\bcondition\s*(?:is|=|:|should be)\s*(\d+)\b",
                    r"\bchoose\s*condition\s*(\d+)\b",
                    r"\bbest\s*condition\s*(?:is|=)\s*(\d+)\b",
                ],
            )
            if condition is None:
                condition = LLMAgent._extract_first_int(
                    compact,
                    [r"\bthe best condition is (\d+)\b", r"\bso the best condition is (\d+)\b"],
                )
            if condition is None:
                choose_matches = re.findall(r"\b(?:so|thus|therefore|then)?\s*choose\s+(\d+)\b", lower)
                if choose_matches:
                    condition = int(choose_matches[-1])
            if condition is None:
                condition_matches = re.findall(
                    r"\bcondition\s*(\d+)\s*(?:is|seems|looks|would be)?\s*(?:best|better|optimal|safest|safe)\b",
                    lower,
                )
                if condition_matches:
                    condition = int(condition_matches[-1])
            if condition is None:
                # Some models answer BCCM prompts with a policy description
                # instead of an explicit numeric condition.
                if re.search(r"\b(?:never|do not|don't)\s+(?:invest|contribute)\b", lower):
                    condition = 5
                elif (
                    re.search(r"\bcondition\s*5\b", lower)
                    and (
                        re.search(r"\b(?:best|optimal|safest|safe|dominant)\b", lower)
                        or re.search(r"\bkeep(?:ing)?\s+(?:my|your|the)?\s*10 points\b", lower)
                        or re.search(r"\bstill benefit(?:ing)? from (?:any |others'? )?contributions\b", lower)
                    )
                ):
                    condition = 5
                elif re.search(r"\b(?:always|unconditionally)\s+(?:invest|contribute)\b", lower):
                    condition = 0
                else:
                    threshold_phrases = [
                        (4, [
                            r"\bif all (?:four|4) others\b.*\b(?:contribute|contributes|invest|invests|do so|does so)\b",
                            r"\bif everyone else\b.*\b(?:contributes|invests|does so)\b",
                            r"\bif all other players\b.*\b(?:contribute|contributes|invest|invests|do so|does so)\b",
                        ]),
                        (3, [r"\bif at least (?:three|3) others\b.*\b(?:contribute|contributes|invest|invests|do so|does so)\b"]),
                        (2, [r"\bif at least (?:two|2) others\b.*\b(?:contribute|contributes|invest|invests|do so|does so)\b"]),
                        (1, [r"\bif at least (?:one|1) other(?: player)?\b.*\b(?:contribute|contributes|invest|invests|do so|does so)\b"]),
                    ]
                    for implied_condition, patterns in threshold_phrases:
                        if any(re.search(pattern, lower) for pattern in patterns):
                            condition = implied_condition
                            break
            if condition is not None:
                parsed["condition"] = condition

        if "choice_options" in observation:
            min_choice = min(observation["choice_options"])
            max_choice = max(observation["choice_options"])
            choice = LLMAgent._extract_first_int(
                compact,
                [
                    r"\bchoice\s*(?:is|=|:)\s*(-?\d+)\b",
                    r"\bchoose\s*(-?\d+)\b",
                    r"\bcontribute\s*(-?\d+)\b",
                    r"\binvest\s*(-?\d+)\b",
                    r"\breturn\s*\{?\s*\"?choice\"?\s*:\s*(-?\d+)\b",
                ],
            )
            if choice is None:
                if re.search(r"\b(do not|don't|not)\s+(invest|contribute)\b", lower) or re.search(r"\bkeep (?:my|your|the)?\s*10 points\b", lower):
                    choice = 0
                elif re.search(r"\b(invest|contribute)\b", lower):
                    choice = max_choice
            if choice is not None:
                choice = max(min_choice, min(max_choice, choice))
                parsed["choice"] = choice

        if "contribution_options" in observation and "threshold_options" in observation:
            offer_count = int(observation.get("offer_count", 2))
            tuples = [
                (int(match.group(1)), int(match.group(2)))
                for match in re.finditer(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", compact)
            ]
            if offer_count == 1 and tuples:
                parsed["contribution_1"], parsed["threshold_1"] = tuples[0]
            elif len(tuples) >= 2:
                parsed["contribution_1"], parsed["threshold_1"] = tuples[0]
                parsed["contribution_2"], parsed["threshold_2"] = tuples[1]
            else:
                offers = []
                for match in re.finditer(
                    r"(?:offer\s*\d+\s*[:=]?\s*)?(?:i\s*(?:will|would|am willing to)?\s*)?(?:contribute|give)\s*(\d+)\s*(?:points?\s*)?(?:to\s*the\s*project\s*)?(?:if|when|provided that)?\s*(?:total(?: group)? contributions?\s*)?(?:are\s*)?(?:at least\s*)?(?:>=|=|is\s+at\s+least\s+)?(\d+)?",
                    lower,
                ):
                    contribution = int(match.group(1))
                    threshold = int(match.group(2)) if match.group(2) is not None else 0
                    offers.append((contribution, threshold))
                if offer_count == 1 and offers:
                    parsed["contribution_1"], parsed["threshold_1"] = offers[0]
                elif len(offers) >= 2:
                    parsed["contribution_1"], parsed["threshold_1"] = offers[0]
                    parsed["contribution_2"], parsed["threshold_2"] = offers[1]

        if observation.get("ccf_source_required") and "ccf_source" not in parsed:
            code_block = re.search(
                r"```(?:python)?\s*(def\s+ccf\s*\(.*?)(?:```|$)",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if code_block:
                parsed["ccf_source"] = code_block.group(1).strip()
            else:
                function_match = re.search(r"(def\s+ccf\s*\(\s*others_total.*)", text, flags=re.DOTALL)
                if function_match:
                    parsed["ccf_source"] = function_match.group(1).strip()

        reason = LLMAgent._extract_reason_text(compact)
        if reason:
            parsed["reason"] = reason
        return parsed

    @staticmethod
    def _extract_first_int(text: str, patterns: list[str]) -> Optional[int]:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _extract_reason_text(text: str) -> Optional[str]:
        fragments = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
        filtered = [
            segment for segment in fragments
            if not re.search(r"\b(valid json|single json object|output only|schema|return only|we need to provide json|we must produce json)\b", segment.lower())
        ]
        if not filtered:
            return None
        return " ".join(filtered[:2])[:300].strip()

    @staticmethod
    def _decision_json_schema(observation: dict) -> dict[str, object]:
        properties: dict[str, object] = {}

        if "choice_options" in observation:
            properties["choice"] = {
                "type": "integer",
                "minimum": min(observation["choice_options"]),
                "maximum": max(observation["choice_options"]),
            }

        if "prediction_options" in observation:
            properties["prediction"] = {
                "type": "integer",
                "minimum": min(observation["prediction_options"]),
                "maximum": max(observation["prediction_options"]),
            }

        if "condition_options" in observation:
            properties["condition"] = {
                "type": "integer",
                "minimum": min(observation["condition_options"]),
                "maximum": max(observation["condition_options"]),
            }

        if "contribution_options" in observation:
            offer_count = int(observation.get("offer_count", 2))
            for key in ("contribution_1", "contribution_2")[:offer_count]:
                properties[key] = {
                    "type": "integer",
                    "minimum": min(observation["contribution_options"]),
                    "maximum": max(observation["contribution_options"]),
                }

        if "threshold_options" in observation:
            offer_count = int(observation.get("offer_count", 2))
            for key in ("threshold_1", "threshold_2")[:offer_count]:
                properties[key] = {
                    "type": "integer",
                    "minimum": min(observation["threshold_options"]),
                    "maximum": max(observation["threshold_options"]),
                }

        if observation.get("ccf_source_required"):
            properties["ccf_source"] = {"type": "string"}

        if observation.get("reason_style", "none") != "none":
            properties["reason"] = {"type": "string"}

        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }