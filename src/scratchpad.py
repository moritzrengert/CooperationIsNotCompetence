import json
from dataclasses import dataclass
from typing import Any, ClassVar, Optional


@dataclass(frozen=True)
class ScratchpadComponent:
    name: ClassVar[str] = "scratchpad"
    enabled: bool = False
    max_characters: Optional[int] = None

    @classmethod
    def from_config(cls, config: Optional[dict[str, Any]]) -> "ScratchpadComponent":
        config = config or {}
        if not isinstance(config, dict):
            raise ValueError("agent_components.scratchpad must be a mapping.")
        unknown = set(config) - {"enabled", "max_characters"}
        if unknown:
            raise ValueError(f"Unknown scratchpad settings: {sorted(unknown)}")

        enabled = config.get("enabled", False)
        limit = config.get("max_characters")
        if not isinstance(enabled, bool):
            raise ValueError("agent_components.scratchpad.enabled must be a boolean.")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError("agent_components.scratchpad.max_characters must be a positive integer.")
        if enabled and limit is None:
            raise ValueError("agent_components.scratchpad.max_characters is required when enabled.")
        return cls(enabled=enabled, max_characters=limit)

    def configuration(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "max_characters": self.max_characters}

    def metadata(self) -> dict[str, Any]:
        return {
            "scratchpad_enabled": self.enabled,
            "scratchpad_max_characters": self.max_characters,
        }

    def initial_state(self) -> str:
        return ""

    def augment_prompt(self, prompt: str, state: str) -> str:
        current = json.dumps(state, ensure_ascii=False)
        return f"""{prompt}

Private persistent scratchpad

Your scratchpad is private to you and persists to your next period. It does not directly affect the game. You may
write anything you consider useful or leave it empty. Return the complete scratchpad for the next period in the
additional JSON field \"scratchpad\". Its value must be a string of at most {self.max_characters} characters.

Current scratchpad: {current}
"""

    def augment_schema(self, schema: dict[str, Any]) -> None:
        schema["properties"][self.name] = {
            "type": "string",
            "maxLength": int(self.max_characters),
        }
        schema["required"].append(self.name)

    def output(self, parsed: dict[str, Any]) -> object:
        return parsed.get(self.name)

    def update(self, state: str, submitted: object) -> tuple[str, dict[str, Any]]:
        invalid = not isinstance(submitted, str)
        if invalid:
            after = state
            error = "scratchpad must be a string; previous scratchpad preserved"
            truncated = False
            submitted_characters = None
        else:
            limit = int(self.max_characters)
            after = submitted[:limit]
            error = None
            truncated = len(submitted) > limit
            submitted_characters = len(submitted)

        return after, {
            "scratchpad_before": state,
            "scratchpad_after": after,
            "scratchpad_invalid": invalid,
            "scratchpad_truncated": truncated,
            "scratchpad_validation_error": error,
            "scratchpad_characters_submitted": submitted_characters,
        }

    @staticmethod
    def inactive_log() -> dict[str, Any]:
        return {
            "scratchpad_before": None,
            "scratchpad_after": None,
            "scratchpad_invalid": False,
            "scratchpad_truncated": False,
            "scratchpad_validation_error": None,
            "scratchpad_characters_submitted": None,
        }
