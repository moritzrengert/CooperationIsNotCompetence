from typing import Any, Optional

from src.scratchpad import ScratchpadComponent


class AgentComponents:
    def __init__(self, components: list[Any]):
        self._all = components
        self._enabled = [component for component in components if component.enabled]

    @classmethod
    def from_config(cls, config: Optional[dict[str, Any]]) -> "AgentComponents":
        config = config or {}
        if not isinstance(config, dict):
            raise ValueError("agent_components must be a mapping.")
        unknown = set(config) - {ScratchpadComponent.name}
        if unknown:
            raise ValueError(f"Unknown agent components: {sorted(unknown)}")
        return cls([ScratchpadComponent.from_config(config.get(ScratchpadComponent.name))])

    def configuration(self) -> dict[str, Any]:
        return {component.name: component.configuration() for component in self._all}

    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for component in self._all:
            result.update(component.metadata())
        return result

    def initial_state(self) -> dict[str, Any]:
        return {component.name: component.initial_state() for component in self._enabled}

    def augment_prompt(self, prompt: str, state: dict[str, Any]) -> str:
        for component in self._enabled:
            prompt = component.augment_prompt(prompt, state[component.name])
        return prompt

    def augment_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        for component in self._enabled:
            component.augment_schema(schema)
        return schema

    def outputs(self, parsed: dict[str, Any]) -> dict[str, object]:
        return {component.name: component.output(parsed) for component in self._enabled}

    def apply(self, state: dict[str, Any], decision: Any) -> None:
        for component in self._enabled:
            name = component.name
            state[name], log = component.update(state[name], decision.component_outputs.get(name))
            decision.component_log.update(log)

    def mark_inactive(self, decision: Any) -> None:
        for component in self._enabled:
            decision.component_log.update(component.inactive_log())
