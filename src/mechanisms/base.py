from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Mechanism(ABC):
    name: str

    @abstractmethod
    def augment_observation(self, observation: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        pass

    @abstractmethod
    def augment_prompt(self, prompt: str, observation: dict[str, Any]) -> str:
        pass

    @abstractmethod
    def resolve(self, decisions: list[Any], context: dict[str, Any]) -> dict[str, Any]:
        pass
