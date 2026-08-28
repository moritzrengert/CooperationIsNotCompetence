from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Game(ABC):
    name: str

    @abstractmethod
    def observation(self, context: dict[str, Any]) -> dict[str, Any]:
        pass

    @abstractmethod
    def prompt(self, observation: dict[str, Any], reason_instruction: str) -> str:
        pass

    @abstractmethod
    def rows(self, context: dict[str, Any], decisions: list[Any], outcome: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        pass
