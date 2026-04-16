from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.capability.models import ToolDefinition, ToolResult


class Tool(ABC):
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition: ...

    @property
    def side_effect_class(self) -> str:
        return "safe"

    @property
    def requires_approval(self) -> bool:
        return False

    @property
    def environment_check_kind(self) -> str:
        return "none"

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult: ...
