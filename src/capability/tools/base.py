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

    @property
    def governance_path_arg_keys(self) -> tuple[str, ...] | None:
        """Argument names the CapabilityBoundary should treat as
        filesystem-path-bearing when governing this tool.

        - `None` (default) → boundary uses its own default whitelist
          (native-tool-oriented). Appropriate for native filesystem tools.
        - `()` → opt out: the tool does not have filesystem-path semantics
          on any argument, and the boundary's containment / protected-prefix
          check would produce false positives (e.g. git pathspecs).
        - a non-empty tuple → use exactly these argument names.
        """
        return None

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult: ...
