from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.capability.models import CapabilityLayer, ToolDefinition, ToolResult


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

    @property
    def reveal_group(self) -> str:
        """Name of the progressive-exposure reveal group this tool belongs
        to. Used by the Knowledge Surface assembler to compute which tools
        are visible to the provider on a given turn. The default `"default"`
        group is always exposed when the tool also sets `default_exposed`.
        """
        return "default"

    @property
    def default_exposed(self) -> bool:
        """Whether the tool is visible to the provider without an explicit
        reveal request. When `False`, the tool is only exposed after the
        model requests its `reveal_group` through the discovery tool.

        Default `True` preserves backward compatibility with tools that
        predate progressive exposure (Handoff 19). Explicit opt-in to
        staged exposure means setting this to `False`.
        """
        return True

    @property
    def capability_layer(self) -> CapabilityLayer:
        """ADR-0013 architectural capability layer."""
        return CapabilityLayer.RAW_PRIMITIVE

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult: ...
