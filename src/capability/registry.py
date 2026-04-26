from __future__ import annotations

from src.capability.models import CapabilityMetadata, ToolDefinition
from src.capability.tools import Tool


class CapabilityRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.definition.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return sorted(self._tools.keys())

    def list_definitions(self) -> list[ToolDefinition]:
        return [self._tools[name].definition for name in sorted(self._tools)]

    def list_metadata(self) -> list[CapabilityMetadata]:
        return [
            CapabilityMetadata(
                name=tool.definition.name,
                description=tool.definition.description,
                reveal_group=tool.reveal_group,
                default_exposed=tool.default_exposed,
                side_effect_class=tool.side_effect_class,
                requires_approval=tool.requires_approval,
                environment_check_kind=tool.environment_check_kind,
                capability_layer=tool.capability_layer,
            )
            for tool in (self._tools[name] for name in sorted(self._tools))
        ]
