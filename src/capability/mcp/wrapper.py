from __future__ import annotations

from typing import Any

from src.capability.models import ToolDefinition, ToolResult
from src.capability.mcp.client import McpClient
from src.capability.mcp.models import McpToolDescriptor
from src.capability.tools.base import Tool


class McpToolWrapper(Tool):
    """Wrap one MCP-discovered tool as an Orbit2 `Tool`.

    The wrapper is the Capability Surface seam for MCP-backed capabilities:
    the same `CapabilityRegistry` + `CapabilityBoundary` pair governs both
    native and MCP tools. MCP execution never bypasses Orbit2's attachment
    path — every call enters through `CapabilityBoundary.execute`, has its
    arguments validated against the schema, and returns a `CapabilityResult`.

    First-slice governance posture: MCP tools are opaque by default. Side
    effects and approval requirements are not inferrable from the MCP
    descriptor alone, so the wrapper declares `side_effect_class="unknown"`
    and `requires_approval=True` as a conservative default. These are
    metadata-only until the Governance Surface approval-gate slice wires a
    consumer — same posture as Handoff 09's deferred HIGH-1 finding.
    """

    def __init__(self, *, descriptor: McpToolDescriptor, client: McpClient) -> None:
        self._descriptor = descriptor
        self._client = client

    @property
    def descriptor(self) -> McpToolDescriptor:
        return self._descriptor

    @property
    def definition(self) -> ToolDefinition:
        parameters = self._descriptor.input_schema or {
            "type": "object",
            "properties": {},
        }
        description = self._descriptor.description or (
            f"MCP tool {self._descriptor.original_name!r} "
            f"from server {self._descriptor.server_name!r}."
        )
        return ToolDefinition(
            name=self._descriptor.orbit_tool_name,
            description=description,
            parameters=parameters,
        )

    @property
    def side_effect_class(self) -> str:
        return "unknown"

    @property
    def requires_approval(self) -> bool:
        return True

    @property
    def environment_check_kind(self) -> str:
        return "none"

    def execute(self, **kwargs: Any) -> ToolResult:
        return self._client.call_tool(self._descriptor.original_name, kwargs)
