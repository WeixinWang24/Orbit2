from __future__ import annotations

from pathlib import Path

from src.capability.models import CapabilityResult, GovernanceOutcome, ToolDefinition
from src.capability.registry import CapabilityRegistry
from src.capability.tools import Tool
from src.runtime.models import ToolRequest


class CapabilityBoundary:
    def __init__(
        self,
        registry: CapabilityRegistry,
        workspace_root: Path,
    ) -> None:
        self._registry = registry
        self._workspace_root = workspace_root.resolve()

    @property
    def registry(self) -> CapabilityRegistry:
        return self._registry

    def list_definitions(self) -> list[ToolDefinition]:
        return self._registry.list_definitions()

    def execute(self, request: ToolRequest) -> CapabilityResult:
        tool = self._registry.get(request.tool_name)
        if tool is None:
            available = ", ".join(self._registry.list_names()) or "(none)"
            return CapabilityResult(
                tool_call_id=request.tool_call_id,
                tool_name=request.tool_name,
                ok=False,
                content=f"unknown tool: {request.tool_name}. available: {available}",
                governance_outcome="denied_unknown_tool",
            )

        outcome = self._govern(tool, request)
        if not outcome.allowed:
            return CapabilityResult(
                tool_call_id=request.tool_call_id,
                tool_name=request.tool_name,
                ok=False,
                content=f"governance denied: {outcome.reason}",
                governance_outcome=f"denied: {outcome.reason}",
            )

        # Validate arguments against declared parameter schema
        arg_error = self._validate_arguments(tool, request)
        if arg_error is not None:
            return CapabilityResult(
                tool_call_id=request.tool_call_id,
                tool_name=request.tool_name,
                ok=False,
                content=f"argument validation failed: {arg_error}",
                governance_outcome="denied_invalid_arguments",
            )

        result = tool.execute(**request.arguments)
        return CapabilityResult(
            tool_call_id=request.tool_call_id,
            tool_name=request.tool_name,
            ok=result.ok,
            content=result.content,
            data=result.data,
            governance_outcome="allowed",
        )

    def _govern(self, tool: Tool, request: ToolRequest) -> GovernanceOutcome:
        path_arg = request.arguments.get("path")
        if path_arg is not None:
            target = (self._workspace_root / path_arg).resolve()
            try:
                target.relative_to(self._workspace_root)
            except ValueError:
                return GovernanceOutcome(
                    allowed=False,
                    reason="path escapes workspace boundary",
                )
        return GovernanceOutcome(allowed=True)

    @staticmethod
    def _validate_arguments(tool: Tool, request: ToolRequest) -> str | None:
        schema = tool.definition.parameters
        if not schema:
            return None

        declared_props = set(schema.get("properties", {}).keys())
        required = set(schema.get("required", []))
        provided = set(request.arguments.keys())

        missing = required - provided
        if missing:
            return f"missing required arguments: {', '.join(sorted(missing))}"

        unexpected = provided - declared_props
        if unexpected:
            return f"unexpected arguments: {', '.join(sorted(unexpected))}"

        return None
