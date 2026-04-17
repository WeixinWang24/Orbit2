from __future__ import annotations

from pathlib import Path

from src.capability.models import (
    CapabilityResult,
    GovernanceOutcome,
    ToolDefinition,
    is_protected_relative_path,
)
from src.capability.mcp.wrapper import GOVERNANCE_DENIED_MARKER
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
        governance_outcome = "allowed"
        if result.data is not None:
            # Tool-layer governance denials (e.g. family-aware MCP wrappers)
            # surface through a data marker so the boundary reports them as a
            # denial rather than a plain tool failure.
            denied_reason = result.data.get(GOVERNANCE_DENIED_MARKER)
            if isinstance(denied_reason, str) and denied_reason:
                governance_outcome = f"denied: {denied_reason}"
        return CapabilityResult(
            tool_call_id=request.tool_call_id,
            tool_name=request.tool_name,
            ok=result.ok,
            content=result.content,
            data=result.data,
            governance_outcome=governance_outcome,
        )

    # Argument names that governance treats as filesystem-path-bearing. Only
    # names whose label unambiguously means "a path" are listed; ambiguous
    # labels like "target" or "location" are left out to avoid false
    # positives on unrelated tool arguments. This is still a literal-name
    # check, not a semantic one — see Handoff 13 audit follow-ups for the
    # remaining gap (MCP tools that carry paths under arbitrary argument
    # names require Governance-Surface expansion to fully close).
    _PATH_ARG_KEYS: tuple[str, ...] = (
        "path",
        "file",
        "file_path",
        "filepath",
        "filename",
        "source_path",
        "target_path",
        "src_path",
        "dst_path",
    )

    def _govern(self, tool: Tool, request: ToolRequest) -> GovernanceOutcome:
        path_arg_keys = tool.governance_path_arg_keys
        if path_arg_keys is None:
            path_arg_keys = self._PATH_ARG_KEYS
        for key in path_arg_keys:
            path_arg = request.arguments.get(key)
            if not isinstance(path_arg, str) or not path_arg:
                continue
            target = (self._workspace_root / path_arg).resolve()
            try:
                relative = target.relative_to(self._workspace_root).as_posix()
            except ValueError:
                return GovernanceOutcome(
                    allowed=False,
                    reason="path escapes workspace boundary",
                )
            matched = is_protected_relative_path(relative)
            if matched is not None:
                return GovernanceOutcome(
                    allowed=False,
                    reason=f"path targets protected location: {matched}",
                )
        return GovernanceOutcome(allowed=True)

    @staticmethod
    def _validate_arguments(tool: Tool, request: ToolRequest) -> str | None:
        schema = tool.definition.parameters
        if not schema:
            return None

        has_properties_decl = "properties" in schema
        declared_props = set(schema.get("properties", {}).keys())
        required = set(schema.get("required", []))
        provided = set(request.arguments.keys())

        missing = required - provided
        if missing:
            return f"missing required arguments: {', '.join(sorted(missing))}"

        # When `properties` is absent, the schema is under-specified (e.g. an
        # MCP server that emits `{"type":"object","additionalProperties":true}`
        # without enumerating fields). Only enforce `required` in that case;
        # do not reject legitimate arguments as "unexpected".
        if has_properties_decl:
            unexpected = provided - declared_props
            if unexpected:
                return f"unexpected arguments: {', '.join(sorted(unexpected))}"

        return None
