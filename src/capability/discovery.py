"""Progressive capability discovery (Handoff 19; Handoff 28 continuation-bridge shaping).

Ships a single tool — `list_available_tools` — that helps the provider
navigate a capability surface whose full inventory is NOT all exposed on
every turn. The tool reports:

- the tools currently visible to the provider this turn (for self-check),
- the hidden reveal groups available for request (family-level, not
  per-tool exhaustive),
- a short hint describing how to use a revealed group.

When the tool is called with `reveal=<group_name>`, its result carries a
`reveal_request` marker that the Knowledge Surface assembler consumes on
the NEXT model step to widen the exposed-tool surface to include that
group. "Next model step" is NOT "next user message" — the session manager's
tool loop keeps calling the model after each tool result within the SAME
user turn, so the revealed tools become available to the provider
immediately on the next tool call inside the same response. Handoff 28
shapes the confirmation strings to make this continuation bridge explicit,
targeting the `session_60ee34acb36e` failure where the model returned a
final text ("please say 继续") after reveal instead of making the downstream
tool call.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from src.capability.models import CapabilityLayer, ToolDefinition, ToolResult
from src.capability.registry import CapabilityRegistry
from src.capability.tool_relationships import relationship_hints_for_tools
from src.capability.tools.base import Tool
from src.governance.disclosure import (
    REVEAL_ALL_SAFE_REQUEST_MARKER,
    REVEAL_BATCH_REQUEST_MARKER,
    REVEAL_REQUEST_MARKER,
)

DISCOVERY_TOOL_NAME = "list_available_tools"
DISCOVERY_REVEAL_GROUP = "discovery"


# Short human-readable descriptions for shipped reveal groups. The
# discovery tool uses these to describe hidden groups to the provider
# without dumping per-tool schemas. Groups not in this map still surface
# via their name + member count, so the tool remains correct for future
# families added before this map is updated.
GROUP_DESCRIPTIONS: dict[str, str] = {
    "discovery": "Always-on capability list for discovering hidden tool groups.",
    "native_fs_read": "Native workspace filesystem read (open, inspect files).",
    "native_fs_mutate": "Native workspace filesystem mutations (write, replace, patch).",
    "mcp_fs_read": "MCP filesystem read tools (read / list / stat).",
    "mcp_fs_mutate": "MCP filesystem mutations (write / replace).",
    "mcp_git_read": "MCP git read tools (status / diff / log).",
    "mcp_git_mutate": "MCP git mutations (add / commit).",
    "mcp_diagnostics": "MCP diagnostics (pytest / ruff / mypy).",
    "mcp_code_intel": "MCP Code Intelligence fact tools (index summaries, symbols, file context, graph fragments).",
    "mcp_structured_filesystem": "MCP structured filesystem evidence tools (bounded file regions and scoped grep).",
    "mcp_structured_git": "MCP structured git evidence tools (bounded diff hunks and revision file regions).",
    "mcp_workflow": "MCP L3 workflows that prepare task-shaped fact packages and explicit provider decision requests.",
    "mcp_obsidian": "MCP Obsidian vault read tools (notes / links / tags).",
}


class ListAvailableToolsTool(Tool):
    """Provider-facing discovery capability for progressive exposure.

    The tool is instantiated with a reference to the `CapabilityRegistry`
    it reports on. It does NOT cache the inventory; every invocation
    re-reads the registry so attachment changes (new MCP families) are
    visible immediately.

    Contract:
    - Always exposed (`default_exposed=True`, `reveal_group="discovery"`).
    - Safe / no approval — it only reads metadata, never executes a tool.
    - When called with `reveal=<group>`, the result payload carries
      `data["reveal_request"]` which the assembler consumes.
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        *,
        active_reveal_groups_provider: Callable[[], list[str]] | None = None,
    ) -> None:
        """Construct with a registry to summarize.

        `active_reveal_groups_provider` optionally yields the list of reveal
        groups that are currently active on THIS turn (i.e. default groups
        plus groups unlocked by earlier reveal requests). When supplied the
        discovery summary reports those tools as exposed rather than hidden.
        Without it the tool falls back to each tool's static `default_exposed`
        flag, which is honest only for the very first turn of a session.

        The session manager wires this callback so the summary the model
        sees matches the `tool_definitions` it actually receives on the
        same turn — preventing the "hidden list contradicts visible list"
        inconsistency.
        """
        self._registry = registry
        self._active_reveal_groups_provider = active_reveal_groups_provider

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=DISCOVERY_TOOL_NAME,
            description=(
                "List the tools currently visible to you and the hidden "
                "reveal groups you can unlock. Pass `reveal=<group_name>` "
                "to unlock one group, `reveal_batch=[<group>, ...]` to "
                "request several at once, or `reveal_all_safe=true` for a "
                "one-shot overview that unlocks every no-approval group. "
                "Revealed tools become available on your NEXT model step, "
                "which happens immediately inside the same response — do "
                "NOT return a final text asking the user to continue. "
                "After a reveal call succeeds, make the downstream tool "
                "call directly in the same response. Batch modes require "
                "an active batch-reveal disclosure strategy — if your "
                "session is running single-reveal they will be recorded "
                "but ignored."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reveal": {
                        "type": "string",
                        "description": (
                            "Optional. Name of a single reveal group to "
                            "unlock for your next tool call in this response."
                        ),
                    },
                    "reveal_batch": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional. List of reveal group names to unlock "
                            "in one shot. The revealed tools become usable "
                            "on your next tool call in this same response. "
                            "Only honored when the session's disclosure "
                            "strategy is `batch_reveal`."
                        ),
                    },
                    "reveal_all_safe": {
                        "type": "boolean",
                        "description": (
                            "Optional. When true, unlock every reveal group "
                            "whose tools are ALL safe / no-approval. Never "
                            "unlocks mutation-bearing groups. Only honored "
                            "under `batch_reveal`."
                        ),
                    },
                },
            },
        )

    @property
    def reveal_group(self) -> str:
        return DISCOVERY_REVEAL_GROUP

    @property
    def default_exposed(self) -> bool:
        return True

    @property
    def side_effect_class(self) -> str:
        return "safe"

    @property
    def capability_layer(self) -> CapabilityLayer:
        return CapabilityLayer.TOOLCHAIN

    @property
    def requires_approval(self) -> bool:
        return False

    @property
    def environment_check_kind(self) -> str:
        return "none"

    @property
    def governance_path_arg_keys(self) -> tuple[str, ...] | None:
        # Discovery takes no path-bearing arguments.
        return ()

    def execute(
        self,
        *,
        reveal: str | None = None,
        reveal_batch: list[str] | None = None,
        reveal_all_safe: bool | None = None,
    ) -> ToolResult:
        payload = self._build_summary()
        data: dict[str, Any] = {"summary": payload}

        all_registry_groups = {
            self._registry.get(n).reveal_group
            for n in self._registry.list_names()
            if self._registry.get(n) is not None
        }
        active = set(payload.get("active_reveal_groups") or [])

        # Single reveal (back-compat with Handoff 19).
        if isinstance(reveal, str) and reveal.strip():
            self._handle_single_reveal(
                reveal.strip(), all_registry_groups, active, data
            )

        # Batch reveal — a list of group names.
        if isinstance(reveal_batch, list) and reveal_batch:
            cleaned: list[str] = []
            rejected: list[str] = []
            noops: list[str] = []
            for g in reveal_batch:
                if not isinstance(g, str) or not g.strip():
                    continue
                name = g.strip()
                if name == DISCOVERY_REVEAL_GROUP:
                    rejected.append(name)
                    continue
                if name not in all_registry_groups:
                    rejected.append(name)
                    continue
                if name in active:
                    noops.append(name)
                    continue
                cleaned.append(name)
            if cleaned:
                data[REVEAL_BATCH_REQUEST_MARKER] = cleaned
                data["reveal_batch_confirmation"] = (
                    f"groups {sorted(cleaned)!r} are now available for your "
                    "next tool call in this same response — continue by "
                    "calling the revealed tool directly. Do NOT return a "
                    "final message asking the user to continue."
                )
            if rejected:
                data["reveal_batch_rejected"] = sorted(rejected)
            if noops:
                data["reveal_batch_noop"] = sorted(noops)

        # All-safe reveal — boolean trigger.
        if reveal_all_safe is True:
            data[REVEAL_ALL_SAFE_REQUEST_MARKER] = True
            data["reveal_all_safe_confirmation"] = (
                "every reveal group whose tools are all safe / no-approval "
                "is now available for your next tool call in this same "
                "response — continue by calling the revealed tools directly. "
                "Do NOT return a final message asking the user to continue."
            )

        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        return ToolResult(ok=True, content=rendered, data=data)

    def _handle_single_reveal(
        self,
        requested: str,
        all_registry_groups: set[str],
        active: set[str],
        data: dict[str, Any],
    ) -> None:
        if requested == DISCOVERY_REVEAL_GROUP:
            data["reveal_error"] = (
                "the 'discovery' group is always active; "
                "no reveal needed"
            )
            return
        if requested not in all_registry_groups:
            data["reveal_error"] = f"unknown reveal group: {requested!r}"
            return
        if requested in active:
            data["reveal_noop"] = (
                f"reveal group {requested!r} is already active"
            )
            return
        data[REVEAL_REQUEST_MARKER] = requested
        data["reveal_request_confirmation"] = (
            f"reveal group {requested!r} is now available for your next tool "
            "call in this same response — continue by calling the revealed "
            "tool directly. Do NOT return a final message asking the user to "
            "continue; the tool loop keeps running until you stop calling tools."
        )

    def _build_summary(self) -> dict[str, Any]:
        # Active reveal groups for THIS turn. When the provider callback
        # is wired the summary uses it as the truth source; otherwise we
        # fall back to each tool's static `default_exposed` flag (which is
        # correct only for the very first turn of a session).
        active_groups: set[str] | None = None
        if self._active_reveal_groups_provider is not None:
            try:
                supplied = self._active_reveal_groups_provider() or []
                if isinstance(supplied, list):
                    active_groups = {g for g in supplied if isinstance(g, str)}
            except Exception:
                # A broken provider must not crash discovery; degrade to
                # the static flag behavior.
                active_groups = None

        exposed: list[str] = []
        hidden_by_group: dict[str, list[str]] = {}
        layer_counts: dict[str, int] = {}
        exposed_layer_counts: dict[str, int] = {}
        hidden_layer_counts: dict[str, int] = {}
        for name in self._registry.list_names():
            tool = self._registry.get(name)
            if tool is None:  # defensive; should not happen with our registry
                continue
            group = tool.reveal_group
            layer = tool.capability_layer.value
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
            if active_groups is not None:
                is_exposed_now = group in active_groups
            else:
                is_exposed_now = tool.default_exposed
            if is_exposed_now:
                exposed.append(name)
                exposed_layer_counts[layer] = exposed_layer_counts.get(layer, 0) + 1
            else:
                hidden_by_group.setdefault(group, []).append(name)
                hidden_layer_counts[layer] = hidden_layer_counts.get(layer, 0) + 1

        reveal_groups = [
            {
                "name": group,
                "description": GROUP_DESCRIPTIONS.get(
                    group, "Reveal group (no description registered)."
                ),
                "tool_count": len(tools),
                "sample_tools": sorted(tools)[:4],
                "layers": sorted({
                    self._registry.get(tool_name).capability_layer.value
                    for tool_name in tools
                    if self._registry.get(tool_name) is not None
                }),
            }
            for group, tools in sorted(hidden_by_group.items())
        ]

        return {
            "exposed_tools": sorted(exposed),
            "exposed_tool_count": len(exposed),
            "active_reveal_groups": sorted(active_groups) if active_groups is not None else None,
            "reveal_groups": reveal_groups,
            "relationship_hints": relationship_hints_for_tools(exposed),
            "capability_layers": {
                "total": dict(sorted(layer_counts.items())),
                "exposed": dict(sorted(exposed_layer_counts.items())),
                "hidden": dict(sorted(hidden_layer_counts.items())),
            },
            "hidden_tool_count": sum(len(v) for v in hidden_by_group.values()),
            "hint": (
                "Call this tool with `reveal=<group_name>` to unlock a "
                "hidden group. The revealed tools become usable on your "
                "NEXT tool call in the SAME response — the session's tool "
                "loop keeps calling you until you stop emitting tool "
                "requests. Do NOT return a final text message asking the "
                "user to say 'continue' after a successful reveal; make the "
                "downstream tool call directly."
            ),
        }
