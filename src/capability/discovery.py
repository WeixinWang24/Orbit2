"""Progressive capability discovery (Handoff 19).

Ships a single tool — `list_available_tools` — that helps the provider
navigate a capability surface whose full inventory is NOT all exposed on
every turn. The tool reports:

- the tools currently visible to the provider this turn (for self-check),
- the hidden reveal groups available for request (family-level, not
  per-tool exhaustive),
- a short hint describing how to request a group on the next turn.

When the tool is called with `reveal=<group_name>`, its result carries a
`reveal_request` marker that the Knowledge Surface assembler consumes on
the next turn to widen the exposed-tool surface to include that group.
That marker is the only cross-surface coupling — the discovery tool has
no privileged access to session state, and the assembler makes the
decision explicitly from transcript material.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from src.capability.models import ToolDefinition, ToolResult
from src.capability.registry import CapabilityRegistry
from src.capability.tools.base import Tool

DISCOVERY_TOOL_NAME = "list_available_tools"
DISCOVERY_REVEAL_GROUP = "discovery"

# Marker key surfaced on `ToolResult.data`. The Knowledge Surface assembler
# scans recent transcript TOOL messages for this field to compute which
# reveal groups are currently unlocked for the next turn.
REVEAL_REQUEST_MARKER = "reveal_request"


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
                "to request that the named group be made visible on the "
                "next turn."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reveal": {
                        "type": "string",
                        "description": (
                            "Optional. Name of a reveal group to request "
                            "for the next turn. The group's tools become "
                            "visible on the next provider turn only; this "
                            "call itself only records the request."
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
    def requires_approval(self) -> bool:
        return False

    @property
    def environment_check_kind(self) -> str:
        return "none"

    @property
    def governance_path_arg_keys(self) -> tuple[str, ...] | None:
        # Discovery takes no path-bearing arguments.
        return ()

    def execute(self, *, reveal: str | None = None) -> ToolResult:
        payload = self._build_summary()
        data: dict[str, Any] = {"summary": payload}
        if isinstance(reveal, str) and reveal.strip():
            requested = reveal.strip()
            # All reveal groups the registry actually exposes — both the
            # hidden ones surfaced in the summary and the currently-active
            # ones (which wouldn't appear in `reveal_groups` because
            # they're no longer hidden). Reveals that request an already-
            # active group are a no-op; reveals that request the discovery
            # group itself are explicitly rejected because the discovery
            # tool is always exposed and has no meaningful "reveal".
            all_registry_groups = {
                self._registry.get(n).reveal_group
                for n in self._registry.list_names()
                if self._registry.get(n) is not None
            }
            if requested == DISCOVERY_REVEAL_GROUP:
                data["reveal_error"] = (
                    "the 'discovery' group is always active; "
                    "no reveal needed"
                )
            elif requested not in all_registry_groups:
                # Invalid group — still return discovery output but include
                # an explicit error marker instead of a reveal_request so
                # the assembler does not widen exposure to a nonexistent
                # group.
                data["reveal_error"] = f"unknown reveal group: {requested!r}"
            elif requested in (payload.get("active_reveal_groups") or []):
                data["reveal_noop"] = (
                    f"reveal group {requested!r} is already active"
                )
            else:
                data[REVEAL_REQUEST_MARKER] = requested
                data["reveal_request_confirmation"] = (
                    f"reveal group {requested!r} will be exposed on the next turn"
                )
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        return ToolResult(ok=True, content=rendered, data=data)

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
        for name in self._registry.list_names():
            tool = self._registry.get(name)
            if tool is None:  # defensive; should not happen with our registry
                continue
            group = tool.reveal_group
            if active_groups is not None:
                is_exposed_now = group in active_groups
            else:
                is_exposed_now = tool.default_exposed
            if is_exposed_now:
                exposed.append(name)
            else:
                hidden_by_group.setdefault(group, []).append(name)

        reveal_groups = [
            {
                "name": group,
                "description": GROUP_DESCRIPTIONS.get(
                    group, "Reveal group (no description registered)."
                ),
                "tool_count": len(tools),
                "sample_tools": sorted(tools)[:4],
            }
            for group, tools in sorted(hidden_by_group.items())
        ]

        return {
            "exposed_tools": sorted(exposed),
            "exposed_tool_count": len(exposed),
            "active_reveal_groups": sorted(active_groups) if active_groups is not None else None,
            "reveal_groups": reveal_groups,
            "hidden_tool_count": sum(len(v) for v in hidden_by_group.values()),
            "hint": (
                "Call this tool again with `reveal=<group_name>` to request "
                "that group be exposed on the next turn. You do NOT need to "
                "call list_available_tools again after that — the requested "
                "group's tools will appear directly on the next turn."
            ),
        }
