"""Capability-awareness context block (Handoff 27).

Addresses the task-triggered capability-incompleteness awareness gap
observed in session 7b475ca42e77: when a user request clearly mapped to a
hidden reveal group (filesystem mutation), the provider prematurely declared
inability and suggested a shell-command fallback rather than calling
`list_available_tools` to discover and reveal the relevant group.

Per ADR-0011 (agent awareness shaping), runtime-to-provider augmentations
are awareness-shaping interventions rather than neutral prompt stuffing. This
block is bounded, typed, governance-conditioned, and inspectable: it reports
the current turn's visible tool count, active reveal groups, hidden reveal
groups with short descriptions, and a short posture statement that targets
the specific admissibility-model failure (goal-preserving bypass to shell
commands when local tool surface appears insufficient).

The snapshot is sourced from authoritative runtime structures: the
CapabilityRegistry supplies the total group inventory and group descriptions,
the ExposureDecision (when available) supplies the active reveal groups for
THIS turn. When no exposure decision is supplied (e.g. the very first call
before the session manager computes one), the collector falls back to each
tool's static `default_exposed` flag — correct for the first turn, honest
for later turns only if the disclosure strategy has not widened exposure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.capability.discovery import GROUP_DESCRIPTIONS as DISCOVERY_GROUP_DESCRIPTIONS
from src.capability.tool_relationships import relationship_hints_for_tools
from src.knowledge.models import ContextFragment

if TYPE_CHECKING:
    from src.capability.registry import CapabilityRegistry
    from src.governance.disclosure import ExposureDecision


CAPABILITY_AWARENESS_FRAGMENT_NAME = "capability_awareness"
CAPABILITY_AWARENESS_VISIBILITY_SCOPE = "capability_awareness"
CAPABILITY_AWARENESS_PRIORITY = 85

_OPEN_TAG = "<capability-awareness>"
_CLOSE_TAG = "</capability-awareness>"

CAPABILITY_AWARENESS_POSTURE_TEXT = (
    "Your visible tool set is a deliberately minimal subset under progressive "
    "disclosure. Hidden reveal groups are available but not exposed until "
    "requested. When a user request clearly maps to a hidden reveal group "
    "(e.g. filesystem mutation, git operations, running a subprocess), call "
    "`list_available_tools` first and request the relevant reveal group "
    "BEFORE declaring inability or suggesting shell-command workarounds. "
    "Missing tools are usually hidden pending discovery, not absent. "
    "After a reveal call succeeds and the original task is still pending, "
    "the newly visible tools are available to you IMMEDIATELY on your next "
    "tool call in the SAME response — the session's tool loop keeps "
    "calling you after each tool result without waiting for user input. "
    "Do NOT return a final text asking the user to say '继续' / 'continue' / "
    "repeat the request; make the downstream tool call directly."
)


@dataclass(frozen=True)
class CapabilityAwarenessSnapshot:
    visible_tool_count: int
    visible_reveal_groups: tuple[str, ...]
    hidden_reveal_groups: tuple[str, ...]
    hidden_group_descriptions: dict[str, str] = field(default_factory=dict)
    relationship_hints: tuple[dict, ...] = field(default_factory=tuple)


class CapabilityAwarenessCollector:
    def __init__(self, registry: "CapabilityRegistry") -> None:
        self._registry = registry

    def collect(
        self, *, exposure_decision: "ExposureDecision | None" = None
    ) -> CapabilityAwarenessSnapshot:
        tool_group: dict[str, str] = {}
        tool_default: dict[str, bool] = {}
        for name in self._registry.list_names():
            tool = self._registry.get(name)
            if tool is None:
                continue
            tool_group[name] = tool.reveal_group
            tool_default[name] = tool.default_exposed

        all_groups = set(tool_group.values())
        if exposure_decision is not None:
            visible_groups = set(exposure_decision.active_reveal_groups)
            visible_tool_names = set(exposure_decision.exposed_tool_names)
            visible_tool_count = len(visible_tool_names)
        else:
            visible_tool_names = {
                n for n, is_default in tool_default.items() if is_default
            }
            visible_tool_count = len(visible_tool_names)
            visible_groups = {
                tool_group[n] for n, is_default in tool_default.items() if is_default
            }

        hidden_groups = tuple(sorted(all_groups - visible_groups))
        visible_groups_sorted = tuple(sorted(visible_groups))

        hidden_descriptions = {
            g: DISCOVERY_GROUP_DESCRIPTIONS.get(
                g, "Reveal group (no description registered)."
            )
            for g in hidden_groups
        }

        return CapabilityAwarenessSnapshot(
            visible_tool_count=visible_tool_count,
            visible_reveal_groups=visible_groups_sorted,
            hidden_reveal_groups=hidden_groups,
            hidden_group_descriptions=hidden_descriptions,
            relationship_hints=tuple(relationship_hints_for_tools(visible_tool_names)),
        )


def build_capability_awareness_fragment(
    snapshot: CapabilityAwarenessSnapshot,
    *,
    policy_name: str,
) -> ContextFragment:
    visible_groups_str = ", ".join(snapshot.visible_reveal_groups) or "(none)"
    hidden_lines: list[str] = []
    for group in snapshot.hidden_reveal_groups:
        desc = snapshot.hidden_group_descriptions.get(group, "")
        if desc:
            hidden_lines.append(f"  - {group}: {desc}")
        else:
            hidden_lines.append(f"  - {group}")
    hidden_section = "\n".join(hidden_lines) if hidden_lines else "  (none)"
    relationship_lines = [
        (
            f"  - {hint['primary_tool']} {hint['relationship']} "
            f"{', '.join(hint['related_tools'])}: {hint['reason']}"
        )
        for hint in snapshot.relationship_hints
    ]
    relationship_section = "\n".join(relationship_lines) if relationship_lines else "  (none)"

    content_lines = [
        _OPEN_TAG,
        f"visible_tool_count: {snapshot.visible_tool_count}",
        f"visible_reveal_groups: {visible_groups_str}",
        f"hidden_reveal_groups ({len(snapshot.hidden_reveal_groups)}):",
        hidden_section,
        f"relationship_hints ({len(snapshot.relationship_hints)}):",
        relationship_section,
        f"posture: {CAPABILITY_AWARENESS_POSTURE_TEXT}",
        _CLOSE_TAG,
    ]
    return ContextFragment(
        fragment_name=CAPABILITY_AWARENESS_FRAGMENT_NAME,
        visibility_scope=CAPABILITY_AWARENESS_VISIBILITY_SCOPE,
        content="\n".join(content_lines),
        priority=CAPABILITY_AWARENESS_PRIORITY,
        metadata={
            "origin": "capability_awareness",
            "policy_name": policy_name,
            "visible_tool_count": snapshot.visible_tool_count,
            "visible_reveal_groups": list(snapshot.visible_reveal_groups),
            "hidden_reveal_groups": list(snapshot.hidden_reveal_groups),
            "relationship_hints": list(snapshot.relationship_hints),
        },
    )
