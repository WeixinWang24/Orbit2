from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ToolRelationshipHint:
    primary_tool: str
    relationship: str
    related_tools: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict:
        return {
            "primary_tool": self.primary_tool,
            "relationship": self.relationship,
            "related_tools": list(self.related_tools),
            "reason": self.reason,
        }


RELATIONSHIP_HINTS: tuple[ToolRelationshipHint, ...] = (
    ToolRelationshipHint(
        primary_tool="mcp__repo_scout__repo_scout_repository_overview",
        relationship="includes_summary_of",
        related_tools=("mcp__git__git_status", "mcp__git__git_log"),
        reason=(
            "repository_overview includes git branch, clean/dirty state, "
            "staged/unstaged/untracked counts, and recent commits"
        ),
    ),
    ToolRelationshipHint(
        primary_tool="mcp__repo_scout__repo_scout_changed_context",
        relationship="includes_summary_of",
        related_tools=("mcp__git__git_status", "mcp__git__git_changed_files"),
        reason="changed_context includes normalized git status and changed-file manifest facts",
    ),
    ToolRelationshipHint(
        primary_tool="mcp__repo_scout__repo_scout_diff_digest",
        relationship="includes_summary_of",
        related_tools=("mcp__repo_scout__repo_scout_changed_context",),
        reason="diff_digest includes the normalized changed manifest before per-file diff facts",
    ),
    ToolRelationshipHint(
        primary_tool="mcp__workflow__repo_recon_workflow",
        relationship="includes_toolchains",
        related_tools=(
            "mcp__repo_scout__repo_scout_repository_overview",
            "mcp__repo_scout__repo_scout_changed_context",
            "mcp__repo_scout__repo_scout_diff_digest",
        ),
        reason="repo_recon_workflow runs overview, changed context, and dirty-workspace diff digest facts",
    ),
    ToolRelationshipHint(
        primary_tool="mcp__workflow__inspect_change_set_workflow",
        relationship="includes_toolchains",
        related_tools=(
            "mcp__repo_scout__repo_scout_diff_digest",
            "mcp__repo_scout__repo_scout_impact_scope",
        ),
        reason="inspect_change_set_workflow runs diff digest and impact-scope fact collection",
    ),
)


def relationship_hints_for_tools(tool_names: Iterable[str]) -> list[dict]:
    available = set(tool_names)
    hints: list[dict] = []
    for hint in RELATIONSHIP_HINTS:
        if hint.primary_tool not in available:
            continue
        related = [name for name in hint.related_tools if name in available]
        if not related:
            continue
        hints.append({
            **hint.to_dict(),
            "related_tools": related,
        })
    return hints


def overlap_notices_for_tool_names(tool_names: Iterable[str]) -> list[dict]:
    requested = list(tool_names)
    requested_set = set(requested)
    notices: list[dict] = []
    for hint in RELATIONSHIP_HINTS:
        if hint.primary_tool not in requested_set:
            continue
        overlapping = [name for name in hint.related_tools if name in requested_set]
        if not overlapping:
            continue
        notices.append({
            "primary_tool": hint.primary_tool,
            "overlapping_tools": overlapping,
            "relationship": hint.relationship,
            "reason": hint.reason,
            "policy": "record_only_execute_all_requested_tools",
        })
    return notices
