"""Structured git MCP server for evidence-bearing git reads."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.capability.models import is_protected_relative_path
from src.capability.mcp_servers.git import stdio_server as raw_git

SERVER_NAME = "structured_git"
MAX_HUNK_LINES_ENV = "ORBIT2_MCP_STRUCTURED_GIT_MAX_HUNK_LINES"
DEFAULT_MAX_CHARS_ENV = "ORBIT2_MCP_STRUCTURED_GIT_DEFAULT_MAX_CHARS"
HARD_MAX_CHARS_ENV = "ORBIT2_MCP_STRUCTURED_GIT_HARD_MAX_CHARS"
DIFF_SOURCE_MAX_CHARS_ENV = "ORBIT2_MCP_STRUCTURED_GIT_DIFF_SOURCE_MAX_CHARS"
SHOW_MAX_LINE_SPAN_ENV = "ORBIT2_MCP_STRUCTURED_GIT_SHOW_MAX_LINE_SPAN"
SHOW_SOURCE_MAX_CHARS_ENV = "ORBIT2_MCP_STRUCTURED_GIT_SHOW_SOURCE_MAX_CHARS"
FALLBACK_MAX_HUNK_LINES = 240
FALLBACK_DEFAULT_MAX_CHARS = 12_000
FALLBACK_HARD_MAX_CHARS = 24_000
FALLBACK_DIFF_SOURCE_MAX_CHARS = 200_000
FALLBACK_SHOW_MAX_LINE_SPAN = 120
FALLBACK_SHOW_SOURCE_MAX_CHARS = 200_000

_HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<section>.*)$"
)


def _positive_int_env(env_name: str, fallback: int, *, label: str) -> int:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return fallback
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{label} must be > 0")
    return value


def _max_hunk_lines() -> int:
    return _positive_int_env(
        MAX_HUNK_LINES_ENV,
        FALLBACK_MAX_HUNK_LINES,
        label="structured git max hunk lines",
    )


def _max_char_limits() -> tuple[int, int]:
    default = _positive_int_env(
        DEFAULT_MAX_CHARS_ENV,
        FALLBACK_DEFAULT_MAX_CHARS,
        label="structured git default max chars",
    )
    hard = _positive_int_env(
        HARD_MAX_CHARS_ENV,
        FALLBACK_HARD_MAX_CHARS,
        label="structured git hard max chars",
    )
    if default > hard:
        raise ValueError("structured git default max chars must be <= hard max chars")
    return default, hard


def _diff_source_max_chars() -> int:
    return _positive_int_env(
        DIFF_SOURCE_MAX_CHARS_ENV,
        FALLBACK_DIFF_SOURCE_MAX_CHARS,
        label="structured git diff source max chars",
    )


def _show_max_line_span() -> int:
    return _positive_int_env(
        SHOW_MAX_LINE_SPAN_ENV,
        FALLBACK_SHOW_MAX_LINE_SPAN,
        label="structured git show max line span",
    )


def _show_source_max_chars() -> int:
    return _positive_int_env(
        SHOW_SOURCE_MAX_CHARS_ENV,
        FALLBACK_SHOW_SOURCE_MAX_CHARS,
        label="structured git show source max chars",
    )


def _bounded_max_chars(max_chars: int | None) -> int:
    default, hard = _max_char_limits()
    if isinstance(max_chars, int) and max_chars > 0:
        return min(max_chars, hard)
    return default


def _validate_evidence_gap(evidence_gap: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(evidence_gap, dict):
        raise ValueError("evidence_gap must be an object")
    description = evidence_gap.get("description")
    needed_evidence = evidence_gap.get("needed_evidence")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("evidence_gap.description is required")
    if not isinstance(needed_evidence, str) or not needed_evidence.strip():
        raise ValueError("evidence_gap.needed_evidence is required")
    normalized: dict[str, Any] = {
        "description": description.strip(),
        "needed_evidence": needed_evidence.strip(),
    }
    linked = evidence_gap.get("linked_context_pack_item")
    if isinstance(linked, str) and linked.strip():
        normalized["linked_context_pack_item"] = linked.strip()
    return normalized


def _evidence_gap_from_fields(
    *,
    evidence_gap_description: str,
    needed_evidence: str,
    linked_context_pack_item: str | None = None,
) -> dict[str, Any]:
    if not isinstance(evidence_gap_description, str) or not evidence_gap_description.strip():
        raise ValueError("evidence_gap_description is required")
    if not isinstance(needed_evidence, str) or not needed_evidence.strip():
        raise ValueError("needed_evidence is required")
    normalized: dict[str, Any] = {
        "description": evidence_gap_description.strip(),
        "needed_evidence": needed_evidence.strip(),
    }
    if isinstance(linked_context_pack_item, str) and linked_context_pack_item.strip():
        normalized["linked_context_pack_item"] = linked_context_pack_item.strip()
    return normalized


def _resolve_safe_git_path(path: str, *, cwd: str | None) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty workspace-relative string")
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError("absolute paths are not allowed")
    resolved_cwd = raw_git._resolve_cwd(cwd)
    workspace = raw_git._workspace_root()
    target = (resolved_cwd / candidate).resolve()
    try:
        relative = target.relative_to(workspace).as_posix()
    except ValueError as exc:
        raise ValueError("path escapes workspace") from exc
    matched = is_protected_relative_path(relative)
    if matched is not None:
        raise ValueError(f"path targets protected location: {matched}")
    return relative


def _line_bounds(start_line: int, end_line: int) -> tuple[int, int]:
    if not isinstance(start_line, int) or not isinstance(end_line, int):
        raise ValueError("start_line and end_line must be integers")
    if start_line < 1 or end_line < 1:
        raise ValueError("start_line and end_line must be 1-based positive integers")
    if start_line > end_line:
        raise ValueError("start_line must be <= end_line")
    max_span = _show_max_line_span()
    if end_line - start_line + 1 > max_span:
        raise ValueError(f"line range exceeds max span of {max_span} lines")
    return start_line, end_line


def _validate_rev_for_file_show(rev: str) -> str:
    if not isinstance(rev, str) or not rev.strip():
        raise ValueError("rev must be a non-empty string")
    cleaned = rev.strip()
    if ":" in cleaned:
        raise ValueError("rev must not contain ':'; pass file path separately")
    return cleaned


def _parse_diff_hunks(diff: str) -> list[dict[str, Any]]:
    hunks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    lines: list[str] = []
    for line in diff.splitlines():
        match = _HUNK_RE.match(line)
        if match is not None:
            if current is not None:
                current["content"] = "\n".join(lines)
                current["line_count"] = len(lines)
                hunks.append(current)
            current = {
                "header": line,
                "old_start": int(match.group("old_start")),
                "old_count": int(match.group("old_count") or "1"),
                "new_start": int(match.group("new_start")),
                "new_count": int(match.group("new_count") or "1"),
                "section": match.group("section").strip(),
            }
            lines = [line]
            continue
        if current is not None:
            lines.append(line)
    if current is not None:
        current["content"] = "\n".join(lines)
        current["line_count"] = len(lines)
        hunks.append(current)
    return hunks


def _read_diff_hunk_result(
    *,
    path: str,
    hunk_index: int,
    evidence_gap: dict[str, Any],
    reason_context_pack_insufficient: str,
    cwd: str | None = None,
    staged: bool = False,
    max_chars: int | None = None,
) -> dict[str, Any]:
    relative = _resolve_safe_git_path(path, cwd=cwd)
    if not isinstance(hunk_index, int) or hunk_index < 1:
        raise ValueError("hunk_index must be a 1-based positive integer")
    normalized_gap = _validate_evidence_gap(evidence_gap)
    if (
        not isinstance(reason_context_pack_insufficient, str)
        or not reason_context_pack_insufficient.strip()
    ):
        raise ValueError("reason_context_pack_insufficient is required")

    diff_result = raw_git._git_diff_result(
        cwd=cwd,
        path=relative,
        staged=staged,
        max_chars=_diff_source_max_chars(),
    )
    if diff_result["truncated"]:
        raise ValueError("source diff exceeded structured git diff source max chars")
    hunks = _parse_diff_hunks(diff_result["diff"])
    if hunk_index > len(hunks):
        raise ValueError(f"hunk_index {hunk_index} out of range; hunk_count={len(hunks)}")
    hunk = dict(hunks[hunk_index - 1])
    max_hunk_lines = _max_hunk_lines()
    if hunk["line_count"] > max_hunk_lines:
        raise ValueError(f"hunk exceeds max span of {max_hunk_lines} lines")

    cap = _bounded_max_chars(max_chars)
    content = hunk["content"]
    truncated = len(content) > cap
    if truncated:
        content = content[:cap]

    return {
        "ok": True,
        "evidence_type": "diff_hunk",
        "target": {
            "path": relative,
            "staged": bool(staged),
            "hunk_index": hunk_index,
            "hunk_count": len(hunks),
        },
        "content": content,
        "hunk_range": {
            "old_start": hunk["old_start"],
            "old_count": hunk["old_count"],
            "new_start": hunk["new_start"],
            "new_count": hunk["new_count"],
            "section": hunk["section"],
            "line_count": hunk["line_count"],
        },
        "diff_hash": {
            "algorithm": "sha256",
            "value": hashlib.sha256(diff_result["diff"].encode("utf-8")).hexdigest(),
        },
        "limits": {
            "max_hunk_lines": max_hunk_lines,
            "max_chars": cap,
            "chars_returned": len(content),
            "truncated": truncated,
        },
        "audit": {
            "capability_layer": "structured_primitive",
            "evidence_gap": normalized_gap,
            "reason_context_pack_insufficient": reason_context_pack_insufficient.strip(),
            "policy_decision": "evidence_gap_declared",
            "substrate": "git_diff",
        },
    }


def _read_git_show_region_result(
    *,
    rev: str,
    path: str,
    start_line: int,
    end_line: int,
    evidence_gap_description: str,
    needed_evidence: str,
    reason_context_pack_insufficient: str,
    cwd: str | None = None,
    max_chars: int | None = None,
    linked_context_pack_item: str | None = None,
) -> dict[str, Any]:
    clean_rev = _validate_rev_for_file_show(rev)
    relative = _resolve_safe_git_path(path, cwd=cwd)
    start, end = _line_bounds(start_line, end_line)
    normalized_gap = _evidence_gap_from_fields(
        evidence_gap_description=evidence_gap_description,
        needed_evidence=needed_evidence,
        linked_context_pack_item=linked_context_pack_item,
    )
    if (
        not isinstance(reason_context_pack_insufficient, str)
        or not reason_context_pack_insufficient.strip()
    ):
        raise ValueError("reason_context_pack_insufficient is required")

    show_rev = f"{clean_rev}:{relative}"
    show_result = raw_git._git_show_result(
        show_rev,
        cwd=cwd,
        max_chars=_show_source_max_chars(),
    )
    if show_result["truncated"]:
        raise ValueError("source git show output exceeded structured git show source max chars")

    content_all = show_result["output"]
    lines = content_all.splitlines()
    selected = lines[start - 1 : min(end, len(lines))]
    content = "\n".join(selected)
    cap = _bounded_max_chars(max_chars)
    truncated = len(content) > cap
    if truncated:
        content = content[:cap]

    actual_end = min(end, len(lines))
    if start > len(lines):
        actual_end = start - 1

    return {
        "ok": True,
        "evidence_type": "git_show_region",
        "target": {
            "rev": clean_rev,
            "path": relative,
            "start_line": start,
            "end_line": end,
        },
        "content": content,
        "line_range": {
            "requested_start_line": start,
            "requested_end_line": end,
            "actual_start_line": start if start <= len(lines) else None,
            "actual_end_line": actual_end if start <= len(lines) else None,
            "total_lines": len(lines),
        },
        "source_hash": {
            "algorithm": "sha256",
            "value": hashlib.sha256(content_all.encode("utf-8")).hexdigest(),
        },
        "limits": {
            "max_lines": _show_max_line_span(),
            "max_chars": cap,
            "chars_returned": len(content),
            "truncated": truncated,
        },
        "audit": {
            "capability_layer": "structured_primitive",
            "evidence_gap": normalized_gap,
            "reason_context_pack_insufficient": reason_context_pack_insufficient.strip(),
            "policy_decision": "evidence_gap_declared",
            "substrate": "git_show",
        },
    }


mcp = FastMCP(SERVER_NAME)


@mcp.tool()
def read_diff_hunk(
    path: str,
    hunk_index: int,
    evidence_gap: dict[str, Any],
    reason_context_pack_insufficient: str,
    cwd: str | None = None,
    staged: bool = False,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Read one bounded diff hunk; path is workspace-relative and first hunk_index is 1."""
    return _read_diff_hunk_result(
        path=path,
        hunk_index=hunk_index,
        evidence_gap=evidence_gap,
        reason_context_pack_insufficient=reason_context_pack_insufficient,
        cwd=cwd,
        staged=staged,
        max_chars=max_chars,
    )


@mcp.tool()
def read_git_show_region(
    rev: str,
    path: str,
    start_line: int,
    end_line: int,
    evidence_gap_description: str,
    needed_evidence: str,
    reason_context_pack_insufficient: str,
    cwd: str | None = None,
    max_chars: int | None = None,
    linked_context_pack_item: str | None = None,
) -> dict[str, Any]:
    """Read 1-based lines from rev:path; pass rev and workspace-relative path separately."""
    return _read_git_show_region_result(
        rev=rev,
        path=path,
        start_line=start_line,
        end_line=end_line,
        evidence_gap_description=evidence_gap_description,
        needed_evidence=needed_evidence,
        reason_context_pack_insufficient=reason_context_pack_insufficient,
        cwd=cwd,
        max_chars=max_chars,
        linked_context_pack_item=linked_context_pack_item,
    )


if __name__ == "__main__":
    mcp.run()
