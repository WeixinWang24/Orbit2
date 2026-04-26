"""Structured filesystem MCP server for evidence-bearing file regions."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.capability.models import is_protected_relative_path
from src.capability.mcp_servers.filesystem import stdio_server as raw_filesystem

SERVER_NAME = "structured_filesystem"
MAX_LINE_SPAN_ENV = "ORBIT2_MCP_STRUCTURED_FS_MAX_LINE_SPAN"
DEFAULT_MAX_CHARS_ENV = "ORBIT2_MCP_STRUCTURED_FS_DEFAULT_MAX_CHARS"
HARD_MAX_CHARS_ENV = "ORBIT2_MCP_STRUCTURED_FS_HARD_MAX_CHARS"
GREP_MAX_MATCHES_ENV = "ORBIT2_MCP_STRUCTURED_FS_GREP_MAX_MATCHES"
GREP_MAX_FILE_BYTES_ENV = "ORBIT2_MCP_STRUCTURED_FS_GREP_MAX_FILE_BYTES"
GREP_MAX_TOTAL_BYTES_ENV = "ORBIT2_MCP_STRUCTURED_FS_GREP_MAX_TOTAL_BYTES"
FALLBACK_MAX_LINE_SPAN = 120
FALLBACK_DEFAULT_MAX_CHARS = 12_000
FALLBACK_HARD_MAX_CHARS = 24_000
FALLBACK_GREP_MAX_MATCHES = 20
FALLBACK_GREP_MAX_FILE_BYTES = 256 * 1024
FALLBACK_GREP_MAX_TOTAL_BYTES = 2 * 1024 * 1024


def _positive_int_env(env_name: str, fallback: int, *, label: str) -> int:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return fallback
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{label} must be > 0")
    return value


def _max_line_span() -> int:
    return _positive_int_env(
        MAX_LINE_SPAN_ENV,
        FALLBACK_MAX_LINE_SPAN,
        label="structured filesystem max line span",
    )


def _max_char_limits() -> tuple[int, int]:
    default = _positive_int_env(
        DEFAULT_MAX_CHARS_ENV,
        FALLBACK_DEFAULT_MAX_CHARS,
        label="structured filesystem default max chars",
    )
    hard = _positive_int_env(
        HARD_MAX_CHARS_ENV,
        FALLBACK_HARD_MAX_CHARS,
        label="structured filesystem hard max chars",
    )
    if default > hard:
        raise ValueError("structured filesystem default max chars must be <= hard max chars")
    return default, hard


def _grep_limits() -> tuple[int, int, int]:
    return (
        _positive_int_env(
            GREP_MAX_MATCHES_ENV,
            FALLBACK_GREP_MAX_MATCHES,
            label="structured filesystem grep max matches",
        ),
        _positive_int_env(
            GREP_MAX_FILE_BYTES_ENV,
            FALLBACK_GREP_MAX_FILE_BYTES,
            label="structured filesystem grep max file bytes",
        ),
        _positive_int_env(
            GREP_MAX_TOTAL_BYTES_ENV,
            FALLBACK_GREP_MAX_TOTAL_BYTES,
            label="structured filesystem grep max total bytes",
        ),
    )


def _resolve_safe_existing_file(path: str) -> tuple[Path, str]:
    target = raw_filesystem._resolve_safe_existing_file(path)
    workspace_root = raw_filesystem._workspace_root()
    try:
        relative = target.relative_to(workspace_root).as_posix()
    except ValueError as exc:
        raise ValueError("path escapes workspace") from exc

    matched = is_protected_relative_path(relative)
    if matched is not None:
        raise ValueError(f"path targets protected location: {matched}")
    if not target.exists():
        raise ValueError("path not found")
    if not target.is_file():
        raise ValueError("path is not a file")
    return target, relative


def _resolve_safe_scope(path: str) -> tuple[Path, str]:
    target = raw_filesystem._resolve_safe_path(path)
    workspace_root = raw_filesystem._workspace_root()
    try:
        relative = target.relative_to(workspace_root).as_posix()
    except ValueError as exc:
        raise ValueError("path escapes workspace") from exc
    matched = is_protected_relative_path(relative)
    if matched is not None:
        raise ValueError(f"path targets protected location: {matched}")
    if not target.exists():
        raise ValueError("path not found")
    if not target.is_dir():
        raise ValueError("path is not a directory")
    return target, relative


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


def _bounded_max_chars(max_chars: int | None) -> int:
    default, hard = _max_char_limits()
    if isinstance(max_chars, int) and max_chars > 0:
        return min(max_chars, hard)
    return default


def _line_bounds(start_line: int, end_line: int) -> tuple[int, int]:
    if not isinstance(start_line, int) or not isinstance(end_line, int):
        raise ValueError("start_line and end_line must be integers")
    if start_line < 1 or end_line < 1:
        raise ValueError("start_line and end_line must be 1-based positive integers")
    if start_line > end_line:
        raise ValueError("start_line must be <= end_line")
    max_line_span = _max_line_span()
    if end_line - start_line + 1 > max_line_span:
        raise ValueError(f"line range exceeds max span of {max_line_span} lines")
    return start_line, end_line


def _read_file_region_result(
    *,
    path: str,
    start_line: int,
    end_line: int,
    evidence_gap: dict[str, Any],
    reason_context_pack_insufficient: str,
    max_chars: int | None = None,
) -> dict[str, Any]:
    target, relative = _resolve_safe_existing_file(path)
    start, end = _line_bounds(start_line, end_line)
    normalized_gap = _validate_evidence_gap(evidence_gap)
    if (
        not isinstance(reason_context_pack_insufficient, str)
        or not reason_context_pack_insufficient.strip()
    ):
        raise ValueError("reason_context_pack_insufficient is required")

    raw_bytes = target.read_bytes()
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
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
        "evidence_type": "file_region",
        "target": {
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
        "file_hash": {
            "algorithm": "sha256",
            "value": hashlib.sha256(raw_bytes).hexdigest(),
        },
        "limits": {
            "max_lines": _max_line_span(),
            "max_chars": cap,
            "chars_returned": len(content),
            "truncated": truncated,
        },
        "audit": {
            "capability_layer": "structured_primitive",
            "evidence_gap": normalized_gap,
            "reason_context_pack_insufficient": reason_context_pack_insufficient.strip(),
            "policy_decision": "evidence_gap_declared",
        },
    }


def _grep_scoped_result(
    *,
    pattern: str,
    path: str,
    evidence_gap_description: str,
    needed_evidence: str,
    reason_context_pack_insufficient: str,
    file_pattern: str = "*",
    case_insensitive: bool = False,
    max_matches: int | None = None,
    linked_context_pack_item: str | None = None,
) -> dict[str, Any]:
    _target, relative = _resolve_safe_scope(path)
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

    default_max_matches, max_file_bytes, max_total_bytes = _grep_limits()
    effective_max_matches = (
        min(max_matches, default_max_matches)
        if isinstance(max_matches, int) and max_matches > 0
        else default_max_matches
    )
    raw = raw_filesystem._grep_result(
        pattern,
        path=path,
        file_pattern=file_pattern,
        max_matches=effective_max_matches,
        case_insensitive=case_insensitive,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
    )
    matches = [
        match for match in raw["matches"]
        if is_protected_relative_path(str(match.get("path", ""))) is None
    ]
    return {
        "ok": True,
        "evidence_type": "grep_scope",
        "query": {
            "pattern": pattern,
            "file_pattern": file_pattern,
            "case_insensitive": bool(case_insensitive),
        },
        "scope": {
            "path": "." if relative == "." else relative,
        },
        "matches": matches,
        "match_count": len(matches),
        "scanned_files": raw["scanned_files"],
        "scanned_bytes": raw["scanned_bytes"],
        "limits": {
            "max_matches": effective_max_matches,
            "max_file_bytes": max_file_bytes,
            "max_total_bytes": max_total_bytes,
            "truncated": raw["truncated"],
            "total_byte_budget_exhausted": raw["total_byte_budget_exhausted"],
        },
        "audit": {
            "capability_layer": "structured_primitive",
            "evidence_gap": normalized_gap,
            "reason_context_pack_insufficient": reason_context_pack_insufficient.strip(),
            "policy_decision": "evidence_gap_declared",
            "substrate": "grep",
        },
    }


mcp = FastMCP(SERVER_NAME)


@mcp.tool()
def read_file_region(
    path: str,
    start_line: int,
    end_line: int,
    evidence_gap: dict[str, Any],
    reason_context_pack_insufficient: str,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Read 1-based lines from a workspace-relative file path with declared evidence intent."""
    return _read_file_region_result(
        path=path,
        start_line=start_line,
        end_line=end_line,
        evidence_gap=evidence_gap,
        reason_context_pack_insufficient=reason_context_pack_insufficient,
        max_chars=max_chars,
    )


@mcp.tool()
def grep_scoped(
    pattern: str,
    path: str,
    evidence_gap_description: str,
    needed_evidence: str,
    reason_context_pack_insufficient: str,
    file_pattern: str = "*",
    case_insensitive: bool = False,
    max_matches: int | None = None,
    linked_context_pack_item: str | None = None,
) -> dict[str, Any]:
    """Search a workspace-relative directory scope; return bounded file/line matches with declared evidence intent."""
    return _grep_scoped_result(
        pattern=pattern,
        path=path,
        evidence_gap_description=evidence_gap_description,
        needed_evidence=needed_evidence,
        reason_context_pack_insufficient=reason_context_pack_insufficient,
        file_pattern=file_pattern,
        case_insensitive=case_insensitive,
        max_matches=max_matches,
        linked_context_pack_item=linked_context_pack_item,
    )


if __name__ == "__main__":
    mcp.run()
