"""Bounded filesystem MCP server for Orbit2 (Handoff 16, first slice).

Ships with five tools only: `read_file`, `list_directory`, `get_file_info`,
`write_file`, `replace_in_file`. Each is workspace-scoped: operations must
target paths inside the allowed root (set via `ORBIT_WORKSPACE_ROOT` env var
or the trailing positional argument at server launch).

Not migrated from Orbit1's 1151-line filesystem server: symbol search, glob,
grep, directory tree, media reads, todo_read/write, web_fetch, unified-patch
application. Deferred to future slices consistent with Handoff 16's bounded
intent.

Orbit2-side family-aware governance metadata for these tools lives at
`src/capability/mcp/governance.py`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

SERVER_NAME = "filesystem"
WORKSPACE_ROOT_ENV = "ORBIT_WORKSPACE_ROOT"
DEFAULT_MAX_READ_BYTES = 64 * 1024
DEFAULT_MAX_DIRECTORY_ENTRIES = 200


def _workspace_root() -> Path:
    """Resolve the allowed filesystem root at tool-call time."""
    raw = os.environ.get(WORKSPACE_ROOT_ENV, "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
    elif len(sys.argv) > 1 and sys.argv[-1].strip():
        root = Path(sys.argv[-1]).expanduser().resolve()
    else:
        raise ValueError(
            f"filesystem MCP server requires allowed root via {WORKSPACE_ROOT_ENV} "
            "env var or trailing positional arg"
        )
    if not root.exists() or not root.is_dir():
        raise ValueError(f"workspace root is invalid: {root}")
    return root


def _resolve_safe_path(path: str) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError("absolute paths are not allowed")
    workspace_root = _workspace_root()
    target = (workspace_root / candidate).resolve()
    try:
        target.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError("path escapes workspace") from exc
    return target


def _resolve_safe_existing_file(path: str) -> Path:
    target = _resolve_safe_path(path)
    if not target.exists():
        raise ValueError("path not found")
    if not target.is_file():
        raise ValueError("path is not a file")
    return target


# ---------------------------------------------------------------------------
# Pure result helpers (importable by tests without running the MCP server)
# ---------------------------------------------------------------------------


def _read_file_result(path: str, *, max_bytes: int | None = None) -> dict[str, Any]:
    target = _resolve_safe_existing_file(path)
    cap = max_bytes if isinstance(max_bytes, int) and max_bytes > 0 else DEFAULT_MAX_READ_BYTES
    data = target.read_bytes()
    truncated = len(data) > cap
    payload = data[:cap].decode("utf-8", errors="replace")
    return {
        "ok": True,
        "path": target.as_posix(),
        "content": payload,
        "truncated": truncated,
        "byte_size": len(data),
    }


def _list_directory_result(path: str, *, max_entries: int | None = None) -> dict[str, Any]:
    target = _resolve_safe_path(path)
    if not target.exists() or not target.is_dir():
        raise ValueError("path is not a directory")
    cap = max_entries if isinstance(max_entries, int) and max_entries > 0 else DEFAULT_MAX_DIRECTORY_ENTRIES
    entries: list[dict[str, Any]] = []
    for child in sorted(target.iterdir(), key=lambda p: p.name):
        entries.append({
            "name": child.name,
            "kind": "directory" if child.is_dir() else ("file" if child.is_file() else "other"),
        })
        if len(entries) >= cap:
            break
    return {
        "ok": True,
        "path": target.as_posix(),
        "entries": entries,
        "truncated": len(entries) >= cap,
    }


def _get_file_info_result(path: str) -> dict[str, Any]:
    target = _resolve_safe_path(path)
    if not target.exists():
        raise ValueError("path not found")
    stat = target.stat()
    return {
        "ok": True,
        "path": target.as_posix(),
        "kind": "directory" if target.is_dir() else ("file" if target.is_file() else "other"),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def _write_file_result(path: str, content: str) -> dict[str, Any]:
    target = _resolve_safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {
        "ok": True,
        "path": target.as_posix(),
        "mutation_kind": "write_file",
    }


def _replace_in_file_result(path: str, old_text: str, new_text: str) -> dict[str, Any]:
    target = _resolve_safe_existing_file(path)
    current = target.read_text(encoding="utf-8")
    if old_text not in current:
        return {
            "ok": False,
            "path": target.as_posix(),
            "mutation_kind": "replace_in_file",
            "failure_kind": "old_text_not_found",
            "replacement_count": 0,
        }
    updated = current.replace(old_text, new_text, 1)
    target.write_text(updated, encoding="utf-8")
    return {
        "ok": True,
        "path": target.as_posix(),
        "mutation_kind": "replace_in_file",
        "replacement_count": 1,
    }


# ---------------------------------------------------------------------------
# FastMCP server wiring
# ---------------------------------------------------------------------------


mcp = FastMCP(SERVER_NAME)


@mcp.tool()
def read_file(path: str) -> dict[str, Any]:
    """Read a workspace-relative text file. Returns up to 64 KiB."""
    return _read_file_result(path)


@mcp.tool()
def list_directory(path: str) -> dict[str, Any]:
    """List entries under a workspace-relative directory (first 200)."""
    return _list_directory_result(path)


@mcp.tool()
def get_file_info(path: str) -> dict[str, Any]:
    """Return size / mtime / kind for a workspace-relative path."""
    return _get_file_info_result(path)


@mcp.tool()
def write_file(path: str, content: str) -> dict[str, Any]:
    """Write UTF-8 text to a workspace-relative path. Creates parent dirs."""
    return _write_file_result(path, content)


@mcp.tool()
def replace_in_file(path: str, old_text: str, new_text: str) -> dict[str, Any]:
    """Replace the first occurrence of `old_text` with `new_text`."""
    return _replace_in_file_result(path, old_text, new_text)


if __name__ == "__main__":
    mcp.run()
