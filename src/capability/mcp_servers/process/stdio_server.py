"""Bounded process MCP server for Orbit2 (Handoff 24, first slice).

Single tool: `run_process` — synchronous, workspace-scoped subprocess
execution with bounded output and timeout. All calls are approval-required
per the family-aware governance overlay at `src/capability/mcp/governance.py`.

Intentionally simpler than Orbit1's persistent-handle model (start_process /
read_process_output / wait_process / terminate_process). The persistent-handle
API requires a full background-process runtime that is out of scope for this
first slice. `run_process` covers the common synchronous use case cleanly
within Orbit2's existing approval-gate discipline.

The command list is run via `subprocess.run`. The cwd must resolve inside the
workspace root (`ORBIT_WORKSPACE_ROOT` env var or trailing positional arg).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.capability.mcp_servers.timing import timed_mcp_tool

SERVER_NAME = "process"
WORKSPACE_ROOT_ENV = "ORBIT_WORKSPACE_ROOT"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_OUTPUT_CHARS = 12_000


def _workspace_root() -> Path:
    raw = os.environ.get(WORKSPACE_ROOT_ENV, "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
    elif len(sys.argv) > 1 and sys.argv[-1].strip():
        root = Path(sys.argv[-1]).expanduser().resolve()
    else:
        raise ValueError(
            f"process MCP server requires allowed root via {WORKSPACE_ROOT_ENV} "
            "env var or trailing positional arg"
        )
    if not root.exists() or not root.is_dir():
        raise ValueError(f"workspace root is invalid: {root}")
    return root


def _resolve_cwd(cwd: str | None) -> Path:
    workspace = _workspace_root()
    if cwd is None or not str(cwd).strip() or cwd == ".":
        return workspace
    candidate = Path(cwd)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (workspace / candidate).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("cwd escapes workspace") from exc
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError("cwd is not an existing directory")
    return resolved


def _run_process_result(
    command: list[str],
    cwd: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> dict[str, Any]:
    if not isinstance(command, list) or not command:
        raise ValueError("command must be a non-empty list of strings")
    for item in command:
        if not isinstance(item, str) or not item:
            raise ValueError("command items must be non-empty strings")
    resolved = _resolve_cwd(cwd)
    timeout = (
        float(timeout_seconds)
        if isinstance(timeout_seconds, (int, float)) and timeout_seconds > 0
        else DEFAULT_TIMEOUT_SECONDS
    )
    cap = (
        int(max_output_chars)
        if isinstance(max_output_chars, int) and max_output_chars > 0
        else DEFAULT_MAX_OUTPUT_CHARS
    )
    try:
        completed = subprocess.run(
            command,
            cwd=str(resolved),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "cwd": str(resolved),
            "command": command,
            "mutation_kind": "run_process",
            "failure_kind": "timeout",
            "timed_out": True,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "stdout_truncated": False,
            "stderr_truncated": False,
        }
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    return {
        "ok": True,
        "cwd": str(resolved),
        "command": command,
        "mutation_kind": "run_process",
        "exit_code": completed.returncode,
        "stdout": stdout[:cap],
        "stderr": stderr[:cap],
        "timed_out": False,
        "stdout_truncated": len(stdout) > cap,
        "stderr_truncated": len(stderr) > cap,
    }


mcp = FastMCP(SERVER_NAME)


@timed_mcp_tool(mcp, SERVER_NAME)
def run_process(
    command: list[str],
    cwd: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> dict[str, Any]:
    """Run a command synchronously in the workspace. Returns exit_code, stdout, stderr."""
    return _run_process_result(command, cwd, timeout_seconds, max_output_chars)


if __name__ == "__main__":
    mcp.run()
