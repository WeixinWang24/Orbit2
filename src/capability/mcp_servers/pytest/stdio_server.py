"""Bounded pytest MCP server for Orbit2 (Handoff 17, first slice).

Ships one tool only: `run_pytest_structured(args, max_chars)`. Invokes pytest
in the workspace with a single target expression (or default pytest discovery),
captures stdout + stderr + exit code, and truncates to a caller-supplied cap.

Not migrated from Orbit1's 717-line pytest server: structured JSON parsing,
warnings classification, capability-gated test-run semantics, pytest plugin
orchestration. Those land in a future per-family slice when Orbit2 needs
them; the first slice is the bounded shell-out.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

SERVER_NAME = "pytest"
WORKSPACE_ROOT_ENV = "ORBIT_WORKSPACE_ROOT"
DEFAULT_MAX_CHARS = 12_000
PYTEST_TIMEOUT_SECONDS = 120.0


def _workspace_root() -> Path:
    raw = os.environ.get(WORKSPACE_ROOT_ENV, "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
    elif len(sys.argv) > 1 and sys.argv[-1].strip():
        root = Path(sys.argv[-1]).expanduser().resolve()
    else:
        raise ValueError(
            f"pytest MCP server requires allowed root via {WORKSPACE_ROOT_ENV} "
            "env var or trailing positional arg"
        )
    if not root.exists() or not root.is_dir():
        raise ValueError(f"workspace root is invalid: {root}")
    return root


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _run_pytest_structured_result(
    *,
    args: list[str] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout_seconds: float = PYTEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    workspace = _workspace_root()
    cli_args = _validate_args(args)
    if not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", *cli_args],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "cwd": str(workspace),
            "args": cli_args,
            "returncode": None,
            "stdout": "",
            "stderr": f"pytest timed out after {timeout_seconds}s",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "failure_kind": "pytest_timeout",
            "timeout_seconds": timeout_seconds,
            "_exc": repr(exc),
        }
    stdout, stdout_truncated = _truncate(completed.stdout or "", max_chars)
    stderr, stderr_truncated = _truncate(completed.stderr or "", max_chars)
    return {
        "ok": completed.returncode == 0,
        "cwd": str(workspace),
        "args": cli_args,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "failure_kind": None if completed.returncode == 0 else "pytest_nonzero_exit",
    }


def _validate_args(args: list[str] | None) -> list[str]:
    """Reject absolute-path args as basic defense-in-depth so a caller cannot
    trivially steer pytest/ruff/mypy at paths outside the workspace via the
    forwarded `args` list. Workspace-containment of relative paths still
    relies on the server running with cwd=workspace.
    """
    cli_args = list(args) if args else []
    for a in cli_args:
        if not isinstance(a, str):
            raise ValueError(f"args entries must be strings; got {type(a).__name__}")
        if a.startswith("/") or a.startswith("\\\\"):
            raise ValueError(f"absolute paths are not allowed in args: {a!r}")
    return cli_args


mcp = FastMCP(SERVER_NAME)


@mcp.tool()
def run_pytest_structured(
    args: list[str] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    """Run pytest inside the workspace and return bounded stdout/stderr output.

    `args` forwards directly to pytest (e.g. `["-q", "tests/test_x.py"]`).
    Use `max_chars` to bound captured output per stream.
    """
    return _run_pytest_structured_result(args=args, max_chars=max_chars)


if __name__ == "__main__":
    mcp.run()
