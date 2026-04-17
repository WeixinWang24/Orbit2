"""Bounded ruff MCP server for Orbit2 (Handoff 17, first slice).

Ships one tool only: `run_ruff_structured(args, max_chars)`. Invokes ruff
(expected on PATH in the Orbit conda env) inside the workspace, captures
stdout/stderr and exit code, truncates per caller cap.

Not migrated from Orbit1's 302-line ruff server: rule classification,
finding-severity rollup, auto-fix semantics. Future per-family slice.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

SERVER_NAME = "ruff"
WORKSPACE_ROOT_ENV = "ORBIT_WORKSPACE_ROOT"
DEFAULT_MAX_CHARS = 12_000
RUFF_TIMEOUT_SECONDS = 60.0


def _workspace_root() -> Path:
    raw = os.environ.get(WORKSPACE_ROOT_ENV, "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
    elif len(sys.argv) > 1 and sys.argv[-1].strip():
        root = Path(sys.argv[-1]).expanduser().resolve()
    else:
        raise ValueError(
            f"ruff MCP server requires allowed root via {WORKSPACE_ROOT_ENV} "
            "env var or trailing positional arg"
        )
    if not root.exists() or not root.is_dir():
        raise ValueError(f"workspace root is invalid: {root}")
    return root


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _ruff_binary() -> str:
    binary = shutil.which("ruff")
    if binary is None:
        raise ValueError("ruff binary not found on PATH")
    return binary


def _validate_args(args: list[str] | None, default: list[str]) -> list[str]:
    """Reject absolute-path args in the forwarded CLI args (defense-in-depth).
    See the equivalent helper in the pytest server for rationale.
    """
    cli_args = list(args) if args else list(default)
    for a in cli_args:
        if not isinstance(a, str):
            raise ValueError(f"args entries must be strings; got {type(a).__name__}")
        if a.startswith("/") or a.startswith("\\\\"):
            raise ValueError(f"absolute paths are not allowed in args: {a!r}")
    return cli_args


def _run_ruff_structured_result(
    *,
    args: list[str] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout_seconds: float = RUFF_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    workspace = _workspace_root()
    cli_args = _validate_args(args, default=["check", "."])
    if not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    try:
        completed = subprocess.run(
            [_ruff_binary(), *cli_args],
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
            "stderr": f"ruff timed out after {timeout_seconds}s",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "failure_kind": "ruff_timeout",
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
        "failure_kind": None if completed.returncode == 0 else "ruff_nonzero_exit",
    }


mcp = FastMCP(SERVER_NAME)


@mcp.tool()
def run_ruff_structured(
    args: list[str] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    """Run ruff inside the workspace and return bounded output.

    `args` forwards directly to ruff (default: `["check", "."]`).
    """
    return _run_ruff_structured_result(args=args, max_chars=max_chars)


if __name__ == "__main__":
    mcp.run()
