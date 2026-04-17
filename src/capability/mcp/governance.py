"""Family-aware MCP governance overlay for Orbit2 (Handoff 16).

Scope is bounded to two concrete MCP families: `filesystem` and `git`. Other
Orbit1 families (bash, process, browser, pytest, ruff, mypy, obsidian) are
explicitly NOT migrated here; unknown server/tool pairs fall through to the
conservative `DEFAULT_MCP_GOVERNANCE`.

This overlay is Orbit2-side only: it classifies what the runtime believes
about a tool's side effects so that Capability Surface closure and future
Governance Surface approval gating have accurate per-tool metadata rather
than treating every MCP call as a generic opaque action.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypedDict


class McpGovernanceMetadata(TypedDict):
    side_effect_class: str
    requires_approval: bool
    governance_policy_group: str
    environment_check_kind: str


DEFAULT_MCP_GOVERNANCE: McpGovernanceMetadata = {
    "side_effect_class": "unknown",
    "requires_approval": True,
    "governance_policy_group": "permission_authority",
    "environment_check_kind": "none",
}


# Filesystem family — bounded first-slice tool set.
# Keep these sets synchronized with src/mcp_servers/filesystem/stdio_server.py.
FILESYSTEM_READ_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "list_directory",
    "get_file_info",
})

FILESYSTEM_WRITE_TOOLS: frozenset[str] = frozenset({
    "write_file",
    "replace_in_file",
})

# Tools whose semantics require the target path to already exist.
_FILESYSTEM_PATH_EXISTS_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "list_directory",
    "get_file_info",
    "replace_in_file",
})


# Git family — bounded first-slice tool set.
# Keep these sets synchronized with src/mcp_servers/git/stdio_server.py.
GIT_READ_TOOLS: frozenset[str] = frozenset({
    "git_status",
    "git_diff",
    "git_log",
})

GIT_WRITE_TOOLS: frozenset[str] = frozenset({
    "git_add",
    "git_commit",
})


def resolve_mcp_tool_governance(
    *, server_name: str, original_tool_name: str
) -> McpGovernanceMetadata:
    """Classify a known MCP tool by its family. Returns the conservative
    default for any server/tool pair that this first slice does not recognize.
    Callers should NOT use this to decide whether a tool is "safe to call
    without a boundary" — it is governance metadata, not a capability gate.
    """
    server = server_name.strip().lower()
    tool = original_tool_name.strip().lower()

    if server == "filesystem":
        if tool in FILESYSTEM_READ_TOOLS:
            return {
                "side_effect_class": "safe",
                "requires_approval": False,
                "governance_policy_group": "system_environment",
                "environment_check_kind": (
                    "path_exists" if tool in _FILESYSTEM_PATH_EXISTS_TOOLS else "none"
                ),
            }
        if tool in FILESYSTEM_WRITE_TOOLS:
            return {
                "side_effect_class": "write",
                "requires_approval": True,
                "governance_policy_group": "permission_authority",
                "environment_check_kind": (
                    "path_exists" if tool in _FILESYSTEM_PATH_EXISTS_TOOLS else "none"
                ),
            }

    if server == "git":
        if tool in GIT_READ_TOOLS:
            return {
                "side_effect_class": "safe",
                "requires_approval": False,
                "governance_policy_group": "system_environment",
                "environment_check_kind": "none",
            }
        if tool in GIT_WRITE_TOOLS:
            return {
                "side_effect_class": "write",
                "requires_approval": True,
                "governance_policy_group": "permission_authority",
                "environment_check_kind": "none",
            }

    return dict(DEFAULT_MCP_GOVERNANCE)


# ---------------------------------------------------------------------------
# Filesystem target-path resolution
#
# Filesystem MCP calls may pass the target path under `path`; the effective
# path depends on the server's allowed-root (workspace root). Orbit2 needs
# to resolve that here so it can check whether the MCP call is touching
# workspace state vs escaping the boundary, without relying on the MCP
# server implementation itself to enforce the check.
# ---------------------------------------------------------------------------


WORKSPACE_ROOT_ENV = "ORBIT_WORKSPACE_ROOT"


def filesystem_server_allowed_root(
    server_args: list[str] | tuple[str, ...] | None,
    server_env: dict[str, str] | None = None,
) -> Path | None:
    """Resolve the allowed filesystem root the MCP server will operate under.

    Preference order (matches Orbit1):
    1. `ORBIT_WORKSPACE_ROOT` environment variable in the server env.
    2. The last positional argument in `server_args`, if any.
    3. None (unresolved — caller must treat this as a governance failure).
    """
    env = server_env or {}
    env_root = env.get(WORKSPACE_ROOT_ENV)
    if isinstance(env_root, str) and env_root.strip():
        try:
            return Path(env_root).resolve()
        except OSError:
            return None
    if server_args:
        candidate = server_args[-1]
        if isinstance(candidate, str) and candidate.strip():
            try:
                return Path(candidate).resolve()
            except OSError:
                return None
    return None


def resolve_filesystem_mcp_target_path(
    *,
    input_payload: dict[str, Any],
    server_args: list[str] | tuple[str, ...] | None,
    server_env: dict[str, str] | None = None,
) -> Path | None:
    """Resolve the effective filesystem target path for a filesystem MCP call.

    Returns None when the payload has no `path` (or it is empty), or when
    the allowed root cannot be determined — callers must treat either as
    "no filesystem target to govern". Returns the resolved absolute Path
    otherwise; the caller is responsible for enforcing containment.
    """
    allowed_root = filesystem_server_allowed_root(server_args, server_env)
    if allowed_root is None:
        return None
    path_value = input_payload.get("path")
    if path_value == "":
        path_value = "."
    if not isinstance(path_value, str) or not path_value:
        return None
    path_obj = Path(path_value)
    return path_obj.resolve() if path_obj.is_absolute() else (allowed_root / path_obj).resolve()
