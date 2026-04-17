"""Family-aware MCP governance overlay for Orbit2.

Handoff 16 introduced this overlay for `filesystem` and `git`. Handoff 17
extends it to the remaining high-value Orbit1 families — `process`,
`browser`, `pytest`, `ruff`, `mypy`, and `obsidian` — so they no longer
fall through to the conservative `DEFAULT_MCP_GOVERNANCE` when their
semantics are already well-understood. The `bash` family (also present in
Orbit1) remains intentionally unrecognized: its semantics are broad enough
to warrant a dedicated future slice.

Recognition here is governance metadata only. It does NOT imply Orbit2
ships a server implementation for the family — `process`, `browser`, and
`obsidian` are governance-overlay-only this slice because their servers
have out-of-scope external dependencies (persistent-process runtime,
playwright, vault integration). Those server implementations are tracked
as deferred follow-ups.

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
# Keep these sets synchronized with src/capability/mcp_servers/filesystem/stdio_server.py.
FILESYSTEM_READ_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "list_directory",
    "get_file_info",
    # Handoff 23 widening — all safe / no-approval, all workspace-scoped
    # through the same `_resolve_safe_path` path discipline.
    "glob",
    "search_files",
    "grep",
    "directory_tree",
    "read_multiple_files",
    "list_directory_with_sizes",
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
    # Handoff 23 widening read tools that operate under an existing path.
    "list_directory_with_sizes",
    "directory_tree",
    "search_files",
    "glob",
    "grep",
})


# Git family — bounded first-slice tool set.
# Keep these sets synchronized with src/capability/mcp_servers/git/stdio_server.py.
GIT_READ_TOOLS: frozenset[str] = frozenset({
    "git_status",
    "git_diff",
    "git_log",
})

GIT_WRITE_TOOLS: frozenset[str] = frozenset({
    "git_add",
    "git_commit",
})


# Process family (Handoff 17) — spawn / observe / control workspace-scoped
# subprocesses. All mutation-bearing by Orbit1 convention; approval-required.
# NOTE: `read_process_output` is classified as write/approval-required despite
# its "read" verb because it advances a stateful read offset on the process's
# persisted output files (non-idempotent) and depends on a prior spawn having
# already happened. Treating it as a safe read would be misleading at the
# approval layer.
# Server implementation NOT migrated this slice (Orbit1's version depends on
# a full persistent-process runtime); governance metadata allows callers who
# bring their own compatible server to be classified correctly.
PROCESS_WRITE_TOOLS: frozenset[str] = frozenset({
    "start_process",
    "read_process_output",
    "wait_process",
    "terminate_process",
})


# Browser family (Handoff 17) — control a headless browser through MCP.
# Split by actual side-effect semantics rather than Orbit1's coarser "all safe"
# classification: `browser_open` / `browser_snapshot` / `browser_console` /
# `browser_screenshot` are informational and produce no remote-state mutation,
# but `browser_click` and `browser_type` DO mutate remote state (submit forms,
# trigger actions, exfiltrate data). Orbit2 treats the latter as
# write/approval-required so a future Governance-Surface approval gate sees
# them correctly.
# Server implementation NOT migrated this slice (requires playwright).
BROWSER_READ_TOOLS: frozenset[str] = frozenset({
    "browser_open",
    "browser_snapshot",
    "browser_console",
    "browser_screenshot",
})

BROWSER_WRITE_TOOLS: frozenset[str] = frozenset({
    "browser_click",
    "browser_type",
})


# Pytest family (Handoff 17) — structured test-runner diagnostics.
# Classified safe / no-approval so a coding agent can run tests without
# approval gates; test execution does spawn arbitrary user code and may
# write `.pytest_cache`, accepted as benign for first-slice diagnostics.
PYTEST_READ_TOOLS: frozenset[str] = frozenset({
    "run_pytest_structured",
})


# Ruff family (Handoff 17) — structured lint diagnostics. Pure read/safe.
RUFF_READ_TOOLS: frozenset[str] = frozenset({
    "run_ruff_structured",
})


# Mypy family (Handoff 17) — structured static-type diagnostics. Pure read/safe.
MYPY_READ_TOOLS: frozenset[str] = frozenset({
    "run_mypy_structured",
})


# Obsidian family (Handoff 17) — vault read surfaces. All safe/no-approval.
# Server implementation NOT migrated this slice (requires vault integration).
OBSIDIAN_READ_TOOLS: frozenset[str] = frozenset({
    "obsidian_list_notes",
    "obsidian_read_note",
    "obsidian_search_notes",
    "obsidian_get_note_links",
    "obsidian_get_vault_metadata",
    "obsidian_check_availability",
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

    if server == "process" and tool in PROCESS_WRITE_TOOLS:
        return {
            "side_effect_class": "write",
            "requires_approval": True,
            "governance_policy_group": "permission_authority",
            "environment_check_kind": "none",
        }

    if server == "browser":
        if tool in BROWSER_READ_TOOLS:
            return {
                "side_effect_class": "safe",
                "requires_approval": False,
                "governance_policy_group": "system_environment",
                "environment_check_kind": "none",
            }
        if tool in BROWSER_WRITE_TOOLS:
            return {
                "side_effect_class": "write",
                "requires_approval": True,
                "governance_policy_group": "permission_authority",
                "environment_check_kind": "none",
            }

    if server == "pytest" and tool in PYTEST_READ_TOOLS:
        return {
            "side_effect_class": "safe",
            "requires_approval": False,
            "governance_policy_group": "system_environment",
            "environment_check_kind": "none",
        }

    if server == "ruff" and tool in RUFF_READ_TOOLS:
        return {
            "side_effect_class": "safe",
            "requires_approval": False,
            "governance_policy_group": "system_environment",
            "environment_check_kind": "none",
        }

    if server == "mypy" and tool in MYPY_READ_TOOLS:
        return {
            "side_effect_class": "safe",
            "requires_approval": False,
            "governance_policy_group": "system_environment",
            "environment_check_kind": "none",
        }

    if server == "obsidian" and tool in OBSIDIAN_READ_TOOLS:
        return {
            "side_effect_class": "safe",
            "requires_approval": False,
            "governance_policy_group": "system_environment",
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
