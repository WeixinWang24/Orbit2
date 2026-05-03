"""Unified Orbit2 MCP server for Codex-side attachment.

This server keeps the existing layered MCP implementations intact and exposes
them through one external server namespace named `orbit2`.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.capability.mcp_servers.timing import timed_mcp_tool

from src.capability.mcp_servers.filesystem import stdio_server as filesystem
from src.capability.mcp_servers.git import stdio_server as git
from src.capability.mcp_servers.l1_structured.filesystem import (
    stdio_server as structured_filesystem,
)
from src.capability.mcp_servers.l1_structured.git import stdio_server as structured_git
from src.capability.mcp_servers.l2_toolchain.code_intel import (
    stdio_server as code_intel_toolchain,
)
from src.capability.mcp_servers.l2_toolchain.pytest import stdio_server as pytest_toolchain
from src.capability.mcp_servers.l2_toolchain.repo_scout import (
    stdio_server as repo_scout_toolchain,
)
from src.capability.mcp_servers.l3_workflow import stdio_server as workflow
from src.capability.mcp_servers.mypy import stdio_server as mypy
from src.capability.mcp_servers.process import stdio_server as process
from src.capability.mcp_servers.ruff import stdio_server as ruff


SERVER_NAME = "orbit2"
PROFILE_ENV = "ORBIT2_EXTERNAL_MCP_PROFILE"
PROFILE_FULL = "full"
PROFILE_READ_ONLY = "read-only"
SUPPORTED_PROFILES = (PROFILE_FULL, PROFILE_READ_ONLY)


def _configure_from_argv() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace")
    parser.add_argument("--vault")
    parser.add_argument(
        "--profile",
        choices=SUPPORTED_PROFILES,
        default=os.environ.get(PROFILE_ENV, PROFILE_FULL),
    )
    parser.add_argument("workspace_positional", nargs="?")
    args, _unknown = parser.parse_known_args(sys.argv[1:])
    workspace = (args.workspace or args.workspace_positional or "").strip()
    if workspace:
        os.environ.setdefault("ORBIT_WORKSPACE_ROOT", workspace)
    vault = (args.vault or "").strip()
    if vault:
        os.environ.setdefault("ORBIT_OBSIDIAN_VAULT_ROOT", vault)
    if args.profile not in SUPPORTED_PROFILES:
        raise SystemExit(
            f"{PROFILE_ENV} must be one of: {', '.join(SUPPORTED_PROFILES)}"
        )
    return args.profile


def _obsidian() -> Any:
    if not os.environ.get("ORBIT_OBSIDIAN_VAULT_ROOT", "").strip():
        raise ValueError(
            "Orbit2 Obsidian tools require ORBIT_OBSIDIAN_VAULT_ROOT or --vault"
        )
    from src.capability.mcp_servers.obsidian import stdio_server as obsidian

    return obsidian


_PROFILE = _configure_from_argv()
mcp = FastMCP(SERVER_NAME)


def _read_tool():
    return timed_mcp_tool(mcp, SERVER_NAME)


def _mutation_tool():
    def decorator(fn):
        if _PROFILE == PROFILE_READ_ONLY:
            return fn
        return timed_mcp_tool(mcp, SERVER_NAME)(fn)

    return decorator


@_read_tool()
def orbit2_filesystem_read_file(path: str) -> dict[str, Any]:
    """Read a workspace-relative file through Orbit2 filesystem MCP."""
    return filesystem._read_file_result(path)


@_read_tool()
def orbit2_filesystem_list_directory(path: str) -> dict[str, Any]:
    """List a workspace-relative directory through Orbit2 filesystem MCP."""
    return filesystem._list_directory_result(path)


@_read_tool()
def orbit2_filesystem_get_file_info(path: str) -> dict[str, Any]:
    """Return metadata for a workspace-relative file or directory."""
    return filesystem._get_file_info_result(path)


@_mutation_tool()
def orbit2_filesystem_write_file(path: str, content: str) -> dict[str, Any]:
    """Write a workspace-relative file through Orbit2 filesystem MCP."""
    return filesystem._write_file_result(path, content)


@_mutation_tool()
def orbit2_filesystem_replace_in_file(
    path: str,
    old_text: str,
    new_text: str,
) -> dict[str, Any]:
    """Replace one exact text occurrence in a workspace-relative file."""
    return filesystem._replace_in_file_result(path, old_text, new_text)


@_mutation_tool()
def orbit2_filesystem_replace_all_in_file(
    path: str,
    old_text: str,
    new_text: str,
) -> dict[str, Any]:
    """Replace all exact text occurrences in a workspace-relative file."""
    return filesystem._replace_all_in_file_result(path, old_text, new_text)


@_mutation_tool()
def orbit2_filesystem_create_directory(path: str) -> dict[str, Any]:
    """Create a workspace-relative directory."""
    return filesystem._create_directory_result(path)


@_mutation_tool()
def orbit2_filesystem_move_file(source: str, destination: str) -> dict[str, Any]:
    """Move a workspace-relative file or directory."""
    return filesystem._move_file_result(source, destination)


@_read_tool()
def orbit2_filesystem_glob(
    pattern: str,
    path: str = ".",
    max_results: int | None = None,
) -> dict[str, Any]:
    """Run a bounded workspace-relative glob."""
    return filesystem._glob_result(pattern, path=path, max_results=max_results)


@_read_tool()
def orbit2_filesystem_search_files(
    path: str,
    pattern: str,
    max_results: int | None = None,
) -> dict[str, Any]:
    """Search workspace-relative filenames under a directory."""
    return filesystem._search_files_result(path, pattern, max_results=max_results)


@_read_tool()
def orbit2_filesystem_grep(
    path: str,
    pattern: str,
    max_results: int | None = None,
    context_lines: int = 0,
) -> dict[str, Any]:
    """Run bounded grep under a workspace-relative file or directory."""
    return filesystem._grep_result(
        path,
        pattern,
        max_results=max_results,
        context_lines=context_lines,
    )


@_read_tool()
def orbit2_filesystem_directory_tree(
    path: str,
    max_depth: int | None = None,
    max_entries: int | None = None,
) -> dict[str, Any]:
    """Return a bounded workspace-relative directory tree."""
    return filesystem._directory_tree_result(
        path,
        max_depth=max_depth,
        max_entries=max_entries,
    )


@_read_tool()
def orbit2_filesystem_read_multiple_files(paths: list[str]) -> dict[str, Any]:
    """Read multiple workspace-relative files without failing the whole call."""
    return filesystem._read_multiple_files_result(paths)


@_read_tool()
def orbit2_filesystem_list_directory_with_sizes(
    path: str,
    max_entries: int | None = None,
) -> dict[str, Any]:
    """List a directory with bounded size metadata."""
    return filesystem._list_directory_with_sizes_result(path, max_entries=max_entries)


@_read_tool()
def orbit2_git_status(cwd: str | None = None) -> dict[str, Any]:
    """Return structured git status for the Orbit2 workspace."""
    return git._git_status_result(cwd)


@_read_tool()
def orbit2_git_diff(
    cwd: str | None = None,
    path: str | None = None,
    staged: bool = False,
    max_chars: int = git.DEFAULT_MAX_DIFF_CHARS,
) -> dict[str, Any]:
    """Return bounded git diff output."""
    return git._git_diff_result(cwd=cwd, path=path, staged=staged, max_chars=max_chars)


@_read_tool()
def orbit2_git_log(cwd: str | None = None, limit: int = 10) -> dict[str, Any]:
    """Return a bounded git log summary."""
    return git._git_log_result(cwd=cwd, limit=limit)


@_read_tool()
def orbit2_git_show(
    rev: str,
    cwd: str | None = None,
    max_chars: int = git.DEFAULT_MAX_DIFF_CHARS,
) -> dict[str, Any]:
    """Return bounded git show output for a revision or rev:path."""
    return git._git_show_result(rev, cwd=cwd, max_chars=max_chars)


@_read_tool()
def orbit2_git_changed_files(cwd: str | None = None) -> dict[str, Any]:
    """Return changed files from git status porcelain."""
    return git._git_changed_files_result(cwd)


@_mutation_tool()
def orbit2_git_restore(paths: list[str], cwd: str | None = None) -> dict[str, Any]:
    """Run git restore on workspace-relative paths."""
    return git._git_restore_result(paths, cwd=cwd)


@_mutation_tool()
def orbit2_git_unstage(paths: list[str], cwd: str | None = None) -> dict[str, Any]:
    """Run git restore --staged on workspace-relative paths."""
    return git._git_unstage_result(paths, cwd=cwd)


@_mutation_tool()
def orbit2_git_checkout_branch(branch: str, cwd: str | None = None) -> dict[str, Any]:
    """Checkout an existing git branch."""
    return git._git_checkout_branch_result(branch, cwd=cwd)


@_mutation_tool()
def orbit2_git_add(paths: list[str], cwd: str | None = None) -> dict[str, Any]:
    """Run git add on workspace-relative paths."""
    return git._git_add_result(paths=paths, cwd=cwd)


@_mutation_tool()
def orbit2_git_commit(message: str, cwd: str | None = None) -> dict[str, Any]:
    """Create a git commit with the current index."""
    return git._git_commit_result(message, cwd=cwd)


@_mutation_tool()
def orbit2_process_run_process(
    command: list[str],
    cwd: str | None = None,
    timeout_seconds: float = process.DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = process.DEFAULT_MAX_OUTPUT_CHARS,
) -> dict[str, Any]:
    """Run a bounded synchronous workspace-scoped process."""
    return process._run_process_result(command, cwd, timeout_seconds, max_output_chars)


@_read_tool()
def orbit2_structured_filesystem_read_file_region(
    path: str,
    start_line: int,
    end_line: int,
    evidence_gap: dict[str, Any],
    reason_context_pack_insufficient: str,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Read a bounded evidence-bearing file line region."""
    return structured_filesystem._read_file_region_result(
        path=path,
        start_line=start_line,
        end_line=end_line,
        evidence_gap=evidence_gap,
        reason_context_pack_insufficient=reason_context_pack_insufficient,
        max_chars=max_chars,
    )


@_read_tool()
def orbit2_structured_filesystem_grep_scoped(
    path: str,
    pattern: str,
    evidence_gap_description: str,
    needed_evidence: str,
    reason_context_pack_insufficient: str,
    max_matches: int | None = None,
    context_lines: int = 0,
    linked_context_pack_item: str | None = None,
) -> dict[str, Any]:
    """Run a bounded evidence-bearing scoped grep."""
    return structured_filesystem._grep_scoped_result(
        path=path,
        pattern=pattern,
        evidence_gap_description=evidence_gap_description,
        needed_evidence=needed_evidence,
        reason_context_pack_insufficient=reason_context_pack_insufficient,
        max_matches=max_matches,
        context_lines=context_lines,
        linked_context_pack_item=linked_context_pack_item,
    )


@_read_tool()
def orbit2_structured_git_read_diff_hunk(
    path: str,
    hunk_index: int,
    evidence_gap: dict[str, Any],
    reason_context_pack_insufficient: str,
    cwd: str | None = None,
    staged: bool = False,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Read one bounded evidence-bearing git diff hunk."""
    return structured_git._read_diff_hunk_result(
        path=path,
        hunk_index=hunk_index,
        evidence_gap=evidence_gap,
        reason_context_pack_insufficient=reason_context_pack_insufficient,
        cwd=cwd,
        staged=staged,
        max_chars=max_chars,
    )


@_read_tool()
def orbit2_structured_git_read_git_show_region(
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
    """Read a bounded evidence-bearing file region from a git revision."""
    return structured_git._read_git_show_region_result(
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


@_read_tool()
def orbit2_pytest_run_pytest_structured(
    args: list[str] | None = None,
    max_chars: int = pytest_toolchain.DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    """Run pytest and return bounded stdout/stderr diagnostics."""
    return pytest_toolchain._run_pytest_structured_result(args=args, max_chars=max_chars)


@_read_tool()
def orbit2_pytest_diagnose_failures(
    args: list[str] | None = None,
    max_chars: int = pytest_toolchain.DEFAULT_MAX_CHARS,
    include_evidence: bool = True,
) -> dict[str, Any]:
    """Run the L2 pytest diagnostic recipe with persisted run trace."""
    return pytest_toolchain._pytest_diagnose_failures_result(
        args=args,
        max_chars=max_chars,
        include_evidence=include_evidence,
    )


@_read_tool()
def orbit2_pytest_toolchain_get_run(run_id: str) -> dict[str, Any]:
    """Read a persisted pytest L2 toolchain run."""
    return pytest_toolchain.toolchain_get_run(run_id)


@_read_tool()
def orbit2_pytest_toolchain_get_step(run_id: str, step_id: str) -> dict[str, Any]:
    """Read one persisted pytest L2 toolchain step."""
    return pytest_toolchain.toolchain_get_step(run_id, step_id)


@_read_tool()
def orbit2_pytest_toolchain_read_artifact_region(
    run_id: str,
    artifact_id: str,
    start_char: int = 0,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Read a bounded region from a persisted pytest L2 artifact."""
    return pytest_toolchain.toolchain_read_artifact_region(
        run_id,
        artifact_id,
        start_char=start_char,
        max_chars=max_chars,
    )


@_read_tool()
def orbit2_code_intel_repository_summary(
    path: str = ".",
    repo_id: str = "workspace",
    label: str | None = None,
) -> dict[str, Any]:
    """Index the workspace and return a compact Code Intelligence summary."""
    return code_intel_toolchain._repository_summary_result(
        path=path,
        repo_id=repo_id,
        label=label,
    )


@_read_tool()
def orbit2_code_intel_find_symbols(
    repo_id: str = "workspace",
    name: str | None = None,
    kind: str | None = None,
    path_prefix: str | None = None,
    limit: int | None = None,
    refresh_index: bool = True,
    path: str = ".",
    label: str | None = None,
) -> dict[str, Any]:
    """Find indexed symbols by name, kind, and path prefix."""
    return code_intel_toolchain._find_symbols_result(
        repo_id=repo_id,
        name=name,
        kind=kind,
        path_prefix=path_prefix,
        limit=limit,
        refresh_index=refresh_index,
        path=path,
        label=label,
    )


@_read_tool()
def orbit2_code_intel_file_context(
    path: str,
    repo_id: str = "workspace",
    refresh_index: bool = True,
    label: str | None = None,
    max_symbols: int | None = None,
    max_edges: int | None = None,
) -> dict[str, Any]:
    """Return symbols/imports/calls for one indexed workspace file."""
    return code_intel_toolchain._file_context_result(
        path=path,
        repo_id=repo_id,
        refresh_index=refresh_index,
        label=label,
        max_symbols=max_symbols,
        max_edges=max_edges,
    )


@_read_tool()
def orbit2_code_intel_export_fragment_summary(
    repo_id: str = "workspace",
    path: str = ".",
    label: str | None = None,
    refresh_index: bool = True,
    max_nodes: int | None = None,
    max_edges: int | None = None,
) -> dict[str, Any]:
    """Return a bounded detached Code Intelligence fragment summary."""
    return code_intel_toolchain._export_fragment_summary_result(
        repo_id=repo_id,
        path=path,
        label=label,
        refresh_index=refresh_index,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )


@_read_tool()
def orbit2_repo_scout_repository_overview(
    path: str = ".",
    repo_id: str = "workspace",
    label: str | None = None,
    max_tree_depth: int | None = None,
    max_tree_entries: int | None = None,
    max_symbols: int | None = None,
) -> dict[str, Any]:
    """Build summary-first repository overview context."""
    return repo_scout_toolchain._repo_scout_repository_overview_result(
        path=path,
        repo_id=repo_id,
        label=label,
        max_tree_depth=max_tree_depth,
        max_tree_entries=max_tree_entries,
        max_symbols=max_symbols,
    )


@_read_tool()
def orbit2_repo_scout_diff_digest(
    cwd: str | None = None,
    repo_id: str = "workspace",
    label: str | None = None,
    include_untracked: bool = True,
    max_diff_files: int | None = None,
) -> dict[str, Any]:
    """Build a summary-first fact digest of changed diffs and touched symbols."""
    return repo_scout_toolchain._repo_scout_diff_digest_result(
        cwd=cwd,
        repo_id=repo_id,
        label=label,
        include_untracked=include_untracked,
        max_diff_files=max_diff_files,
    )


@_read_tool()
def orbit2_repo_scout_impact_scope(
    cwd: str | None = None,
    repo_id: str = "workspace",
    label: str | None = None,
    include_untracked: bool = True,
    max_impact_files: int | None = None,
    max_impact_symbols: int | None = None,
    max_impact_edges: int | None = None,
) -> dict[str, Any]:
    """Build a non-decisional fact report of direct changed-symbol impact scope."""
    return repo_scout_toolchain._repo_scout_impact_scope_result(
        cwd=cwd,
        repo_id=repo_id,
        label=label,
        include_untracked=include_untracked,
        max_impact_files=max_impact_files,
        max_impact_symbols=max_impact_symbols,
        max_impact_edges=max_impact_edges,
    )


@_read_tool()
def orbit2_repo_scout_changed_context(
    cwd: str | None = None,
    repo_id: str = "workspace",
    label: str | None = None,
    include_untracked: bool = True,
    max_symbols_per_file: int | None = None,
    max_edges_per_file: int | None = None,
) -> dict[str, Any]:
    """Build summary-first changed-file Repo Scout context."""
    return repo_scout_toolchain._repo_scout_changed_context_result(
        cwd=cwd,
        repo_id=repo_id,
        label=label,
        include_untracked=include_untracked,
        max_symbols_per_file=max_symbols_per_file,
        max_edges_per_file=max_edges_per_file,
    )


@_read_tool()
def orbit2_repo_scout_toolchain_get_run(run_id: str) -> dict[str, Any]:
    """Read a persisted Repo Scout L2 toolchain run."""
    return repo_scout_toolchain.toolchain_get_run(run_id)


@_read_tool()
def orbit2_repo_scout_toolchain_get_step(
    run_id: str,
    step_id: str,
) -> dict[str, Any]:
    """Read one persisted Repo Scout L2 toolchain step."""
    return repo_scout_toolchain.toolchain_get_step(run_id, step_id)


@_read_tool()
def orbit2_repo_scout_toolchain_read_artifact_region(
    run_id: str,
    artifact_id: str,
    start_char: int = 0,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Read a bounded region from a persisted Repo Scout L2 artifact."""
    return repo_scout_toolchain.toolchain_read_artifact_region(
        run_id,
        artifact_id,
        start_char=start_char,
        max_chars=max_chars,
    )


@_read_tool()
def orbit2_workflow_inspect_change_set_workflow(
    cwd: str | None = None,
    repo_id: str = "workspace",
    label: str | None = None,
    include_untracked: bool = True,
    max_diff_files: int | None = None,
    max_impact_files: int | None = None,
    max_impact_symbols: int | None = None,
    max_impact_edges: int | None = None,
) -> dict[str, Any]:
    """Run the L3 change-set inspection workflow and emit a provider decision request.

    Includes Repo Scout diff digest and impact-scope facts; prefer this over
    separate diff_digest + impact_scope calls when the user asks to inspect a
    dirty change set and a provider-mediated next-step decision is desired.
    """
    return workflow.inspect_change_set_workflow(
        cwd=cwd,
        repo_id=repo_id,
        label=label,
        include_untracked=include_untracked,
        max_diff_files=max_diff_files,
        max_impact_files=max_impact_files,
        max_impact_symbols=max_impact_symbols,
        max_impact_edges=max_impact_edges,
    )


@_read_tool()
def orbit2_workflow_repo_recon_workflow(
    cwd: str | None = None,
    repo_id: str = "workspace",
    label: str | None = None,
    include_untracked: bool = True,
    max_tree_depth: int | None = None,
    max_tree_entries: int | None = None,
    max_symbols: int | None = None,
    max_symbols_per_file: int | None = None,
    max_edges_per_file: int | None = None,
    max_diff_files: int | None = None,
) -> dict[str, Any]:
    """Run the L3 repository reconnaissance workflow and emit a provider decision request.

    Includes repository overview, changed workspace context, and dirty-workspace
    diff digest; prefer this over separate repository_overview + git_status +
    git_log calls when the user asks to scout or inspect the current workspace.
    """
    return workflow.repo_recon_workflow(
        cwd=cwd,
        repo_id=repo_id,
        label=label,
        include_untracked=include_untracked,
        max_tree_depth=max_tree_depth,
        max_tree_entries=max_tree_entries,
        max_symbols=max_symbols,
        max_symbols_per_file=max_symbols_per_file,
        max_edges_per_file=max_edges_per_file,
        max_diff_files=max_diff_files,
    )


@_read_tool()
def orbit2_ruff_run_ruff_structured(
    args: list[str] | None = None,
    max_chars: int = ruff.DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    """Run ruff and return bounded stdout/stderr diagnostics."""
    return ruff._run_ruff_structured_result(args=args, max_chars=max_chars)


@_read_tool()
def orbit2_mypy_run_mypy_structured(
    args: list[str] | None = None,
    max_chars: int = mypy.DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    """Run mypy and return bounded stdout/stderr diagnostics."""
    return mypy._run_mypy_structured_result(args=args, max_chars=max_chars)


@_read_tool()
def orbit2_obsidian_list_notes(
    path: str | None = None,
    recursive: bool = False,
    max_results: int | None = None,
) -> dict[str, Any]:
    """List Markdown notes in the configured Obsidian vault."""
    return _obsidian()._list_notes_result(path, recursive, max_results)


@_read_tool()
def orbit2_obsidian_read_note(
    path: str,
    include_raw_content: bool | None = False,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Read one Obsidian note by path, stem, or wikilink target."""
    return _obsidian()._read_note_result(path, include_raw_content, max_chars)


@_read_tool()
def orbit2_obsidian_read_notes(
    paths: list[str],
    include_raw_content: bool | None = False,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Read multiple Obsidian notes without failing the whole call."""
    return _obsidian()._read_notes_result(paths, include_raw_content, max_chars)


@_read_tool()
def orbit2_obsidian_search_notes(
    query: str,
    path: str | None = None,
    max_results: int | None = None,
    search_in: list[str] | None = None,
) -> dict[str, Any]:
    """Search Obsidian notes across title, metadata, tags, summary, and body."""
    return _obsidian()._search_notes_result(query, path, max_results, search_in)


@_read_tool()
def orbit2_obsidian_get_note_links(path: str) -> dict[str, Any]:
    """Return outgoing local links for an Obsidian note."""
    return _obsidian()._get_note_links_result(path)


@_read_tool()
def orbit2_obsidian_get_backlinks(path: str) -> dict[str, Any]:
    """Return notes linking to the target Obsidian note."""
    return _obsidian()._get_backlinks_result(path)


@_read_tool()
def orbit2_obsidian_get_unresolved_links(
    path: str | None = None,
    max_results: int | None = None,
) -> dict[str, Any]:
    """Return Obsidian links that do not resolve to Markdown notes."""
    return _obsidian()._get_unresolved_links_result(path, max_results)


@_read_tool()
def orbit2_obsidian_get_tag_summary(
    path: str | None = None,
    max_results: int | None = None,
) -> dict[str, Any]:
    """Return Obsidian tags and their note locations."""
    return _obsidian()._get_tag_summary_result(path, max_results)


@_read_tool()
def orbit2_obsidian_get_vault_metadata(
    path: str | None = None,
    include_top_level_entries: bool | None = True,
    max_entries: int | None = None,
) -> dict[str, Any]:
    """Return bounded Obsidian vault metadata."""
    return _obsidian()._get_vault_metadata_result(
        path,
        include_top_level_entries,
        max_entries,
    )


@_read_tool()
def orbit2_obsidian_check_availability() -> dict[str, Any]:
    """Check configured Obsidian vault accessibility."""
    return _obsidian()._check_availability_result()


if __name__ == "__main__":
    mcp.run()
