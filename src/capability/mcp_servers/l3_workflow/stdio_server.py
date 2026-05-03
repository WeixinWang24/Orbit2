from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from src.capability.mcp_servers.timing import timed_mcp_tool

from src.capability.mcp_servers.l3_workflow.inspect_change_set import (
    inspect_change_set_workflow_result,
)
from src.capability.mcp_servers.l3_workflow.repo_recon import (
    repo_recon_workflow_result,
)


SERVER_NAME = "workflow"

mcp = FastMCP(SERVER_NAME)


@timed_mcp_tool(mcp, SERVER_NAME)
def inspect_change_set_workflow(
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
    return inspect_change_set_workflow_result(
        cwd=cwd,
        repo_id=repo_id,
        label=label,
        include_untracked=include_untracked,
        max_diff_files=max_diff_files,
        max_impact_files=max_impact_files,
        max_impact_symbols=max_impact_symbols,
        max_impact_edges=max_impact_edges,
    )


@timed_mcp_tool(mcp, SERVER_NAME)
def repo_recon_workflow(
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
    return repo_recon_workflow_result(
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


if __name__ == "__main__":
    mcp.run()
