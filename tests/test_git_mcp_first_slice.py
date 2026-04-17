"""Git MCP server first-slice tests (Handoff 16).

Uses throwaway git repos under tmp_path so nothing in this test file touches
the surrounding Orbit2 working tree. Covers pure result helpers (fast) plus
an integration path through the real stdio subprocess routed through
Orbit2's capability attachment.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.capability.boundary import CapabilityBoundary
from src.capability.mcp import (
    McpClientBootstrap,
    StdioMcpClient,
    attach_mcp_server,
)
from src.capability.registry import CapabilityRegistry
from src.capability.mcp_servers.git import stdio_server as git_server
from src.runtime.models import ToolRequest


_GIT_MODULE = "src.capability.mcp_servers.git.stdio_server"


def _run_git(args: list[str], *, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


@pytest.fixture
def git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ORBIT_WORKSPACE_ROOT", str(tmp_path))
    _run_git(["init"], cwd=tmp_path)
    _run_git(["config", "user.name", "Orbit2 Test"], cwd=tmp_path)
    _run_git(["config", "user.email", "orbit2@example.com"], cwd=tmp_path)
    _run_git(["config", "commit.gpgsign", "false"], cwd=tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Pure result helpers (fast)
# ---------------------------------------------------------------------------


class TestGitResultHelpers:
    def test_git_status_untracked_file(self, git_repo: Path) -> None:
        (git_repo / "notes.txt").write_text("hello\n", encoding="utf-8")
        r = git_server._git_status_result()
        assert r["ok"] is True
        assert "notes.txt" in r["untracked"]
        assert r["untracked_count"] == 1
        assert r["staged_count"] == 0
        assert r["clean"] is False

    def test_git_status_clean_repo(self, git_repo: Path) -> None:
        (git_repo / "a.txt").write_text("1\n", encoding="utf-8")
        _run_git(["add", "a.txt"], cwd=git_repo)
        _run_git(["commit", "-m", "init"], cwd=git_repo)
        r = git_server._git_status_result()
        assert r["clean"] is True
        assert r["branch"]  # some branch name

    def test_git_diff_working_tree(self, git_repo: Path) -> None:
        target = git_repo / "a.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        _run_git(["add", "a.txt"], cwd=git_repo)
        _run_git(["commit", "-m", "init"], cwd=git_repo)
        target.write_text("alpha\nBETA\n", encoding="utf-8")
        r = git_server._git_diff_result(path="a.txt", staged=False)
        assert r["has_diff"] is True
        assert "-beta" in r["diff"]
        assert "+BETA" in r["diff"]

    def test_git_diff_staged(self, git_repo: Path) -> None:
        target = git_repo / "a.txt"
        target.write_text("alpha\n", encoding="utf-8")
        _run_git(["add", "a.txt"], cwd=git_repo)
        _run_git(["commit", "-m", "init"], cwd=git_repo)
        target.write_text("ALPHA\n", encoding="utf-8")
        _run_git(["add", "a.txt"], cwd=git_repo)
        r = git_server._git_diff_result(path="a.txt", staged=True)
        assert r["staged"] is True
        assert "+ALPHA" in r["diff"]

    def test_git_diff_truncation(self, git_repo: Path) -> None:
        target = git_repo / "big.txt"
        target.write_text("a\n" * 5000, encoding="utf-8")
        _run_git(["add", "big.txt"], cwd=git_repo)
        _run_git(["commit", "-m", "init"], cwd=git_repo)
        target.write_text("b\n" * 5000, encoding="utf-8")
        r = git_server._git_diff_result(path="big.txt", max_chars=200)
        assert r["truncated"] is True
        assert len(r["diff"]) == 200

    def test_git_log_returns_commits(self, git_repo: Path) -> None:
        target = git_repo / "a.txt"
        target.write_text("alpha\n", encoding="utf-8")
        _run_git(["add", "a.txt"], cwd=git_repo)
        _run_git(["commit", "-m", "init"], cwd=git_repo)
        target.write_text("beta\n", encoding="utf-8")
        _run_git(["add", "a.txt"], cwd=git_repo)
        _run_git(["commit", "-m", "expand"], cwd=git_repo)
        r = git_server._git_log_result(limit=2)
        assert r["commit_count"] == 2
        assert r["commits"][0]["subject"] == "expand"
        assert r["commits"][1]["subject"] == "init"

    def test_git_add_stages_path(self, git_repo: Path) -> None:
        (git_repo / "a.txt").write_text("hi\n", encoding="utf-8")
        r = git_server._git_add_result(paths=["a.txt"])
        assert r["ok"] is True
        assert r["mutation_kind"] == "git_add"
        status = git_server._git_status_result()
        assert status["staged_count"] == 1

    def test_git_add_rejects_empty_list(self, git_repo: Path) -> None:
        with pytest.raises(ValueError):
            git_server._git_add_result(paths=[])

    def test_git_add_rejects_path_escape(self, git_repo: Path) -> None:
        """Audit MED-4 fix: individual paths-list entries must be validated
        against workspace containment before git is invoked."""
        with pytest.raises(ValueError, match="escapes workspace"):
            git_server._git_add_result(paths=["../../outside"])

    def test_git_add_rejects_absolute_path_outside_workspace(self, git_repo: Path) -> None:
        with pytest.raises(ValueError, match="escapes workspace"):
            git_server._git_add_result(paths=["/etc/passwd"])

    def test_git_commit_succeeds(self, git_repo: Path) -> None:
        (git_repo / "a.txt").write_text("hi\n", encoding="utf-8")
        _run_git(["add", "a.txt"], cwd=git_repo)
        r = git_server._git_commit_result("init")
        assert r["ok"] is True
        assert r["mutation_kind"] == "git_commit"
        assert r["subject"] == "init"
        assert r["commit"]  # non-empty sha

    def test_git_commit_fails_cleanly_when_nothing_staged(self, git_repo: Path) -> None:
        r = git_server._git_commit_result("noop")
        assert r["ok"] is False
        assert r["failure_kind"] == "git_commit_failed"

    def test_cwd_escape_refused(self, git_repo: Path) -> None:
        with pytest.raises(ValueError):
            git_server._git_status_result(cwd="../outside")


# ---------------------------------------------------------------------------
# Integration: real stdio subprocess + Orbit2 capability attachment
# ---------------------------------------------------------------------------


@pytest.fixture
def git_integration_bootstrap(git_repo: Path) -> McpClientBootstrap:
    env = {**os.environ, "ORBIT_WORKSPACE_ROOT": str(git_repo)}
    return McpClientBootstrap(
        server_name="git",
        command=sys.executable,
        args=("-m", _GIT_MODULE, str(git_repo)),
        env=env,
        transport="stdio",
    )


class TestGitMcpIntegration:
    def test_server_lists_first_slice_tools(
        self, git_integration_bootstrap: McpClientBootstrap
    ) -> None:
        client = StdioMcpClient(git_integration_bootstrap)
        descriptors = client.list_tools()
        names = {d.original_name for d in descriptors}
        assert {"git_status", "git_diff", "git_log", "git_add", "git_commit"} <= names

    def test_status_path_through_capability_boundary(
        self, git_repo: Path, git_integration_bootstrap: McpClientBootstrap
    ) -> None:
        (git_repo / "note.txt").write_text("hi\n", encoding="utf-8")
        registry = CapabilityRegistry()
        attach_mcp_server(git_integration_bootstrap, registry)
        boundary = CapabilityBoundary(registry, git_repo)

        result = boundary.execute(ToolRequest(
            tool_call_id="c1", tool_name="mcp__git__git_status", arguments={},
        ))
        assert result.ok is True
        assert result.governance_outcome == "allowed"
        assert "note.txt" in result.content  # surfaces as JSON in content

    def test_commit_path_through_capability_boundary(
        self, git_repo: Path, git_integration_bootstrap: McpClientBootstrap
    ) -> None:
        (git_repo / "note.txt").write_text("hi\n", encoding="utf-8")
        registry = CapabilityRegistry()
        attach_mcp_server(git_integration_bootstrap, registry)
        boundary = CapabilityBoundary(registry, git_repo)

        # Stage
        add_result = boundary.execute(ToolRequest(
            tool_call_id="c1", tool_name="mcp__git__git_add",
            arguments={"paths": ["note.txt"]},
        ))
        assert add_result.ok is True
        assert add_result.governance_outcome == "allowed"

        # Commit
        commit_result = boundary.execute(ToolRequest(
            tool_call_id="c2", tool_name="mcp__git__git_commit",
            arguments={"message": "initial commit via MCP"},
        ))
        assert commit_result.ok is True

        # Verify commit actually landed on disk
        log = subprocess.run(
            ["git", "log", "--oneline", "-n1"],
            cwd=str(git_repo), capture_output=True, text=True, check=True,
        )
        assert "initial commit via MCP" in log.stdout

    def test_git_mutation_wrapper_carries_approval_metadata(
        self, git_integration_bootstrap: McpClientBootstrap
    ) -> None:
        """Handoff 16 acceptance: git mutation tools must stay on the
        approval-required side of the seam even when no approval runtime
        consumes the flag yet."""
        registry = CapabilityRegistry()
        attach_mcp_server(git_integration_bootstrap, registry)
        commit_wrapper = registry.get("mcp__git__git_commit")
        status_wrapper = registry.get("mcp__git__git_status")
        assert commit_wrapper.requires_approval is True
        assert commit_wrapper.side_effect_class == "write"
        assert status_wrapper.requires_approval is False
        assert status_wrapper.side_effect_class == "safe"
