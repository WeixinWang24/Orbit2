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
from src.core.runtime.models import ToolRequest


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

    def test_git_show_returns_commit_content(self, git_repo: Path) -> None:
        target = git_repo / "a.txt"
        target.write_text("hello\n", encoding="utf-8")
        _run_git(["add", "a.txt"], cwd=git_repo)
        _run_git(["commit", "-m", "first commit"], cwd=git_repo)
        r = git_server._git_show_result("HEAD")
        assert r["ok"] is True
        assert "first commit" in r["output"]
        assert r["truncated"] is False

    def test_git_show_truncation(self, git_repo: Path) -> None:
        (git_repo / "big.txt").write_text("x\n" * 5000, encoding="utf-8")
        _run_git(["add", "big.txt"], cwd=git_repo)
        _run_git(["commit", "-m", "big"], cwd=git_repo)
        r = git_server._git_show_result("HEAD", max_chars=100)
        assert r["truncated"] is True
        assert len(r["output"]) == 100

    def test_git_show_invalid_rev_raises(self, git_repo: Path) -> None:
        (git_repo / "a.txt").write_text("x\n", encoding="utf-8")
        _run_git(["add", "a.txt"], cwd=git_repo)
        _run_git(["commit", "-m", "init"], cwd=git_repo)
        with pytest.raises(ValueError):
            git_server._git_show_result("nonexistent-sha-12345")

    def test_git_changed_files_returns_file_lists(self, git_repo: Path) -> None:
        (git_repo / "staged.txt").write_text("s\n", encoding="utf-8")
        (git_repo / "untracked.txt").write_text("u\n", encoding="utf-8")
        _run_git(["add", "staged.txt"], cwd=git_repo)
        r = git_server._git_changed_files_result()
        assert r["ok"] is True
        assert "staged.txt" in r["staged_files"]
        assert "untracked.txt" in r["untracked_files"]
        assert r["staged_count"] == 1
        assert r["untracked_count"] == 1
        assert r["total_changed_count"] >= 2

    def test_git_restore_discards_changes(self, git_repo: Path) -> None:
        target = git_repo / "a.txt"
        target.write_text("original\n", encoding="utf-8")
        _run_git(["add", "a.txt"], cwd=git_repo)
        _run_git(["commit", "-m", "init"], cwd=git_repo)
        target.write_text("modified\n", encoding="utf-8")
        assert target.read_text(encoding="utf-8") == "modified\n"
        r = git_server._git_restore_result(["a.txt"])
        assert r["ok"] is True
        assert r["mutation_kind"] == "git_restore"
        assert target.read_text(encoding="utf-8") == "original\n"

    def test_git_restore_rejects_path_escape(self, git_repo: Path) -> None:
        with pytest.raises(ValueError, match="escapes workspace"):
            git_server._git_restore_result(["../../outside"])

    def test_git_restore_rejects_empty_list(self, git_repo: Path) -> None:
        with pytest.raises(ValueError):
            git_server._git_restore_result([])

    def test_git_unstage_removes_from_index(self, git_repo: Path) -> None:
        target = git_repo / "a.txt"
        target.write_text("x\n", encoding="utf-8")
        _run_git(["add", "a.txt"], cwd=git_repo)
        _run_git(["commit", "-m", "init"], cwd=git_repo)
        target.write_text("y\n", encoding="utf-8")
        _run_git(["add", "a.txt"], cwd=git_repo)
        status_before = git_server._git_status_result()
        assert status_before["staged_count"] == 1
        r = git_server._git_unstage_result(["a.txt"])
        assert r["ok"] is True
        assert r["mutation_kind"] == "git_unstage"
        status_after = git_server._git_status_result()
        assert status_after["staged_count"] == 0

    def test_git_unstage_rejects_path_escape(self, git_repo: Path) -> None:
        with pytest.raises(ValueError, match="escapes workspace"):
            git_server._git_unstage_result(["../../outside"])

    def test_git_checkout_branch_switches_branch(self, git_repo: Path) -> None:
        (git_repo / "a.txt").write_text("init\n", encoding="utf-8")
        _run_git(["add", "a.txt"], cwd=git_repo)
        _run_git(["commit", "-m", "init"], cwd=git_repo)
        _run_git(["branch", "feature"], cwd=git_repo)
        r = git_server._git_checkout_branch_result("feature")
        assert r["ok"] is True
        assert r["mutation_kind"] == "git_checkout_branch"
        assert r["branch"] == "feature"

    def test_git_checkout_branch_fails_on_missing_branch(self, git_repo: Path) -> None:
        r = git_server._git_checkout_branch_result("nonexistent-branch")
        assert r["ok"] is False
        assert r["failure_kind"] == "branch_not_found"

    def test_git_checkout_branch_rejects_empty_name(self, git_repo: Path) -> None:
        with pytest.raises(ValueError):
            git_server._git_checkout_branch_result("")


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
    def test_server_lists_all_tools(
        self, git_integration_bootstrap: McpClientBootstrap
    ) -> None:
        client = StdioMcpClient(git_integration_bootstrap)
        descriptors = client.list_tools()
        names = {d.original_name for d in descriptors}
        expected = {
            "git_status", "git_diff", "git_log", "git_add", "git_commit",
            "git_show", "git_changed_files",
            "git_restore", "git_unstage", "git_checkout_branch",
        }
        assert expected <= names

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
        assert result.governance_outcome.startswith("allowed")
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
        assert add_result.governance_outcome.startswith("allowed")

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
        registry = CapabilityRegistry()
        attach_mcp_server(git_integration_bootstrap, registry)
        for name in ("git_commit", "git_add", "git_restore", "git_unstage", "git_checkout_branch"):
            wrapper = registry.get(f"mcp__git__{name}")
            assert wrapper.requires_approval is True, name
            assert wrapper.side_effect_class == "write", name
        for name in ("git_status", "git_diff", "git_log", "git_show", "git_changed_files"):
            wrapper = registry.get(f"mcp__git__{name}")
            assert wrapper.requires_approval is False, name
            assert wrapper.side_effect_class == "safe", name
