"""Tests for the L1 structured git MCP first slice."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.capability.boundary import CapabilityBoundary
from src.capability.mcp import McpClientBootstrap, StdioMcpClient, attach_mcp_server
from src.capability.models import CapabilityLayer
from src.capability.registry import CapabilityRegistry
from src.core.runtime.models import ToolRequest

from src.capability.mcp_servers.l1_structured.git import stdio_server as l1_git_server

_L1_GIT_MODULE = "src.capability.mcp_servers.l1_structured.git.stdio_server"


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


def _gap() -> dict[str, Any]:
    return {
        "description": "Need the exact diff hunk before assessing risk.",
        "needed_evidence": "The bounded hunk for the changed file.",
    }


def _make_two_hunk_change(repo: Path) -> None:
    lines = [f"line {i}" for i in range(1, 31)]
    target = repo / "a.txt"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _run_git(["add", "a.txt"], cwd=repo)
    _run_git(["commit", "-m", "init"], cwd=repo)

    updated = list(lines)
    updated[1] = "line 2 changed"
    updated[24] = "line 25 changed"
    target.write_text("\n".join(updated) + "\n", encoding="utf-8")


def _make_revision_file(repo: Path) -> str:
    lines = [f"rev line {i}" for i in range(1, 11)]
    target = repo / "history.txt"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _run_git(["add", "history.txt"], cwd=repo)
    _run_git(["commit", "-m", "history"], cwd=repo)
    cp = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    )
    return cp.stdout.strip()


class TestReadDiffHunkPureHelper:
    def test_parses_diff_hunks(self) -> None:
        diff = "\n".join([
            "diff --git a/a.txt b/a.txt",
            "index 111..222 100644",
            "--- a/a.txt",
            "+++ b/a.txt",
            "@@ -1,2 +1,2 @@",
            " alpha",
            "-beta",
            "+BETA",
            "@@ -20,2 +20,2 @@ tail",
            "-old",
            "+new",
        ])

        hunks = l1_git_server._parse_diff_hunks(diff)

        assert len(hunks) == 2
        assert hunks[0]["old_start"] == 1
        assert hunks[0]["new_start"] == 1
        assert "-beta" in hunks[0]["content"]
        assert hunks[1]["section"] == "tail"

    def test_reads_selected_hunk_with_audit_metadata(self, git_repo: Path) -> None:
        _make_two_hunk_change(git_repo)

        result = l1_git_server._read_diff_hunk_result(
            path="a.txt",
            hunk_index=2,
            evidence_gap=_gap(),
            reason_context_pack_insufficient="Changed files only named the file, not the relevant hunk.",
        )

        raw_diff = l1_git_server.raw_git._git_diff_result(path="a.txt", max_chars=200_000)["diff"]
        assert result["ok"] is True
        assert result["evidence_type"] == "diff_hunk"
        assert result["target"]["path"] == "a.txt"
        assert result["target"]["hunk_index"] == 2
        assert result["target"]["hunk_count"] == 2
        assert "-line 25" in result["content"]
        assert "+line 25 changed" in result["content"]
        assert "line 2 changed" not in result["content"]
        assert result["diff_hash"]["value"] == hashlib.sha256(raw_diff.encode()).hexdigest()
        assert result["audit"]["capability_layer"] == "structured_primitive"
        assert result["audit"]["substrate"] == "git_diff"

    def test_reuses_raw_git_diff_substrate(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (git_repo / "a.txt").write_text("x\n", encoding="utf-8")
        calls: list[dict[str, Any]] = []

        def _fake_diff(**kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {
                "ok": True,
                "cwd": str(git_repo),
                "staged": False,
                "diff": "@@ -1 +1 @@\n-x\n+y",
                "has_diff": True,
                "truncated": False,
            }

        monkeypatch.setattr(l1_git_server.raw_git, "_git_diff_result", _fake_diff)

        result = l1_git_server._read_diff_hunk_result(
            path="a.txt",
            hunk_index=1,
            evidence_gap=_gap(),
            reason_context_pack_insufficient="Need exact hunk.",
        )

        assert calls and calls[0]["path"] == "a.txt"
        assert result["content"] == "@@ -1 +1 @@\n-x\n+y"

    def test_requires_declared_evidence_gap(self, git_repo: Path) -> None:
        _make_two_hunk_change(git_repo)

        with pytest.raises(ValueError, match="evidence_gap.description"):
            l1_git_server._read_diff_hunk_result(
                path="a.txt",
                hunk_index=1,
                evidence_gap={"needed_evidence": "diff hunk"},
                reason_context_pack_insufficient="Need exact hunk.",
            )

    def test_rejects_out_of_range_hunk_index(self, git_repo: Path) -> None:
        _make_two_hunk_change(git_repo)

        with pytest.raises(ValueError, match="out of range"):
            l1_git_server._read_diff_hunk_result(
                path="a.txt",
                hunk_index=3,
                evidence_gap=_gap(),
                reason_context_pack_insufficient="Need exact hunk.",
            )

    def test_hunk_limits_come_from_mcp_environment(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _make_two_hunk_change(git_repo)
        monkeypatch.setenv(l1_git_server.MAX_HUNK_LINES_ENV, "2")

        with pytest.raises(ValueError, match="max span of 2"):
            l1_git_server._read_diff_hunk_result(
                path="a.txt",
                hunk_index=1,
                evidence_gap=_gap(),
                reason_context_pack_insufficient="Need exact hunk.",
            )

    def test_char_limits_come_from_mcp_environment(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _make_two_hunk_change(git_repo)
        monkeypatch.setenv(l1_git_server.DEFAULT_MAX_CHARS_ENV, "20")
        monkeypatch.setenv(l1_git_server.HARD_MAX_CHARS_ENV, "25")

        result = l1_git_server._read_diff_hunk_result(
            path="a.txt",
            hunk_index=1,
            evidence_gap=_gap(),
            reason_context_pack_insufficient="Need exact hunk.",
            max_chars=80,
        )

        assert result["limits"]["max_chars"] == 25
        assert result["limits"]["chars_returned"] == 25
        assert result["limits"]["truncated"] is True

    def test_rejects_protected_paths(self, git_repo: Path) -> None:
        with pytest.raises(ValueError, match="protected location"):
            l1_git_server._read_diff_hunk_result(
                path=".git/config",
                hunk_index=1,
                evidence_gap=_gap(),
                reason_context_pack_insufficient="Need exact hunk.",
            )


class TestReadGitShowRegionPureHelper:
    def test_reads_revision_file_region_with_audit_metadata(self, git_repo: Path) -> None:
        rev = _make_revision_file(git_repo)

        result = l1_git_server._read_git_show_region_result(
            rev=rev,
            path="history.txt",
            start_line=2,
            end_line=4,
            evidence_gap_description="Need historical file lines for comparison.",
            needed_evidence="The exact lines from the committed version.",
            reason_context_pack_insufficient="The context pack only identified the commit and file.",
        )

        raw_show = l1_git_server.raw_git._git_show_result(f"{rev}:history.txt", max_chars=200_000)["output"]
        assert result["ok"] is True
        assert result["evidence_type"] == "git_show_region"
        assert result["target"] == {
            "rev": rev,
            "path": "history.txt",
            "start_line": 2,
            "end_line": 4,
        }
        assert result["content"] == "rev line 2\nrev line 3\nrev line 4"
        assert result["source_hash"]["value"] == hashlib.sha256(raw_show.encode()).hexdigest()
        assert result["audit"]["substrate"] == "git_show"

    def test_reuses_raw_git_show_substrate(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (git_repo / "history.txt").write_text("x\n", encoding="utf-8")
        calls: list[dict[str, Any]] = []

        def _fake_show(rev: str, path: str | None = None, **kwargs: Any) -> dict[str, Any]:
            calls.append({"rev": rev, "path": path, **kwargs})
            return {
                "ok": True,
                "cwd": str(git_repo),
                "rev": rev,
                "path": path,
                "output": "one\ntwo\nthree",
                "truncated": False,
            }

        monkeypatch.setattr(l1_git_server.raw_git, "_git_show_result", _fake_show)

        result = l1_git_server._read_git_show_region_result(
            rev="HEAD",
            path="history.txt",
            start_line=2,
            end_line=2,
            evidence_gap_description="Need historical line.",
            needed_evidence="The exact line from the revision.",
            reason_context_pack_insufficient="Need line-level evidence.",
        )

        assert calls and calls[0]["rev"] == "HEAD:history.txt"
        assert calls[0]["path"] is None
        assert result["content"] == "two"

    def test_uses_flat_evidence_fields(self, git_repo: Path) -> None:
        _make_revision_file(git_repo)

        with pytest.raises(ValueError, match="needed_evidence"):
            l1_git_server._read_git_show_region_result(
                rev="HEAD",
                path="history.txt",
                start_line=1,
                end_line=1,
                evidence_gap_description="Need historical line.",
                needed_evidence="",
                reason_context_pack_insufficient="Need line-level evidence.",
            )

    def test_rejects_rev_with_colon(self, git_repo: Path) -> None:
        _make_revision_file(git_repo)

        with pytest.raises(ValueError, match="must not contain ':'"):
            l1_git_server._read_git_show_region_result(
                rev="HEAD:history.txt",
                path="history.txt",
                start_line=1,
                end_line=1,
                evidence_gap_description="Need historical line.",
                needed_evidence="The exact line from the revision.",
                reason_context_pack_insufficient="Need line-level evidence.",
            )

    def test_line_limits_come_from_environment(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _make_revision_file(git_repo)
        monkeypatch.setenv(l1_git_server.SHOW_MAX_LINE_SPAN_ENV, "2")

        with pytest.raises(ValueError, match="max span of 2"):
            l1_git_server._read_git_show_region_result(
                rev="HEAD",
                path="history.txt",
                start_line=1,
                end_line=3,
                evidence_gap_description="Need historical lines.",
                needed_evidence="The exact lines from the revision.",
                reason_context_pack_insufficient="Need line-level evidence.",
            )

    def test_char_limits_apply_to_selected_region(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _make_revision_file(git_repo)
        monkeypatch.setenv(l1_git_server.DEFAULT_MAX_CHARS_ENV, "8")
        monkeypatch.setenv(l1_git_server.HARD_MAX_CHARS_ENV, "10")

        result = l1_git_server._read_git_show_region_result(
            rev="HEAD",
            path="history.txt",
            start_line=1,
            end_line=3,
            evidence_gap_description="Need historical lines.",
            needed_evidence="The exact lines from the revision.",
            reason_context_pack_insufficient="Need line-level evidence.",
            max_chars=20,
        )

        assert result["content"] == "rev line 1"
        assert result["limits"]["max_chars"] == 10
        assert result["limits"]["chars_returned"] == 10
        assert result["limits"]["truncated"] is True

    def test_rejects_protected_paths_for_show_region(self, git_repo: Path) -> None:
        with pytest.raises(ValueError, match="protected location"):
            l1_git_server._read_git_show_region_result(
                rev="HEAD",
                path=".git/config",
                start_line=1,
                end_line=1,
                evidence_gap_description="Need historical line.",
                needed_evidence="The exact line from the revision.",
                reason_context_pack_insufficient="Need line-level evidence.",
            )


@pytest.fixture
def l1_git_integration_bootstrap(git_repo: Path) -> McpClientBootstrap:
    _make_two_hunk_change(git_repo)
    env = {**os.environ, "ORBIT_WORKSPACE_ROOT": str(git_repo)}
    return McpClientBootstrap(
        server_name="structured_git",
        command=sys.executable,
        args=("-m", _L1_GIT_MODULE, str(git_repo)),
        env=env,
        transport="stdio",
    )


class TestStructuredGitMcpIntegration:
    def test_server_lists_read_diff_hunk(
        self, l1_git_integration_bootstrap: McpClientBootstrap
    ) -> None:
        client = StdioMcpClient(l1_git_integration_bootstrap)
        descriptors = client.list_tools()
        names = {d.original_name for d in descriptors}
        assert names == {"read_diff_hunk", "read_git_show_region"}
        descriptions = {d.original_name: d.description or "" for d in descriptors}
        assert "first hunk_index is 1" in descriptions["read_diff_hunk"]
        assert "workspace-relative path separately" in descriptions["read_git_show_region"]

    def test_attached_tool_has_l1_metadata_and_governance(
        self, l1_git_integration_bootstrap: McpClientBootstrap
    ) -> None:
        registry = CapabilityRegistry()
        _client, registered = attach_mcp_server(l1_git_integration_bootstrap, registry)

        assert set(registered) == {
            "mcp__structured_git__read_diff_hunk",
            "mcp__structured_git__read_git_show_region",
        }
        tool = registry.get("mcp__structured_git__read_diff_hunk")
        assert tool is not None
        assert tool.capability_layer == CapabilityLayer.STRUCTURED_PRIMITIVE
        assert tool.side_effect_class == "safe"
        assert tool.requires_approval is False
        assert tool.reveal_group == "mcp_structured_git"

    def test_read_hunk_path_through_capability_boundary(
        self, git_repo: Path, l1_git_integration_bootstrap: McpClientBootstrap
    ) -> None:
        registry = CapabilityRegistry()
        attach_mcp_server(l1_git_integration_bootstrap, registry)
        boundary = CapabilityBoundary(registry, git_repo)

        result = boundary.execute(ToolRequest(
            tool_call_id="c1",
            tool_name="mcp__structured_git__read_diff_hunk",
            arguments={
                "path": "a.txt",
                "hunk_index": 1,
                "evidence_gap": _gap(),
                "reason_context_pack_insufficient": "Changed-files context did not include the hunk.",
            },
        ))

        assert result.ok is True
        assert result.governance_outcome.startswith("allowed")
        assert "-line 2" in result.content
        assert "+line 2 changed" in result.content

    def test_read_git_show_region_path_through_capability_boundary(
        self, git_repo: Path, l1_git_integration_bootstrap: McpClientBootstrap
    ) -> None:
        registry = CapabilityRegistry()
        attach_mcp_server(l1_git_integration_bootstrap, registry)
        boundary = CapabilityBoundary(registry, git_repo)

        result = boundary.execute(ToolRequest(
            tool_call_id="c1",
            tool_name="mcp__structured_git__read_git_show_region",
            arguments={
                "rev": "HEAD",
                "path": "a.txt",
                "start_line": 1,
                "end_line": 3,
                "evidence_gap_description": "Need committed file lines.",
                "needed_evidence": "The exact lines from HEAD for comparison.",
                "reason_context_pack_insufficient": "The current context does not include revision file content.",
            },
        ))

        assert result.ok is True
        assert result.governance_outcome.startswith("allowed")
        assert "rev" in result.content
        assert "line 1" in result.content
