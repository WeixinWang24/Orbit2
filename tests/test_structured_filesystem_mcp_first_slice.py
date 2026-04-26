"""Tests for the L1 structured filesystem MCP first slice."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from src.capability.boundary import CapabilityBoundary
from src.capability.mcp import McpClientBootstrap, StdioMcpClient, attach_mcp_server
from src.capability.models import CapabilityLayer
from src.capability.registry import CapabilityRegistry
from src.core.runtime.models import ToolRequest

from src.capability.mcp_servers.l1_structured.filesystem import stdio_server as l1_fs_server

_L1_FS_MODULE = "src.capability.mcp_servers.l1_structured.filesystem.stdio_server"


@pytest.fixture(autouse=True)
def workspace_root_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ORBIT_WORKSPACE_ROOT", str(tmp_path))


def _gap() -> dict[str, Any]:
    return {
        "description": "Need exact source lines before making a claim.",
        "needed_evidence": "The local file region containing the implementation.",
    }


class TestReadFileRegionPureHelper:
    def test_reuses_raw_filesystem_path_resolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "main.py").write_text("alpha\nbeta\n", encoding="utf-8")
        calls: list[str] = []

        def _fake_resolve(path: str) -> Path:
            calls.append(path)
            return tmp_path / path

        monkeypatch.setattr(
            l1_fs_server.raw_filesystem,
            "_resolve_safe_existing_file",
            _fake_resolve,
        )
        monkeypatch.setattr(
            l1_fs_server.raw_filesystem,
            "_workspace_root",
            lambda: tmp_path,
        )

        result = l1_fs_server._read_file_region_result(
            path="main.py",
            start_line=1,
            end_line=1,
            evidence_gap=_gap(),
            reason_context_pack_insufficient="Need exact source line.",
        )

        assert calls == ["main.py"]
        assert result["target"]["path"] == "main.py"
        assert result["content"] == "alpha"

    def test_reads_bounded_region_with_audit_metadata(self, tmp_path: Path) -> None:
        source = "alpha\nbeta\ngamma\ndelta\n"
        (tmp_path / "main.py").write_text(source, encoding="utf-8")

        result = l1_fs_server._read_file_region_result(
            path="main.py",
            start_line=2,
            end_line=3,
            evidence_gap=_gap(),
            reason_context_pack_insufficient="The context pack named the file but not the relevant lines.",
        )

        assert result["ok"] is True
        assert result["evidence_type"] == "file_region"
        assert result["target"] == {"path": "main.py", "start_line": 2, "end_line": 3}
        assert result["content"] == "beta\ngamma"
        assert result["file_hash"]["value"] == hashlib.sha256(source.encode()).hexdigest()
        assert result["audit"]["capability_layer"] == "structured_primitive"
        assert result["audit"]["policy_decision"] == "evidence_gap_declared"

    def test_requires_declared_evidence_gap(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("alpha\n", encoding="utf-8")

        with pytest.raises(ValueError, match="evidence_gap.description"):
            l1_fs_server._read_file_region_result(
                path="main.py",
                start_line=1,
                end_line=1,
                evidence_gap={"needed_evidence": "line text"},
                reason_context_pack_insufficient="Missing exact line.",
            )

    def test_requires_context_pack_insufficiency_reason(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("alpha\n", encoding="utf-8")

        with pytest.raises(ValueError, match="reason_context_pack_insufficient"):
            l1_fs_server._read_file_region_result(
                path="main.py",
                start_line=1,
                end_line=1,
                evidence_gap=_gap(),
                reason_context_pack_insufficient="",
            )

    def test_rejects_overwide_line_spans(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("\n".join(str(i) for i in range(140)), encoding="utf-8")

        with pytest.raises(ValueError, match="max span"):
            l1_fs_server._read_file_region_result(
                path="main.py",
                start_line=1,
                end_line=121,
                evidence_gap=_gap(),
                reason_context_pack_insufficient="Need a smaller region.",
            )

    def test_line_span_limit_comes_from_mcp_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "main.py").write_text("a\nb\nc\n", encoding="utf-8")
        monkeypatch.setenv(l1_fs_server.MAX_LINE_SPAN_ENV, "2")

        with pytest.raises(ValueError, match="max span of 2"):
            l1_fs_server._read_file_region_result(
                path="main.py",
                start_line=1,
                end_line=3,
                evidence_gap=_gap(),
                reason_context_pack_insufficient="Need a smaller region.",
            )

    def test_char_limits_come_from_mcp_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "main.py").write_text("0123456789\n", encoding="utf-8")
        monkeypatch.setenv(l1_fs_server.DEFAULT_MAX_CHARS_ENV, "4")
        monkeypatch.setenv(l1_fs_server.HARD_MAX_CHARS_ENV, "6")

        default_limited = l1_fs_server._read_file_region_result(
            path="main.py",
            start_line=1,
            end_line=1,
            evidence_gap=_gap(),
            reason_context_pack_insufficient="Need exact line.",
        )
        requested_over_hard = l1_fs_server._read_file_region_result(
            path="main.py",
            start_line=1,
            end_line=1,
            evidence_gap=_gap(),
            reason_context_pack_insufficient="Need exact line.",
            max_chars=9,
        )

        assert default_limited["content"] == "0123"
        assert default_limited["limits"]["max_chars"] == 4
        assert requested_over_hard["content"] == "012345"
        assert requested_over_hard["limits"]["max_chars"] == 6

    def test_invalid_char_limit_environment_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "main.py").write_text("alpha\n", encoding="utf-8")
        monkeypatch.setenv(l1_fs_server.DEFAULT_MAX_CHARS_ENV, "20")
        monkeypatch.setenv(l1_fs_server.HARD_MAX_CHARS_ENV, "10")

        with pytest.raises(ValueError, match="default max chars must be <= hard max chars"):
            l1_fs_server._read_file_region_result(
                path="main.py",
                start_line=1,
                end_line=1,
                evidence_gap=_gap(),
                reason_context_pack_insufficient="Need exact line.",
            )

    def test_grep_scoped_reuses_raw_grep_substrate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "src").mkdir()
        calls: list[dict[str, Any]] = []

        def _fake_grep(pattern: str, **kwargs: Any) -> dict[str, Any]:
            calls.append({"pattern": pattern, **kwargs})
            return {
                "ok": True,
                "path": str(tmp_path / "src"),
                "pattern": pattern,
                "file_pattern": kwargs["file_pattern"],
                "case_insensitive": kwargs["case_insensitive"],
                "matches": [{"path": "src/app.py", "line": 3, "text": "def target():"}],
                "match_count": 1,
                "scanned_files": 1,
                "scanned_bytes": 42,
                "truncated": False,
                "total_byte_budget_exhausted": False,
            }

        monkeypatch.setattr(l1_fs_server.raw_filesystem, "_grep_result", _fake_grep)

        result = l1_fs_server._grep_scoped_result(
            pattern="target",
            path="src",
            evidence_gap_description="Need candidate locations for the target symbol.",
            needed_evidence="Bounded grep matches that identify likely source lines.",
            reason_context_pack_insufficient="The context pack named the symbol but not its file or line.",
            file_pattern="*.py",
            max_matches=5,
        )

        assert calls and calls[0]["pattern"] == "target"
        assert calls[0]["path"] == "src"
        assert calls[0]["max_matches"] == 5
        assert result["evidence_type"] == "grep_scope"
        assert result["matches"] == [{"path": "src/app.py", "line": 3, "text": "def target():"}]
        assert result["audit"]["substrate"] == "grep"

    def test_grep_scoped_returns_typed_matches(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("alpha\nneedle\n", encoding="utf-8")
        (tmp_path / "src" / "note.txt").write_text("needle\n", encoding="utf-8")

        result = l1_fs_server._grep_scoped_result(
            pattern="needle",
            path="src",
            evidence_gap_description="Need a scoped search hit before reading lines.",
            needed_evidence="The file and line where the query appears.",
            reason_context_pack_insufficient="No line-level evidence is currently available.",
            file_pattern="*.py",
        )

        assert result["ok"] is True
        assert result["query"]["pattern"] == "needle"
        assert result["scope"]["path"] == "src"
        assert result["match_count"] == 1
        assert result["matches"][0]["path"] == "src/app.py"
        assert result["matches"][0]["line"] == 2

    def test_grep_scoped_uses_flat_evidence_fields(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("needle\n", encoding="utf-8")

        with pytest.raises(ValueError, match="needed_evidence"):
            l1_fs_server._grep_scoped_result(
                pattern="needle",
                path="src",
                evidence_gap_description="Need a scoped search hit.",
                needed_evidence="",
                reason_context_pack_insufficient="No line-level evidence is currently available.",
            )

    def test_grep_scoped_limits_come_from_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("needle\n", encoding="utf-8")
        (tmp_path / "src" / "b.py").write_text("needle\n", encoding="utf-8")
        monkeypatch.setenv(l1_fs_server.GREP_MAX_MATCHES_ENV, "1")

        result = l1_fs_server._grep_scoped_result(
            pattern="needle",
            path="src",
            evidence_gap_description="Need one scoped search hit.",
            needed_evidence="A bounded match list.",
            reason_context_pack_insufficient="Need to locate the evidence before reading lines.",
            max_matches=10,
        )

        assert result["limits"]["max_matches"] == 1
        assert result["match_count"] == 1
        assert result["limits"]["truncated"] is True

    def test_grep_scoped_rejects_protected_scope(self, tmp_path: Path) -> None:
        (tmp_path / ".runtime").mkdir()

        with pytest.raises(ValueError, match="protected location"):
            l1_fs_server._grep_scoped_result(
                pattern="token",
                path=".runtime",
                evidence_gap_description="Need scoped search.",
                needed_evidence="A bounded match list.",
                reason_context_pack_insufficient="Need to locate the evidence before reading lines.",
            )

    def test_raw_grep_skips_protected_prefixes(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("needle\n", encoding="utf-8")
        (tmp_path / ".runtime").mkdir()
        (tmp_path / ".runtime" / "secret.txt").write_text("needle secret\n", encoding="utf-8")
        (tmp_path / "src" / "__pycache__").mkdir()
        (tmp_path / "src" / "__pycache__" / "app.cpython-311.pyc").write_bytes(b"needle bytecode")

        result = l1_fs_server.raw_filesystem._grep_result("needle", path=".")

        paths = {match["path"] for match in result["matches"]}
        assert "src/app.py" in paths
        assert ".runtime/secret.txt" not in paths
        assert "src/__pycache__/app.cpython-311.pyc" not in paths

    def test_raw_grep_ignore_rules_come_from_mcp_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("needle\n", encoding="utf-8")
        (tmp_path / "src" / "cache_dir").mkdir()
        (tmp_path / "src" / "cache_dir" / "kept.py").write_text("needle cached\n", encoding="utf-8")
        (tmp_path / "src" / "generated.tmp").write_text("needle tmp\n", encoding="utf-8")
        monkeypatch.setenv(
            l1_fs_server.raw_filesystem.GREP_IGNORED_PARTS_ENV,
            "cache_dir",
        )
        monkeypatch.setenv(
            l1_fs_server.raw_filesystem.GREP_IGNORED_SUFFIXES_ENV,
            ".tmp",
        )

        result = l1_fs_server.raw_filesystem._grep_result("needle", path=".")

        paths = {match["path"] for match in result["matches"]}
        assert "src/app.py" in paths
        assert "src/cache_dir/kept.py" not in paths
        assert "src/generated.tmp" not in paths

    def test_raw_grep_ignore_rules_can_be_cleared_by_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "__pycache__").mkdir()
        (tmp_path / "src" / "__pycache__" / "app.cpython-311.pyc").write_bytes(b"needle bytecode")
        monkeypatch.setenv(l1_fs_server.raw_filesystem.GREP_IGNORED_PARTS_ENV, "")
        monkeypatch.setenv(l1_fs_server.raw_filesystem.GREP_IGNORED_SUFFIXES_ENV, "")

        result = l1_fs_server.raw_filesystem._grep_result("needle", path=".")

        paths = {match["path"] for match in result["matches"]}
        assert "src/__pycache__/app.cpython-311.pyc" in paths

    def test_rejects_absolute_and_protected_paths(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("alpha\n", encoding="utf-8")
        (tmp_path / ".runtime").mkdir()
        (tmp_path / ".runtime" / "secret.toml").write_text("token='x'\n", encoding="utf-8")

        with pytest.raises(ValueError, match="absolute paths"):
            l1_fs_server._read_file_region_result(
                path=str(tmp_path / "main.py"),
                start_line=1,
                end_line=1,
                evidence_gap=_gap(),
                reason_context_pack_insufficient="Need exact line.",
            )

        with pytest.raises(ValueError, match="protected location"):
            l1_fs_server._read_file_region_result(
                path=".runtime/secret.toml",
                start_line=1,
                end_line=1,
                evidence_gap=_gap(),
                reason_context_pack_insufficient="Need exact line.",
            )

    def test_truncates_by_char_cap(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("0123456789\nabcdefghij\n", encoding="utf-8")

        result = l1_fs_server._read_file_region_result(
            path="main.py",
            start_line=1,
            end_line=2,
            evidence_gap=_gap(),
            reason_context_pack_insufficient="Need exact line.",
            max_chars=7,
        )

        assert result["content"] == "0123456"
        assert result["limits"]["truncated"] is True
        assert result["limits"]["chars_returned"] == 7


@pytest.fixture
def l1_fs_integration_bootstrap(tmp_path: Path) -> McpClientBootstrap:
    (tmp_path / "main.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    env = {**os.environ, "ORBIT_WORKSPACE_ROOT": str(tmp_path)}
    return McpClientBootstrap(
        server_name="structured_filesystem",
        command=sys.executable,
        args=("-m", _L1_FS_MODULE, str(tmp_path)),
        env=env,
        transport="stdio",
    )


class TestStructuredFilesystemMcpIntegration:
    def test_server_lists_read_file_region(
        self, l1_fs_integration_bootstrap: McpClientBootstrap
    ) -> None:
        client = StdioMcpClient(l1_fs_integration_bootstrap)
        names = {d.original_name for d in client.list_tools()}
        assert names == {"read_file_region", "grep_scoped"}

    def test_attached_tool_has_l1_metadata_and_governance(
        self, l1_fs_integration_bootstrap: McpClientBootstrap
    ) -> None:
        registry = CapabilityRegistry()
        _client, registered = attach_mcp_server(l1_fs_integration_bootstrap, registry)

        assert set(registered) == {
            "mcp__structured_filesystem__read_file_region",
            "mcp__structured_filesystem__grep_scoped",
        }
        tool = registry.get("mcp__structured_filesystem__read_file_region")
        assert tool is not None
        assert tool.capability_layer == CapabilityLayer.STRUCTURED_PRIMITIVE
        assert tool.side_effect_class == "safe"
        assert tool.requires_approval is False
        assert tool.reveal_group == "mcp_structured_filesystem"

    def test_read_region_path_through_capability_boundary(
        self, tmp_path: Path, l1_fs_integration_bootstrap: McpClientBootstrap
    ) -> None:
        registry = CapabilityRegistry()
        attach_mcp_server(l1_fs_integration_bootstrap, registry)
        boundary = CapabilityBoundary(registry, tmp_path)

        result = boundary.execute(ToolRequest(
            tool_call_id="c1",
            tool_name="mcp__structured_filesystem__read_file_region",
            arguments={
                "path": "main.py",
                "start_line": 2,
                "end_line": 3,
                "evidence_gap": _gap(),
                "reason_context_pack_insufficient": "The visible context did not include these lines.",
            },
        ))

        assert result.ok is True
        assert result.governance_outcome.startswith("allowed")
        assert "two" in result.content
        assert "three" in result.content

    def test_grep_scoped_path_through_capability_boundary(
        self, tmp_path: Path, l1_fs_integration_bootstrap: McpClientBootstrap
    ) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("needle\n", encoding="utf-8")
        registry = CapabilityRegistry()
        attach_mcp_server(l1_fs_integration_bootstrap, registry)
        boundary = CapabilityBoundary(registry, tmp_path)

        result = boundary.execute(ToolRequest(
            tool_call_id="c1",
            tool_name="mcp__structured_filesystem__grep_scoped",
            arguments={
                "pattern": "needle",
                "path": "src",
                "evidence_gap_description": "Need candidate source locations.",
                "needed_evidence": "Bounded grep matches with file and line.",
                "reason_context_pack_insufficient": "The current context does not contain source locations.",
                "file_pattern": "*.py",
            },
        ))

        assert result.ok is True
        assert result.governance_outcome.startswith("allowed")
        assert "grep_scope" in result.content
        assert "src/app.py" in result.content
