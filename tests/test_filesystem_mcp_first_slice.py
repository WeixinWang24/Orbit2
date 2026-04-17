"""Filesystem MCP server first-slice tests (Handoff 16).

Covers both the pure-result helpers (for fast unit coverage) and a real
stdio subprocess round trip (integration) to prove the server works through
Orbit2's capability attachment path.
"""
from __future__ import annotations

import os
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
from src.capability.mcp_servers.filesystem import stdio_server as fs_server
from src.core.runtime.models import ToolRequest


_FS_MODULE = "src.capability.mcp_servers.filesystem.stdio_server"


# ---------------------------------------------------------------------------
# Pure result helpers (fast)
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ORBIT_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "hello.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.txt").write_text("nested content", encoding="utf-8")
    return tmp_path


class TestFilesystemResultHelpers:
    def test_read_file_returns_content(self, workspace: Path) -> None:
        r = fs_server._read_file_result("hello.txt")
        assert r["ok"] is True
        assert r["content"] == "hello world"
        assert r["path"].endswith("hello.txt")
        assert r["truncated"] is False

    def test_read_file_truncation(self, workspace: Path) -> None:
        (workspace / "big.txt").write_text("x" * 1000, encoding="utf-8")
        r = fs_server._read_file_result("big.txt", max_bytes=100)
        assert r["truncated"] is True
        assert len(r["content"]) == 100

    def test_read_file_rejects_absolute_path(self, workspace: Path) -> None:
        with pytest.raises(ValueError):
            fs_server._read_file_result("/etc/passwd")

    def test_read_file_rejects_path_escape(self, workspace: Path) -> None:
        with pytest.raises(ValueError):
            fs_server._read_file_result("../outside")

    def test_read_file_rejects_missing(self, workspace: Path) -> None:
        with pytest.raises(ValueError):
            fs_server._read_file_result("does_not_exist.txt")

    def test_list_directory_returns_entries(self, workspace: Path) -> None:
        r = fs_server._list_directory_result(".")
        names = {e["name"] for e in r["entries"]}
        assert {"hello.txt", "subdir"}.issubset(names)
        assert r["ok"] is True

    def test_list_directory_truncation(self, workspace: Path) -> None:
        # Create enough entries to exceed the cap
        many = workspace / "many"
        many.mkdir()
        for i in range(20):
            (many / f"f{i}.txt").write_text("x", encoding="utf-8")
        r = fs_server._list_directory_result("many", max_entries=5)
        assert len(r["entries"]) == 5
        assert r["truncated"] is True

    def test_get_file_info(self, workspace: Path) -> None:
        r = fs_server._get_file_info_result("hello.txt")
        assert r["kind"] == "file"
        assert r["size"] == len("hello world")

    def test_get_file_info_directory(self, workspace: Path) -> None:
        r = fs_server._get_file_info_result("subdir")
        assert r["kind"] == "directory"

    def test_write_file_creates(self, workspace: Path) -> None:
        r = fs_server._write_file_result("new.txt", "brand new")
        assert r["ok"] is True
        assert r["mutation_kind"] == "write_file"
        assert (workspace / "new.txt").read_text(encoding="utf-8") == "brand new"

    def test_write_file_creates_parent_dirs(self, workspace: Path) -> None:
        fs_server._write_file_result("deep/sub/file.txt", "ok")
        assert (workspace / "deep" / "sub" / "file.txt").read_text(encoding="utf-8") == "ok"

    def test_write_file_overwrites(self, workspace: Path) -> None:
        fs_server._write_file_result("hello.txt", "replaced")
        assert (workspace / "hello.txt").read_text(encoding="utf-8") == "replaced"

    def test_write_file_rejects_escape(self, workspace: Path) -> None:
        with pytest.raises(ValueError):
            fs_server._write_file_result("../outside.txt", "bad")

    def test_replace_in_file_first_occurrence(self, workspace: Path) -> None:
        (workspace / "t.txt").write_text("foo foo foo", encoding="utf-8")
        r = fs_server._replace_in_file_result("t.txt", "foo", "bar")
        assert r["ok"] is True
        assert r["replacement_count"] == 1
        assert (workspace / "t.txt").read_text(encoding="utf-8") == "bar foo foo"

    def test_replace_in_file_missing_text(self, workspace: Path) -> None:
        r = fs_server._replace_in_file_result("hello.txt", "absent", "x")
        assert r["ok"] is False
        assert r["failure_kind"] == "old_text_not_found"

    def test_workspace_missing_env_and_arg_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ORBIT_WORKSPACE_ROOT", raising=False)
        # Clear sys.argv trailing arg too
        monkeypatch.setattr(sys, "argv", ["prog"])
        with pytest.raises(ValueError):
            fs_server._read_file_result("hello.txt")


# ---------------------------------------------------------------------------
# Integration: real stdio subprocess + Orbit2 capability attachment
# ---------------------------------------------------------------------------


@pytest.fixture
def fs_integration_bootstrap(tmp_path: Path) -> McpClientBootstrap:
    (tmp_path / "hello.txt").write_text("hello world", encoding="utf-8")
    env = {**os.environ, "ORBIT_WORKSPACE_ROOT": str(tmp_path)}
    return McpClientBootstrap(
        server_name="filesystem",
        command=sys.executable,
        args=("-m", _FS_MODULE, str(tmp_path)),
        env=env,
        transport="stdio",
    )


class TestFilesystemMcpIntegration:
    def test_server_lists_first_slice_tools(
        self, fs_integration_bootstrap: McpClientBootstrap
    ) -> None:
        client = StdioMcpClient(fs_integration_bootstrap)
        descriptors = client.list_tools()
        names = {d.original_name for d in descriptors}
        assert {"read_file", "list_directory", "get_file_info", "write_file", "replace_in_file"} <= names

    def test_read_path_through_capability_boundary(
        self, tmp_path: Path, fs_integration_bootstrap: McpClientBootstrap
    ) -> None:
        registry = CapabilityRegistry()
        client, registered = attach_mcp_server(fs_integration_bootstrap, registry)
        assert "mcp__filesystem__read_file" in registered

        boundary = CapabilityBoundary(registry, tmp_path)
        result = boundary.execute(ToolRequest(
            tool_call_id="c1",
            tool_name="mcp__filesystem__read_file",
            arguments={"path": "hello.txt"},
        ))
        assert result.ok is True
        assert "hello world" in result.content
        assert result.governance_outcome == "allowed"

    def test_write_path_through_capability_boundary(
        self, tmp_path: Path, fs_integration_bootstrap: McpClientBootstrap
    ) -> None:
        registry = CapabilityRegistry()
        attach_mcp_server(fs_integration_bootstrap, registry)
        boundary = CapabilityBoundary(registry, tmp_path)

        result = boundary.execute(ToolRequest(
            tool_call_id="c1",
            tool_name="mcp__filesystem__write_file",
            arguments={"path": "created.txt", "content": "mcp filesystem mutation"},
        ))
        assert result.ok is True
        assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "mcp filesystem mutation"
        assert result.governance_outcome == "allowed"

    def test_escape_path_refused_by_orbit2_before_reaching_server(
        self, tmp_path: Path, fs_integration_bootstrap: McpClientBootstrap
    ) -> None:
        """Orbit2-side governance must deny the call before it ever reaches
        the MCP server — proving closure is Orbit2-side, not delegated."""
        registry = CapabilityRegistry()
        attach_mcp_server(fs_integration_bootstrap, registry)
        boundary = CapabilityBoundary(registry, tmp_path)

        result = boundary.execute(ToolRequest(
            tool_call_id="c1",
            tool_name="mcp__filesystem__write_file",
            arguments={"path": "../outside.txt", "content": "leak"},
        ))
        assert result.ok is False
        assert result.governance_outcome.startswith("denied:")
        assert not (tmp_path.parent / "outside.txt").exists()

    def test_protected_prefix_refused_for_filesystem_write(
        self, tmp_path: Path, fs_integration_bootstrap: McpClientBootstrap
    ) -> None:
        registry = CapabilityRegistry()
        attach_mcp_server(fs_integration_bootstrap, registry)
        boundary = CapabilityBoundary(registry, tmp_path)

        result = boundary.execute(ToolRequest(
            tool_call_id="c1",
            tool_name="mcp__filesystem__write_file",
            arguments={"path": ".env.production", "content": "SECRET=leaked"},
        ))
        assert result.ok is False
        assert "protected location" in result.content
        assert not (tmp_path / ".env.production").exists()
