"""Ruff MCP server first-slice tests (Handoff 17)."""
from __future__ import annotations

import os
import shutil
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
from src.capability.mcp_servers.ruff import stdio_server as ruff_server
from src.runtime.models import ToolRequest


_RUFF_MODULE = "src.capability.mcp_servers.ruff.stdio_server"


pytestmark = pytest.mark.skipif(
    shutil.which("ruff") is None, reason="ruff binary not installed"
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ORBIT_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


class TestRuffResultHelpers:
    def test_ruff_clean_file_reports_ok(self, workspace: Path) -> None:
        result = ruff_server._run_ruff_structured_result(args=["check", "clean.py"])
        assert result["ok"] is True
        assert result["returncode"] == 0

    def test_ruff_dirty_file_reports_failure(self, workspace: Path) -> None:
        (workspace / "dirty.py").write_text("import os\n", encoding="utf-8")
        result = ruff_server._run_ruff_structured_result(args=["check", "--select=F401", "dirty.py"])
        assert result["ok"] is False
        assert result["failure_kind"] == "ruff_nonzero_exit"
        assert "F401" in result["stdout"] or "unused" in result["stdout"].lower()

    def test_rejects_zero_max_chars(self, workspace: Path) -> None:
        with pytest.raises(ValueError):
            ruff_server._run_ruff_structured_result(args=["check", "."], max_chars=0)

    def test_rejects_absolute_path_args(self, workspace: Path) -> None:
        with pytest.raises(ValueError, match="absolute paths"):
            ruff_server._run_ruff_structured_result(args=["check", "/etc"])

    def test_timeout_surfaces_as_structured_failure(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess as _sp

        def _raise_timeout(*args, **kwargs):
            raise _sp.TimeoutExpired(cmd="ruff", timeout=kwargs.get("timeout", 60.0))

        monkeypatch.setattr(ruff_server.subprocess, "run", _raise_timeout)
        result = ruff_server._run_ruff_structured_result(args=["check", "clean.py"], timeout_seconds=0.01)
        assert result["ok"] is False
        assert result["failure_kind"] == "ruff_timeout"


@pytest.fixture
def ruff_bootstrap(workspace: Path) -> McpClientBootstrap:
    env = {**os.environ, "ORBIT_WORKSPACE_ROOT": str(workspace)}
    return McpClientBootstrap(
        server_name="ruff",
        command=sys.executable,
        args=("-m", _RUFF_MODULE, str(workspace)),
        env=env,
        transport="stdio",
    )


class TestRuffMcpIntegration:
    def test_server_lists_first_slice_tool(
        self, ruff_bootstrap: McpClientBootstrap
    ) -> None:
        client = StdioMcpClient(ruff_bootstrap)
        names = {d.original_name for d in client.list_tools()}
        assert "run_ruff_structured" in names

    def test_ruff_path_through_capability_boundary(
        self, workspace: Path, ruff_bootstrap: McpClientBootstrap
    ) -> None:
        registry = CapabilityRegistry()
        _client, registered = attach_mcp_server(ruff_bootstrap, registry)
        assert "mcp__ruff__run_ruff_structured" in registered
        boundary = CapabilityBoundary(registry, workspace)

        result = boundary.execute(ToolRequest(
            tool_call_id="c1",
            tool_name="mcp__ruff__run_ruff_structured",
            arguments={"args": ["check", "clean.py"]},
        ))
        assert result.ok is True
        assert result.governance_outcome == "allowed"

    def test_ruff_wrapper_metadata_matches_governance(
        self, ruff_bootstrap: McpClientBootstrap
    ) -> None:
        registry = CapabilityRegistry()
        attach_mcp_server(ruff_bootstrap, registry)
        wrapper = registry.get("mcp__ruff__run_ruff_structured")
        assert wrapper.side_effect_class == "safe"
        assert wrapper.requires_approval is False
