"""Mypy MCP server first-slice tests (Handoff 17)."""
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
from src.capability.mcp_servers.mypy import stdio_server as mypy_server
from src.runtime.models import ToolRequest


_MYPY_MODULE = "src.capability.mcp_servers.mypy.stdio_server"


pytestmark = pytest.mark.skipif(
    shutil.which("mypy") is None, reason="mypy binary not installed"
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ORBIT_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "clean.py").write_text("x: int = 1\n", encoding="utf-8")
    return tmp_path


class TestMypyResultHelpers:
    def test_mypy_clean_file_reports_ok(self, workspace: Path) -> None:
        result = mypy_server._run_mypy_structured_result(args=["--no-incremental", "clean.py"])
        assert result["ok"] is True
        assert result["returncode"] == 0

    def test_mypy_type_error_is_ok_with_diagnostics(self, workspace: Path) -> None:
        (workspace / "dirty.py").write_text("x: int = 'not an int'\n", encoding="utf-8")
        result = mypy_server._run_mypy_structured_result(
            args=["--show-error-codes", "--no-incremental", "dirty.py"]
        )
        # returncode=1 (type errors) → still ok=True because mypy RAN successfully
        assert result["ok"] is True
        assert result["returncode"] == 1
        assert "incompatible" in result["stdout"].lower() or "assign" in result["stdout"].lower()

    def test_rejects_zero_max_chars(self, workspace: Path) -> None:
        with pytest.raises(ValueError):
            mypy_server._run_mypy_structured_result(args=["."], max_chars=0)

    def test_rejects_absolute_path_args(self, workspace: Path) -> None:
        with pytest.raises(ValueError, match="absolute paths"):
            mypy_server._run_mypy_structured_result(args=["/etc/passwd"])

    def test_timeout_surfaces_as_structured_failure(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess as _sp

        def _raise_timeout(*args, **kwargs):
            raise _sp.TimeoutExpired(cmd="mypy", timeout=kwargs.get("timeout", 120.0))

        monkeypatch.setattr(mypy_server.subprocess, "run", _raise_timeout)
        result = mypy_server._run_mypy_structured_result(args=["clean.py"], timeout_seconds=0.01)
        assert result["ok"] is False
        assert result["failure_kind"] == "mypy_timeout"


@pytest.fixture
def mypy_bootstrap(workspace: Path) -> McpClientBootstrap:
    env = {**os.environ, "ORBIT_WORKSPACE_ROOT": str(workspace)}
    return McpClientBootstrap(
        server_name="mypy",
        command=sys.executable,
        args=("-m", _MYPY_MODULE, str(workspace)),
        env=env,
        transport="stdio",
    )


class TestMypyMcpIntegration:
    def test_server_lists_first_slice_tool(
        self, mypy_bootstrap: McpClientBootstrap
    ) -> None:
        client = StdioMcpClient(mypy_bootstrap)
        names = {d.original_name for d in client.list_tools()}
        assert "run_mypy_structured" in names

    def test_mypy_path_through_capability_boundary(
        self, workspace: Path, mypy_bootstrap: McpClientBootstrap
    ) -> None:
        registry = CapabilityRegistry()
        _client, registered = attach_mcp_server(mypy_bootstrap, registry)
        assert "mcp__mypy__run_mypy_structured" in registered
        boundary = CapabilityBoundary(registry, workspace)

        result = boundary.execute(ToolRequest(
            tool_call_id="c1",
            tool_name="mcp__mypy__run_mypy_structured",
            arguments={"args": ["--no-incremental", "clean.py"]},
        ))
        assert result.ok is True
        assert result.governance_outcome == "allowed"

    def test_mypy_wrapper_metadata_matches_governance(
        self, mypy_bootstrap: McpClientBootstrap
    ) -> None:
        registry = CapabilityRegistry()
        attach_mcp_server(mypy_bootstrap, registry)
        wrapper = registry.get("mcp__mypy__run_mypy_structured")
        assert wrapper.side_effect_class == "safe"
        assert wrapper.requires_approval is False
