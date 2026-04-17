"""Pytest MCP server first-slice tests (Handoff 17)."""
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
from src.capability.mcp_servers.pytest import stdio_server as pytest_server
from src.core.runtime.models import ToolRequest


_PYTEST_MODULE = "src.capability.mcp_servers.pytest.stdio_server"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ORBIT_WORKSPACE_ROOT", str(tmp_path))
    # Write a trivial passing test file
    (tmp_path / "test_trivial.py").write_text(
        "def test_one():\n    assert 1 + 1 == 2\n", encoding="utf-8"
    )
    # Minimal pyproject so pytest has a predictable root
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["."]\n', encoding="utf-8"
    )
    return tmp_path


class TestPytestResultHelpers:
    def test_pytest_runs_and_reports_pass(self, workspace: Path) -> None:
        result = pytest_server._run_pytest_structured_result(args=["-q", "test_trivial.py"])
        assert result["ok"] is True
        assert result["returncode"] == 0
        assert "1 passed" in result["stdout"]

    def test_pytest_runs_and_reports_failure(self, workspace: Path) -> None:
        (workspace / "test_fail.py").write_text(
            "def test_fail():\n    assert False\n", encoding="utf-8"
        )
        result = pytest_server._run_pytest_structured_result(args=["-q", "test_fail.py"])
        assert result["ok"] is False
        assert result["returncode"] != 0
        assert result["failure_kind"] == "pytest_nonzero_exit"

    def test_truncation(self, workspace: Path) -> None:
        result = pytest_server._run_pytest_structured_result(
            args=["-q", "test_trivial.py"], max_chars=50
        )
        assert result["stdout_truncated"] is True or len(result["stdout"]) <= 50

    def test_rejects_zero_max_chars(self, workspace: Path) -> None:
        with pytest.raises(ValueError):
            pytest_server._run_pytest_structured_result(args=[], max_chars=0)

    def test_rejects_absolute_path_args(self, workspace: Path) -> None:
        """Audit MED-1 fix: absolute-path args rejected as defense-in-depth."""
        with pytest.raises(ValueError, match="absolute paths"):
            pytest_server._run_pytest_structured_result(args=["/etc/passwd"])

    def test_timeout_surfaces_as_structured_failure(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Audit HIGH-1 fix: subprocess.TimeoutExpired must surface as
        ok=False with failure_kind='pytest_timeout', not crash the server."""
        import subprocess as _sp

        def _raise_timeout(*args, **kwargs):
            raise _sp.TimeoutExpired(cmd="pytest", timeout=kwargs.get("timeout", 120.0))

        monkeypatch.setattr(pytest_server.subprocess, "run", _raise_timeout)
        result = pytest_server._run_pytest_structured_result(args=["-q"], timeout_seconds=0.01)
        assert result["ok"] is False
        assert result["failure_kind"] == "pytest_timeout"
        assert result["timeout_seconds"] == 0.01


@pytest.fixture
def pytest_bootstrap(workspace: Path) -> McpClientBootstrap:
    env = {**os.environ, "ORBIT_WORKSPACE_ROOT": str(workspace)}
    return McpClientBootstrap(
        server_name="pytest",
        command=sys.executable,
        args=("-m", _PYTEST_MODULE, str(workspace)),
        env=env,
        transport="stdio",
    )


class TestPytestMcpIntegration:
    def test_server_lists_first_slice_tool(
        self, pytest_bootstrap: McpClientBootstrap
    ) -> None:
        client = StdioMcpClient(pytest_bootstrap)
        names = {d.original_name for d in client.list_tools()}
        assert "run_pytest_structured" in names

    def test_pytest_path_through_capability_boundary(
        self, workspace: Path, pytest_bootstrap: McpClientBootstrap
    ) -> None:
        registry = CapabilityRegistry()
        _client, registered = attach_mcp_server(pytest_bootstrap, registry)
        assert "mcp__pytest__run_pytest_structured" in registered
        boundary = CapabilityBoundary(registry, workspace)

        result = boundary.execute(ToolRequest(
            tool_call_id="c1",
            tool_name="mcp__pytest__run_pytest_structured",
            arguments={"args": ["-q", "test_trivial.py"]},
        ))
        assert result.ok is True
        assert result.governance_outcome == "allowed"
        assert "1 passed" in result.content

    def test_pytest_wrapper_metadata_matches_governance(
        self, pytest_bootstrap: McpClientBootstrap
    ) -> None:
        registry = CapabilityRegistry()
        attach_mcp_server(pytest_bootstrap, registry)
        wrapper = registry.get("mcp__pytest__run_pytest_structured")
        assert wrapper.side_effect_class == "safe"
        assert wrapper.requires_approval is False
