"""Process MCP server first-slice tests (Handoff 24).

Covers the `run_process` tool: pure result helper tests (fast) plus an
integration path through the real stdio subprocess to verify governance
metadata and capability attachment. Uses `tmp_path` as workspace root so
nothing touches the surrounding Orbit2 working tree.
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
from src.capability.mcp_servers.process import stdio_server as proc_server
from src.core.runtime.models import ToolRequest


_PROC_MODULE = "src.capability.mcp_servers.process.stdio_server"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ORBIT_WORKSPACE_ROOT", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Pure result helpers (fast)
# ---------------------------------------------------------------------------


class TestProcessResultHelpers:
    def test_run_echo_succeeds(self, workspace: Path) -> None:
        r = proc_server._run_process_result(["echo", "hello"])
        assert r["ok"] is True
        assert r["exit_code"] == 0
        assert "hello" in r["stdout"]
        assert r["timed_out"] is False
        assert r["mutation_kind"] == "run_process"

    def test_run_failing_command_returns_nonzero(self, workspace: Path) -> None:
        r = proc_server._run_process_result(["false"])
        assert r["ok"] is True  # subprocess ran OK; exit code indicates failure
        assert r["exit_code"] != 0

    def test_run_process_stderr_captured(self, workspace: Path) -> None:
        r = proc_server._run_process_result(
            [sys.executable, "-c", "import sys; sys.stderr.write('err\\n')"]
        )
        assert r["ok"] is True
        assert "err" in r["stderr"]

    def test_run_process_stdout_truncation(self, workspace: Path) -> None:
        r = proc_server._run_process_result(
            [sys.executable, "-c", "print('x' * 200)"],
            max_output_chars=50,
        )
        assert r["ok"] is True
        assert len(r["stdout"]) <= 50
        assert r["stdout_truncated"] is True

    def test_run_process_cwd_escape_refused(self, workspace: Path) -> None:
        with pytest.raises(ValueError, match="escapes workspace"):
            proc_server._run_process_result(["echo", "hi"], cwd="../outside")

    def test_run_process_empty_command_raises(self, workspace: Path) -> None:
        with pytest.raises(ValueError):
            proc_server._run_process_result([])

    def test_run_process_invalid_command_item_raises(self, workspace: Path) -> None:
        with pytest.raises(ValueError):
            proc_server._run_process_result(["echo", ""])

    def test_run_process_in_subdirectory(self, workspace: Path) -> None:
        sub = workspace / "sub"
        sub.mkdir()
        r = proc_server._run_process_result(["pwd"], cwd="sub")
        assert r["ok"] is True
        assert r["exit_code"] == 0


# ---------------------------------------------------------------------------
# Integration: real stdio subprocess + Orbit2 capability attachment
# ---------------------------------------------------------------------------


@pytest.fixture
def proc_bootstrap(workspace: Path) -> McpClientBootstrap:
    env = {**os.environ, "ORBIT_WORKSPACE_ROOT": str(workspace)}
    return McpClientBootstrap(
        server_name="process",
        command=sys.executable,
        args=("-m", _PROC_MODULE, str(workspace)),
        env=env,
        transport="stdio",
    )


class TestProcessMcpIntegration:
    def test_server_lists_run_process(self, proc_bootstrap: McpClientBootstrap) -> None:
        client = StdioMcpClient(proc_bootstrap)
        descriptors = client.list_tools()
        names = {d.original_name for d in descriptors}
        assert "run_process" in names

    def test_run_process_through_capability_boundary(
        self, workspace: Path, proc_bootstrap: McpClientBootstrap
    ) -> None:
        registry = CapabilityRegistry()
        attach_mcp_server(proc_bootstrap, registry)
        boundary = CapabilityBoundary(registry, workspace)
        result = boundary.execute(ToolRequest(
            tool_call_id="p1",
            tool_name="mcp__process__run_process",
            arguments={"command": ["echo", "orbit2"]},
        ))
        assert result.ok is True
        assert "orbit2" in result.content
        assert result.governance_outcome.startswith("allowed")

    def test_run_process_wrapper_requires_approval(
        self, proc_bootstrap: McpClientBootstrap
    ) -> None:
        registry = CapabilityRegistry()
        attach_mcp_server(proc_bootstrap, registry)
        wrapper = registry.get("mcp__process__run_process")
        assert wrapper is not None
        assert wrapper.requires_approval is True
        assert wrapper.side_effect_class == "write"

    def test_run_process_reveal_group_is_mcp_process(
        self, proc_bootstrap: McpClientBootstrap
    ) -> None:
        registry = CapabilityRegistry()
        attach_mcp_server(proc_bootstrap, registry)
        wrapper = registry.get("mcp__process__run_process")
        assert wrapper is not None
        assert wrapper.reveal_group == "mcp_process"
