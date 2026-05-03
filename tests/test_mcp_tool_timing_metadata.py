from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from src.capability.mcp import McpClientBootstrap, StdioMcpClient
from src.capability.mcp_compare import compare_mcp_calls
from src.capability.mcp_servers.timing import attach_tool_call_timing


FILESYSTEM_MODULE = "src.capability.mcp_servers.filesystem.stdio_server"
ORBIT2_MODULE = "src.capability.mcp_servers.orbit2.stdio_server"


def _client(server_name: str, module: str, workspace: Path) -> StdioMcpClient:
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
        "ORBIT_WORKSPACE_ROOT": str(workspace),
    }
    return StdioMcpClient(
        McpClientBootstrap(
            server_name=server_name,
            command=sys.executable,
            args=("-m", module, str(workspace)),
            env=env,
        )
    )


def _assert_tool_call_metadata(
    payload: dict,
    *,
    server_name: str,
    tool_name: str,
) -> None:
    metadata = payload["metadata"]
    tool_call = metadata["tool_call"]
    assert tool_call["server_name"] == server_name
    assert tool_call["tool_name"] == tool_name
    assert tool_call["clock"] == "perf_counter_ns"
    assert tool_call["elapsed_ns"] > 0
    assert tool_call["elapsed_ms"] >= 0


def test_timing_helper_preserves_existing_metadata() -> None:
    payload = attach_tool_call_timing(
        {"ok": True, "metadata": {"capability_layer": "toolchain"}},
        server_name="repo_scout",
        tool_name="repo_scout_changed_context",
        elapsed_ns=1_234_567,
    )

    assert payload["metadata"]["capability_layer"] == "toolchain"
    assert payload["metadata"]["tool_call"]["elapsed_ms"] == 1.235


def test_family_mcp_tool_result_carries_tool_call_timing(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")
    client = _client("filesystem", FILESYSTEM_MODULE, tmp_path)

    result = client.call_tool("read_file", {"path": "main.py"})
    payload = json.loads(result.content)

    assert result.ok is True
    _assert_tool_call_metadata(
        payload,
        server_name="filesystem",
        tool_name="read_file",
    )


def test_orbit2_aggregate_tool_result_carries_aggregate_tool_timing(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")
    client = _client("orbit2", ORBIT2_MODULE, tmp_path)

    result = client.call_tool("orbit2_filesystem_read_file", {"path": "main.py"})
    payload = json.loads(result.content)

    assert result.ok is True
    _assert_tool_call_metadata(
        payload,
        server_name="orbit2",
        tool_name="orbit2_filesystem_read_file",
    )


def test_mcp_compare_keeps_tool_call_time_separate_from_transport(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")
    [result] = compare_mcp_calls([
        (
            _client("filesystem", FILESYSTEM_MODULE, tmp_path),
            "family",
            "read_file",
            {"path": "main.py"},
        )
    ])

    assert result.transport_elapsed_ms > 0
    assert result.elapsed_ms == result.transport_elapsed_ms
    assert result.tool_call_elapsed_ms is not None
    assert result.tool_call_elapsed_ms >= 0
