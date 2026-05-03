from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from src.capability.boundary import CapabilityBoundary
from src.capability.mcp import McpClientBootstrap, StdioMcpClient, attach_mcp_server
from src.capability.mcp_servers.l2_toolchain.code_intel import (
    stdio_server as code_intel_server,
)
from src.capability.registry import CapabilityRegistry
from src.core.runtime.models import ToolRequest


CODE_INTEL_MODULE = "src.capability.mcp_servers.l2_toolchain.code_intel.stdio_server"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _workspace(tmp_path: Path, monkeypatch) -> Path:
    _write(
        tmp_path / "src" / "app.py",
        "from pathlib import Path\n\n\ndef main():\n    return helper(Path.cwd())\n\n\ndef helper(path):\n    return path\n",
    )
    _write(
        tmp_path / "tests" / "test_app.py",
        "from src.app import main\n\n\ndef test_main():\n    assert main()\n",
    )
    monkeypatch.setenv("ORBIT_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "ORBIT2_MCP_CODE_INTEL_DB_PATH",
        str(tmp_path / ".runtime" / "code_intel_mcp_test.db"),
    )
    return tmp_path


def _bootstrap(workspace: Path) -> McpClientBootstrap:
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
        "ORBIT_WORKSPACE_ROOT": str(workspace),
        "ORBIT2_MCP_CODE_INTEL_DB_PATH": str(
            workspace / ".runtime" / "code_intel_mcp_test.db"
        ),
    }
    return McpClientBootstrap(
        server_name="code_intel",
        command=sys.executable,
        args=("-m", CODE_INTEL_MODULE, str(workspace)),
        env=env,
    )


class TestCodeIntelMcpResultHelpers:
    def test_repository_summary_indexes_workspace(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        workspace = _workspace(tmp_path, monkeypatch)

        result = code_intel_server._repository_summary_result(
            repo_id="fixture",
            label="Fixture",
        )

        assert result["ok"] is True
        assert result["toolchain_name"] == "code_intel_repository_summary"
        assert result["audit"]["message_type"] == "fact_report"
        assert result["audit"]["decision_posture"] == "non_decisional"
        assert result["summary"]["file_count"] == 2
        assert result["summary"]["symbol_count"] >= 4
        assert result["summary"]["edge_count"] >= 2
        assert (workspace / ".runtime" / "code_intel_mcp_test.db").exists()

    def test_find_symbols_returns_bounded_fact_report(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _workspace(tmp_path, monkeypatch)

        result = code_intel_server._find_symbols_result(
            repo_id="fixture_symbols",
            name="main",
            limit=10,
        )

        assert result["ok"] is True
        assert result["toolchain_name"] == "code_intel_find_symbols"
        assert result["summary"]["symbol_count"] == 1
        assert result["symbols"][0]["qualified_name"] == "src.app.main"
        assert result["audit"]["epistemic_posture"] == "observed_or_mechanically_derived"

    def test_file_context_returns_symbols_imports_and_calls(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _workspace(tmp_path, monkeypatch)

        result = code_intel_server._file_context_result(
            path="src/app.py",
            repo_id="fixture_context",
        )

        assert result["ok"] is True
        assert result["target"]["path"] == "src/app.py"
        assert any(symbol["name"] == "main" for symbol in result["symbols"])
        assert any(edge["target_name"] == "pathlib.Path" for edge in result["imports"])
        assert any(edge["target_name"] == "helper" for edge in result["calls"])

    def test_export_fragment_summary_is_bounded(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _workspace(tmp_path, monkeypatch)

        result = code_intel_server._export_fragment_summary_result(
            repo_id="fixture_fragment",
            max_nodes=3,
            max_edges=3,
        )

        assert result["ok"] is True
        assert result["toolchain_name"] == "code_intel_export_fragment_summary"
        assert result["summary"]["node_count"] >= 3
        assert result["limits"]["nodes_returned"] == 3
        assert result["limits"]["max_edges"] == 3
        assert result["audit"]["lower_level_reuse"] == [
            "code_intel.index",
            "code_intel.fragment",
        ]

    def test_rejects_protected_path_prefix(self, tmp_path: Path, monkeypatch) -> None:
        _workspace(tmp_path, monkeypatch)

        try:
            code_intel_server._file_context_result(
                path=".runtime/code_intel_mcp_test.db",
                repo_id="fixture_protected",
            )
        except ValueError as exc:
            assert "protected location" in str(exc)
        else:
            raise AssertionError("protected path was not rejected")


class TestCodeIntelMcpIntegration:
    def test_server_lists_first_slice_tools(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        workspace = _workspace(tmp_path, monkeypatch)
        client = StdioMcpClient(_bootstrap(workspace))

        names = {tool.original_name for tool in client.list_tools()}

        assert names == {
            "code_intel_repository_summary",
            "code_intel_find_symbols",
            "code_intel_file_context",
            "code_intel_export_fragment_summary",
        }

    def test_code_intel_path_through_capability_boundary(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        workspace = _workspace(tmp_path, monkeypatch)
        registry = CapabilityRegistry()
        attach_mcp_server(_bootstrap(workspace), registry)
        boundary = CapabilityBoundary(registry, workspace)

        result = boundary.execute(ToolRequest(
            tool_call_id="c1",
            tool_name="mcp__code_intel__code_intel_find_symbols",
            arguments={"repo_id": "fixture_boundary", "name": "main"},
        ))

        assert result.ok is True
        assert result.governance_outcome.startswith("allowed")
        payload = json.loads(result.content)
        assert payload["toolchain_name"] == "code_intel_find_symbols"
        assert payload["symbols"][0]["name"] == "main"

    def test_code_intel_wrapper_metadata_matches_governance(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        workspace = _workspace(tmp_path, monkeypatch)
        registry = CapabilityRegistry()
        attach_mcp_server(_bootstrap(workspace), registry)

        for tool_name in (
            "mcp__code_intel__code_intel_repository_summary",
            "mcp__code_intel__code_intel_find_symbols",
            "mcp__code_intel__code_intel_file_context",
            "mcp__code_intel__code_intel_export_fragment_summary",
        ):
            wrapper = registry.get(tool_name)
            assert wrapper.side_effect_class == "safe"
            assert wrapper.requires_approval is False
            assert wrapper.capability_layer.value == "toolchain"
            assert wrapper.reveal_group == "mcp_code_intel"
