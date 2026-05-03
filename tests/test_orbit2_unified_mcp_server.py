from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from src.capability.mcp import McpClientBootstrap, StdioMcpClient
from src.capability.mcp_compare import compare_mcp_calls


ORBIT2_MODULE = "src.capability.mcp_servers.orbit2.stdio_server"
FILESYSTEM_MODULE = "src.capability.mcp_servers.filesystem.stdio_server"
STRUCTURED_FS_MODULE = "src.capability.mcp_servers.l1_structured.filesystem.stdio_server"
ORBIT2_MUTATION_TOOLS = {
    "orbit2_filesystem_write_file",
    "orbit2_filesystem_replace_in_file",
    "orbit2_filesystem_replace_all_in_file",
    "orbit2_filesystem_create_directory",
    "orbit2_filesystem_move_file",
    "orbit2_git_restore",
    "orbit2_git_unstage",
    "orbit2_git_checkout_branch",
    "orbit2_git_add",
    "orbit2_git_commit",
    "orbit2_process_run_process",
}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "main.py").write_text(
        "from pathlib import Path\n\n\ndef main():\n    return Path.cwd()\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['.']\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "Index.md").write_text("# Index\n\n[[Second]] #orbit2\n", encoding="utf-8")
    (root / "Second.md").write_text("# Second\n\nBody\n", encoding="utf-8")
    return root


def _client(
    server_name: str,
    module: str,
    workspace: Path,
    vault: Path | None = None,
    profile: str | None = None,
) -> StdioMcpClient:
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
        "ORBIT_WORKSPACE_ROOT": str(workspace),
    }
    args = ["-m", module, str(workspace)]
    if vault is not None:
        env["ORBIT_OBSIDIAN_VAULT_ROOT"] = str(vault)
        args.extend(["--vault", str(vault)])
    if profile is not None:
        args.extend(["--profile", profile])
    return StdioMcpClient(
        McpClientBootstrap(
            server_name=server_name,
            command=sys.executable,
            args=tuple(args),
            env=env,
        )
    )


class TestOrbit2UnifiedMcpServer:
    def test_unified_server_lists_orbit2_namespace_tools(
        self, workspace: Path, vault: Path
    ) -> None:
        client = _client("orbit2", ORBIT2_MODULE, workspace, vault)
        names = {tool.original_name for tool in client.list_tools()}

        assert len(names) == 59
        assert "orbit2_filesystem_read_file" in names
        assert "orbit2_filesystem_write_file" in names
        assert "orbit2_structured_filesystem_read_file_region" in names
        assert "orbit2_structured_git_read_git_show_region" in names
        assert "orbit2_pytest_diagnose_failures" in names
        assert "orbit2_code_intel_repository_summary" in names
        assert "orbit2_code_intel_find_symbols" in names
        assert "orbit2_code_intel_file_context" in names
        assert "orbit2_code_intel_export_fragment_summary" in names
        assert "orbit2_repo_scout_repository_overview" in names
        assert "orbit2_repo_scout_diff_digest" in names
        assert "orbit2_repo_scout_impact_scope" in names
        assert "orbit2_repo_scout_changed_context" in names
        assert "orbit2_repo_scout_toolchain_get_run" in names
        assert "orbit2_workflow_inspect_change_set_workflow" in names
        assert "orbit2_workflow_repo_recon_workflow" in names
        assert "orbit2_obsidian_search_notes" in names

    def test_read_only_profile_hides_mutation_tools(
        self, workspace: Path, vault: Path
    ) -> None:
        client = _client("orbit2", ORBIT2_MODULE, workspace, vault, profile="read-only")
        names = {tool.original_name for tool in client.list_tools()}

        assert len(names) == 48
        assert "orbit2_filesystem_read_file" in names
        assert "orbit2_structured_filesystem_read_file_region" in names
        assert "orbit2_code_intel_repository_summary" in names
        assert "orbit2_repo_scout_repository_overview" in names
        assert "orbit2_workflow_inspect_change_set_workflow" in names
        assert "orbit2_workflow_repo_recon_workflow" in names
        assert names.isdisjoint(ORBIT2_MUTATION_TOOLS)

    def test_unified_filesystem_result_matches_family_server(
        self, workspace: Path, vault: Path
    ) -> None:
        orbit2 = _client("orbit2", ORBIT2_MODULE, workspace, vault)
        filesystem = _client("filesystem", FILESYSTEM_MODULE, workspace)

        aggregate, family = compare_mcp_calls([
            (
                orbit2,
                "orbit2_aggregate",
                "orbit2_filesystem_read_file",
                {"path": "main.py"},
            ),
            (
                filesystem,
                "orbit2_family",
                "read_file",
                {"path": "main.py"},
            ),
        ])

        aggregate_payload = json.loads(aggregate.result.content)
        family_payload = json.loads(family.result.content)
        assert aggregate.result.ok is True
        assert family.result.ok is True
        assert aggregate_payload["content"] == family_payload["content"]
        assert Path(aggregate_payload["path"]).name == Path(family_payload["path"]).name == "main.py"
        assert aggregate.elapsed_ms > 0
        assert family.elapsed_ms > 0

    def test_unified_structured_region_matches_family_server(
        self, workspace: Path, vault: Path
    ) -> None:
        orbit2 = _client("orbit2", ORBIT2_MODULE, workspace, vault)
        structured = _client("structured_filesystem", STRUCTURED_FS_MODULE, workspace)
        evidence_gap = {
            "description": "compare structured read result",
            "needed_evidence": "main.py first lines",
        }
        arguments = {
            "path": "main.py",
            "start_line": 1,
            "end_line": 3,
            "evidence_gap": evidence_gap,
            "reason_context_pack_insufficient": "comparison test needs exact region",
        }

        aggregate, family = compare_mcp_calls([
            (
                orbit2,
                "orbit2_aggregate",
                "orbit2_structured_filesystem_read_file_region",
                arguments,
            ),
            (
                structured,
                "orbit2_family",
                "read_file_region",
                arguments,
            ),
        ])

        aggregate_payload = json.loads(aggregate.result.content)
        family_payload = json.loads(family.result.content)
        assert aggregate_payload["content"] == family_payload["content"]
        assert aggregate_payload["evidence_type"] == "file_region"
        assert aggregate_payload["target"] == family_payload["target"]
        assert aggregate.elapsed_ms > 0
        assert family.elapsed_ms > 0

    def test_unified_obsidian_tool_uses_configured_vault(
        self, workspace: Path, vault: Path
    ) -> None:
        client = _client("orbit2", ORBIT2_MODULE, workspace, vault)
        result = client.call_tool(
            "orbit2_obsidian_search_notes",
            {"query": "orbit2", "max_results": 5},
        )
        payload = json.loads(result.content)
        assert result.ok is True
        assert payload["matches"][0]["path"] == "Index.md"


class TestCodexNativeMcpBaseline:
    def test_codex_cli_mcp_lists_native_tools_when_available(self) -> None:
        if shutil.which("codex") is None:
            pytest.skip("codex CLI not available")

        client = StdioMcpClient(
            McpClientBootstrap(
                server_name="codex_cli",
                command="codex",
                args=("mcp-server",),
            )
        )
        names = {tool.original_name for tool in client.list_tools()}
        assert {"codex", "codex-reply"}.issubset(names)
