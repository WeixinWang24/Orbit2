"""Repo Scout L2 toolchain MCP first-slice tests."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.capability.boundary import CapabilityBoundary
from src.capability.mcp import McpClientBootstrap, StdioMcpClient, attach_mcp_server
from src.capability.mcp_servers.l2_toolchain.repo_scout import (
    stdio_server as repo_scout_server,
)
from src.capability.registry import CapabilityRegistry
from src.core.runtime.models import ToolRequest


REPO_SCOUT_MODULE = "src.capability.mcp_servers.l2_toolchain.repo_scout.stdio_server"


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True, text=True)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def git_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _run_git(tmp_path, "init")
    _run_git(tmp_path, "config", "user.email", "orbit2@example.test")
    _run_git(tmp_path, "config", "user.name", "Orbit2 Test")
    _write(
        tmp_path / "src" / "app.py",
        "from pathlib import Path\n\n\ndef main():\n    return Path.cwd()\n",
    )
    _write(
        tmp_path / "tests" / "test_app.py",
        "from src.app import main\n\n\ndef test_main():\n    assert main()\n",
    )
    _write(tmp_path / "README.md", "# Fixture\n\nRepo Scout fixture.\n")
    _write(tmp_path / ".gitignore", ".runtime/\n__pycache__/\n*.pyc\n")
    _run_git(tmp_path, "add", ".")
    _run_git(tmp_path, "commit", "-m", "initial")
    _write(
        tmp_path / "src" / "app.py",
        "from pathlib import Path\n\n\ndef main():\n    return Path.cwd()\n\n\ndef changed():\n    return main()\n",
    )
    _write(tmp_path / "scratch.py", "def scratch():\n    return 'new'\n")
    _write(
        tmp_path / "tests" / "test_app.py",
        "from src.app import changed\n\n\ndef test_changed():\n    assert changed()\n",
    )
    _write(tmp_path / ".DS_Store", "finder noise\n")
    _write(tmp_path / ".claude" / "settings.json", "{}\n")
    monkeypatch.setenv("ORBIT_WORKSPACE_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def repo_scout_bootstrap(git_workspace: Path) -> McpClientBootstrap:
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
        "ORBIT_WORKSPACE_ROOT": str(git_workspace),
    }
    return McpClientBootstrap(
        server_name="repo_scout",
        command=sys.executable,
        args=("-m", REPO_SCOUT_MODULE, str(git_workspace)),
        env=env,
        transport="stdio",
    )


class TestRepoScoutResultHelpers:
    def test_repository_overview_reuses_tree_git_and_code_intel(
        self,
        git_workspace: Path,
    ) -> None:
        result = repo_scout_server._repo_scout_repository_overview_result(
            repo_id="fixture_overview",
            label="Fixture",
            max_tree_depth=3,
            max_tree_entries=80,
            max_symbols=30,
        )

        assert result["ok"] is True
        assert result["toolchain_name"] == "repo_scout_repository_overview"
        assert result["run_id"].startswith("tcr_")
        assert result["audit"]["message_type"] == "fact_report"
        assert result["audit"]["decision_posture"] == "non_decisional"
        assert result["summary"]["clean"] is False
        assert result["summary"]["indexed_file_count"] >= 2
        assert [step["step_id"] for step in result["trace"]] == [
            "step_001_collect_repository_tree",
            "step_002_collect_git_orientation",
            "step_003_index_code_intel",
            "step_004_build_repository_overview",
        ]
        assert "src" in result["overview"]["tree"]["top_level_dirs"]
        assert result["overview"]["tree"]["file_extension_counts"][".py"] >= 2
        assert any(
            candidate["path"] == "README.md"
            for candidate in result["overview"]["orientation_candidates"]
        )
        assert any(
            item["path"] == "src/app.py"
            for item in result["overview"]["code_intel"]["top_symbol_files"]
        )
        assert result["audit"]["lower_level_reuse"] == [
            "filesystem.directory_tree",
            "git.status",
            "git.log",
            "code_intel.index",
        ]

        persisted = repo_scout_server.toolchain_get_run(result["run_id"])
        assert persisted["ok"] is True
        assert len(persisted["run"]["steps"]) == 4

        region = repo_scout_server.toolchain_read_artifact_region(
            result["run_id"],
            "artifact_002_repository_overview",
            max_chars=4_000,
        )
        assert region["ok"] is True
        payload = json.loads(region["artifact_region"]["content"])
        assert payload["tree"]["entry_count"] >= 3

    def test_changed_manifest_normalizes_noise_and_directories(
        self,
        git_workspace: Path,
    ) -> None:
        manifest = repo_scout_server._changed_manifest(
            cwd=None,
            include_untracked=True,
        )

        paths = {item["path"] for item in manifest["files"]}
        ignored = {item["path"]: item["ignore_reason"] for item in manifest["ignored"]}
        directories = {item["path"] for item in manifest["directories"]}
        assert ".DS_Store" not in paths
        assert ignored[".DS_Store"] == "ignored_path"
        assert ".claude/" in directories
        assert "scratch.py" in paths

    def test_diff_digest_maps_hunks_to_symbols_and_classification_tags(
        self,
        git_workspace: Path,
    ) -> None:
        result = repo_scout_server._repo_scout_diff_digest_result(
            repo_id="fixture_digest",
            label="Fixture Digest",
        )

        assert result["ok"] is True
        assert result["toolchain_name"] == "repo_scout_diff_digest"
        assert result["run_id"].startswith("tcr_")
        assert result["audit"]["message_type"] == "fact_report"
        assert result["audit"]["decision_posture"] == "non_decisional"
        assert result["summary"]["file_count"] >= 2
        assert result["summary"]["hunk_count"] >= 1
        assert result["summary"]["ignored_state_count"] >= 1
        assert [step["step_id"] for step in result["trace"]] == [
            "step_001_collect_changed_manifest",
            "step_002_collect_file_diffs",
            "step_003_index_and_digest",
            "step_004_persist_diff_digest",
        ]

        files = {
            item["path"]: item
            for item in result["diff_digest"]["files"]
        }
        app = files["src/app.py"]
        assert app["diff_available"] is True
        assert app["hunk_count"] >= 1
        assert app["additions"] >= 2
        assert any(symbol["name"] == "changed" for symbol in app["touched_symbols"])
        assert app["evidence_read_candidates"][0]["tool"] == "structured_git.read_diff_hunk"

        scratch = files["scratch.py"]
        assert scratch["diff_available"] is False
        assert scratch["evidence_read_candidates"][0]["tool"] == "structured_filesystem.read_file_region"
        assert ".claude/" in {item["path"] for item in result["diff_digest"]["directories"]}
        assert ".DS_Store" in {item["path"] for item in result["diff_digest"]["ignored"]}

        persisted = repo_scout_server.toolchain_get_run(result["run_id"])
        assert persisted["ok"] is True
        assert len(persisted["run"]["steps"]) == 4

    def test_impact_scope_reports_changed_symbols_and_direct_adjacency(
        self,
        git_workspace: Path,
    ) -> None:
        result = repo_scout_server._repo_scout_impact_scope_result(
            repo_id="fixture_impact",
            label="Fixture Impact",
            max_impact_files=20,
            max_impact_symbols=20,
            max_impact_edges=40,
        )

        assert result["ok"] is True
        assert result["toolchain_name"] == "repo_scout_impact_scope"
        assert result["audit"]["message_type"] == "fact_report"
        assert result["audit"]["decision_posture"] == "non_decisional"
        assert result["audit"]["scope_depth"] == "direct_edge_name_match_only"
        assert [step["step_id"] for step in result["trace"]] == [
            "step_001_collect_changed_manifest",
            "step_002_collect_file_diffs",
            "step_003_index_and_scope",
            "step_004_persist_impact_scope",
        ]

        impact = result["impact_scope"]
        assert impact["boundary"]["decision_posture"] == "non_decisional"
        assert impact["summary"]["changed_symbol_count"] >= 1
        assert any(
            symbol["name"] == "changed"
            for item in impact["changed_symbols"]
            if item["path"] == "src/app.py"
            for symbol in item["symbols"]
        )
        assert any(
            edge["target_name"] == "changed" and edge["file_path"] == "tests/test_app.py"
            for edge in impact["adjacency"]["incoming_references"]
        )
        assert "tests/test_app.py" in {
            item["path"]
            for item in impact["test_adjacency"]["changed_test_files"]
        }

        persisted = repo_scout_server.toolchain_get_run(result["run_id"])
        assert persisted["ok"] is True
        assert len(persisted["run"]["steps"]) == 4

    def test_changed_context_reuses_git_and_code_intel(
        self,
        git_workspace: Path,
    ) -> None:
        result = repo_scout_server._repo_scout_changed_context_result(
            repo_id="fixture",
            label="Fixture",
        )

        assert result["ok"] is True
        assert result["run_id"].startswith("tcr_")
        assert result["audit"]["message_type"] == "fact_report"
        assert result["audit"]["decision_posture"] == "non_decisional"
        assert result["summary"]["clean"] is False
        assert result["summary"]["changed_file_state_count"] >= 2
        assert result["summary"]["indexed_file_count"] >= 2
        assert result["summary"]["indexed_symbol_count"] >= 3
        assert [step["step_id"] for step in result["trace"]] == [
            "step_001_collect_git_manifest",
            "step_002_index_code_intel",
            "step_003_build_scout_context",
        ]

        contexts = {
            ctx["path"]: ctx
            for ctx in result["code_intel"]["file_contexts"]
        }
        assert "src/app.py" in contexts
        assert ".DS_Store" not in contexts
        assert ".claude/" not in contexts
        assert any(symbol["name"] == "changed" for symbol in contexts["src/app.py"]["symbols"])
        assert any(edge["target_name"] == "pathlib.Path" for edge in contexts["src/app.py"]["imports"])
        assert (git_workspace / ".runtime" / "toolchain_runs.sqlite3").exists()
        assert (git_workspace / ".runtime" / "code_intel.db").exists()

        persisted = repo_scout_server.toolchain_get_run(result["run_id"])
        assert persisted["ok"] is True
        assert persisted["run"]["toolchain_name"] == "repo_scout_changed_context"
        assert len(persisted["run"]["steps"]) == 3

        region = repo_scout_server.toolchain_read_artifact_region(
            result["run_id"],
            "artifact_002_file_contexts",
            max_chars=4_000,
        )
        assert region["ok"] is True
        payload = json.loads(region["artifact_region"]["content"])
        assert any(item["path"] == "src/app.py" for item in payload)

    def test_include_untracked_false_excludes_untracked_context(
        self,
        git_workspace: Path,
    ) -> None:
        result = repo_scout_server._repo_scout_changed_context_result(
            repo_id="fixture_no_untracked",
            include_untracked=False,
        )

        paths = {ctx["path"] for ctx in result["code_intel"]["file_contexts"]}
        assert "src/app.py" in paths
        assert "scratch.py" not in paths


class TestRepoScoutMcpIntegration:
    def test_server_lists_first_slice_tools(
        self,
        repo_scout_bootstrap: McpClientBootstrap,
    ) -> None:
        client = StdioMcpClient(repo_scout_bootstrap)
        names = {d.original_name for d in client.list_tools()}
        assert "repo_scout_repository_overview" in names
        assert "repo_scout_diff_digest" in names
        assert "repo_scout_impact_scope" in names
        assert "repo_scout_changed_context" in names
        assert "toolchain_get_run" in names
        assert "toolchain_get_step" in names
        assert "toolchain_read_artifact_region" in names

    def test_repo_scout_path_through_capability_boundary(
        self,
        git_workspace: Path,
        repo_scout_bootstrap: McpClientBootstrap,
    ) -> None:
        registry = CapabilityRegistry()
        attach_mcp_server(repo_scout_bootstrap, registry)
        boundary = CapabilityBoundary(registry, git_workspace)

        result = boundary.execute(ToolRequest(
            tool_call_id="c1",
            tool_name="mcp__repo_scout__repo_scout_changed_context",
            arguments={"repo_id": "fixture_boundary", "include_untracked": False},
        ))

        assert result.ok is True
        assert result.governance_outcome.startswith("allowed")
        payload = json.loads(result.content)
        assert payload["toolchain_name"] == "repo_scout_changed_context"
        assert payload["audit"]["lower_level_reuse"] == [
            "git.status",
            "git.changed_files",
            "code_intel.index",
        ]

    def test_repo_scout_wrapper_metadata_matches_governance(
        self,
        repo_scout_bootstrap: McpClientBootstrap,
    ) -> None:
        registry = CapabilityRegistry()
        attach_mcp_server(repo_scout_bootstrap, registry)
        wrapper = registry.get("mcp__repo_scout__repo_scout_changed_context")
        assert wrapper.side_effect_class == "safe"
        assert wrapper.requires_approval is False
        assert wrapper.capability_layer.value == "toolchain"

        overview = registry.get("mcp__repo_scout__repo_scout_repository_overview")
        assert overview.side_effect_class == "safe"
        assert overview.requires_approval is False
        assert overview.capability_layer.value == "toolchain"

        digest = registry.get("mcp__repo_scout__repo_scout_diff_digest")
        assert digest.side_effect_class == "safe"
        assert digest.requires_approval is False
        assert digest.capability_layer.value == "toolchain"

        impact = registry.get("mcp__repo_scout__repo_scout_impact_scope")
        assert impact.side_effect_class == "safe"
        assert impact.requires_approval is False
        assert impact.capability_layer.value == "toolchain"
