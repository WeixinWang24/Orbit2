from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.capability.mcp_servers.l3_workflow import (
    DecisionResponse,
    SQLiteWorkflowRunStore,
    default_workflow_db_path,
    inspect_change_set_workflow_result,
    persist_decision_response,
    repo_recon_workflow_result,
    validate_decision_response,
)
from src.core.runtime.workflow_decision import dispatch_workflow_resume_branch
from src.capability.mcp_servers.l3_workflow import stdio_server as workflow_server


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
        "def main():\n    return 'ok'\n",
    )
    _write(
        tmp_path / "tests" / "test_app.py",
        "from src.app import main\n\n\ndef test_main():\n    assert main()\n",
    )
    _write(tmp_path / ".gitignore", ".runtime/\n__pycache__/\n*.pyc\n")
    _run_git(tmp_path, "add", ".")
    _run_git(tmp_path, "commit", "-m", "initial")
    _write(
        tmp_path / "src" / "app.py",
        "def main():\n    return changed()\n\n\ndef changed():\n    return 'ok'\n",
    )
    _write(
        tmp_path / "tests" / "test_app.py",
        "from src.app import changed\n\n\ndef test_changed():\n    assert changed()\n",
    )
    monkeypatch.setenv("ORBIT_WORKSPACE_ROOT", str(tmp_path))
    return tmp_path


def test_inspect_change_set_workflow_emits_decision_request(
    git_workspace: Path,
) -> None:
    result = inspect_change_set_workflow_result(
        repo_id="workflow_fixture",
        label="Workflow Fixture",
        max_diff_files=10,
        max_impact_files=10,
        max_impact_symbols=20,
        max_impact_edges=40,
    )

    assert result["ok"] is True
    assert result["workflow_name"] == "inspect_change_set_workflow"
    assert result["status"] == "waiting_for_decision"
    assert result["audit"]["provider_roundtrip_owner"] == "runtime_core"
    assert result["audit"]["provider_wait"] == "not_performed_inside_tool_call"

    package = result["decision_request_package"]
    assert package["message_type"] == "decision_request"
    assert package["status"] == "waiting_for_decision"
    assert package["workflow_run_id"] == result["workflow_run_id"]
    assert package["factual_context"]["decision_posture"] == "non_decisional"
    assert {
        option["option_id"]
        for option in package["admissible_options"]
    } >= {"read_exact_l1_evidence", "assemble_review_summary", "stop_and_report"}
    assert all("recommend" not in option["description"].lower() for option in package["admissible_options"])
    assert "admissible_options_are_not_recommendations" in package["constraints"]

    store = SQLiteWorkflowRunStore(default_workflow_db_path(git_workspace))
    try:
        persisted = store.get_run(result["workflow_run_id"])
    finally:
        store.close()
    assert persisted is not None
    assert persisted["status"] == "waiting_for_decision"
    assert persisted["current_decision_id"] == package["decision_id"]
    assert len(persisted["steps"]) == 3
    assert len(persisted["decision_requests"]) == 1


def test_inspect_change_set_workflow_mcp_entrypoint_delegates(
    git_workspace: Path,
) -> None:
    result = workflow_server.inspect_change_set_workflow(
        repo_id="workflow_mcp_fixture",
        label="Workflow MCP Fixture",
        max_diff_files=10,
        max_impact_files=10,
        max_impact_symbols=20,
        max_impact_edges=40,
    )

    assert result["ok"] is True
    assert result["workflow_name"] == "inspect_change_set_workflow"
    assert result["status"] == "waiting_for_decision"


def test_repo_recon_workflow_emits_decision_request(
    git_workspace: Path,
) -> None:
    result = repo_recon_workflow_result(
        repo_id="repo_recon_fixture",
        label="Repo Recon Fixture",
        max_tree_depth=3,
        max_tree_entries=80,
        max_symbols=40,
        max_diff_files=10,
    )

    assert result["ok"] is True
    assert result["workflow_name"] == "repo_recon_workflow"
    assert result["status"] == "waiting_for_decision"
    assert result["summary"]["repo_clean"] is False
    assert result["summary"]["diff_digest_run_id"] is not None
    assert result["audit"]["provider_roundtrip_owner"] == "runtime_core"

    package = result["decision_request_package"]
    assert package["message_type"] == "decision_request"
    assert package["factual_context"]["fact_domain"] == "repo_recon"
    assert package["factual_context"]["decision_posture"] == "non_decisional"
    option_ids = {option["option_id"] for option in package["admissible_options"]}
    assert option_ids >= {
        "summarize_workspace_orientation",
        "inspect_change_set",
        "read_repo_structure_evidence",
        "stop_and_report",
    }
    assert "admissible_options_are_not_recommendations" in package["constraints"]

    store = SQLiteWorkflowRunStore(default_workflow_db_path(git_workspace))
    try:
        persisted = store.get_run(result["workflow_run_id"])
    finally:
        store.close()
    assert persisted is not None
    assert persisted["status"] == "waiting_for_decision"
    assert persisted["current_decision_id"] == package["decision_id"]
    assert len(persisted["steps"]) == 4
    assert len(persisted["decision_requests"]) == 1


def test_repo_recon_workflow_mcp_entrypoint_delegates(
    git_workspace: Path,
) -> None:
    result = workflow_server.repo_recon_workflow(
        repo_id="repo_recon_mcp_fixture",
        label="Repo Recon MCP Fixture",
        max_tree_depth=3,
        max_tree_entries=80,
        max_symbols=40,
        max_diff_files=10,
    )

    assert result["ok"] is True
    assert result["workflow_name"] == "repo_recon_workflow"
    assert result["status"] == "waiting_for_decision"


def test_decision_response_validation_and_persistence(
    git_workspace: Path,
) -> None:
    result = inspect_change_set_workflow_result(
        repo_id="workflow_resume_fixture",
        label="Workflow Resume Fixture",
        max_diff_files=10,
        max_impact_files=10,
    )
    package = result["decision_request_package"]
    response = DecisionResponse(
        workflow_run_id=package["workflow_run_id"],
        decision_id=package["decision_id"],
        selected_option_id="assemble_review_summary",
        rationale_summary="Use existing facts for a compact review summary.",
    )

    resume = validate_decision_response(
        decision_request_package=package,
        response=response,
    )
    assert resume.selected_option["option_id"] == "assemble_review_summary"

    persisted_resume = persist_decision_response(
        workspace_root=git_workspace,
        decision_request_package=package,
        response=response,
    )
    assert persisted_resume["selected_option"]["option_id"] == "assemble_review_summary"

    store = SQLiteWorkflowRunStore(default_workflow_db_path(git_workspace))
    try:
        persisted = store.get_run(result["workflow_run_id"])
    finally:
        store.close()
    assert persisted is not None
    assert persisted["decision_responses"][0]["selected_option_id"] == "assemble_review_summary"


def test_repo_recon_workflow_transition_resume_dispatches_change_set_workflow(
    git_workspace: Path,
) -> None:
    result = repo_recon_workflow_result(
        repo_id="repo_recon_transition_fixture",
        label="Repo Recon Transition Fixture",
        max_tree_depth=3,
        max_tree_entries=80,
        max_symbols=40,
        max_diff_files=10,
    )
    package = result["decision_request_package"]
    dispatch_result = dispatch_workflow_resume_branch(
        workspace_root=git_workspace,
        decision_request_package=package,
        resume_request={
            "decision_response": {
                "workflow_run_id": package["workflow_run_id"],
                "decision_id": package["decision_id"],
                "selected_option_id": "inspect_change_set",
                "rationale_summary": "Continue from repo recon into change-set facts.",
            }
        },
    )

    assert dispatch_result["ok"] is True
    assert dispatch_result["branch_type"] == "workflow_transition"
    nested = dispatch_result["branch_result"]["workflow_result"]
    assert nested["workflow_name"] == "inspect_change_set_workflow"
    assert nested["status"] == "waiting_for_decision"


def test_invalid_decision_response_option_is_rejected(
    git_workspace: Path,
) -> None:
    result = inspect_change_set_workflow_result(
        repo_id="workflow_invalid_fixture",
        label="Workflow Invalid Fixture",
        max_diff_files=10,
        max_impact_files=10,
    )
    package = result["decision_request_package"]

    with pytest.raises(ValueError, match="not admissible"):
        validate_decision_response(
            decision_request_package=package,
            response={
                "workflow_run_id": package["workflow_run_id"],
                "decision_id": package["decision_id"],
                "selected_option_id": "hidden_planner_branch",
            },
        )
