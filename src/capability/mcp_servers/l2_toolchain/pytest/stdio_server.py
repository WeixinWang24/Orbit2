"""L2 pytest toolchain MCP server with persisted run traces."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.capability.mcp_servers.l1_structured.filesystem import (
    stdio_server as structured_filesystem,
)
from src.capability.mcp_servers.l2_toolchain.base import ToolchainRunRecorder
from src.capability.mcp_servers.l2_toolchain.schemas import (
    ToolchainEvidenceRef,
    ToolchainFinding,
    now_iso,
)
from src.capability.mcp_servers.l2_toolchain.store import (
    SQLiteToolchainRunStore,
    default_toolchain_db_path,
)

SERVER_NAME = "pytest"
WORKSPACE_ROOT_ENV = "ORBIT_WORKSPACE_ROOT"
DEFAULT_MAX_CHARS = 12_000
PYTEST_TIMEOUT_SECONDS = 120.0
MAX_EVIDENCE_REGIONS_ENV = "ORBIT2_MCP_TOOLCHAIN_PYTEST_MAX_EVIDENCE_REGIONS"
FALLBACK_MAX_EVIDENCE_REGIONS = 3

_FAILED_RE = re.compile(r"^FAILED\s+(?P<test_id>\S+)(?:\s+-\s+(?P<message>.*))?$")
_PY_FILE_LINE_RE = re.compile(r"^(?P<path>[^\s:][^:\n]*\.py):(?P<line>\d+):")


def _workspace_root() -> Path:
    raw = os.environ.get(WORKSPACE_ROOT_ENV, "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
    elif len(sys.argv) > 1 and sys.argv[-1].strip():
        root = Path(sys.argv[-1]).expanduser().resolve()
    else:
        raise ValueError(
            f"pytest MCP server requires allowed root via {WORKSPACE_ROOT_ENV} "
            "env var or trailing positional arg"
        )
    if not root.exists() or not root.is_dir():
        raise ValueError(f"workspace root is invalid: {root}")
    return root


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _validate_args(args: list[str] | None) -> list[str]:
    cli_args = list(args) if args else []
    for a in cli_args:
        if not isinstance(a, str):
            raise ValueError(f"args entries must be strings; got {type(a).__name__}")
        if a.startswith("/") or a.startswith("\\\\"):
            raise ValueError(f"absolute paths are not allowed in args: {a!r}")
    return cli_args


def _positive_int_env(env_name: str, fallback: int, *, label: str) -> int:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return fallback
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{label} must be > 0")
    return value


def _max_evidence_regions() -> int:
    return _positive_int_env(
        MAX_EVIDENCE_REGIONS_ENV,
        FALLBACK_MAX_EVIDENCE_REGIONS,
        label="pytest toolchain max evidence regions",
    )


def _run_pytest_structured_result(
    *,
    args: list[str] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout_seconds: float = PYTEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    workspace = _workspace_root()
    cli_args = _validate_args(args)
    if not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", *cli_args],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "cwd": str(workspace),
            "args": cli_args,
            "returncode": None,
            "stdout": "",
            "stderr": f"pytest timed out after {timeout_seconds}s",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "failure_kind": "pytest_timeout",
            "timeout_seconds": timeout_seconds,
            "_exc": repr(exc),
        }
    stdout, stdout_truncated = _truncate(completed.stdout or "", max_chars)
    stderr, stderr_truncated = _truncate(completed.stderr or "", max_chars)
    return {
        "ok": completed.returncode == 0,
        "cwd": str(workspace),
        "args": cli_args,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "failure_kind": None if completed.returncode == 0 else "pytest_nonzero_exit",
    }


def _extract_pytest_findings(output: str) -> list[dict[str, Any]]:
    failed: dict[str, dict[str, Any]] = {}
    locations: list[tuple[str, int]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        failed_match = _FAILED_RE.match(line)
        if failed_match is not None:
            test_id = failed_match.group("test_id")
            message = failed_match.group("message") or "pytest failure"
            path = test_id.split("::", 1)[0]
            failed[test_id] = {
                "severity": "error",
                "code": "pytest_failure",
                "message": message,
                "file_path": path if path.endswith(".py") else None,
                "line": None,
                "column": None,
                "test_id": test_id,
                "evidence_refs": [],
            }
            continue

        location_match = _PY_FILE_LINE_RE.match(line)
        if location_match is not None:
            locations.append(
                (location_match.group("path"), int(location_match.group("line")))
            )

    for test_id, finding in failed.items():
        if finding.get("line") is not None:
            continue
        file_path = finding.get("file_path")
        if not file_path:
            continue
        for location_path, line in locations:
            if location_path == file_path or location_path.endswith(f"/{file_path}"):
                finding["line"] = line
                break

    if failed:
        return [ToolchainFinding(**finding).to_dict() for finding in failed.values()]

    if locations:
        path, line = locations[0]
        return [
            ToolchainFinding(
                severity="error",
                code="pytest_failure",
                message="pytest reported a failure location",
                file_path=path,
                line=line,
            ).to_dict()
        ]
    return []


def _collect_failure_evidence(
    findings: list[dict[str, Any]],
    *,
    max_regions: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence_items: list[dict[str, Any]] = []
    evidence_refs: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for finding in findings:
        path = finding.get("file_path")
        line = finding.get("line")
        if not isinstance(path, str) or not isinstance(line, int):
            continue
        key = (path, line)
        if key in seen:
            continue
        seen.add(key)
        if len(evidence_items) >= max_regions:
            break
        start = max(1, line - 3)
        end = line + 3
        try:
            evidence = structured_filesystem._read_file_region_result(
                path=path,
                start_line=start,
                end_line=end,
                evidence_gap={
                    "description": "pytest failure location needs bounded source evidence",
                    "needed_evidence": f"source region around {path}:{line}",
                },
                reason_context_pack_insufficient=(
                    "pytest output identifies a failure location but not enough "
                    "source context for diagnosis"
                ),
                max_chars=4_000,
            )
        except Exception as exc:
            evidence = {
                "ok": False,
                "evidence_type": "file_region",
                "target": {"path": path, "line": line},
                "error": str(exc),
            }
        evidence_id = f"ev_{len(evidence_items) + 1:03d}"
        evidence["evidence_id"] = evidence_id
        target = evidence.get("target", {"path": path, "line": line})
        summary = f"{path}:{line}"
        ref = ToolchainEvidenceRef(
            evidence_id=evidence_id,
            evidence_type=str(evidence.get("evidence_type", "file_region")),
            target=target if isinstance(target, dict) else {"path": path, "line": line},
            summary=summary,
        ).to_dict()
        evidence_refs.append(ref)
        finding.setdefault("evidence_refs", []).append(ref)
        evidence_items.append(evidence)
    return evidence_items, evidence_refs


def _pytest_diagnose_failures_result(
    *,
    args: list[str] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    include_evidence: bool = True,
) -> dict[str, Any]:
    workspace = _workspace_root()
    cli_args = _validate_args(args)
    request = {
        "args": cli_args,
        "max_chars": max_chars,
        "include_evidence": bool(include_evidence),
    }
    recorder = ToolchainRunRecorder(
        toolchain_name="pytest_diagnose_failures",
        workspace_root=workspace,
        request=request,
    )
    try:
        step_started = now_iso()
        pytest_result = _run_pytest_structured_result(args=cli_args, max_chars=max_chars)
        stdout_ref = recorder.save_artifact(
            step_id="step_001_run_pytest",
            artifact_id="artifact_001_stdout",
            name="pytest_stdout",
            content=pytest_result["stdout"],
            metadata={"stdout_truncated_by_pytest_tool": pytest_result["stdout_truncated"]},
        )
        stderr_ref = recorder.save_artifact(
            step_id="step_001_run_pytest",
            artifact_id="artifact_002_stderr",
            name="pytest_stderr",
            content=pytest_result["stderr"],
            metadata={"stderr_truncated_by_pytest_tool": pytest_result["stderr_truncated"]},
        )
        output = "\n".join([pytest_result["stdout"], pytest_result["stderr"]]).strip()
        findings = _extract_pytest_findings(output)
        run_status = "passed" if pytest_result["ok"] else "failed"
        recorder.record_step(
            step_id="step_001_run_pytest",
            name="run pytest",
            kind="process",
            status=run_status,
            started_at=step_started,
            input_summary=f"pytest args={cli_args!r}",
            output_summary=(
                f"returncode={pytest_result['returncode']} "
                f"findings={len(findings)}"
            ),
            artifact_refs=[stdout_ref, stderr_ref],
            error_summary=pytest_result["failure_kind"],
        )

        evidence: list[dict[str, Any]] = []
        evidence_refs: list[dict[str, Any]] = []
        if include_evidence and findings:
            evidence_started = now_iso()
            evidence, evidence_refs = _collect_failure_evidence(
                findings,
                max_regions=_max_evidence_regions(),
            )
            recorder.record_step(
                step_id="step_002_collect_evidence",
                name="collect bounded source evidence",
                kind="structured_primitive",
                status="completed",
                started_at=evidence_started,
                input_summary=f"findings={len(findings)}",
                output_summary=f"evidence_regions={len(evidence)}",
                evidence_refs=evidence_refs,
            )

        summary = (
            "pytest passed"
            if pytest_result["ok"]
            else f"pytest failed with {len(findings)} structured finding(s)"
        )
        report: dict[str, Any] = {
            "ok": pytest_result["ok"],
            "run_id": recorder.run_id,
            "status": run_status,
            "toolchain_name": "pytest_diagnose_failures",
            "cwd": str(workspace),
            "request": request,
            "returncode": pytest_result["returncode"],
            "summary": summary,
            "findings": findings,
            "evidence": evidence,
            "trace": recorder.steps,
            "trace_available": True,
            "artifact_refs": [stdout_ref, stderr_ref],
            "limits": {
                "max_chars": max_chars,
                "max_evidence_regions": _max_evidence_regions(),
            },
            "audit": {
                "capability_layer": "toolchain",
                "provider_control": "provider_opaque_during_execution",
                "provider_audit": "provider_auditable_after_execution",
            },
        }
        recorder.finalize(status=run_status, report=report)
        return report
    except Exception as exc:
        report = {
            "ok": False,
            "run_id": recorder.run_id,
            "status": "error",
            "toolchain_name": "pytest_diagnose_failures",
            "cwd": str(workspace),
            "request": request,
            "summary": f"pytest diagnostic toolchain errored: {exc}",
            "findings": [],
            "evidence": [],
            "trace": recorder.steps,
            "trace_available": True,
            "audit": {
                "capability_layer": "toolchain",
                "provider_control": "provider_opaque_during_execution",
                "provider_audit": "provider_auditable_after_execution",
            },
        }
        recorder.finalize(status="error", report=report)
        return report
    finally:
        recorder.close()


def _toolchain_store() -> SQLiteToolchainRunStore:
    workspace = _workspace_root()
    return SQLiteToolchainRunStore(default_toolchain_db_path(workspace))


mcp = FastMCP(SERVER_NAME)


@mcp.tool()
def run_pytest_structured(
    args: list[str] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    """Run pytest inside the workspace and return bounded stdout/stderr output."""
    return _run_pytest_structured_result(args=args, max_chars=max_chars)


@mcp.tool()
def pytest_diagnose_failures(
    args: list[str] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    include_evidence: bool = True,
) -> dict[str, Any]:
    """Run a fixed pytest diagnostic recipe and persist an auditable L2 run."""
    return _pytest_diagnose_failures_result(
        args=args,
        max_chars=max_chars,
        include_evidence=include_evidence,
    )


@mcp.tool()
def toolchain_get_run(run_id: str) -> dict[str, Any]:
    """Read a persisted L2 toolchain run by run_id."""
    store = _toolchain_store()
    try:
        run = store.get_run(run_id)
        if run is None:
            return {"ok": False, "error": f"toolchain run not found: {run_id}"}
        return {"ok": True, "run": run}
    finally:
        store.close()


@mcp.tool()
def toolchain_get_step(run_id: str, step_id: str) -> dict[str, Any]:
    """Read a persisted L2 toolchain step by run_id and step_id."""
    store = _toolchain_store()
    try:
        step = store.get_step(run_id, step_id)
        if step is None:
            return {"ok": False, "error": f"toolchain step not found: {run_id}/{step_id}"}
        return {"ok": True, "step": step}
    finally:
        store.close()


@mcp.tool()
def toolchain_read_artifact_region(
    run_id: str,
    artifact_id: str,
    start_char: int = 0,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Read a bounded text region from a persisted L2 toolchain artifact."""
    store = _toolchain_store()
    try:
        region = store.read_artifact_region(
            run_id=run_id,
            artifact_id=artifact_id,
            start_char=start_char,
            max_chars=max_chars,
        )
        if region is None:
            return {
                "ok": False,
                "error": f"toolchain artifact not found: {run_id}/{artifact_id}",
            }
        return {"ok": True, "artifact_region": region}
    finally:
        store.close()


if __name__ == "__main__":
    mcp.run()
