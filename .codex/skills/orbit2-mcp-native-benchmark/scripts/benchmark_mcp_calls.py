#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


MUTATION_TOKENS = (
    "write_file",
    "replace_in_file",
    "replace_all_in_file",
    "create_directory",
    "move_file",
    "git_add",
    "git_commit",
    "git_restore",
    "git_unstage",
    "checkout_branch",
    "run_process",
)


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_client_types(repo_root: Path) -> tuple[Any, Any]:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.capability.mcp import McpClientBootstrap, StdioMcpClient

    return McpClientBootstrap, StdioMcpClient


def _payload_from_result(result: Any) -> dict[str, Any]:
    content = getattr(result, "content", "") or ""
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        payload = {"raw_content": content}
    if not isinstance(payload, dict):
        payload = {"raw_content": payload}
    return payload


def _tool_call_timing(payload: dict[str, Any]) -> dict[str, Any] | None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    tool_call = metadata.get("tool_call")
    if not isinstance(tool_call, dict):
        return None
    elapsed_ms = tool_call.get("elapsed_ms")
    elapsed_ns = tool_call.get("elapsed_ns")
    if not isinstance(elapsed_ms, (int, float)) or not isinstance(elapsed_ns, int):
        return None
    return {
        "server_name": tool_call.get("server_name"),
        "tool_name": tool_call.get("tool_name"),
        "elapsed_ms": round(float(elapsed_ms), 3),
        "elapsed_ns": elapsed_ns,
        "clock": tool_call.get("clock"),
    }


def _payload_check(case: str, payload: dict[str, Any]) -> dict[str, Any]:
    if case == "read_file_main_py":
        return {
            "byte_size": payload.get("byte_size"),
            "path_endswith_main_py": str(payload.get("path", "")).endswith("main.py"),
        }
    if case == "read_file_region_main_py_1_5":
        target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
        return {
            "evidence_type": payload.get("evidence_type"),
            "target": target,
            "content_chars": len(str(payload.get("content", ""))),
        }
    if case == "git_changed_files":
        return {
            "branch": payload.get("branch"),
            "staged_count": payload.get("staged_count"),
            "unstaged_count": payload.get("unstaged_count"),
            "untracked_count": payload.get("untracked_count"),
            "total_changed_count": payload.get("total_changed_count"),
        }
    return {"content_chars": len(json.dumps(payload, ensure_ascii=False))}


def _make_client(
    *,
    bootstrap_type: Any,
    client_type: Any,
    server_name: str,
    python_bin: str,
    args: list[str],
    env: dict[str, str],
    timeout_seconds: float,
) -> Any:
    return client_type(
        bootstrap_type(
            server_name=server_name,
            command=python_bin,
            args=tuple(args),
            env=env,
        ),
        timeout_seconds=timeout_seconds,
    )


def _measure_call(
    *,
    bootstrap_type: Any,
    client_type: Any,
    server_name: str,
    python_bin: str,
    args: list[str],
    env: dict[str, str],
    timeout_seconds: float,
    tool: str,
    arguments: dict[str, Any],
    case: str,
) -> dict[str, Any]:
    client = _make_client(
        bootstrap_type=bootstrap_type,
        client_type=client_type,
        server_name=server_name,
        python_bin=python_bin,
        args=args,
        env=env,
        timeout_seconds=timeout_seconds,
    )
    started = time.perf_counter()
    result = client.call_tool(tool, arguments)
    transport_elapsed_ms = (time.perf_counter() - started) * 1000.0
    payload = _payload_from_result(result)
    tool_call_timing = _tool_call_timing(payload)
    return {
        "tool_call_elapsed_ms": (
            tool_call_timing["elapsed_ms"] if tool_call_timing is not None else None
        ),
        "transport_elapsed_ms": round(transport_elapsed_ms, 3),
        "tool_call_timing": tool_call_timing,
        "ok": bool(getattr(result, "ok", False)) and bool(payload.get("ok", True)),
        "content_chars": len(getattr(result, "content", "") or ""),
        "payload_check": _payload_check(case, payload),
    }


def _measure_side(
    *,
    bootstrap_type: Any,
    client_type: Any,
    label: str,
    server_name: str,
    module: str,
    python_bin: str,
    args: list[str],
    env: dict[str, str],
    timeout_seconds: float,
    tool: str,
    arguments: dict[str, Any],
    case: str,
    repeats: int,
) -> dict[str, Any]:
    samples = [
        _measure_call(
            bootstrap_type=bootstrap_type,
            client_type=client_type,
            server_name=server_name,
            python_bin=python_bin,
            args=args,
            env=env,
            timeout_seconds=timeout_seconds,
            tool=tool,
            arguments=arguments,
            case=case,
        )
        for _ in range(repeats)
    ]
    tool_call_elapsed = [
        sample["tool_call_elapsed_ms"]
        for sample in samples
        if sample["tool_call_elapsed_ms"] is not None
    ]
    transport_elapsed = [sample["transport_elapsed_ms"] for sample in samples]
    timing_stats: dict[str, float | None] = {
        "mean_ms": None,
        "min_ms": None,
        "max_ms": None,
    }
    if len(tool_call_elapsed) == len(samples):
        timing_stats = {
            "mean_ms": round(statistics.mean(tool_call_elapsed), 3),
            "min_ms": round(min(tool_call_elapsed), 3),
            "max_ms": round(max(tool_call_elapsed), 3),
        }
    return {
        "label": label,
        "server_name": server_name,
        "module": module,
        "tool": tool,
        "measurement_scope": "server-side MCP tool function execution from metadata.tool_call",
        "samples": samples,
        **timing_stats,
        "transport_mean_ms": round(statistics.mean(transport_elapsed), 3),
        "transport_min_ms": round(min(transport_elapsed), 3),
        "transport_max_ms": round(max(transport_elapsed), 3),
        "ok": all(sample["ok"] for sample in samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Orbit2 external aggregate MCP against family MCP servers."
    )
    parser.add_argument("--workspace", default=str(Path.cwd()))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()

    workspace = str(Path(args.workspace).resolve())
    repo_root = _repo_root_from_script()
    os.environ["ORBIT2_MCP_STDERR_LOG"] = "/dev/null"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    env["ORBIT_WORKSPACE_ROOT"] = workspace
    env["ORBIT2_MCP_STDERR_LOG"] = "/dev/null"

    bootstrap_type, client_type = _load_client_types(repo_root)

    aggregate_module = "src.capability.mcp_servers.orbit2.stdio_server"
    aggregate_args = [
        "-m",
        aggregate_module,
        "--workspace",
        workspace,
        "--profile",
        "read-only",
    ]
    aggregate_client = _make_client(
        bootstrap_type=bootstrap_type,
        client_type=client_type,
        server_name="orbit2",
        python_bin=args.python,
        args=aggregate_args,
        env=env,
        timeout_seconds=args.timeout_seconds,
    )
    started = time.perf_counter()
    tools = aggregate_client.list_tools()
    list_tools_ms = (time.perf_counter() - started) * 1000.0
    names = [tool.original_name for tool in tools]

    cases = [
        {
            "case": "read_file_main_py",
            "arguments": {"path": "main.py"},
            "aggregate": ("orbit2", aggregate_module, aggregate_args, "orbit2_filesystem_read_file"),
            "family": (
                "filesystem",
                "src.capability.mcp_servers.filesystem.stdio_server",
                ["-m", "src.capability.mcp_servers.filesystem.stdio_server", workspace],
                "read_file",
            ),
        },
        {
            "case": "read_file_region_main_py_1_5",
            "arguments": {
                "path": "main.py",
                "start_line": 1,
                "end_line": 5,
                "evidence_gap": {
                    "description": "mcp native benchmark",
                    "needed_evidence": "main.py lines 1-5",
                },
                "reason_context_pack_insufficient": "timing comparison requires exact lines",
            },
            "aggregate": (
                "orbit2",
                aggregate_module,
                aggregate_args,
                "orbit2_structured_filesystem_read_file_region",
            ),
            "family": (
                "structured_filesystem",
                "src.capability.mcp_servers.l1_structured.filesystem.stdio_server",
                ["-m", "src.capability.mcp_servers.l1_structured.filesystem.stdio_server", workspace],
                "read_file_region",
            ),
        },
        {
            "case": "git_changed_files",
            "arguments": {"cwd": "."},
            "aggregate": ("orbit2", aggregate_module, aggregate_args, "orbit2_git_changed_files"),
            "family": (
                "git",
                "src.capability.mcp_servers.git.stdio_server",
                ["-m", "src.capability.mcp_servers.git.stdio_server", workspace],
                "git_changed_files",
            ),
        },
    ]

    results = []
    for case in cases:
        group: dict[str, Any] = {"case": case["case"]}
        for label in ("aggregate", "family"):
            server_name, module, server_args, tool = case[label]
            group[label] = _measure_side(
                bootstrap_type=bootstrap_type,
                client_type=client_type,
                label=label,
                server_name=server_name,
                module=module,
                python_bin=args.python,
                args=server_args,
                env=env,
                timeout_seconds=args.timeout_seconds,
                tool=tool,
                arguments=case["arguments"],
                case=case["case"],
                repeats=args.repeats,
            )
        results.append(group)

    output = {
        "method": "orbit2_mcp_native_benchmark",
        "measurement_scope": "server-side MCP tool function execution from metadata.tool_call",
        "transport_scope": "StdioMcpClient cold-start + MCP handshake + tool call, reported separately as transport_*",
        "workspace": workspace,
        "python": args.python,
        "repeats": args.repeats,
        "list_tools": {
            "server_name": "orbit2",
            "profile": "read-only",
            "tool_count": len(tools),
            "transport_elapsed_ms": round(list_tools_ms, 3),
            "mutation_visible": [
                name for name in names if any(token in name for token in MUTATION_TOKENS)
            ],
        },
        "results": results,
    }
    print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
