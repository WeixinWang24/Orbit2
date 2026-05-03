from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from src.capability.models import ToolResult
from src.capability.mcp.client import McpClient


@dataclass(frozen=True)
class TimedMcpResult:
    label: str
    tool_name: str
    transport_elapsed_ms: float
    tool_call_elapsed_ms: float | None
    result: ToolResult

    @property
    def elapsed_ms(self) -> float:
        return self.transport_elapsed_ms


def _tool_call_elapsed_ms(result: ToolResult) -> float | None:
    try:
        payload = json.loads(result.content)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    tool_call = metadata.get("tool_call")
    if not isinstance(tool_call, dict):
        return None
    value = tool_call.get("elapsed_ms")
    if isinstance(value, int | float):
        return float(value)
    return None


def call_mcp_tool_timed(
    *,
    client: McpClient,
    label: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> TimedMcpResult:
    start = time.perf_counter()
    result = client.call_tool(tool_name, arguments)
    transport_elapsed_ms = (time.perf_counter() - start) * 1000
    return TimedMcpResult(
        label=label,
        tool_name=tool_name,
        transport_elapsed_ms=transport_elapsed_ms,
        tool_call_elapsed_ms=_tool_call_elapsed_ms(result),
        result=result,
    )


def compare_mcp_calls(
    calls: list[tuple[McpClient, str, str, dict[str, Any]]],
) -> list[TimedMcpResult]:
    return [
        call_mcp_tool_timed(
            client=client,
            label=label,
            tool_name=tool_name,
            arguments=arguments,
        )
        for client, label, tool_name, arguments in calls
    ]
