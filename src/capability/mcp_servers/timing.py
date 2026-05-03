from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def attach_tool_call_timing(
    result: Any,
    *,
    server_name: str,
    tool_name: str,
    elapsed_ns: int,
) -> Any:
    if not isinstance(result, dict):
        return result
    payload = dict(result)
    existing_metadata = payload.get("metadata")
    metadata = dict(existing_metadata) if isinstance(existing_metadata, dict) else {}
    metadata["tool_call"] = {
        "server_name": server_name,
        "tool_name": tool_name,
        "elapsed_ms": round(elapsed_ns / 1_000_000, 3),
        "elapsed_ns": elapsed_ns,
        "clock": "perf_counter_ns",
    }
    payload["metadata"] = metadata
    return payload


def timed_mcp_tool(mcp: Any, server_name: str) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        tool_name = fn.__name__

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started_ns = time.perf_counter_ns()
            result = fn(*args, **kwargs)
            elapsed_ns = time.perf_counter_ns() - started_ns
            return attach_tool_call_timing(
                result,
                server_name=server_name,
                tool_name=tool_name,
                elapsed_ns=elapsed_ns,
            )

        return mcp.tool()(wrapper)  # type: ignore[return-value]

    return decorator
