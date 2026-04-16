from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Protocol

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from src.capability.models import ToolResult
from src.capability.mcp.models import (
    McpClientBootstrap,
    McpToolDescriptor,
    build_orbit_tool_name,
)

DEFAULT_MCP_TIMEOUT_SECONDS: float = 15.0


def _default_stderr_log_path() -> Path:
    override = os.environ.get("ORBIT2_MCP_STDERR_LOG")
    if override:
        return Path(override)
    # src/capability/mcp/client.py → parents[3] = repo root
    return Path(__file__).resolve().parents[3] / ".runtime" / "mcp_stderr.log"


class McpClient(Protocol):
    """Minimal client contract the wrapper relies on.

    Tests inject a stub that satisfies this without spawning a subprocess.
    """

    def list_tools(self) -> list[McpToolDescriptor]: ...

    def call_tool(self, original_tool_name: str, arguments: dict[str, Any]) -> ToolResult: ...


class StdioMcpClient:
    """Stateless stdio MCP client.

    Each `list_tools` and `call_tool` spawns a fresh subprocess, runs the MCP
    handshake, issues the request, and tears down. Simple, predictable, and
    sufficient for a first bounded capability-surface attachment slice.

    Persistent-session lifecycle is explicitly deferred; see Handoff 13
    out-of-scope notes.

    Sync-only. Must not be called from inside a running asyncio event loop —
    this class uses `anyio.run` which creates its own loop. Callers that
    already live in async context should run `list_tools` / `call_tool` in a
    worker thread.
    """

    def __init__(
        self,
        bootstrap: McpClientBootstrap,
        *,
        timeout_seconds: float = DEFAULT_MCP_TIMEOUT_SECONDS,
    ) -> None:
        if bootstrap.transport != "stdio":
            raise ValueError(
                f"StdioMcpClient only supports transport='stdio', got {bootstrap.transport!r}"
            )
        self._bootstrap = bootstrap
        self._timeout_seconds = timeout_seconds

    @property
    def bootstrap(self) -> McpClientBootstrap:
        return self._bootstrap

    def list_tools(self) -> list[McpToolDescriptor]:
        _assert_no_running_loop()
        return anyio.run(self._async_list_tools)

    def call_tool(self, original_tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        _assert_no_running_loop()
        return anyio.run(self._async_call_tool, original_tool_name, arguments)

    async def _async_list_tools(self) -> list[McpToolDescriptor]:
        async with _mcp_session(self._bootstrap, self._timeout_seconds) as session:
            result = await session.list_tools()
            return _descriptors_from_list_result(self._bootstrap.server_name, result)

    async def _async_call_tool(
        self, original_tool_name: str, arguments: dict[str, Any]
    ) -> ToolResult:
        async with _mcp_session(self._bootstrap, self._timeout_seconds) as session:
            result = await session.call_tool(original_tool_name, arguments)
            return _tool_result_from_call_result(result)


def _assert_no_running_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        "StdioMcpClient.list_tools/call_tool must not be invoked from inside a "
        "running asyncio event loop. Run them in a worker thread instead "
        "(e.g. asyncio.to_thread or concurrent.futures.ThreadPoolExecutor)."
    )


@asynccontextmanager
async def _mcp_session(
    bootstrap: McpClientBootstrap, timeout_seconds: float
) -> AsyncIterator[ClientSession]:
    """Spin up one stdio-backed MCP session with deterministic teardown order.

    Teardown ordering (innermost-first) is handled by the stacked async-with
    blocks so that the subprocess's stdio handles close before the cancel
    scope does — this avoids the hang-on-timeout failure mode where the
    cancel scope's `__exit__` fires before in-flight `await`s get cancelled.
    """
    stderr_log_path = _default_stderr_log_path()
    try:
        stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
        errlog = stderr_log_path.open("a", encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RuntimeError(
            f"MCP stderr log could not be opened at {stderr_log_path} "
            f"(set ORBIT2_MCP_STDERR_LOG to override): {exc}"
        ) from exc
    try:
        params = StdioServerParameters(
            command=bootstrap.command,
            args=list(bootstrap.args),
            env=dict(bootstrap.env) if bootstrap.env else None,
        )
        with anyio.fail_after(timeout_seconds):
            async with stdio_client(params, errlog=errlog) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session
    finally:
        errlog.close()


def _descriptors_from_list_result(server_name: str, result: Any) -> list[McpToolDescriptor]:
    descriptors: list[McpToolDescriptor] = []
    for item in getattr(result, "tools", []) or []:
        original_name = getattr(item, "name", None)
        if not isinstance(original_name, str) or not original_name:
            continue
        try:
            orbit_tool_name = build_orbit_tool_name(server_name, original_name)
        except Exception:
            # Tool name violates naming constraints (e.g. contains __).
            # Skip it rather than letting one bad tool poison attachment.
            continue
        description = getattr(item, "description", None)
        input_schema = getattr(item, "inputSchema", None)
        descriptors.append(
            McpToolDescriptor(
                server_name=server_name,
                original_name=original_name,
                orbit_tool_name=orbit_tool_name,
                description=description if isinstance(description, str) else None,
                input_schema=input_schema if isinstance(input_schema, dict) else None,
            )
        )
    return descriptors


def _tool_result_from_call_result(result: Any) -> ToolResult:
    """Normalize an MCP CallToolResult into an Orbit2 ToolResult.

    Text content items become the primary `content` string. Non-text content
    (images, blobs, or structured errors) cannot be rendered as text, so we
    preserve the raw representation in `data['raw_result']` and, when no text
    is present, fall back to a JSON dump so a caller always sees *something*
    rather than an opaque empty string — especially important on error
    responses.
    """
    content_items = getattr(result, "content", None) or []
    text_parts: list[str] = []
    non_text_kinds: list[str] = []
    if isinstance(content_items, list):
        for item in content_items:
            item_type = getattr(item, "type", None)
            if item_type == "text":
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
            elif isinstance(item_type, str):
                non_text_kinds.append(item_type)
    output_text = "\n".join(text_parts).strip()
    is_error = bool(getattr(result, "isError", False))
    raw = _safe_model_dump(result, fallback={"content": output_text, "isError": is_error})
    if not output_text:
        if is_error:
            output_text = _safe_json_summary(raw) or "MCP tool reported error with no text content"
        elif non_text_kinds:
            output_text = (
                f"MCP tool returned non-text content: {', '.join(sorted(set(non_text_kinds)))} "
                "(see data['raw_result'])"
            )
    return ToolResult(
        ok=not is_error,
        content=output_text,
        data={"raw_result": raw, "non_text_content_kinds": sorted(set(non_text_kinds))},
    )


def _safe_json_summary(raw: Any) -> str:
    try:
        return json.dumps(raw, ensure_ascii=False)[:500]
    except (TypeError, ValueError):
        return ""


def _safe_model_dump(obj: Any, *, fallback: dict[str, Any]) -> dict[str, Any]:
    """Call `.model_dump(mode='json')` on a pydantic-like object, falling back
    gracefully if the object lacks the method or if serialization raises
    (e.g. non-serializable fields on a malformed MCP response). Guarantees we
    never crash inside the normalization path just because a response is
    weird."""
    if not hasattr(obj, "model_dump"):
        return dict(fallback)
    try:
        dumped = obj.model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 — deliberately broad, never raise out
        fallback = dict(fallback)
        fallback["_model_dump_error"] = repr(exc)
        return fallback
    if isinstance(dumped, dict):
        return dumped
    return {"result": dumped}
