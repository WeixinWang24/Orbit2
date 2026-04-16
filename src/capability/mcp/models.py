from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


_NAME_SEPARATOR = "__"


class InvalidMcpNameError(ValueError):
    """Raised when a server or tool name would corrupt the `mcp__<server>__<tool>`
    namespace because it already contains the double-underscore separator."""


@dataclass(frozen=True)
class McpClientBootstrap:
    """Configuration for attaching one MCP server to Orbit2.

    First-slice posture: stdio transport only. The persistent/daemon/unix-socket
    variants from Orbit1 are explicitly deferred to a later capability-surface
    slice. The `transport` field is typed `Literal["stdio"]` so an out-of-scope
    transport cannot be silently threaded through a custom client factory.

    `server_name` must not contain the `__` separator used by
    `build_orbit_tool_name`; permitting it would let two different servers
    collide on a single registry key (`mcp__svc__foo__bar` is ambiguous
    between `server='svc'`+`tool='foo__bar'` and `server='svc__foo'`+`tool='bar'`).
    """

    server_name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    transport: Literal["stdio"] = "stdio"

    def __post_init__(self) -> None:
        if _NAME_SEPARATOR in self.server_name:
            raise InvalidMcpNameError(
                f"server_name must not contain {_NAME_SEPARATOR!r}; got {self.server_name!r}"
            )
        if not self.server_name:
            raise InvalidMcpNameError("server_name must be non-empty")


@dataclass(frozen=True)
class McpToolDescriptor:
    """One MCP-discovered tool, adapted to Orbit2 naming.

    `orbit_tool_name` is the name the tool is registered under in
    `CapabilityRegistry` — it carries the `mcp__<server>__<tool>` prefix so
    MCP-backed tools never collide with `native__*` tools and the source of
    the capability stays visible in transcripts.
    """

    server_name: str
    original_name: str
    orbit_tool_name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None


def build_orbit_tool_name(server_name: str, original_name: str) -> str:
    if _NAME_SEPARATOR in server_name:
        raise InvalidMcpNameError(
            f"server_name must not contain {_NAME_SEPARATOR!r}; got {server_name!r}"
        )
    if _NAME_SEPARATOR in original_name:
        raise InvalidMcpNameError(
            f"original tool name must not contain {_NAME_SEPARATOR!r}; got {original_name!r}"
        )
    return f"mcp{_NAME_SEPARATOR}{server_name}{_NAME_SEPARATOR}{original_name}"
