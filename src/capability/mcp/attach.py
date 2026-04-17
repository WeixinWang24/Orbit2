from __future__ import annotations

from typing import Callable

from src.capability.registry import CapabilityRegistry
from src.capability.mcp.client import McpClient, StdioMcpClient
from src.capability.mcp.governance import resolve_mcp_tool_governance
from src.capability.mcp.models import McpClientBootstrap
from src.capability.mcp.wrapper import FilesystemMcpToolWrapper, McpToolWrapper


ClientFactory = Callable[[McpClientBootstrap], McpClient]


class McpAttachmentCollisionError(RuntimeError):
    """Raised when an MCP tool's Orbit2 name would shadow an already-registered
    capability. Prevents a later-attached server from silently overwriting a
    native tool or a tool from a previously-attached server."""


def attach_mcp_server(
    bootstrap: McpClientBootstrap,
    registry: CapabilityRegistry,
    *,
    client_factory: ClientFactory | None = None,
) -> tuple[McpClient, list[str]]:
    """Attach an MCP server's tools to an Orbit2 `CapabilityRegistry`.

    Lists tools from the server, resolves family-aware governance metadata
    for each (see `src/capability/mcp/governance.py`), wraps each descriptor
    as an `McpToolWrapper` (or a family-specific subclass), and registers
    it under the namespaced `mcp__<server>__<tool>` name. Returns the live
    client plus the list of registered tool names.

    `client_factory` lets tests inject a mock client without spawning a real
    subprocess; production callers pass `None` to get a `StdioMcpClient`.

    Collision safety: if any constructed `mcp__<server>__<tool>` name is
    already present in the registry, attachment is aborted BEFORE any
    registration mutates the registry (all-or-nothing).
    """
    factory: ClientFactory = client_factory or (lambda b: StdioMcpClient(b))
    client = factory(bootstrap)
    descriptors = client.list_tools()

    colliding = [d.orbit_tool_name for d in descriptors if registry.get(d.orbit_tool_name) is not None]
    if colliding:
        raise McpAttachmentCollisionError(
            f"cannot attach MCP server {bootstrap.server_name!r}: "
            f"registry already has tool(s): {sorted(set(colliding))!r}"
        )

    registered: list[str] = []
    for descriptor in descriptors:
        governance = resolve_mcp_tool_governance(
            server_name=descriptor.server_name,
            original_tool_name=descriptor.original_name,
        )
        wrapper = _make_wrapper(descriptor, client, governance, bootstrap)
        registry.register(wrapper)
        registered.append(descriptor.orbit_tool_name)
    return client, registered


def _make_wrapper(
    descriptor,
    client: McpClient,
    governance,
    bootstrap: McpClientBootstrap,
) -> McpToolWrapper:
    if descriptor.server_name.strip().lower() == "filesystem":
        return FilesystemMcpToolWrapper(
            descriptor=descriptor,
            client=client,
            governance=governance,
            bootstrap=bootstrap,
        )
    return McpToolWrapper(descriptor=descriptor, client=client, governance=governance)
