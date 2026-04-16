from __future__ import annotations

from typing import Callable

from src.capability.registry import CapabilityRegistry
from src.capability.mcp.client import McpClient, StdioMcpClient
from src.capability.mcp.models import McpClientBootstrap
from src.capability.mcp.wrapper import McpToolWrapper


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

    Lists tools from the server, wraps each as an `McpToolWrapper`, and
    registers it under the namespaced `mcp__<server>__<tool>` name. Returns
    the live client (so callers may close it) and the list of registered
    tool names.

    `client_factory` lets tests inject a mock client without spawning a real
    subprocess; production callers pass `None` to get a `StdioMcpClient`.

    This is the Orbit2-side capability-closure attachment seam for MCP:
    every attached tool flows through the same `CapabilityRegistry` and
    `CapabilityBoundary` that governs native tools.

    Collision safety: if the constructed `mcp__<server>__<tool>` name is
    already present in the registry, attachment is aborted BEFORE any
    registration mutates the registry. This prevents a malicious or
    misconfigured second server from shadowing an existing native tool or a
    previously-attached MCP server's tool. Raises
    `McpAttachmentCollisionError` with the colliding name(s).
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
        wrapper = McpToolWrapper(descriptor=descriptor, client=client)
        registry.register(wrapper)
        registered.append(descriptor.orbit_tool_name)
    return client, registered
