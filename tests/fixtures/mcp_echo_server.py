"""Minimal stdio MCP server used as a fixture for Orbit2 MCP attachment tests.

Exposes exactly one trivial tool — `echo` — so both pytest integration tests
and the Handoff 13 CLI verification can exercise a real MCP subprocess round
trip without depending on any third-party server.

Run standalone (MCP stdio transport on stdin/stdout):
    python -m tests.fixtures.mcp_echo_server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo_server")


@mcp.tool()
def echo(text: str) -> str:
    """Return the input text unchanged."""
    return text


if __name__ == "__main__":
    mcp.run()
