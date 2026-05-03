from __future__ import annotations

from dataclasses import dataclass

from src.capability.models import CapabilityLayer


@dataclass(frozen=True)
class McpServerModule:
    server_name: str
    capability_layer: CapabilityLayer
    module_path: str
    legacy_module_path: str


L0_RAW_MCP_SERVER_MODULES: tuple[McpServerModule, ...] = (
    McpServerModule(
        server_name="filesystem",
        capability_layer=CapabilityLayer.RAW_PRIMITIVE,
        module_path="src.capability.mcp_servers.l0_raw.filesystem.stdio_server",
        legacy_module_path="src.capability.mcp_servers.filesystem.stdio_server",
    ),
    McpServerModule(
        server_name="git",
        capability_layer=CapabilityLayer.RAW_PRIMITIVE,
        module_path="src.capability.mcp_servers.l0_raw.git.stdio_server",
        legacy_module_path="src.capability.mcp_servers.git.stdio_server",
    ),
    McpServerModule(
        server_name="process",
        capability_layer=CapabilityLayer.RAW_PRIMITIVE,
        module_path="src.capability.mcp_servers.l0_raw.process.stdio_server",
        legacy_module_path="src.capability.mcp_servers.process.stdio_server",
    ),
)

L1_STRUCTURED_MCP_SERVER_MODULES: tuple[McpServerModule, ...] = (
    McpServerModule(
        server_name="structured_filesystem",
        capability_layer=CapabilityLayer.STRUCTURED_PRIMITIVE,
        module_path="src.capability.mcp_servers.l1_structured.filesystem.stdio_server",
        legacy_module_path="src.capability.mcp_servers.l1_structured.filesystem.stdio_server",
    ),
    McpServerModule(
        server_name="structured_git",
        capability_layer=CapabilityLayer.STRUCTURED_PRIMITIVE,
        module_path="src.capability.mcp_servers.l1_structured.git.stdio_server",
        legacy_module_path="src.capability.mcp_servers.l1_structured.git.stdio_server",
    ),
)

L2_TOOLCHAIN_MCP_SERVER_MODULES: tuple[McpServerModule, ...] = (
    McpServerModule(
        server_name="pytest",
        capability_layer=CapabilityLayer.TOOLCHAIN,
        module_path="src.capability.mcp_servers.l2_toolchain.pytest.stdio_server",
        legacy_module_path="src.capability.mcp_servers.pytest.stdio_server",
    ),
    McpServerModule(
        server_name="ruff",
        capability_layer=CapabilityLayer.TOOLCHAIN,
        module_path="src.capability.mcp_servers.l2_toolchain.ruff.stdio_server",
        legacy_module_path="src.capability.mcp_servers.ruff.stdio_server",
    ),
    McpServerModule(
        server_name="mypy",
        capability_layer=CapabilityLayer.TOOLCHAIN,
        module_path="src.capability.mcp_servers.l2_toolchain.mypy.stdio_server",
        legacy_module_path="src.capability.mcp_servers.mypy.stdio_server",
    ),
    McpServerModule(
        server_name="code_intel",
        capability_layer=CapabilityLayer.TOOLCHAIN,
        module_path="src.capability.mcp_servers.l2_toolchain.code_intel.stdio_server",
        legacy_module_path="src.capability.mcp_servers.l2_toolchain.code_intel.stdio_server",
    ),
    McpServerModule(
        server_name="repo_scout",
        capability_layer=CapabilityLayer.TOOLCHAIN,
        module_path="src.capability.mcp_servers.l2_toolchain.repo_scout.stdio_server",
        legacy_module_path="src.capability.mcp_servers.l2_toolchain.repo_scout.stdio_server",
    ),
)

L3_WORKFLOW_MCP_SERVER_MODULES: tuple[McpServerModule, ...] = (
    McpServerModule(
        server_name="workflow",
        capability_layer=CapabilityLayer.WORKFLOW,
        module_path="src.capability.mcp_servers.l3_workflow.stdio_server",
        legacy_module_path="src.capability.mcp_servers.l3_workflow.stdio_server",
    ),
)

OBSIDIAN_MCP_SERVER_MODULE = McpServerModule(
    server_name="obsidian",
    capability_layer=CapabilityLayer.TOOLCHAIN,
    module_path="src.capability.mcp_servers.l2_toolchain.obsidian.stdio_server",
    legacy_module_path="src.capability.mcp_servers.obsidian.stdio_server",
)

DEFAULT_WORKSPACE_MCP_SERVER_MODULES: tuple[McpServerModule, ...] = (
    *L0_RAW_MCP_SERVER_MODULES,
    *L1_STRUCTURED_MCP_SERVER_MODULES,
    *L2_TOOLCHAIN_MCP_SERVER_MODULES,
    *L3_WORKFLOW_MCP_SERVER_MODULES,
)
