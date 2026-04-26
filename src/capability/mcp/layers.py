from __future__ import annotations

from src.capability.models import CapabilityLayer
from src.capability.mcp.governance import (
    GIT_READ_TOOLS,
    MYPY_READ_TOOLS,
    OBSIDIAN_READ_TOOLS,
    PYTEST_READ_TOOLS,
    RUFF_READ_TOOLS,
    STRUCTURED_FILESYSTEM_READ_TOOLS,
    STRUCTURED_GIT_READ_TOOLS,
)


OBSIDIAN_TOOLCHAIN_TOOLS: frozenset[str] = frozenset({
    "obsidian_list_notes",
    "obsidian_search_notes",
    "obsidian_get_note_links",
    "obsidian_get_backlinks",
    "obsidian_get_unresolved_links",
    "obsidian_get_tag_summary",
    "obsidian_get_vault_metadata",
    "obsidian_check_availability",
})


def classify_mcp_capability_layer(
    *, server_name: str, original_tool_name: str
) -> CapabilityLayer:
    server = server_name.strip().lower()
    tool = original_tool_name.strip().lower()

    if server == "git" and tool in {"git_status", "git_changed_files", "git_log"}:
        return CapabilityLayer.TOOLCHAIN
    if server == "git" and tool in GIT_READ_TOOLS:
        return CapabilityLayer.RAW_PRIMITIVE

    if server == "pytest" and tool in PYTEST_READ_TOOLS:
        return CapabilityLayer.TOOLCHAIN
    if server == "ruff" and tool in RUFF_READ_TOOLS:
        return CapabilityLayer.TOOLCHAIN
    if server == "mypy" and tool in MYPY_READ_TOOLS:
        return CapabilityLayer.TOOLCHAIN

    if server == "structured_filesystem" and tool in STRUCTURED_FILESYSTEM_READ_TOOLS:
        return CapabilityLayer.STRUCTURED_PRIMITIVE

    if server == "structured_git" and tool in STRUCTURED_GIT_READ_TOOLS:
        return CapabilityLayer.STRUCTURED_PRIMITIVE

    if server == "obsidian":
        if tool in {"obsidian_read_note", "obsidian_read_notes"}:
            return CapabilityLayer.RAW_PRIMITIVE
        if tool in OBSIDIAN_READ_TOOLS or tool in OBSIDIAN_TOOLCHAIN_TOOLS:
            return CapabilityLayer.TOOLCHAIN

    return CapabilityLayer.RAW_PRIMITIVE
