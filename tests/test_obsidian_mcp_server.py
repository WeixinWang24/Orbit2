"""Obsidian MCP server tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from src.capability.mcp import McpClientBootstrap, StdioMcpClient
from src.capability.mcp_servers.obsidian import stdio_server as obsidian_server


_OBSIDIAN_MODULE = "src.capability.mcp_servers.obsidian.stdio_server"


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "vault"
    root.mkdir()
    (root / ".obsidian").mkdir()
    (root / ".obsidian" / "workspace.json").write_text("{}", encoding="utf-8")
    (root / "Projects").mkdir()
    (root / "Ideas.md").write_text(
        "---\ntags: [thinking]\ntype: principle\n---\n"
        "# Ideas\nLinks to [[Projects/Launch]] and [[Missing]]. #spark\n",
        encoding="utf-8",
    )
    (root / "Projects" / "Launch.md").write_text(
        "---\ntags:\n  - project\n---\n"
        "# Launch\nLaunch plan. Back to [[Ideas|idea board]]. "
        "See [Local](../Ideas.md) and [External](https://example.com).\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ORBIT_OBSIDIAN_VAULT_ROOT", str(root))
    return root


class TestObsidianHelpers:
    def test_list_notes_excludes_hidden_config(self, vault: Path) -> None:
        result = obsidian_server._list_notes_result(recursive=True)
        paths = {note["path"] for note in result["notes"]}
        assert paths == {"Ideas.md", "Projects/Launch.md"}

    def test_read_note_returns_structured_note(self, vault: Path) -> None:
        result = obsidian_server._read_note_result("Ideas", include_raw_content=True)
        assert result["path"] == "Ideas.md"
        assert result["note_type_hint"] == "principle"
        assert "thinking" in result["tags"]
        assert "spark" in result["tags"]
        assert result["headings"] == [{"level": 1, "text": "Ideas"}]
        assert result["links"][0]["resolved_path"] == "Projects/Launch.md"
        assert result["raw_content"]

    def test_search_notes_scores_and_scopes_results(self, vault: Path) -> None:
        result = obsidian_server._search_notes_result("launch project", max_results=5)
        assert result["matches"][0]["path"] == "Projects/Launch.md"
        assert result["matches"][0]["score_hint"] > 0

    def test_backlinks_resolve_wikilinks_and_markdown_links(self, vault: Path) -> None:
        result = obsidian_server._get_backlinks_result("Ideas.md")
        sources = {item["source_path"] for item in result["backlinks"]}
        assert sources == {"Projects/Launch.md"}

    def test_unresolved_links(self, vault: Path) -> None:
        result = obsidian_server._get_unresolved_links_result()
        unresolved = {(item["source_path"], item["target"]) for item in result["unresolved_links"]}
        assert ("Ideas.md", "Missing") in unresolved

    def test_tag_summary(self, vault: Path) -> None:
        result = obsidian_server._get_tag_summary_result()
        tags = {item["tag"]: set(item["notes"]) for item in result["tags"]}
        assert tags["thinking"] == {"Ideas.md"}
        assert tags["project"] == {"Projects/Launch.md"}

    def test_path_escape_refused(self, vault: Path) -> None:
        with pytest.raises(ValueError, match="escapes vault root"):
            obsidian_server._read_note_result("../outside.md")


@pytest.fixture
def obsidian_bootstrap(vault: Path) -> McpClientBootstrap:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"ORBIT_OBSIDIAN_VAULT_ROOT", "OBSIDIAN_VAULT_PATH"}
    }
    return McpClientBootstrap(
        server_name="obsidian",
        command=sys.executable,
        args=("-m", _OBSIDIAN_MODULE, "--vault", str(vault)),
        env=env,
        transport="stdio",
    )


class TestObsidianMcpIntegration:
    def test_server_lists_reusable_tool_surface(
        self, obsidian_bootstrap: McpClientBootstrap
    ) -> None:
        client = StdioMcpClient(obsidian_bootstrap)
        names = {descriptor.original_name for descriptor in client.list_tools()}
        assert {
            "obsidian_list_notes",
            "obsidian_read_note",
            "obsidian_read_notes",
            "obsidian_search_notes",
            "obsidian_get_note_links",
            "obsidian_get_backlinks",
            "obsidian_get_unresolved_links",
            "obsidian_get_tag_summary",
            "obsidian_get_vault_metadata",
            "obsidian_check_availability",
        }.issubset(names)

    def test_server_call_returns_readable_content(
        self, obsidian_bootstrap: McpClientBootstrap
    ) -> None:
        client = StdioMcpClient(obsidian_bootstrap)
        result = client.call_tool(
            "obsidian_search_notes",
            {"query": "launch", "max_results": 5},
        )
        assert result.ok is True
        assert "Projects/Launch.md" in result.content
