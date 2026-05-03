"""Tests for the Orbit2 family-aware MCP governance overlay (Handoff 16).

Covers:
- resolve_mcp_tool_governance for filesystem read/write and git read/write
- conservative-default fallback for unknown server/tool pairs
- filesystem_server_allowed_root resolution (env, args, absent)
- resolve_filesystem_mcp_target_path resolution (relative, absolute, absent)
- McpToolWrapper uses governance metadata
- FilesystemMcpToolWrapper denies path escapes + protected prefixes
- CapabilityBoundary surfaces tool-layer governance denials via
  governance_outcome
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.capability.boundary import CapabilityBoundary
from src.capability.mcp import (
    DEFAULT_MCP_GOVERNANCE,
    FilesystemMcpToolWrapper,
    McpClientBootstrap,
    McpToolDescriptor,
    McpToolWrapper,
    attach_mcp_server,
    filesystem_server_allowed_root,
    resolve_filesystem_mcp_target_path,
    resolve_mcp_tool_governance,
)
from src.capability.models import ToolResult
from src.capability.registry import CapabilityRegistry
from src.core.runtime.models import ToolRequest


# ---------------------------------------------------------------------------
# Stub MCP client (shared with test_mcp_attachment.py style)
# ---------------------------------------------------------------------------


class _StubMcpClient:
    def __init__(
        self,
        descriptors: list[McpToolDescriptor],
        responses: dict[str, ToolResult] | None = None,
    ) -> None:
        self._descriptors = descriptors
        self._responses = responses or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_tools(self) -> list[McpToolDescriptor]:
        return list(self._descriptors)

    def call_tool(self, original_tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append((original_tool_name, arguments))
        if original_tool_name in self._responses:
            return self._responses[original_tool_name]
        return ToolResult(ok=True, content=f"stub:{original_tool_name}", data={"raw_result": None})


def _fs_desc(name: str) -> McpToolDescriptor:
    return McpToolDescriptor(
        server_name="filesystem",
        original_name=name,
        orbit_tool_name=f"mcp__filesystem__{name}",
        description=f"filesystem {name}",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )


def _git_desc(name: str) -> McpToolDescriptor:
    return McpToolDescriptor(
        server_name="git",
        original_name=name,
        orbit_tool_name=f"mcp__git__{name}",
        description=f"git {name}",
        input_schema={"type": "object", "properties": {}, "additionalProperties": True},
    )


# ---------------------------------------------------------------------------
# Governance overlay
# ---------------------------------------------------------------------------


class TestGovernanceOverlay:
    @pytest.mark.parametrize("tool", ["read_file", "list_directory", "get_file_info"])
    def test_filesystem_read_tools_are_safe(self, tool: str) -> None:
        g = resolve_mcp_tool_governance(server_name="filesystem", original_tool_name=tool)
        assert g["side_effect_class"] == "safe"
        assert g["requires_approval"] is False
        assert g["governance_policy_group"] == "system_environment"

    def test_filesystem_read_tools_path_exists_check(self) -> None:
        g = resolve_mcp_tool_governance(server_name="filesystem", original_tool_name="read_file")
        assert g["environment_check_kind"] == "path_exists"

    @pytest.mark.parametrize("tool", ["write_file", "replace_in_file"])
    def test_filesystem_write_tools_require_approval(self, tool: str) -> None:
        g = resolve_mcp_tool_governance(server_name="filesystem", original_tool_name=tool)
        assert g["side_effect_class"] == "write"
        assert g["requires_approval"] is True
        assert g["governance_policy_group"] == "permission_authority"

    @pytest.mark.parametrize("tool", ["git_status", "git_diff", "git_log"])
    def test_git_read_tools_are_safe(self, tool: str) -> None:
        g = resolve_mcp_tool_governance(server_name="git", original_tool_name=tool)
        assert g["side_effect_class"] == "safe"
        assert g["requires_approval"] is False

    @pytest.mark.parametrize("tool", ["git_add", "git_commit"])
    def test_git_write_tools_require_approval(self, tool: str) -> None:
        g = resolve_mcp_tool_governance(server_name="git", original_tool_name=tool)
        assert g["side_effect_class"] == "write"
        assert g["requires_approval"] is True

    def test_unknown_server_falls_back_to_default(self) -> None:
        g = resolve_mcp_tool_governance(server_name="foosvc", original_tool_name="bar")
        assert g == DEFAULT_MCP_GOVERNANCE

    def test_unknown_filesystem_tool_falls_back_to_default(self) -> None:
        """A filesystem tool not in the recognized set must NOT be silently
        promoted to safe. `glob` was added to the overlay in Handoff 23; this
        test now uses an unrecognized name to exercise the fallback path."""
        g = resolve_mcp_tool_governance(
            server_name="filesystem", original_tool_name="not_a_real_fs_tool",
        )
        assert g == DEFAULT_MCP_GOVERNANCE

    def test_governance_is_case_insensitive_on_names(self) -> None:
        g = resolve_mcp_tool_governance(server_name="Filesystem", original_tool_name="Read_File")
        assert g["side_effect_class"] == "safe"

    def test_bash_family_not_recognized_yet(self) -> None:
        """Handoff 17 bounds: bash remains the only MCP family not covered by
        Orbit2's governance overlay. Future per-family slice will widen this."""
        g = resolve_mcp_tool_governance(server_name="bash", original_tool_name="run_bash")
        assert g == DEFAULT_MCP_GOVERNANCE


class TestProcessFamilyGovernance:
    @pytest.mark.parametrize(
        "tool", ["start_process", "read_process_output", "wait_process", "terminate_process"]
    )
    def test_process_tools_are_write_approval_required(self, tool: str) -> None:
        g = resolve_mcp_tool_governance(server_name="process", original_tool_name=tool)
        assert g["side_effect_class"] == "write"
        assert g["requires_approval"] is True
        assert g["governance_policy_group"] == "permission_authority"

    def test_unknown_process_tool_falls_back(self) -> None:
        g = resolve_mcp_tool_governance(server_name="process", original_tool_name="unknown_process_tool")
        assert g == DEFAULT_MCP_GOVERNANCE


class TestBrowserFamilyGovernance:
    @pytest.mark.parametrize(
        "tool",
        ["browser_open", "browser_snapshot", "browser_console", "browser_screenshot"],
    )
    def test_browser_read_tools_are_safe(self, tool: str) -> None:
        g = resolve_mcp_tool_governance(server_name="browser", original_tool_name=tool)
        assert g["side_effect_class"] == "safe"
        assert g["requires_approval"] is False
        assert g["governance_policy_group"] == "system_environment"

    @pytest.mark.parametrize("tool", ["browser_click", "browser_type"])
    def test_browser_interaction_tools_are_write_approval(self, tool: str) -> None:
        """Audit HIGH-2: click/type mutate remote state (submit forms, trigger
        actions). They must be classified write/approval-required so the
        future Governance-Surface gate catches them."""
        g = resolve_mcp_tool_governance(server_name="browser", original_tool_name=tool)
        assert g["side_effect_class"] == "write"
        assert g["requires_approval"] is True
        assert g["governance_policy_group"] == "permission_authority"

    def test_unknown_browser_tool_falls_back(self) -> None:
        g = resolve_mcp_tool_governance(server_name="browser", original_tool_name="browser_unknown")
        assert g == DEFAULT_MCP_GOVERNANCE


class TestPytestFamilyGovernance:
    def test_run_pytest_structured_is_safe(self) -> None:
        g = resolve_mcp_tool_governance(server_name="pytest", original_tool_name="run_pytest_structured")
        assert g["side_effect_class"] == "safe"
        assert g["requires_approval"] is False
        assert g["governance_policy_group"] == "system_environment"

    def test_unknown_pytest_tool_falls_back(self) -> None:
        g = resolve_mcp_tool_governance(server_name="pytest", original_tool_name="run_pytest_live")
        assert g == DEFAULT_MCP_GOVERNANCE


class TestRepoScoutFamilyGovernance:
    @pytest.mark.parametrize(
        "tool",
        [
            "repo_scout_repository_overview",
            "repo_scout_diff_digest",
            "repo_scout_impact_scope",
            "repo_scout_changed_context",
            "toolchain_get_run",
            "toolchain_get_step",
            "toolchain_read_artifact_region",
        ],
    )
    def test_repo_scout_tools_are_safe(self, tool: str) -> None:
        g = resolve_mcp_tool_governance(server_name="repo_scout", original_tool_name=tool)
        assert g["side_effect_class"] == "safe"
        assert g["requires_approval"] is False
        assert g["governance_policy_group"] == "system_environment"

    def test_unknown_repo_scout_tool_falls_back(self) -> None:
        g = resolve_mcp_tool_governance(
            server_name="repo_scout",
            original_tool_name="repo_scout_mutate",
        )
        assert g == DEFAULT_MCP_GOVERNANCE


class TestCodeIntelFamilyGovernance:
    @pytest.mark.parametrize(
        "tool",
        [
            "code_intel_repository_summary",
            "code_intel_find_symbols",
            "code_intel_file_context",
            "code_intel_export_fragment_summary",
        ],
    )
    def test_code_intel_tools_are_safe(self, tool: str) -> None:
        g = resolve_mcp_tool_governance(server_name="code_intel", original_tool_name=tool)
        assert g["side_effect_class"] == "safe"
        assert g["requires_approval"] is False
        assert g["governance_policy_group"] == "system_environment"

    def test_unknown_code_intel_tool_falls_back(self) -> None:
        g = resolve_mcp_tool_governance(
            server_name="code_intel",
            original_tool_name="code_intel_mutate",
        )
        assert g == DEFAULT_MCP_GOVERNANCE


class TestRuffFamilyGovernance:
    def test_run_ruff_structured_is_safe(self) -> None:
        g = resolve_mcp_tool_governance(server_name="ruff", original_tool_name="run_ruff_structured")
        assert g["side_effect_class"] == "safe"
        assert g["requires_approval"] is False


class TestMypyFamilyGovernance:
    def test_run_mypy_structured_is_safe(self) -> None:
        g = resolve_mcp_tool_governance(server_name="mypy", original_tool_name="run_mypy_structured")
        assert g["side_effect_class"] == "safe"
        assert g["requires_approval"] is False


class TestStructuredFilesystemFamilyGovernance:
    @pytest.mark.parametrize("tool", ["read_file_region", "grep_scoped"])
    def test_structured_filesystem_reads_are_safe_path_exists(self, tool: str) -> None:
        g = resolve_mcp_tool_governance(
            server_name="structured_filesystem",
            original_tool_name=tool,
        )
        assert g["side_effect_class"] == "safe"
        assert g["requires_approval"] is False
        assert g["governance_policy_group"] == "system_environment"
        assert g["environment_check_kind"] == "path_exists"

    def test_unknown_structured_filesystem_tool_falls_back(self) -> None:
        g = resolve_mcp_tool_governance(
            server_name="structured_filesystem",
            original_tool_name="read_file_summary",
        )
        assert g == DEFAULT_MCP_GOVERNANCE


class TestStructuredGitFamilyGovernance:
    @pytest.mark.parametrize("tool", ["read_diff_hunk", "read_git_show_region"])
    def test_structured_git_reads_are_safe(self, tool: str) -> None:
        g = resolve_mcp_tool_governance(
            server_name="structured_git",
            original_tool_name=tool,
        )
        assert g["side_effect_class"] == "safe"
        assert g["requires_approval"] is False
        assert g["governance_policy_group"] == "system_environment"
        assert g["environment_check_kind"] == "none"

    def test_unknown_structured_git_tool_falls_back(self) -> None:
        g = resolve_mcp_tool_governance(
            server_name="structured_git",
            original_tool_name="read_revision_hunk",
        )
        assert g == DEFAULT_MCP_GOVERNANCE


class TestObsidianFamilyGovernance:
    @pytest.mark.parametrize(
        "tool",
        [
            "obsidian_list_notes", "obsidian_read_note", "obsidian_search_notes",
            "obsidian_get_note_links", "obsidian_get_vault_metadata",
            "obsidian_check_availability",
        ],
    )
    def test_obsidian_tools_are_safe(self, tool: str) -> None:
        g = resolve_mcp_tool_governance(server_name="obsidian", original_tool_name=tool)
        assert g["side_effect_class"] == "safe"
        assert g["requires_approval"] is False

    def test_unknown_obsidian_tool_falls_back(self) -> None:
        g = resolve_mcp_tool_governance(server_name="obsidian", original_tool_name="obsidian_mutate_vault")
        assert g == DEFAULT_MCP_GOVERNANCE


# ---------------------------------------------------------------------------
# filesystem_server_allowed_root + target-path resolution
# ---------------------------------------------------------------------------


class TestFilesystemAllowedRoot:
    def test_env_variable_wins(self, tmp_path: Path) -> None:
        root = tmp_path / "env_root"
        root.mkdir()
        result = filesystem_server_allowed_root(
            server_args=["python", "-m", "something", str(tmp_path / "ignored")],
            server_env={"ORBIT_WORKSPACE_ROOT": str(root)},
        )
        assert result == root.resolve()

    def test_falls_back_to_trailing_arg(self, tmp_path: Path) -> None:
        root = tmp_path / "arg_root"
        root.mkdir()
        result = filesystem_server_allowed_root(
            server_args=["python", "-m", "src.capability.mcp_servers.filesystem.stdio_server", str(root)],
            server_env={},
        )
        assert result == root.resolve()

    def test_returns_none_when_no_root(self) -> None:
        assert filesystem_server_allowed_root([], {}) is None
        assert filesystem_server_allowed_root(None, None) is None


class TestFilesystemTargetPath:
    def test_relative_path_joined_to_root(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        (root / "sub").mkdir()
        target = resolve_filesystem_mcp_target_path(
            input_payload={"path": "sub/file.txt"},
            server_args=["python", "-m", "svc", str(root)],
        )
        assert target == (root / "sub" / "file.txt").resolve()

    def test_absolute_path_preserved(self, tmp_path: Path) -> None:
        other = tmp_path / "other" / "foo.txt"
        target = resolve_filesystem_mcp_target_path(
            input_payload={"path": str(other)},
            server_args=["python", "-m", "svc", str(tmp_path)],
        )
        assert target == other.resolve()

    def test_empty_path_becomes_dot(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        target = resolve_filesystem_mcp_target_path(
            input_payload={"path": ""},
            server_args=["svc", str(root)],
        )
        assert target == root.resolve()

    def test_missing_path_returns_none(self, tmp_path: Path) -> None:
        target = resolve_filesystem_mcp_target_path(
            input_payload={},
            server_args=["svc", str(tmp_path)],
        )
        assert target is None

    def test_missing_allowed_root_returns_none(self) -> None:
        target = resolve_filesystem_mcp_target_path(
            input_payload={"path": "anything"},
            server_args=[],
        )
        assert target is None


# ---------------------------------------------------------------------------
# Wrapper governance-metadata propagation
# ---------------------------------------------------------------------------


class TestWrapperGovernanceMetadata:
    def test_wrapper_exposes_provided_governance(self) -> None:
        desc = _fs_desc("read_file")
        g = resolve_mcp_tool_governance(server_name="filesystem", original_tool_name="read_file")
        wrapper = McpToolWrapper(descriptor=desc, client=_StubMcpClient([desc]), governance=g)
        assert wrapper.side_effect_class == "safe"
        assert wrapper.requires_approval is False
        assert wrapper.environment_check_kind == "path_exists"

    def test_wrapper_defaults_to_conservative_when_governance_absent(self) -> None:
        desc = _fs_desc("read_file")
        wrapper = McpToolWrapper(descriptor=desc, client=_StubMcpClient([desc]))
        assert wrapper.side_effect_class == "unknown"
        assert wrapper.requires_approval is True

    def test_filesystem_write_tool_wrapper_has_write_metadata(
        self, tmp_path: Path
    ) -> None:
        desc = _fs_desc("write_file")
        bootstrap = McpClientBootstrap(
            server_name="filesystem",
            command="python",
            args=("-m", "svc", str(tmp_path)),
        )
        g = resolve_mcp_tool_governance(server_name="filesystem", original_tool_name="write_file")
        wrapper = FilesystemMcpToolWrapper(
            descriptor=desc, client=_StubMcpClient([desc]), governance=g, bootstrap=bootstrap,
        )
        assert wrapper.side_effect_class == "write"
        assert wrapper.requires_approval is True


# ---------------------------------------------------------------------------
# Filesystem wrapper — target-path governance
# ---------------------------------------------------------------------------


class TestFilesystemWrapperTargetPathGovernance:
    def _make(
        self, tmp_path: Path, tool: str
    ) -> tuple[FilesystemMcpToolWrapper, _StubMcpClient]:
        desc = _fs_desc(tool)
        stub = _StubMcpClient([desc])
        bootstrap = McpClientBootstrap(
            server_name="filesystem",
            command="python",
            args=("-m", "svc", str(tmp_path)),
        )
        governance = resolve_mcp_tool_governance(
            server_name="filesystem", original_tool_name=tool
        )
        return (
            FilesystemMcpToolWrapper(
                descriptor=desc, client=stub, governance=governance, bootstrap=bootstrap,
            ),
            stub,
        )

    def test_inside_workspace_allowed(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("hi")
        wrapper, stub = self._make(tmp_path, "read_file")
        result = wrapper.execute(path="file.txt")
        assert result.ok is True
        assert stub.calls == [("read_file", {"path": "file.txt"})]

    def test_path_escape_denied_before_call(self, tmp_path: Path) -> None:
        wrapper, stub = self._make(tmp_path, "read_file")
        result = wrapper.execute(path="../../etc/passwd")
        assert result.ok is False
        assert "governance denied" in result.content
        assert stub.calls == [], "MCP server must not be called on governance deny"

    def test_protected_prefix_denied(self, tmp_path: Path) -> None:
        wrapper, stub = self._make(tmp_path, "write_file")
        result = wrapper.execute(path=".env.production", content="SECRET=bad")
        assert result.ok is False
        assert "protected location" in result.content
        assert stub.calls == []

    def test_git_subtree_denied(self, tmp_path: Path) -> None:
        wrapper, stub = self._make(tmp_path, "write_file")
        result = wrapper.execute(path=".git/hooks/pre-commit", content="#!/bin/sh\nexploit")
        assert result.ok is False
        assert "protected location" in result.content
        assert stub.calls == []

    def test_envrc_denied(self, tmp_path: Path) -> None:
        wrapper, stub = self._make(tmp_path, "write_file")
        result = wrapper.execute(path=".envrc", content="export TOKEN=abc")
        assert result.ok is False
        assert stub.calls == []

    def test_no_allowed_root_denies(self, tmp_path: Path) -> None:
        """Filesystem MCP wrapper must refuse when its bootstrap declares no
        allowed root — a misconfigured server must not be usable."""
        desc = _fs_desc("read_file")
        bootstrap = McpClientBootstrap(
            server_name="filesystem",
            command="python",
            args=(),  # no trailing root
        )
        g = resolve_mcp_tool_governance(server_name="filesystem", original_tool_name="read_file")
        stub = _StubMcpClient([desc])
        wrapper = FilesystemMcpToolWrapper(
            descriptor=desc, client=stub, governance=g, bootstrap=bootstrap,
        )
        result = wrapper.execute(path="anything")
        assert result.ok is False
        assert "no resolvable allowed root" in result.content
        assert stub.calls == []


# ---------------------------------------------------------------------------
# Boundary surfaces tool-layer governance denials through governance_outcome
# ---------------------------------------------------------------------------


class TestBoundarySurfacesToolLayerGovernance:
    def test_filesystem_deny_reports_denied_governance_outcome(
        self, tmp_path: Path
    ) -> None:
        registry = CapabilityRegistry()
        bootstrap = McpClientBootstrap(
            server_name="filesystem",
            command="python",
            args=("-m", "svc", str(tmp_path)),
        )
        stub = _StubMcpClient(
            [_fs_desc("read_file")],
            responses={"read_file": ToolResult(ok=True, content="x")},
        )
        attach_mcp_server(bootstrap, registry, client_factory=lambda b: stub)
        boundary = CapabilityBoundary(registry, tmp_path)

        # Denial path — governance refuses
        result = boundary.execute(ToolRequest(
            tool_call_id="c1",
            tool_name="mcp__filesystem__read_file",
            arguments={"path": "../outside"},
        ))
        assert result.ok is False
        assert result.governance_outcome.startswith("denied:")
        assert stub.calls == []

    def test_filesystem_allow_reports_allowed_governance_outcome(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "hi.txt").write_text("hello")
        registry = CapabilityRegistry()
        bootstrap = McpClientBootstrap(
            server_name="filesystem",
            command="python",
            args=("-m", "svc", str(tmp_path)),
        )
        stub = _StubMcpClient(
            [_fs_desc("read_file")],
            responses={"read_file": ToolResult(ok=True, content="hello")},
        )
        attach_mcp_server(bootstrap, registry, client_factory=lambda b: stub)
        boundary = CapabilityBoundary(registry, tmp_path)

        result = boundary.execute(ToolRequest(
            tool_call_id="c1",
            tool_name="mcp__filesystem__read_file",
            arguments={"path": "hi.txt"},
        ))
        assert result.ok is True
        assert result.governance_outcome.startswith("allowed")

    def test_git_mutation_registers_with_write_metadata(
        self, tmp_path: Path
    ) -> None:
        """Git mutation tools remain on the approval-required side of the
        Orbit2 seam even though no approval loop runs yet."""
        registry = CapabilityRegistry()
        bootstrap = McpClientBootstrap(
            server_name="git",
            command="python",
            args=("-m", "svc", str(tmp_path)),
        )
        descriptors = [_git_desc("git_status"), _git_desc("git_commit")]
        stub = _StubMcpClient(descriptors)
        attach_mcp_server(bootstrap, registry, client_factory=lambda b: stub)

        status_wrapper = registry.get("mcp__git__git_status")
        commit_wrapper = registry.get("mcp__git__git_commit")
        assert status_wrapper.requires_approval is False
        assert status_wrapper.side_effect_class == "safe"
        assert commit_wrapper.requires_approval is True
        assert commit_wrapper.side_effect_class == "write"


# ---------------------------------------------------------------------------
# attach_mcp_server picks the filesystem-family wrapper
# ---------------------------------------------------------------------------


class TestGitMcpBoundaryGovernanceOptOut:
    """Audit HIGH-1 fix: git MCP wrappers must opt out of the boundary's
    filesystem-path whitelist because git `path` is a pathspec, not a
    filesystem path relative to workspace."""

    def test_git_wrapper_declares_empty_path_arg_keys(self, tmp_path: Path) -> None:
        desc = _git_desc("git_diff")
        wrapper = McpToolWrapper(descriptor=desc, client=_StubMcpClient([desc]))
        assert wrapper.governance_path_arg_keys == ()

    def test_filesystem_wrapper_owns_its_path_governance(self, tmp_path: Path) -> None:
        """FilesystemMcpToolWrapper opts out of the boundary's filesystem-path
        whitelist because it runs its OWN path-governance check against the
        MCP server's allowed_root (which may differ from the boundary's
        workspace_root). Prevents double-governance with inconsistent roots."""
        desc = _fs_desc("read_file")
        bootstrap = McpClientBootstrap(
            server_name="filesystem", command="python",
            args=("-m", "svc", str(tmp_path)),
        )
        wrapper = FilesystemMcpToolWrapper(
            descriptor=desc, client=_StubMcpClient([desc]),
            governance=None, bootstrap=bootstrap,
        )
        assert wrapper.governance_path_arg_keys == ()

    def test_boundary_skips_path_check_for_git_path_arg(self, tmp_path: Path) -> None:
        """A `path` arg to git_diff whose value coincidentally resembles a
        protected prefix (e.g. `.git/HEAD` diffs) must NOT be refused by the
        boundary — git governs its own pathspec semantics."""
        registry = CapabilityRegistry()
        desc = McpToolDescriptor(
            server_name="git",
            original_name="git_diff",
            orbit_tool_name="mcp__git__git_diff",
            description="git diff",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        )
        stub = _StubMcpClient(
            [desc], responses={"git_diff": ToolResult(ok=True, content="diff-output")},
        )
        bootstrap = McpClientBootstrap(
            server_name="git", command="python", args=("-m", "svc", str(tmp_path)),
        )
        attach_mcp_server(bootstrap, registry, client_factory=lambda b: stub)
        boundary = CapabilityBoundary(registry, tmp_path)
        result = boundary.execute(ToolRequest(
            tool_call_id="c1", tool_name="mcp__git__git_diff",
            arguments={"path": ".git/HEAD"},
        ))
        assert result.ok is True, "git_diff path arg must not be path-governed"
        assert stub.calls == [("git_diff", {"path": ".git/HEAD"})]


class TestNativeToolsRetainDefaultPathGovernance:
    """Regression guard: the governance_path_arg_keys opt-out must NOT leak
    into native tools."""

    def test_native_tool_still_path_governed(self, tmp_path: Path) -> None:
        from src.capability.registry import CapabilityRegistry as R
        from src.capability.tools import ReadFileTool

        registry = R()
        registry.register(ReadFileTool(tmp_path))
        boundary = CapabilityBoundary(registry, tmp_path)
        result = boundary.execute(ToolRequest(
            tool_call_id="c1", tool_name="native__read_file",
            arguments={"path": ".env.production"},
        ))
        assert result.ok is False
        assert "protected location" in result.content


class TestFilesystemWrapperDeniesWhenAllowedRootMissing:
    """Audit HIGH-2 fix: filesystem wrapper must refuse ALL calls when the
    bootstrap has no resolvable allowed root, regardless of whether the
    individual call carries a path arg."""

    def _make_no_root_wrapper(self, tool: str) -> tuple[FilesystemMcpToolWrapper, _StubMcpClient]:
        desc = _fs_desc(tool)
        stub = _StubMcpClient([desc])
        bootstrap = McpClientBootstrap(
            server_name="filesystem", command="python", args=(),
        )
        governance = resolve_mcp_tool_governance(
            server_name="filesystem", original_tool_name=tool
        )
        return (
            FilesystemMcpToolWrapper(
                descriptor=desc, client=stub, governance=governance, bootstrap=bootstrap,
            ),
            stub,
        )

    def test_no_root_no_path_arg_still_denied(self) -> None:
        wrapper, stub = self._make_no_root_wrapper("list_directory")
        result = wrapper.execute()
        assert result.ok is False
        assert "no resolvable allowed root" in result.content
        assert stub.calls == []

    def test_no_root_with_path_arg_denied(self) -> None:
        wrapper, stub = self._make_no_root_wrapper("read_file")
        result = wrapper.execute(path="anything")
        assert result.ok is False
        assert "no resolvable allowed root" in result.content
        assert stub.calls == []


class TestAttachPicksFamilyWrapper:
    def test_filesystem_server_uses_filesystem_wrapper(self, tmp_path: Path) -> None:
        registry = CapabilityRegistry()
        bootstrap = McpClientBootstrap(
            server_name="filesystem",
            command="python",
            args=("-m", "svc", str(tmp_path)),
        )
        stub = _StubMcpClient([_fs_desc("read_file")])
        attach_mcp_server(bootstrap, registry, client_factory=lambda b: stub)
        wrapper = registry.get("mcp__filesystem__read_file")
        assert isinstance(wrapper, FilesystemMcpToolWrapper)

    def test_git_server_uses_generic_wrapper(self, tmp_path: Path) -> None:
        registry = CapabilityRegistry()
        bootstrap = McpClientBootstrap(
            server_name="git",
            command="python",
            args=("-m", "svc", str(tmp_path)),
        )
        stub = _StubMcpClient([_git_desc("git_status")])
        attach_mcp_server(bootstrap, registry, client_factory=lambda b: stub)
        wrapper = registry.get("mcp__git__git_status")
        assert isinstance(wrapper, McpToolWrapper)
        assert not isinstance(wrapper, FilesystemMcpToolWrapper)
