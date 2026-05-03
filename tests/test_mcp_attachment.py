"""Tests for Orbit2 MCP capability attachment (Handoff 13).

Covers:
- McpClientBootstrap / McpToolDescriptor / orbit_tool_name naming
- McpToolWrapper definition + execute delegation + governance metadata
- attach_mcp_server attachment path into CapabilityRegistry
- MCP tools flow through CapabilityBoundary with identical governance
- Integration round trip against a real stdio MCP subprocess (the echo fixture)
- Native filesystem tools remain unaffected after MCP attachment
"""
from __future__ import annotations

import os
import sys
import importlib
from pathlib import Path
from typing import Any

import pytest

from src.capability.boundary import CapabilityBoundary
from src.capability.models import CapabilityLayer, CapabilityResult, ToolDefinition, ToolResult
from src.capability.mcp import (
    McpClientBootstrap,
    McpToolDescriptor,
    McpToolWrapper,
    StdioMcpClient,
    attach_mcp_server,
)
from src.capability.mcp.client import McpClient
from src.capability.mcp.models import build_orbit_tool_name
from src.capability.registry import CapabilityRegistry
from src.capability.tools import ReadFileTool
from src.core.runtime.models import ToolRequest


REPO_ROOT = Path(__file__).resolve().parent.parent


class TestLayeredMcpServerLayout:
    def test_default_workspace_modules_use_layered_entrypoints(self) -> None:
        from src.capability.mcp_servers import DEFAULT_WORKSPACE_MCP_SERVER_MODULES

        modules = {m.server_name: m for m in DEFAULT_WORKSPACE_MCP_SERVER_MODULES}

        assert modules["filesystem"].capability_layer == CapabilityLayer.RAW_PRIMITIVE
        assert modules["filesystem"].module_path.endswith(
            "mcp_servers.l0_raw.filesystem.stdio_server"
        )
        assert modules["structured_filesystem"].capability_layer == CapabilityLayer.STRUCTURED_PRIMITIVE
        assert modules["structured_filesystem"].module_path.endswith(
            "mcp_servers.l1_structured.filesystem.stdio_server"
        )
        assert modules["structured_git"].capability_layer == CapabilityLayer.STRUCTURED_PRIMITIVE
        assert modules["structured_git"].module_path.endswith(
            "mcp_servers.l1_structured.git.stdio_server"
        )
        assert modules["pytest"].capability_layer == CapabilityLayer.TOOLCHAIN
        assert modules["pytest"].module_path.endswith(
            "mcp_servers.l2_toolchain.pytest.stdio_server"
        )
        assert modules["code_intel"].capability_layer == CapabilityLayer.TOOLCHAIN
        assert modules["code_intel"].module_path.endswith(
            "mcp_servers.l2_toolchain.code_intel.stdio_server"
        )
        assert modules["repo_scout"].capability_layer == CapabilityLayer.TOOLCHAIN
        assert modules["repo_scout"].module_path.endswith(
            "mcp_servers.l2_toolchain.repo_scout.stdio_server"
        )
        assert modules["workflow"].capability_layer == CapabilityLayer.WORKFLOW
        assert modules["workflow"].module_path.endswith(
            "mcp_servers.l3_workflow.stdio_server"
        )

    def test_layered_entrypoint_modules_reuse_legacy_mcp_objects(self) -> None:
        from src.capability.mcp_servers import DEFAULT_WORKSPACE_MCP_SERVER_MODULES

        for server_module in DEFAULT_WORKSPACE_MCP_SERVER_MODULES:
            layered = importlib.import_module(server_module.module_path)
            legacy = importlib.import_module(server_module.legacy_module_path)
            assert layered.mcp is legacy.mcp

    def test_obsidian_module_uses_l2_toolchain_entrypoint(self) -> None:
        from src.capability.mcp_servers import OBSIDIAN_MCP_SERVER_MODULE

        assert OBSIDIAN_MCP_SERVER_MODULE.capability_layer == CapabilityLayer.TOOLCHAIN
        assert OBSIDIAN_MCP_SERVER_MODULE.module_path.endswith(
            "mcp_servers.l2_toolchain.obsidian.stdio_server"
        )


# ---------------------------------------------------------------------------
# Stub client used by every non-integration test below
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
        return ToolResult(ok=True, content=f"stub:{original_tool_name}:{arguments}", data={"raw_result": None})


def _make_descriptor(
    server: str = "demo",
    name: str = "greet",
    description: str = "Return a greeting.",
    schema: dict[str, Any] | None = None,
) -> McpToolDescriptor:
    return McpToolDescriptor(
        server_name=server,
        original_name=name,
        orbit_tool_name=build_orbit_tool_name(server, name),
        description=description,
        input_schema=schema
        or {"type": "object", "properties": {"who": {"type": "string"}}, "required": ["who"]},
    )


# ---------------------------------------------------------------------------
# Models + naming
# ---------------------------------------------------------------------------


class TestMcpModels:
    def test_bootstrap_defaults(self) -> None:
        b = McpClientBootstrap(server_name="svc", command="python")
        assert b.transport == "stdio"
        assert b.args == ()
        assert b.env == {}

    def test_bootstrap_is_frozen(self) -> None:
        b = McpClientBootstrap(server_name="svc", command="python")
        with pytest.raises(Exception):
            b.server_name = "other"  # type: ignore[misc]

    def test_orbit_tool_name_namespaced(self) -> None:
        assert build_orbit_tool_name("svc", "greet") == "mcp__svc__greet"
        assert build_orbit_tool_name("echo_server", "echo") == "mcp__echo_server__echo"

    def test_stdio_client_rejects_non_stdio_transport(self) -> None:
        b = McpClientBootstrap(server_name="svc", command="python", transport="unix_socket")
        with pytest.raises(ValueError):
            StdioMcpClient(b)


# ---------------------------------------------------------------------------
# McpToolWrapper
# ---------------------------------------------------------------------------


class TestMcpToolWrapper:
    def test_definition_uses_namespaced_name_and_schema(self) -> None:
        descriptor = _make_descriptor()
        client = _StubMcpClient([descriptor])
        wrapper = McpToolWrapper(descriptor=descriptor, client=client)
        defn = wrapper.definition
        assert isinstance(defn, ToolDefinition)
        assert defn.name == "mcp__demo__greet"
        assert defn.description == "Return a greeting."
        assert defn.parameters["required"] == ["who"]

    def test_definition_falls_back_when_description_absent(self) -> None:
        descriptor = McpToolDescriptor(
            server_name="svc",
            original_name="t",
            orbit_tool_name="mcp__svc__t",
            description=None,
            input_schema=None,
        )
        wrapper = McpToolWrapper(descriptor=descriptor, client=_StubMcpClient([]))
        defn = wrapper.definition
        assert "MCP tool" in defn.description
        assert defn.parameters == {"type": "object", "properties": {}}

    def test_governance_posture(self) -> None:
        wrapper = McpToolWrapper(descriptor=_make_descriptor(), client=_StubMcpClient([]))
        assert wrapper.side_effect_class == "unknown"
        assert wrapper.requires_approval is True
        assert wrapper.environment_check_kind == "none"

    def test_workflow_tool_classifies_as_l3(self) -> None:
        descriptor = _make_descriptor(
            server="workflow",
            name="inspect_change_set_workflow",
        )
        wrapper = McpToolWrapper(
            descriptor=descriptor,
            client=_StubMcpClient([]),
            governance={
                "side_effect_class": "safe",
                "requires_approval": False,
                "governance_policy_group": "system_environment",
                "environment_check_kind": "none",
            },
        )

        assert wrapper.reveal_group == "mcp_workflow"
        assert wrapper.capability_layer == CapabilityLayer.WORKFLOW

    def test_execute_delegates_to_client_with_original_name(self) -> None:
        descriptor = _make_descriptor()
        client = _StubMcpClient(
            [descriptor],
            responses={"greet": ToolResult(ok=True, content="hi Vio", data={"raw_result": {}})},
        )
        wrapper = McpToolWrapper(descriptor=descriptor, client=client)
        result = wrapper.execute(who="Vio")
        assert result.ok is True
        assert result.content == "hi Vio"
        assert client.calls == [("greet", {"who": "Vio"})]


# ---------------------------------------------------------------------------
# attach_mcp_server
# ---------------------------------------------------------------------------


class TestAttachMcpServer:
    def test_registers_each_tool_under_namespaced_name(self) -> None:
        descriptors = [
            _make_descriptor(server="svc", name="alpha"),
            _make_descriptor(server="svc", name="beta"),
        ]
        registry = CapabilityRegistry()
        bootstrap = McpClientBootstrap(server_name="svc", command="python")

        stub = _StubMcpClient(descriptors)
        client, registered = attach_mcp_server(
            bootstrap, registry, client_factory=lambda b: stub
        )

        assert registered == ["mcp__svc__alpha", "mcp__svc__beta"]
        assert client is stub
        assert registry.get("mcp__svc__alpha") is not None
        assert registry.get("mcp__svc__beta") is not None

    def test_attachment_coexists_with_native_tools(self, tmp_path: Path) -> None:
        registry = CapabilityRegistry()
        registry.register(ReadFileTool(tmp_path))
        descriptors = [_make_descriptor(server="svc", name="alpha")]
        bootstrap = McpClientBootstrap(server_name="svc", command="python")
        attach_mcp_server(bootstrap, registry, client_factory=lambda b: _StubMcpClient(descriptors))

        names = set(registry.list_names())
        assert "native__read_file" in names
        assert "mcp__svc__alpha" in names

    def test_attachment_preserves_stub_client_reference(self) -> None:
        descriptor = _make_descriptor(server="svc", name="alpha")
        stub = _StubMcpClient([descriptor])
        registry = CapabilityRegistry()
        bootstrap = McpClientBootstrap(server_name="svc", command="python")
        attach_mcp_server(bootstrap, registry, client_factory=lambda b: stub)

        wrapper = registry.get("mcp__svc__alpha")
        assert isinstance(wrapper, McpToolWrapper)
        assert wrapper._client is stub  # noqa: SLF001


# ---------------------------------------------------------------------------
# CapabilityBoundary routing — MCP flows through the same governed seam
# ---------------------------------------------------------------------------


class TestMcpBoundaryRouting:
    def _setup(
        self,
        tmp_path: Path,
        descriptor: McpToolDescriptor,
        response: ToolResult,
    ) -> tuple[CapabilityBoundary, _StubMcpClient]:
        registry = CapabilityRegistry()
        stub = _StubMcpClient(
            [descriptor], responses={descriptor.original_name: response}
        )
        bootstrap = McpClientBootstrap(server_name=descriptor.server_name, command="python")
        attach_mcp_server(bootstrap, registry, client_factory=lambda b: stub)
        return CapabilityBoundary(registry, tmp_path), stub

    def test_successful_mcp_call_routes_through_boundary(self, tmp_path: Path) -> None:
        descriptor = _make_descriptor()
        response = ToolResult(ok=True, content="hi Vio", data={"raw_result": {}})
        boundary, stub = self._setup(tmp_path, descriptor, response)

        request = ToolRequest(
            tool_call_id="c1",
            tool_name="mcp__demo__greet",
            arguments={"who": "Vio"},
        )
        result = boundary.execute(request)

        assert isinstance(result, CapabilityResult)
        assert result.ok is True
        assert result.content == "hi Vio"
        assert result.governance_outcome.startswith("allowed")
        assert stub.calls == [("greet", {"who": "Vio"})], "MCP original name must be used, not the orbit-namespaced one"

    def test_mcp_tool_error_surfaces_as_not_ok(self, tmp_path: Path) -> None:
        descriptor = _make_descriptor()
        response = ToolResult(ok=False, content="oops", data={"raw_result": {}})
        boundary, _ = self._setup(tmp_path, descriptor, response)
        result = boundary.execute(ToolRequest(
            tool_call_id="c1", tool_name="mcp__demo__greet", arguments={"who": "X"},
        ))
        assert result.ok is False
        assert result.content == "oops"
        # Tool-layer failure still reports governance allowed
        assert result.governance_outcome.startswith("allowed")

    def test_missing_required_argument_rejected_before_mcp_call(self, tmp_path: Path) -> None:
        descriptor = _make_descriptor()
        boundary, stub = self._setup(tmp_path, descriptor, ToolResult(ok=True, content="x"))
        result = boundary.execute(ToolRequest(
            tool_call_id="c1", tool_name="mcp__demo__greet", arguments={},
        ))
        assert result.ok is False
        assert "missing required arguments" in result.content
        assert result.governance_outcome == "denied_invalid_arguments"
        assert stub.calls == [], "MCP server must not be called when validation fails"

    def test_unknown_mcp_tool_rejected(self, tmp_path: Path) -> None:
        descriptor = _make_descriptor()
        boundary, stub = self._setup(tmp_path, descriptor, ToolResult(ok=True, content="x"))
        result = boundary.execute(ToolRequest(
            tool_call_id="c1", tool_name="mcp__demo__does_not_exist", arguments={"who": "x"},
        ))
        assert result.ok is False
        assert result.governance_outcome == "denied_unknown_tool"
        assert stub.calls == []

    def test_mcp_tool_with_literal_path_key_hits_protected_prefix(self, tmp_path: Path) -> None:
        """KNOWN BOUNDED GUARANTEE: the existing governance checks the literal
        argument name `"path"`. An MCP tool that happens to use exactly that
        key does get the protected-prefix denial. An MCP tool that carries a
        filesystem path under a different key (`file`, `target`, etc.) is NOT
        currently governed — see Handoff 13 deferred audit follow-ups. This
        test only asserts the literal-key case."""
        descriptor = McpToolDescriptor(
            server_name="filesvc",
            original_name="read",
            orbit_tool_name="mcp__filesvc__read",
            description="Read a file.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        boundary, stub = self._setup(tmp_path, descriptor, ToolResult(ok=True, content="x"))
        result = boundary.execute(ToolRequest(
            tool_call_id="c1", tool_name="mcp__filesvc__read",
            arguments={"path": ".env.production"},
        ))
        assert result.ok is False
        assert "protected location" in result.content
        assert stub.calls == [], "Governance must deny before invoking the MCP server"


# ---------------------------------------------------------------------------
# Native tools unaffected by MCP subpackage addition
# ---------------------------------------------------------------------------


class TestNativeToolsUnaffectedByMcp:
    def test_native_read_file_still_works_after_mcp_import(self, tmp_path: Path) -> None:
        (tmp_path / "hello.txt").write_text("hello")
        registry = CapabilityRegistry()
        registry.register(ReadFileTool(tmp_path))
        stub = _StubMcpClient([_make_descriptor(server="svc", name="t")])
        bootstrap = McpClientBootstrap(server_name="svc", command="python")
        attach_mcp_server(bootstrap, registry, client_factory=lambda b: stub)

        boundary = CapabilityBoundary(registry, tmp_path)
        result = boundary.execute(ToolRequest(
            tool_call_id="c1", tool_name="native__read_file", arguments={"path": "hello.txt"},
        ))
        assert result.ok is True
        assert result.content == "hello"


# ---------------------------------------------------------------------------
# Integration: real stdio subprocess against the echo_server fixture
# ---------------------------------------------------------------------------


_FIXTURE_MODULE = "tests.fixtures.mcp_echo_server"


@pytest.fixture
def echo_bootstrap() -> McpClientBootstrap:
    # Run the fixture server under the same Python interpreter running tests so
    # the `mcp` package is guaranteed to be available.
    env = {**os.environ}
    return McpClientBootstrap(
        server_name="echo_server",
        command=sys.executable,
        args=("-m", _FIXTURE_MODULE),
        env=env,
        transport="stdio",
    )


class TestStdioMcpClientIntegration:
    """One real stdio round trip. Proves attachment works end-to-end against a
    live MCP subprocess — not just against stubs."""

    def test_list_tools_returns_echo_descriptor(self, echo_bootstrap: McpClientBootstrap) -> None:
        client = StdioMcpClient(echo_bootstrap)
        descriptors = client.list_tools()
        names = {d.original_name for d in descriptors}
        assert "echo" in names
        echo_d = [d for d in descriptors if d.original_name == "echo"][0]
        assert echo_d.orbit_tool_name == "mcp__echo_server__echo"
        assert echo_d.input_schema is not None
        # FastMCP derives schema from the function signature
        assert "text" in echo_d.input_schema.get("properties", {})

    def test_call_tool_echoes_text(self, echo_bootstrap: McpClientBootstrap) -> None:
        client = StdioMcpClient(echo_bootstrap)
        result = client.call_tool("echo", {"text": "hello orbit2"})
        assert result.ok is True
        assert "hello orbit2" in result.content

    def test_attach_and_execute_through_boundary(
        self, echo_bootstrap: McpClientBootstrap, tmp_path: Path
    ) -> None:
        registry = CapabilityRegistry()
        client, registered = attach_mcp_server(echo_bootstrap, registry)
        assert "mcp__echo_server__echo" in registered

        boundary = CapabilityBoundary(registry, tmp_path)
        result = boundary.execute(ToolRequest(
            tool_call_id="call_1",
            tool_name="mcp__echo_server__echo",
            arguments={"text": "orbit2 routes through CapabilityBoundary"},
        ))
        assert result.ok is True
        assert "orbit2 routes through CapabilityBoundary" in result.content
        assert result.governance_outcome.startswith("allowed")


# ---------------------------------------------------------------------------
# Audit-driven hardening tests
# ---------------------------------------------------------------------------


class TestMcpNamespaceCollisionGuards:
    """HIGH-2 from audit: the `mcp__<server>__<tool>` namespace must be safe
    against server names or tool names that contain the separator."""

    def test_bootstrap_rejects_double_underscore_server_name(self) -> None:
        from src.capability.mcp.models import InvalidMcpNameError
        with pytest.raises(InvalidMcpNameError):
            McpClientBootstrap(server_name="svc__foo", command="python")

    def test_bootstrap_rejects_empty_server_name(self) -> None:
        from src.capability.mcp.models import InvalidMcpNameError
        with pytest.raises(InvalidMcpNameError):
            McpClientBootstrap(server_name="", command="python")

    def test_build_orbit_tool_name_rejects_separator_in_original(self) -> None:
        from src.capability.mcp.models import InvalidMcpNameError
        with pytest.raises(InvalidMcpNameError):
            build_orbit_tool_name("svc", "foo__bar")

    def test_attachment_aborts_on_collision_with_existing_tool(
        self, tmp_path: Path
    ) -> None:
        from src.capability.mcp.attach import McpAttachmentCollisionError
        registry = CapabilityRegistry()
        registry.register(ReadFileTool(tmp_path))
        colliding_descriptor = McpToolDescriptor(
            server_name="svc",
            original_name="x",
            orbit_tool_name="native__read_file",  # deliberate collision
            description=None,
            input_schema=None,
        )
        bootstrap = McpClientBootstrap(server_name="svc", command="python")
        stub = _StubMcpClient([colliding_descriptor])
        with pytest.raises(McpAttachmentCollisionError):
            attach_mcp_server(bootstrap, registry, client_factory=lambda b: stub)

    def test_collision_aborts_before_registering_any_tool(self, tmp_path: Path) -> None:
        """If one tool in a multi-tool server collides, no tools from that
        server may be registered (all-or-nothing)."""
        from src.capability.mcp.attach import McpAttachmentCollisionError
        registry = CapabilityRegistry()
        registry.register(ReadFileTool(tmp_path))
        descriptors = [
            _make_descriptor(server="svc", name="alpha"),
            McpToolDescriptor(
                server_name="svc",
                original_name="collide",
                orbit_tool_name="native__read_file",  # collision
                description=None,
                input_schema=None,
            ),
        ]
        bootstrap = McpClientBootstrap(server_name="svc", command="python")
        with pytest.raises(McpAttachmentCollisionError):
            attach_mcp_server(
                bootstrap, registry, client_factory=lambda b: _StubMcpClient(descriptors)
            )
        # alpha must NOT have been registered
        assert registry.get("mcp__svc__alpha") is None


class TestMcpSchemaValidationEdgeCases:
    """MED-1 from audit: schemas that lack `properties` must only enforce
    `required` (not flag every provided arg as unexpected)."""

    def test_schema_without_properties_but_additional_true_accepts_args(
        self, tmp_path: Path
    ) -> None:
        descriptor = McpToolDescriptor(
            server_name="svc",
            original_name="t",
            orbit_tool_name="mcp__svc__t",
            description="loose schema",
            input_schema={"type": "object", "additionalProperties": True},
        )
        stub = _StubMcpClient(
            [descriptor],
            responses={"t": ToolResult(ok=True, content="ok", data={"raw_result": {}})},
        )
        registry = CapabilityRegistry()
        bootstrap = McpClientBootstrap(server_name="svc", command="python")
        attach_mcp_server(bootstrap, registry, client_factory=lambda b: stub)
        boundary = CapabilityBoundary(registry, tmp_path)
        result = boundary.execute(ToolRequest(
            tool_call_id="c1", tool_name="mcp__svc__t",
            arguments={"anything": "at all", "and_also": 42},
        ))
        assert result.ok is True
        assert stub.calls == [("t", {"anything": "at all", "and_also": 42})]

    def test_schema_without_properties_still_enforces_required(
        self, tmp_path: Path
    ) -> None:
        descriptor = McpToolDescriptor(
            server_name="svc",
            original_name="t",
            orbit_tool_name="mcp__svc__t",
            description="loose schema with required",
            input_schema={"type": "object", "required": ["must_have"]},
        )
        stub = _StubMcpClient([descriptor])
        registry = CapabilityRegistry()
        bootstrap = McpClientBootstrap(server_name="svc", command="python")
        attach_mcp_server(bootstrap, registry, client_factory=lambda b: stub)
        boundary = CapabilityBoundary(registry, tmp_path)
        result = boundary.execute(ToolRequest(
            tool_call_id="c1", tool_name="mcp__svc__t", arguments={},
        ))
        assert result.ok is False
        assert "missing required arguments" in result.content
        assert result.governance_outcome == "denied_invalid_arguments"


class TestMcpErrorSurface:
    """HIGH-1 from audit: empty-text error responses and non-text content
    must surface *something* to the caller rather than an opaque blank."""

    def test_call_result_error_with_no_text_produces_raw_summary(self) -> None:
        from src.capability.mcp.client import _tool_result_from_call_result

        class _Item:
            type = "structured"

        class _Result:
            isError = True
            content = [_Item()]

            def model_dump(self, *, mode: str = "json") -> dict:
                return {"isError": True, "detail": "something went wrong"}

        result = _tool_result_from_call_result(_Result())
        assert result.ok is False
        assert result.content != ""
        assert "something went wrong" in result.content or "structured" in result.content

    def test_non_text_content_kinds_surfaced_when_no_text(self) -> None:
        from src.capability.mcp.client import _tool_result_from_call_result

        class _Item:
            def __init__(self, type_: str) -> None:
                self.type = type_

        class _Result:
            isError = False
            content = [_Item("image"), _Item("image"), _Item("resource")]

        result = _tool_result_from_call_result(_Result())
        assert result.ok is True
        assert "non-text content" in result.content
        assert "image" in result.content
        assert result.data["non_text_content_kinds"] == ["image", "resource"]

    def test_pure_text_result_still_renders_cleanly(self) -> None:
        from src.capability.mcp.client import _tool_result_from_call_result

        class _Item:
            type = "text"
            text = "hello"

        class _Result:
            isError = False
            content = [_Item()]

        result = _tool_result_from_call_result(_Result())
        assert result.ok is True
        assert result.content == "hello"


class TestMcpTransportTyping:
    """LOW-2 from audit: transport field is typed Literal['stdio'] so the
    frozen dataclass carries type-level evidence of the bounded scope."""

    def test_stdio_client_rejects_typing_escape_via_runtime_mutation(self) -> None:
        # Constructing with a non-stdio value — type checker catches at build
        # time; runtime guard in StdioMcpClient catches anything that slips
        # through (e.g. `typing.cast`).
        b = McpClientBootstrap(server_name="svc", command="python")  # default stdio
        # Simulate a cast/mutation attempt via object.__setattr__ (frozen dataclass)
        with pytest.raises(Exception):
            object.__setattr__(b, "transport", "unix_socket")
            StdioMcpClient(b)


class TestMcpPathArgNameWhitelist:
    """HIGH-3 narrowing: governance now checks a small whitelist of
    explicitly-path-bearing argument names (path, file, filepath, ...). Names
    whose meaning is path-unambiguous are covered; ambiguous names like
    `target` or `location` are not, and remain a deferred Governance-Surface
    gap documented in the Handoff 13 return log."""

    @pytest.mark.parametrize("key", ["file", "file_path", "filepath", "filename", "source_path", "target_path", "src_path", "dst_path"])
    def test_protected_prefix_denied_across_alt_path_keys(
        self, tmp_path: Path, key: str
    ) -> None:
        schema = {"type": "object", "properties": {key: {"type": "string"}}, "required": [key]}
        descriptor = McpToolDescriptor(
            server_name="svc",
            original_name="read",
            orbit_tool_name="mcp__svc__read",
            description="",
            input_schema=schema,
        )
        stub = _StubMcpClient([descriptor])
        registry = CapabilityRegistry()
        bootstrap = McpClientBootstrap(server_name="svc", command="python")
        attach_mcp_server(bootstrap, registry, client_factory=lambda b: stub)
        boundary = CapabilityBoundary(registry, tmp_path)
        result = boundary.execute(ToolRequest(
            tool_call_id="c1", tool_name="mcp__svc__read",
            arguments={key: ".env.production"},
        ))
        assert result.ok is False, f"key={key!r} must be governed"
        assert "protected location" in result.content
        assert stub.calls == []

    @pytest.mark.parametrize("key", ["file", "file_path", "filepath", "filename"])
    def test_workspace_escape_denied_across_alt_path_keys(
        self, tmp_path: Path, key: str
    ) -> None:
        schema = {"type": "object", "properties": {key: {"type": "string"}}, "required": [key]}
        descriptor = McpToolDescriptor(
            server_name="svc",
            original_name="read",
            orbit_tool_name="mcp__svc__read",
            description="",
            input_schema=schema,
        )
        stub = _StubMcpClient([descriptor])
        registry = CapabilityRegistry()
        bootstrap = McpClientBootstrap(server_name="svc", command="python")
        attach_mcp_server(bootstrap, registry, client_factory=lambda b: stub)
        boundary = CapabilityBoundary(registry, tmp_path)
        result = boundary.execute(ToolRequest(
            tool_call_id="c1", tool_name="mcp__svc__read",
            arguments={key: "../../etc/passwd"},
        ))
        assert result.ok is False
        assert "escapes workspace" in result.content

    def test_non_path_argument_names_not_governed_by_path_check(
        self, tmp_path: Path
    ) -> None:
        """Explicit test of the known gap: arg names outside the whitelist
        (e.g. `target`, `location`, `query`) are NOT path-governed. This is a
        correctness guarantee about what the bounded slice claims — future
        governance-surface work must widen this."""
        schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        descriptor = McpToolDescriptor(
            server_name="svc",
            original_name="lookup",
            orbit_tool_name="mcp__svc__lookup",
            description="",
            input_schema=schema,
        )
        stub = _StubMcpClient(
            [descriptor],
            responses={"lookup": ToolResult(ok=True, content="x")},
        )
        registry = CapabilityRegistry()
        bootstrap = McpClientBootstrap(server_name="svc", command="python")
        attach_mcp_server(bootstrap, registry, client_factory=lambda b: stub)
        boundary = CapabilityBoundary(registry, tmp_path)
        result = boundary.execute(ToolRequest(
            tool_call_id="c1", tool_name="mcp__svc__lookup",
            arguments={"query": ".env.production"},
        ))
        # query is NOT path-governed — tool is invoked. Governance surface
        # expansion is an explicit deferred follow-up.
        assert result.ok is True
        assert stub.calls == [("lookup", {"query": ".env.production"})]


class TestMcpResultRobustness:
    """Audit new-issue-2: `model_dump` raising must not crash result normalization."""

    def test_model_dump_exception_falls_back_safely(self) -> None:
        from src.capability.mcp.client import _tool_result_from_call_result

        class _Item:
            type = "text"
            text = "hello"

        class _Result:
            isError = False
            content = [_Item()]

            def model_dump(self, *, mode: str = "json") -> dict:
                raise RuntimeError("serialization exploded")

        result = _tool_result_from_call_result(_Result())
        assert result.ok is True
        assert result.content == "hello"
        assert "_model_dump_error" in result.data["raw_result"]
        assert "serialization exploded" in result.data["raw_result"]["_model_dump_error"]


class TestMcpStderrLogErrorContext:
    """Audit new-issue-1: errlog failure must carry MCP context in the error
    message rather than a bare OSError."""

    def test_unwritable_stderr_log_raises_with_mcp_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os as _os

        # Point the stderr log at a path whose parent exists as a FILE, so
        # parent.mkdir fails with a recognizable OSError.
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("")
        monkeypatch.setenv("ORBIT2_MCP_STDERR_LOG", str(blocker / "inside" / "log.txt"))

        # Use a stub client to avoid needing a real MCP server — we only want
        # _mcp_session to try opening the stderr log.
        from src.capability.mcp.client import _mcp_session
        bootstrap = McpClientBootstrap(server_name="svc", command="python")

        async def _drive() -> None:
            async with _mcp_session(bootstrap, timeout_seconds=5.0):
                pass

        import anyio
        with pytest.raises(RuntimeError, match="MCP stderr log"):
            anyio.run(_drive)


class TestMcpRunningLoopGuard:
    """MED-2 from audit: calling StdioMcpClient.list_tools/call_tool from
    inside an async context must raise a clear error, not an opaque one."""

    def test_call_from_running_loop_raises_clear_error(self) -> None:
        import asyncio

        async def offender() -> None:
            client = StdioMcpClient(
                McpClientBootstrap(server_name="svc", command="python")
            )
            client.list_tools()

        with pytest.raises(RuntimeError, match="worker thread"):
            asyncio.run(offender())
