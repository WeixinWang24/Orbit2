"""Tests for the Orbit2 Capability Surface — first governed closure path.

Tests cover:
- Tool execution (ReadFileTool)
- Registry attachment boundary
- Boundary governance (workspace escape, unknown tool)
- Full governed closure path
- SessionManager integration with tool call loop
- Transcript/context separation with tool calls
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

import pytest

from src.capability.boundary import CapabilityBoundary
from src.capability.models import CapabilityResult, GovernanceOutcome, ToolDefinition, ToolResult
from src.capability.registry import CapabilityRegistry
from src.capability.tools import ReadFileTool, Tool
from src.knowledge.assembly import TranscriptContextAssembler
from src.providers.base import ExecutionBackend
from src.runtime.models import (
    ConversationMessage,
    ExecutionPlan,
    Message,
    MessageRole,
    ToolRequest,
    TurnRequest,
)
from src.runtime.session import CapabilityBoundaryUnavailableError, SessionManager
from src.store.sqlite import SQLiteSessionStore


# ---------------------------------------------------------------------------
# ReadFileTool tests
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "hello.txt").write_text("hello world")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.txt").write_text("nested content")
    return tmp_path


class TestReadFileTool:
    def test_read_existing_file(self, workspace: Path) -> None:
        tool = ReadFileTool(workspace)
        result = tool.execute(path="hello.txt")
        assert result.ok is True
        assert result.content == "hello world"
        assert result.data is not None
        assert result.data["path"] == str(workspace / "hello.txt")

    def test_read_nested_file(self, workspace: Path) -> None:
        tool = ReadFileTool(workspace)
        result = tool.execute(path="subdir/nested.txt")
        assert result.ok is True
        assert result.content == "nested content"

    def test_file_not_found(self, workspace: Path) -> None:
        tool = ReadFileTool(workspace)
        result = tool.execute(path="nonexistent.txt")
        assert result.ok is False
        assert "file not found" in result.content

    def test_path_escape_blocked(self, workspace: Path) -> None:
        tool = ReadFileTool(workspace)
        result = tool.execute(path="../../etc/passwd")
        assert result.ok is False
        assert "path escapes workspace" in result.content

    def test_definition(self, workspace: Path) -> None:
        tool = ReadFileTool(workspace)
        defn = tool.definition
        assert defn.name == "native__read_file"
        assert "path" in defn.parameters["properties"]
        assert defn.parameters["required"] == ["path"]

    def test_governance_attributes(self, workspace: Path) -> None:
        tool = ReadFileTool(workspace)
        assert tool.side_effect_class == "safe"
        assert tool.requires_approval is False
        assert tool.environment_check_kind == "path_exists"


# ---------------------------------------------------------------------------
# CapabilityRegistry tests
# ---------------------------------------------------------------------------


class TestCapabilityRegistry:
    def test_register_and_get(self, workspace: Path) -> None:
        registry = CapabilityRegistry()
        tool = ReadFileTool(workspace)
        registry.register(tool)
        assert registry.get("native__read_file") is tool

    def test_get_unknown_returns_none(self) -> None:
        registry = CapabilityRegistry()
        assert registry.get("nonexistent") is None

    def test_list_names(self, workspace: Path) -> None:
        registry = CapabilityRegistry()
        registry.register(ReadFileTool(workspace))
        assert registry.list_names() == ["native__read_file"]

    def test_list_definitions(self, workspace: Path) -> None:
        registry = CapabilityRegistry()
        registry.register(ReadFileTool(workspace))
        defs = registry.list_definitions()
        assert len(defs) == 1
        assert defs[0].name == "native__read_file"


# ---------------------------------------------------------------------------
# CapabilityBoundary governance tests
# ---------------------------------------------------------------------------


class TestCapabilityBoundary:
    def test_governed_execution_success(self, workspace: Path) -> None:
        registry = CapabilityRegistry()
        registry.register(ReadFileTool(workspace))
        boundary = CapabilityBoundary(registry, workspace)

        request = ToolRequest(
            tool_call_id="call_1",
            tool_name="native__read_file",
            arguments={"path": "hello.txt"},
        )
        result = boundary.execute(request)
        assert result.ok is True
        assert result.content == "hello world"
        assert result.tool_call_id == "call_1"
        assert result.tool_name == "native__read_file"
        assert result.governance_outcome == "allowed"

    def test_governance_denies_path_escape(self, workspace: Path) -> None:
        registry = CapabilityRegistry()
        registry.register(ReadFileTool(workspace))
        boundary = CapabilityBoundary(registry, workspace)

        request = ToolRequest(
            tool_call_id="call_2",
            tool_name="native__read_file",
            arguments={"path": "../../etc/passwd"},
        )
        result = boundary.execute(request)
        assert result.ok is False
        assert "governance denied" in result.content
        assert "denied" in result.governance_outcome

    def test_unknown_tool_denied(self, workspace: Path) -> None:
        registry = CapabilityRegistry()
        boundary = CapabilityBoundary(registry, workspace)

        request = ToolRequest(
            tool_call_id="call_3",
            tool_name="nonexistent_tool",
            arguments={},
        )
        result = boundary.execute(request)
        assert result.ok is False
        assert "unknown tool" in result.content
        assert result.governance_outcome == "denied_unknown_tool"

    def test_file_not_found_passes_governance_but_fails_execution(
        self, workspace: Path
    ) -> None:
        """Path within workspace passes governance; tool returns not-found."""
        registry = CapabilityRegistry()
        registry.register(ReadFileTool(workspace))
        boundary = CapabilityBoundary(registry, workspace)

        request = ToolRequest(
            tool_call_id="call_4",
            tool_name="native__read_file",
            arguments={"path": "missing.txt"},
        )
        result = boundary.execute(request)
        assert result.ok is False
        assert "file not found" in result.content
        assert result.governance_outcome == "allowed"


# ---------------------------------------------------------------------------
# SessionManager integration — tool call loop
# ---------------------------------------------------------------------------


class ToolCallBackend(ExecutionBackend):
    """Fake backend that returns one tool call, then a final text response."""

    def __init__(self, tool_calls_sequence: list[list[ToolRequest]]) -> None:
        self._sequence = list(tool_calls_sequence)
        self._call_count = 0

    @property
    def backend_name(self) -> str:
        return "tool-call-fake"

    def plan_from_messages(
        self,
        request: TurnRequest,
        *,
        on_partial_text: Callable[[str], None] | None = None,
    ) -> ExecutionPlan:
        if self._call_count < len(self._sequence):
            tool_requests = self._sequence[self._call_count]
            self._call_count += 1
            return ExecutionPlan(
                source_backend=self.backend_name,
                plan_label="tool-call-fake-tool-calls",
                final_text=None,
                model="fake",
                tool_requests=tool_requests,
            )
        self._call_count += 1
        return ExecutionPlan(
            source_backend=self.backend_name,
            plan_label="tool-call-fake-final-text",
            final_text="I read the file for you.",
            model="fake",
        )


class TestSessionManagerToolLoop:
    def _make_manager(
        self, workspace: Path, tool_calls_sequence: list[list[ToolRequest]]
    ) -> tuple[SessionManager, SQLiteSessionStore]:
        store = SQLiteSessionStore(db_path=":memory:")
        backend = ToolCallBackend(tool_calls_sequence)

        registry = CapabilityRegistry()
        registry.register(ReadFileTool(workspace))
        boundary = CapabilityBoundary(registry, workspace)

        manager = SessionManager(
            backend=backend,
            store=store,
            capability_boundary=boundary,
        )
        return manager, store

    def test_single_tool_call_loop(self, workspace: Path) -> None:
        tool_requests = [
            [ToolRequest(tool_call_id="call_1", tool_name="native__read_file", arguments={"path": "hello.txt"})]
        ]
        manager, store = self._make_manager(workspace, tool_requests)
        session = manager.create_session()

        plan = manager.run_turn(session.session_id, "read hello.txt")
        assert plan.final_text == "I read the file for you."
        assert plan.tool_requests == []

        messages = store.list_messages(session.session_id)
        roles = [m.role for m in messages]
        assert roles == [
            MessageRole.USER,
            MessageRole.ASSISTANT,  # tool call
            MessageRole.TOOL,       # tool result
            MessageRole.ASSISTANT,  # final text
        ]

        # Verify tool call metadata on assistant message
        tool_call_msg = messages[1]
        assert "tool_calls" in tool_call_msg.metadata
        assert tool_call_msg.metadata["tool_calls"][0]["tool_name"] == "native__read_file"

        # Verify tool result metadata
        tool_result_msg = messages[2]
        assert tool_result_msg.metadata["tool_call_id"] == "call_1"
        assert tool_result_msg.metadata["tool_name"] == "native__read_file"
        assert tool_result_msg.metadata["ok"] is True
        assert tool_result_msg.metadata["governance_outcome"] == "allowed"
        assert tool_result_msg.content == "hello world"

    def test_tool_failure_recorded_in_transcript(self, workspace: Path) -> None:
        tool_requests = [
            [ToolRequest(tool_call_id="call_1", tool_name="native__read_file", arguments={"path": "missing.txt"})]
        ]
        manager, store = self._make_manager(workspace, tool_requests)
        session = manager.create_session()

        plan = manager.run_turn(session.session_id, "read missing.txt")
        messages = store.list_messages(session.session_id)

        tool_result_msg = messages[2]
        assert tool_result_msg.role == MessageRole.TOOL
        assert tool_result_msg.metadata["ok"] is False
        assert "file not found" in tool_result_msg.content

    def test_governance_denial_recorded_in_transcript(self, workspace: Path) -> None:
        tool_requests = [
            [ToolRequest(tool_call_id="call_1", tool_name="native__read_file", arguments={"path": "../../etc/passwd"})]
        ]
        manager, store = self._make_manager(workspace, tool_requests)
        session = manager.create_session()

        plan = manager.run_turn(session.session_id, "read /etc/passwd")
        messages = store.list_messages(session.session_id)

        tool_result_msg = messages[2]
        assert tool_result_msg.role == MessageRole.TOOL
        assert "governance denied" in tool_result_msg.content

    def test_no_capability_boundary_works_as_before(self, tmp_path: Path) -> None:
        """Without capability boundary, text-only turns work unchanged."""
        store = SQLiteSessionStore(db_path=":memory:")
        backend = ToolCallBackend([])  # No tool calls
        manager = SessionManager(backend=backend, store=store)
        session = manager.create_session()

        plan = manager.run_turn(session.session_id, "hello")
        assert plan.final_text == "I read the file for you."

        messages = store.list_messages(session.session_id)
        roles = [m.role for m in messages]
        assert roles == [MessageRole.USER, MessageRole.ASSISTANT]

    def test_tool_definitions_sent_to_provider(self, workspace: Path) -> None:
        """When capability boundary is configured, tool definitions appear in TurnRequest."""

        class RecordingBackend(ExecutionBackend):
            def __init__(self):
                self.last_request: TurnRequest | None = None

            @property
            def backend_name(self) -> str:
                return "recording"

            def plan_from_messages(self, request, *, on_partial_text=None):
                self.last_request = request
                return ExecutionPlan(
                    source_backend="recording",
                    plan_label="recording-final",
                    final_text="ok",
                    model="fake",
                )

        store = SQLiteSessionStore(db_path=":memory:")
        backend = RecordingBackend()
        registry = CapabilityRegistry()
        registry.register(ReadFileTool(workspace))
        boundary = CapabilityBoundary(registry, workspace)
        manager = SessionManager(backend=backend, store=store, capability_boundary=boundary)
        session = manager.create_session()

        manager.run_turn(session.session_id, "hi")
        assert backend.last_request is not None
        assert backend.last_request.tool_definitions is not None
        assert len(backend.last_request.tool_definitions) == 1
        assert backend.last_request.tool_definitions[0]["name"] == "native__read_file"


# ---------------------------------------------------------------------------
# Assembler tool-call message handling
# ---------------------------------------------------------------------------


class TestAssemblerToolMessages:
    def test_assembler_handles_tool_role(self) -> None:
        assembler = TranscriptContextAssembler()
        messages = [
            ConversationMessage(
                message_id="m1", session_id="s1", role=MessageRole.USER,
                content="read it", turn_index=1,
                created_at=datetime.now(timezone.utc),
            ),
            ConversationMessage(
                message_id="m2", session_id="s1", role=MessageRole.ASSISTANT,
                content="", turn_index=2,
                created_at=datetime.now(timezone.utc),
                metadata={"tool_calls": [{"tool_call_id": "c1", "tool_name": "native__read_file", "arguments": {"path": "f.txt"}}]},
            ),
            ConversationMessage(
                message_id="m3", session_id="s1", role=MessageRole.TOOL,
                content="file contents", turn_index=3,
                created_at=datetime.now(timezone.utc),
                metadata={"tool_call_id": "c1", "tool_name": "native__read_file"},
            ),
            ConversationMessage(
                message_id="m4", session_id="s1", role=MessageRole.ASSISTANT,
                content="Here you go.", turn_index=4,
                created_at=datetime.now(timezone.utc),
            ),
        ]

        request = assembler.assemble(messages)
        assert len(request.messages) == 4

        # Assistant with tool_calls
        m_asst = request.messages[1]
        assert m_asst.role == "assistant"
        assert m_asst.content is None  # empty string becomes None
        assert m_asst.tool_calls is not None
        assert m_asst.tool_calls[0]["tool_name"] == "native__read_file"

        # Tool result
        m_tool = request.messages[2]
        assert m_tool.role == "tool"
        assert m_tool.content == "file contents"
        assert m_tool.tool_call_id == "c1"

        # Final assistant
        m_final = request.messages[3]
        assert m_final.role == "assistant"
        assert m_final.content == "Here you go."
        assert m_final.tool_calls is None


# ---------------------------------------------------------------------------
# OpenAI-compatible backend tool handling
# ---------------------------------------------------------------------------


class TestOpenAIToolFormatting:
    def test_build_chat_messages_with_tool_calls(self) -> None:
        from src.providers.openai_compatible import OpenAICompatibleBackend, OpenAICompatibleConfig

        backend = OpenAICompatibleBackend(OpenAICompatibleConfig())
        request = TurnRequest(
            messages=[
                Message(role="user", content="read file"),
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[{"tool_call_id": "c1", "tool_name": "read_file", "arguments": {"path": "a.txt"}}],
                ),
                Message(role="tool", content="file data", tool_call_id="c1"),
            ],
        )
        built = backend._build_chat_messages(request)
        assert len(built) == 3

        # Assistant message with tool_calls
        asst = built[1]
        assert asst["role"] == "assistant"
        assert "content" not in asst
        assert len(asst["tool_calls"]) == 1
        assert asst["tool_calls"][0]["id"] == "c1"
        assert asst["tool_calls"][0]["function"]["name"] == "read_file"

        # Tool result message
        tool_msg = built[2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["content"] == "file data"
        assert tool_msg["tool_call_id"] == "c1"

    def test_build_tools_from_definitions(self) -> None:
        from src.providers.openai_compatible import OpenAICompatibleBackend, OpenAICompatibleConfig

        backend = OpenAICompatibleBackend(OpenAICompatibleConfig())
        request = TurnRequest(
            messages=[Message(role="user", content="hi")],
            tool_definitions=[
                {"name": "my_tool", "description": "does things", "parameters": {"type": "object", "properties": {}}},
            ],
        )
        tools = backend._build_tools(request)
        assert tools is not None
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "my_tool"

    def test_build_tools_none_when_no_definitions(self) -> None:
        from src.providers.openai_compatible import OpenAICompatibleBackend, OpenAICompatibleConfig

        backend = OpenAICompatibleBackend(OpenAICompatibleConfig())
        request = TurnRequest(messages=[Message(role="user", content="hi")])
        assert backend._build_tools(request) is None


# ---------------------------------------------------------------------------
# Audit-driven tests (FINDING fixes)
# ---------------------------------------------------------------------------


class TestAuditFindings:
    """Tests for paths identified by the openclaw-audit."""

    def test_no_boundary_with_tool_calls_raises(self, tmp_path: Path) -> None:
        """FINDING-001: Provider returns tool_requests but no boundary configured."""
        tool_requests = [
            [ToolRequest(tool_call_id="c1", tool_name="native__read_file", arguments={"path": "x.txt"})]
        ]
        store = SQLiteSessionStore(db_path=":memory:")
        backend = ToolCallBackend(tool_requests)
        manager = SessionManager(backend=backend, store=store)  # No boundary
        session = manager.create_session()

        with pytest.raises(CapabilityBoundaryUnavailableError):
            manager.run_turn(session.session_id, "read it")

    def test_path_governance_on_tool_without_path_exists_check(self, workspace: Path) -> None:
        """FINDING-002: Governance must check path args on any tool, not just path_exists tools."""

        class UnsafePathTool(Tool):
            @property
            def definition(self) -> ToolDefinition:
                return ToolDefinition(
                    name="unsafe_tool",
                    description="A tool that forgot to set path_exists",
                    parameters={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                )

            @property
            def environment_check_kind(self) -> str:
                return "none"  # Forgot to set path_exists

            def execute(self, *, path: str) -> ToolResult:
                return ToolResult(ok=True, content="should not reach here")

        registry = CapabilityRegistry()
        registry.register(UnsafePathTool())
        boundary = CapabilityBoundary(registry, workspace)

        request = ToolRequest(
            tool_call_id="c1",
            tool_name="unsafe_tool",
            arguments={"path": "../../etc/passwd"},
        )
        result = boundary.execute(request)
        assert result.ok is False
        assert "governance denied" in result.content

    def test_max_tool_turns_exhaustion_recorded(self, workspace: Path) -> None:
        """FINDING-005: MAX_TOOL_TURNS exhaustion must leave a visible transcript record."""
        from src.runtime.session import MAX_TOOL_TURNS

        # Create a backend that always returns tool calls (never finishes)
        infinite_tool_calls = [
            [ToolRequest(tool_call_id=f"c{i}", tool_name="native__read_file", arguments={"path": "hello.txt"})]
            for i in range(MAX_TOOL_TURNS + 5)
        ]
        store = SQLiteSessionStore(db_path=":memory:")
        backend = ToolCallBackend(infinite_tool_calls)
        registry = CapabilityRegistry()
        registry.register(ReadFileTool(workspace))
        boundary = CapabilityBoundary(registry, workspace)
        manager = SessionManager(backend=backend, store=store, capability_boundary=boundary)
        session = manager.create_session()

        plan = manager.run_turn(session.session_id, "loop forever")
        messages = store.list_messages(session.session_id)

        # The last TOOL message should be the sentinel
        tool_messages = [m for m in messages if m.role == MessageRole.TOOL]
        last_tool = tool_messages[-1]
        assert "tool loop exhausted" in last_tool.content
        assert last_tool.metadata["governance_outcome"] == "denied_loop_exhaustion"

        # The last ASSISTANT message should have tool_loop_exhausted flag
        assistant_messages = [m for m in messages if m.role == MessageRole.ASSISTANT]
        last_assistant = assistant_messages[-1]
        assert last_assistant.metadata.get("tool_loop_exhausted") is True

    def test_unexpected_arguments_rejected(self, workspace: Path) -> None:
        """FINDING-009: Extra arguments not in schema should be rejected."""
        registry = CapabilityRegistry()
        registry.register(ReadFileTool(workspace))
        boundary = CapabilityBoundary(registry, workspace)

        request = ToolRequest(
            tool_call_id="c1",
            tool_name="native__read_file",
            arguments={"path": "hello.txt", "extra_arg": "bad"},
        )
        result = boundary.execute(request)
        assert result.ok is False
        assert "unexpected arguments" in result.content
        assert result.governance_outcome == "denied_invalid_arguments"

    def test_missing_required_arguments_rejected(self, workspace: Path) -> None:
        """FINDING-009: Missing required arguments should be rejected."""
        registry = CapabilityRegistry()
        registry.register(ReadFileTool(workspace))
        boundary = CapabilityBoundary(registry, workspace)

        request = ToolRequest(
            tool_call_id="c1",
            tool_name="native__read_file",
            arguments={},
        )
        result = boundary.execute(request)
        assert result.ok is False
        assert "missing required arguments" in result.content

    def test_malformed_json_arguments_handled(self) -> None:
        """FINDING-003: Malformed JSON in tool arguments should not crash."""
        from src.providers.openai_compatible import OpenAICompatibleBackend, OpenAICompatibleConfig

        class FakeFunction:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments

        class FakeToolCall:
            def __init__(self, tc_id, function):
                self.id = tc_id
                self.type = "function"
                self.function = function

        class FakeMessage:
            def __init__(self, content, tool_calls):
                self.content = content
                self.tool_calls = tool_calls

        class FakeChoice:
            def __init__(self, message, finish_reason="tool_calls"):
                self.message = message
                self.finish_reason = finish_reason

        class FakeUsage:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 5

        class FakeResponse:
            def __init__(self):
                self.choices = [FakeChoice(FakeMessage(
                    content=None,
                    tool_calls=[FakeToolCall("c1", FakeFunction("read_file", '{"path": "trunc'))],
                ))]
                self.model = "test"
                self.usage = FakeUsage()

        backend = OpenAICompatibleBackend(OpenAICompatibleConfig())
        plan = backend._normalize_response_to_plan(FakeResponse())
        assert len(plan.tool_requests) == 1
        assert plan.tool_requests[0].arguments.get("_parse_error") is True


# ---------------------------------------------------------------------------
# Handoff 08: Tool-call triggering and transcript persistence debug tests
# ---------------------------------------------------------------------------


class TestCodexToolSupport:
    """Tests for Codex backend tool definition injection and function_call parsing."""

    def _make_codex_backend(self, tmp_path: Path):
        from src.providers.codex import CodexBackend, CodexConfig
        credential_path = tmp_path / "cred.json"
        credential_path.write_text(
            '{"access_token":"tok","refresh_token":"ref",'
            f'"expires_at_epoch_ms":{int(__import__("time").time() * 1000) + 60000}}}',
            encoding="utf-8",
        )
        return CodexBackend(CodexConfig(credential_path=str(credential_path)), repo_root=tmp_path)

    def test_codex_payload_includes_tool_definitions(self, tmp_path: Path) -> None:
        backend = self._make_codex_backend(tmp_path)
        request = TurnRequest(
            messages=[Message(role="user", content="read file")],
            tool_definitions=[
                {"name": "native__read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            ],
        )
        payload = backend._build_request_payload(request)
        assert "tools" in payload
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["type"] == "function"
        assert payload["tools"][0]["name"] == "native__read_file"

    def test_codex_payload_no_tools_when_none(self, tmp_path: Path) -> None:
        backend = self._make_codex_backend(tmp_path)
        request = TurnRequest(messages=[Message(role="user", content="hello")])
        payload = backend._build_request_payload(request)
        assert "tools" not in payload

    def test_codex_normalizes_via_output_item_done(self, tmp_path: Path) -> None:
        """Primary path: response.output_item.done has all fields (real Codex API format)."""
        from src.transports.codex_sse import CodexSSEEvent
        backend = self._make_codex_backend(tmp_path)
        # Real Codex API event sequence (from .runtime/last_events.json capture)
        events = [
            CodexSSEEvent(
                payload={"type": "response.output_item.added", "output_index": 0,
                         "item": {"id": "fc_123", "type": "function_call", "status": "in_progress",
                                  "arguments": "", "call_id": "call_abc", "name": "native__read_file"}},
                raw_line="data: ...",
            ),
            CodexSSEEvent(
                payload={"type": "response.function_call_arguments.delta",
                         "delta": '{"path":"hello.txt"}', "item_id": "fc_123", "output_index": 0},
                raw_line="data: ...",
            ),
            CodexSSEEvent(
                payload={"type": "response.function_call_arguments.done",
                         "arguments": '{"path":"hello.txt"}', "item_id": "fc_123", "output_index": 0},
                raw_line="data: ...",
            ),
            CodexSSEEvent(
                payload={"type": "response.output_item.done", "output_index": 0,
                         "item": {"id": "fc_123", "type": "function_call", "status": "completed",
                                  "arguments": '{"path":"hello.txt"}', "call_id": "call_abc", "name": "native__read_file"}},
                raw_line="data: ...",
            ),
            # response.completed with empty output (real Codex behavior)
            CodexSSEEvent(
                payload={"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "model": "gpt-5.4", "usage": {}, "output": []}},
                raw_line="data: ...",
            ),
        ]
        plan = backend._normalize_events(events)
        assert plan.plan_label == "openai-codex-tool-calls"
        assert len(plan.tool_requests) == 1
        assert plan.tool_requests[0].tool_call_id == "call_abc"
        assert plan.tool_requests[0].tool_name == "native__read_file"
        assert plan.tool_requests[0].arguments == {"path": "hello.txt"}
        assert plan.tool_requests[0].provider_item_id == "fc_123"

    def test_codex_input_includes_tool_results(self, tmp_path: Path) -> None:
        backend = self._make_codex_backend(tmp_path)
        request = TurnRequest(
            messages=[
                Message(role="user", content="read it"),
                Message(role="assistant", content=None,
                        tool_calls=[{"tool_call_id": "call_abc", "tool_name": "native__read_file",
                                     "arguments": {"path": "f.txt"}, "provider_item_id": "fc_123"}]),
                Message(role="tool", content="file data", tool_call_id="call_abc"),
            ],
        )
        items = backend._build_codex_input(request)
        # user, function_call, function_call_output
        assert len(items) == 3
        assert items[0]["role"] == "user"
        assert items[1]["type"] == "function_call"
        assert items[1]["id"] == "fc_123"  # fc_ prefix for Codex API
        assert items[1]["call_id"] == "call_abc"  # call_ prefix
        assert items[1]["name"] == "native__read_file"
        assert items[2]["type"] == "function_call_output"
        assert items[2]["call_id"] == "call_abc"
        assert items[2]["output"] == "file data"

    def test_codex_input_falls_back_to_call_id_when_no_item_id(self, tmp_path: Path) -> None:
        """When provider_item_id is absent, use tool_call_id as id fallback."""
        backend = self._make_codex_backend(tmp_path)
        request = TurnRequest(
            messages=[
                Message(role="user", content="hi"),
                Message(role="assistant", content=None,
                        tool_calls=[{"tool_call_id": "call_xyz", "tool_name": "read_file",
                                     "arguments": {"path": "a.txt"}}]),
                Message(role="tool", content="data", tool_call_id="call_xyz"),
            ],
        )
        items = backend._build_codex_input(request)
        fc_item = [i for i in items if i.get("type") == "function_call"][0]
        assert fc_item["id"] == "call_xyz"  # falls back to tool_call_id
        assert fc_item["call_id"] == "call_xyz"

    def test_codex_malformed_function_call_args(self, tmp_path: Path) -> None:
        from src.transports.codex_sse import CodexSSEEvent
        backend = self._make_codex_backend(tmp_path)
        events = [
            CodexSSEEvent(
                payload={"type": "response.output_item.done", "output_index": 0,
                         "item": {"id": "fc_bad", "type": "function_call", "status": "completed",
                                  "call_id": "call_bad", "name": "read_file",
                                  "arguments": '{"path": "trunc'}},
                raw_line="data: ...",
            ),
            CodexSSEEvent(
                payload={"type": "response.completed", "response": {"id": "r1", "status": "completed", "model": "m", "usage": {}}},
                raw_line="data: ...",
            ),
        ]
        plan = backend._normalize_events(events)
        assert len(plan.tool_requests) == 1
        assert plan.tool_requests[0].arguments.get("_parse_error") is True

    def test_codex_fallback_to_response_completed_output(self, tmp_path: Path) -> None:
        """Fallback: when no output_item.done fires, use response.completed output items."""
        from src.transports.codex_sse import CodexSSEEvent
        backend = self._make_codex_backend(tmp_path)
        events = [
            CodexSSEEvent(
                payload={
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "model": "gpt-5.4",
                        "usage": {},
                        "output": [
                            {
                                "type": "function_call",
                                "id": "fc_123",
                                "call_id": "call_abc",
                                "name": "native__read_file",
                                "arguments": '{"path": "hello.txt"}',
                                "status": "completed",
                            }
                        ],
                    },
                },
                raw_line="data: ...",
            ),
        ]
        plan = backend._normalize_events(events)
        assert plan.plan_label == "openai-codex-tool-calls"
        assert len(plan.tool_requests) == 1
        assert plan.tool_requests[0].tool_call_id == "call_abc"
        assert plan.tool_requests[0].tool_name == "native__read_file"
        assert plan.tool_requests[0].arguments == {"path": "hello.txt"}


class TestRealisticFileReadPath:
    """Handoff 08 requirement: simulate a realistic file-read prompt and verify
    the full closure path and transcript persistence."""

    def test_file_read_end_to_end_transcript(self, workspace: Path) -> None:
        """Simulate a file-read request through the full stack:
        backend -> SessionManager -> CapabilityBoundary -> transcript persistence."""
        tool_requests = [
            [ToolRequest(
                tool_call_id="call_read_1",
                tool_name="native__read_file",
                arguments={"path": "hello.txt"},
            )]
        ]
        store = SQLiteSessionStore(db_path=":memory:")
        backend = ToolCallBackend(tool_requests)
        registry = CapabilityRegistry()
        registry.register(ReadFileTool(workspace))
        boundary = CapabilityBoundary(registry, workspace)
        manager = SessionManager(backend=backend, store=store, capability_boundary=boundary)
        session = manager.create_session(system_prompt="You can read files.")

        plan = manager.run_turn(session.session_id, "Please read hello.txt for me.")
        assert plan.final_text == "I read the file for you."

        # Full transcript verification
        messages = store.list_messages(session.session_id)
        assert len(messages) == 4

        # 1. User message
        assert messages[0].role == MessageRole.USER
        assert messages[0].content == "Please read hello.txt for me."

        # 2. Assistant with tool_calls metadata
        assert messages[1].role == MessageRole.ASSISTANT
        assert "tool_calls" in messages[1].metadata
        tc = messages[1].metadata["tool_calls"][0]
        assert tc["tool_call_id"] == "call_read_1"
        assert tc["tool_name"] == "native__read_file"
        assert tc["arguments"] == {"path": "hello.txt"}

        # 3. Tool result with full metadata
        assert messages[2].role == MessageRole.TOOL
        assert messages[2].content == "hello world"
        assert messages[2].metadata["tool_call_id"] == "call_read_1"
        assert messages[2].metadata["tool_name"] == "native__read_file"
        assert messages[2].metadata["ok"] is True
        assert messages[2].metadata["governance_outcome"] == "allowed"

        # 4. Final assistant text
        assert messages[3].role == MessageRole.ASSISTANT
        assert messages[3].content == "I read the file for you."

    def test_file_read_persists_across_sessions(self, workspace: Path, tmp_path: Path) -> None:
        """Verify tool-call transcript survives store reload (SQLite persistence)."""
        db_file = tmp_path / "persist_test.db"
        try:
            # Write session with tool call
            store1 = SQLiteSessionStore(db_path=db_file)
            tool_requests = [
                [ToolRequest(tool_call_id="c1", tool_name="native__read_file", arguments={"path": "hello.txt"})]
            ]
            backend = ToolCallBackend(tool_requests)
            registry = CapabilityRegistry()
            registry.register(ReadFileTool(workspace))
            boundary = CapabilityBoundary(registry, workspace)
            manager1 = SessionManager(backend=backend, store=store1, capability_boundary=boundary)
            session = manager1.create_session()
            manager1.run_turn(session.session_id, "read hello.txt")
            store1.close()

            # Read back from new store instance
            store2 = SQLiteSessionStore(db_path=db_file)
            messages = store2.list_messages(session.session_id)
            store2.close()

            roles = [m.role for m in messages]
            assert MessageRole.TOOL in roles
            tool_msg = [m for m in messages if m.role == MessageRole.TOOL][0]
            assert tool_msg.content == "hello world"
            assert tool_msg.metadata["tool_call_id"] == "c1"
        finally:
            db_file.unlink(missing_ok=True)


class TestCLICapabilityWiring:
    """Verify the CLI harness wires the capability boundary."""

    def test_build_capability_boundary(self, workspace: Path) -> None:
        from src.cli.harness import _build_capability_boundary
        boundary = _build_capability_boundary(workspace)
        definitions = boundary.list_definitions()
        assert len(definitions) == 1
        assert definitions[0].name == "native__read_file"

    def test_cli_manager_has_capability_boundary(self, workspace: Path) -> None:
        from src.cli.harness import _build_capability_boundary
        store = SQLiteSessionStore(db_path=":memory:")
        backend = ToolCallBackend([])
        boundary = _build_capability_boundary(workspace)
        manager = SessionManager(backend=backend, store=store, capability_boundary=boundary)
        assert manager._capability_boundary is not None


class TestGovernanceSensitivePaths:
    """CRIT-01: Credential and sensitive path access must be denied."""

    def test_runtime_credentials_blocked(self, workspace: Path) -> None:
        (workspace / ".runtime").mkdir()
        (workspace / ".runtime" / "creds.json").write_text('{"secret": true}')

        registry = CapabilityRegistry()
        registry.register(ReadFileTool(workspace))
        boundary = CapabilityBoundary(registry, workspace)

        result = boundary.execute(ToolRequest(
            tool_call_id="c1",
            tool_name="native__read_file",
            arguments={"path": ".runtime/creds.json"},
        ))
        assert result.ok is False
        assert "governance denied" in result.content
        assert "protected location" in result.content

    def test_env_file_blocked(self, workspace: Path) -> None:
        (workspace / ".env").write_text("SECRET=bad")

        registry = CapabilityRegistry()
        registry.register(ReadFileTool(workspace))
        boundary = CapabilityBoundary(registry, workspace)

        result = boundary.execute(ToolRequest(
            tool_call_id="c1",
            tool_name="native__read_file",
            arguments={"path": ".env"},
        ))
        assert result.ok is False
        assert "protected location" in result.content

    def test_normal_files_still_allowed(self, workspace: Path) -> None:
        registry = CapabilityRegistry()
        registry.register(ReadFileTool(workspace))
        boundary = CapabilityBoundary(registry, workspace)

        result = boundary.execute(ToolRequest(
            tool_call_id="c1",
            tool_name="native__read_file",
            arguments={"path": "hello.txt"},
        ))
        assert result.ok is True


class TestCodexInputIdField:
    """function_call items must include correct 'id' (fc_ prefix) for Responses API."""

    def test_function_call_item_uses_provider_item_id(self, tmp_path: Path) -> None:
        from src.providers.codex import CodexBackend, CodexConfig
        import time
        credential_path = tmp_path / "cred.json"
        credential_path.write_text(
            '{"access_token":"tok","refresh_token":"ref",'
            f'"expires_at_epoch_ms":{int(time.time() * 1000) + 60000}}}',
            encoding="utf-8",
        )
        backend = CodexBackend(CodexConfig(credential_path=str(credential_path)), repo_root=tmp_path)
        request = TurnRequest(
            messages=[
                Message(role="user", content="read it"),
                Message(role="assistant", content=None,
                        tool_calls=[{"tool_call_id": "call_xyz", "tool_name": "read_file",
                                     "arguments": {"path": "a.txt"}, "provider_item_id": "fc_abc"}]),
                Message(role="tool", content="data", tool_call_id="call_xyz"),
            ],
        )
        items = backend._build_codex_input(request)
        fc_item = [i for i in items if i.get("type") == "function_call"][0]
        assert fc_item["id"] == "fc_abc"  # Uses provider_item_id (fc_ prefix)
        assert fc_item["call_id"] == "call_xyz"  # Uses tool_call_id (call_ prefix)
