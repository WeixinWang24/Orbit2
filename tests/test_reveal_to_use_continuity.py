"""Reveal-to-use continuity tests — Handoff 28.

Regression anchor: session_60ee34acb36e. The model successfully revealed
`native_fs_mutate` in response to a "create directory" request, but then
returned a final text message asking the user to say "继续" instead of
using the newly-visible `write_file` / `create_directory` tool in the same
response. The session manager's tool loop DOES re-plan after each tool call
— so the model could have made the downstream call immediately. The bug is
the provider's mental model: it treats "next turn" as "next user message"
when the reveal actually takes effect on the NEXT model step within the
same response.

The Handoff-28 fix is awareness-shaping: change the discovery tool's
description, hint, and reveal-confirmation strings — plus the capability
awareness posture — so they make the continuation bridge explicit and warn
against the shell-fallback / "please say continue" response pattern.

These tests verify the shaped strings are present. They do not test a real
provider's behavior; that is an empirical question requiring live runs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.capability.boundary import CapabilityBoundary
from src.capability.discovery import (
    DISCOVERY_TOOL_NAME,
    ListAvailableToolsTool,
)
from src.capability.registry import CapabilityRegistry
from src.capability.tools import ReadFileTool, WriteFileTool
from src.core.providers.base import ExecutionBackend
from src.core.runtime.models import (
    ExecutionPlan,
    ToolRequest,
    TurnRequest,
)
from src.core.runtime.session import SessionManager
from src.core.store.sqlite import SQLiteSessionStore
from src.governance.capability_awareness_disclosure import (
    BasicCapabilityAwarenessDisclosurePolicy,
)
from src.governance.disclosure import REVEAL_REQUEST_MARKER
from src.knowledge import (
    CAPABILITY_AWARENESS_POSTURE_TEXT,
    CapabilityAwarenessCollector,
    StructuredContextAssembler,
)


# ---------------------------------------------------------------------------
# Shaped-string presence: discovery tool outputs
# ---------------------------------------------------------------------------


def _disco_with_registry(tmp_path: Path) -> ListAvailableToolsTool:
    registry = CapabilityRegistry()
    registry.register(ReadFileTool(tmp_path))
    registry.register(WriteFileTool(tmp_path))
    registry.register(ListAvailableToolsTool(registry))
    return registry.get(DISCOVERY_TOOL_NAME)


class TestDiscoveryDefinitionCarriesContinuationBridge:
    def test_description_mentions_same_response_and_anti_pause(
        self, tmp_path: Path
    ) -> None:
        disco = _disco_with_registry(tmp_path)
        desc = disco.definition.description
        # The bridge cue: revealed tools usable in the same response
        assert "same response" in desc
        # The anti-pattern the provider fell into in session_60ee34acb36e
        assert (
            "do NOT return a final text" in desc.lower()
            or "do not return a final text" in desc.lower()
        )

    def test_hint_warns_against_asking_user_to_continue(
        self, tmp_path: Path
    ) -> None:
        disco = _disco_with_registry(tmp_path)
        import json
        result = disco.execute()
        payload = json.loads(result.content)
        hint = payload["hint"]
        assert "same response" in hint.lower() or "same response" in hint
        assert "continue" in hint.lower()
        # Must explicitly address the "ask user to continue" pattern
        assert "do not return" in hint.lower() or "do NOT return" in hint


class TestRevealConfirmationCarriesContinuationBridge:
    def test_single_reveal_confirmation_tells_provider_to_continue(
        self, tmp_path: Path
    ) -> None:
        disco = _disco_with_registry(tmp_path)
        result = disco.execute(reveal="native_fs_mutate")
        assert REVEAL_REQUEST_MARKER in result.data
        confirmation = result.data["reveal_request_confirmation"]
        assert "same response" in confirmation
        # Must specifically warn against the final-text-asking-continuation pattern
        assert "final message" in confirmation.lower()
        # Must NOT carry the old "will be exposed on the next turn" phrasing,
        # which induced the "wait for user" misread in session_60ee34acb36e
        assert "will be exposed on the next turn" not in confirmation

    def test_batch_reveal_confirmation_carries_continuation_bridge(
        self, tmp_path: Path
    ) -> None:
        disco = _disco_with_registry(tmp_path)
        result = disco.execute(reveal_batch=["native_fs_mutate"])
        confirmation = result.data["reveal_batch_confirmation"]
        assert "same response" in confirmation
        assert "final message" in confirmation.lower()

    def test_all_safe_reveal_confirmation_carries_continuation_bridge(
        self, tmp_path: Path
    ) -> None:
        disco = _disco_with_registry(tmp_path)
        result = disco.execute(reveal_all_safe=True)
        confirmation = result.data["reveal_all_safe_confirmation"]
        assert "same response" in confirmation
        assert "final message" in confirmation.lower()


# ---------------------------------------------------------------------------
# Awareness posture carries the continuation bridge
# ---------------------------------------------------------------------------


class TestCapabilityAwarenessPostureContinuationBridge:
    def test_posture_text_mentions_reveal_to_use_continuity(self) -> None:
        text = CAPABILITY_AWARENESS_POSTURE_TEXT
        # Key bridge concepts
        assert "reveal" in text.lower()
        assert "same response" in text or "SAME response" in text
        # Explicitly address the observed failure: asking for "继续"
        assert "继续" in text or "continue" in text.lower()
        # Explicit don't-return-final-text
        assert "final text" in text.lower() or "do NOT return" in text or "do not return" in text.lower()


# ---------------------------------------------------------------------------
# End-to-end regression: session_60ee34acb36e shape via fake backend
#
# The fake backend script:
#   step 1 (user: "create .test folder"): issues reveal request
#   step 2 (after reveal executes, loop re-plans): issues the downstream
#           write_file call — proving the loop keeps the model active in
#           the same response after reveal
#   step 3 (after write_file): returns final text
#
# This proves the continuation bridge IS structurally supported by the
# session manager; the Handoff-28 shaping is what nudges a real provider
# toward following it.
# ---------------------------------------------------------------------------


class _ScriptedBackend(ExecutionBackend):
    def __init__(self, script: list[ExecutionPlan]) -> None:
        self._script = list(script)
        self.call_count = 0
        self.last_system: str | None = None

    @property
    def backend_name(self) -> str:
        return "scripted"

    def plan_from_messages(self, request: TurnRequest, *, on_partial_text=None) -> ExecutionPlan:
        self.last_system = request.system
        if self.call_count < len(self._script):
            plan = self._script[self.call_count]
            self.call_count += 1
            return plan
        return ExecutionPlan(
            source_backend=self.backend_name,
            plan_label="scripted-fallback",
            final_text="done",
            model="fake",
        )


class TestRevealToUseContinuityEndToEnd:
    def test_session_manager_loop_supports_reveal_then_use_in_one_response(
        self, tmp_path: Path
    ) -> None:
        registry = CapabilityRegistry()
        registry.register(ReadFileTool(tmp_path))
        registry.register(WriteFileTool(tmp_path))
        registry.register(ListAvailableToolsTool(registry))
        boundary = CapabilityBoundary(registry, tmp_path)

        script = [
            # step 1: reveal native_fs_mutate
            ExecutionPlan(
                source_backend="scripted", plan_label="scripted-reveal",
                final_text=None, model="fake",
                tool_requests=[ToolRequest(
                    tool_call_id="c1", tool_name="list_available_tools",
                    arguments={"reveal": "native_fs_mutate"},
                )],
            ),
            # step 2: immediately use write_file in the SAME response
            # — no user turn in between; this simulates the well-shaped flow
            ExecutionPlan(
                source_backend="scripted", plan_label="scripted-use",
                final_text=None, model="fake",
                tool_requests=[ToolRequest(
                    tool_call_id="c2", tool_name="native__write_file",
                    arguments={"path": ".test/.gitkeep", "content": ""},
                )],
            ),
            # step 3: final text
            ExecutionPlan(
                source_backend="scripted", plan_label="scripted-final",
                final_text="已创建 .test 文件夹", model="fake",
            ),
        ]
        backend = _ScriptedBackend(script)

        store = SQLiteSessionStore(db_path=":memory:")
        assembler = StructuredContextAssembler(
            capability_awareness_collector=CapabilityAwarenessCollector(boundary.registry),
            capability_awareness_disclosure_policy=BasicCapabilityAwarenessDisclosurePolicy(),
        )
        manager = SessionManager(
            backend=backend, store=store, assembler=assembler,
            capability_boundary=boundary,
        )
        session = manager.create_session(system_prompt="helper")
        manager.run_turn(session.session_id, "在工作目录下创建.test文件夹")

        # The backend should have been called 3 times in ONE user turn:
        # planning(step1) → planning(step2 after reveal) → planning(step3 after write)
        assert backend.call_count == 3

        # The file must have actually been written in the same user turn
        created = tmp_path / ".test" / ".gitkeep"
        assert created.exists(), ".gitkeep should have been written via loop"

        # Only one user message in the transcript — no "继续" was needed
        messages = manager.list_messages(session.session_id)
        user_msgs = [m for m in messages if m.role.value == "user"]
        assert len(user_msgs) == 1
        assert "继续" not in user_msgs[0].content

        store.close()

    def test_second_plan_after_reveal_sees_widened_exposure_in_system_prompt(
        self, tmp_path: Path
    ) -> None:
        """The awareness fragment on step 2 (after reveal) should no longer
        list `native_fs_mutate` as hidden — because the exposure decision
        now reflects the reveal request from the saved tool result."""
        registry = CapabilityRegistry()
        registry.register(ReadFileTool(tmp_path))
        registry.register(WriteFileTool(tmp_path))
        registry.register(ListAvailableToolsTool(registry))
        boundary = CapabilityBoundary(registry, tmp_path)

        captured_systems: list[str | None] = []

        class _CapturingBackend(_ScriptedBackend):
            def plan_from_messages(self, request, *, on_partial_text=None):
                captured_systems.append(request.system)
                return super().plan_from_messages(request, on_partial_text=on_partial_text)

        script = [
            ExecutionPlan(
                source_backend="scripted", plan_label="r",
                final_text=None, model="fake",
                tool_requests=[ToolRequest(
                    tool_call_id="c1", tool_name="list_available_tools",
                    arguments={"reveal": "native_fs_mutate"},
                )],
            ),
            ExecutionPlan(
                source_backend="scripted", plan_label="f",
                final_text="done-after-reveal", model="fake",
            ),
        ]
        backend = _CapturingBackend(script)

        store = SQLiteSessionStore(db_path=":memory:")
        assembler = StructuredContextAssembler(
            capability_awareness_collector=CapabilityAwarenessCollector(boundary.registry),
            capability_awareness_disclosure_policy=BasicCapabilityAwarenessDisclosurePolicy(),
        )
        manager = SessionManager(
            backend=backend, store=store, assembler=assembler,
            capability_boundary=boundary,
        )
        session = manager.create_session(system_prompt=None)
        manager.run_turn(session.session_id, "create .test folder please")

        assert len(captured_systems) == 2
        first, second = captured_systems
        # Step 1 system prompt: native_fs_mutate should be listed as hidden
        assert first is not None and "native_fs_mutate" in first
        # Step 2 system prompt: after reveal, native_fs_mutate should be
        # visible, so the awareness fragment should not list it as hidden
        # (BasicPolicy may even suppress the whole block if nothing is hidden).
        if second is not None and "<capability-awareness>" in second:
            # If block present, native_fs_mutate must not be in hidden list
            cap_start = second.find("<capability-awareness>")
            cap_end = second.find("</capability-awareness>")
            cap_block = second[cap_start:cap_end]
            hidden_section_start = cap_block.find("hidden_reveal_groups")
            hidden_section = cap_block[hidden_section_start:]
            assert "native_fs_mutate" not in hidden_section

        store.close()
