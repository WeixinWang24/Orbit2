"""Tests for `src/operation/cli/approval.py::CLIApprovalInteractor`."""

from __future__ import annotations

import io

from src.governance.approval import ApprovalDecision, ApprovalRequest
from src.operation.cli.approval import CLIApprovalInteractor


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        session_id="session_123",
        tool_name="write_file",
        reveal_group="native_fs_mutate",
        side_effect_class="write",
        requires_approval=True,
        arguments={"path": "notes.md", "content": "hi"},
        summary="Write content to a file in the workspace.",
    )


class _ScriptedReader:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("reader called more than scripted")
        return self._responses.pop(0)


class _EofReader:
    def __call__(self, prompt: str) -> str:
        raise EOFError()


class _InterruptReader:
    def __call__(self, prompt: str) -> str:
        raise KeyboardInterrupt()


class TestCLIApprovalInteractorChoices:
    def test_a_allows_once(self) -> None:
        out = io.StringIO()
        reader = _ScriptedReader(["a"])
        interactor = CLIApprovalInteractor(reader=reader, out=out)
        assert interactor.prompt(_request()) == ApprovalDecision.ALLOW_ONCE

    def test_allow_word_allows_once(self) -> None:
        out = io.StringIO()
        reader = _ScriptedReader(["allow"])
        interactor = CLIApprovalInteractor(reader=reader, out=out)
        assert interactor.prompt(_request()) == ApprovalDecision.ALLOW_ONCE

    def test_s_allows_similar(self) -> None:
        out = io.StringIO()
        reader = _ScriptedReader(["s"])
        interactor = CLIApprovalInteractor(reader=reader, out=out)
        assert interactor.prompt(_request()) == ApprovalDecision.ALLOW_SIMILAR_IN_SESSION

    def test_similar_word_allows_similar(self) -> None:
        out = io.StringIO()
        reader = _ScriptedReader(["similar"])
        interactor = CLIApprovalInteractor(reader=reader, out=out)
        assert interactor.prompt(_request()) == ApprovalDecision.ALLOW_SIMILAR_IN_SESSION

    def test_d_denies(self) -> None:
        out = io.StringIO()
        reader = _ScriptedReader(["d"])
        interactor = CLIApprovalInteractor(reader=reader, out=out)
        assert interactor.prompt(_request()) == ApprovalDecision.DENY

    def test_empty_defaults_to_deny(self) -> None:
        out = io.StringIO()
        reader = _ScriptedReader([""])
        interactor = CLIApprovalInteractor(reader=reader, out=out)
        assert interactor.prompt(_request()) == ApprovalDecision.DENY

    def test_unrecognised_input_denies(self) -> None:
        out = io.StringIO()
        reader = _ScriptedReader(["maybe"])
        interactor = CLIApprovalInteractor(reader=reader, out=out)
        assert interactor.prompt(_request()) == ApprovalDecision.DENY

    def test_case_insensitive(self) -> None:
        out = io.StringIO()
        reader = _ScriptedReader(["  S  "])
        interactor = CLIApprovalInteractor(reader=reader, out=out)
        assert interactor.prompt(_request()) == ApprovalDecision.ALLOW_SIMILAR_IN_SESSION


class TestCLIApprovalInteractorFailSafe:
    def test_eof_denies(self) -> None:
        out = io.StringIO()
        interactor = CLIApprovalInteractor(reader=_EofReader(), out=out)
        assert interactor.prompt(_request()) == ApprovalDecision.DENY

    def test_interrupt_denies(self) -> None:
        out = io.StringIO()
        interactor = CLIApprovalInteractor(reader=_InterruptReader(), out=out)
        assert interactor.prompt(_request()) == ApprovalDecision.DENY


class TestCLIApprovalInteractorRendering:
    def test_header_shows_tool_and_reveal_group(self) -> None:
        out = io.StringIO()
        reader = _ScriptedReader(["d"])
        interactor = CLIApprovalInteractor(reader=reader, out=out)
        interactor.prompt(_request())
        rendered = out.getvalue()
        assert "write_file" in rendered
        assert "native_fs_mutate" in rendered
        assert "Approval required" in rendered

    def test_argument_preview_is_bounded(self) -> None:
        out = io.StringIO()
        reader = _ScriptedReader(["d"])
        interactor = CLIApprovalInteractor(reader=reader, out=out)
        long_req = ApprovalRequest(
            session_id="s1",
            tool_name="write_file",
            reveal_group="native_fs_mutate",
            side_effect_class="write",
            requires_approval=True,
            arguments={"content": "x" * 5000},
            summary="",
        )
        interactor.prompt(long_req)
        rendered = out.getvalue()
        # Argument preview truncated with ellipsis — never dumps the full 5000-char payload
        assert "..." in rendered
        # Must not echo the whole oversized argument blob to the operator console
        assert "x" * 5000 not in rendered

    def test_summary_single_line_only(self) -> None:
        out = io.StringIO()
        reader = _ScriptedReader(["d"])
        interactor = CLIApprovalInteractor(reader=reader, out=out)
        req = ApprovalRequest(
            session_id="s1",
            tool_name="write_file",
            reveal_group="native_fs_mutate",
            side_effect_class="write",
            requires_approval=True,
            arguments={},
            summary="first line\nsecond line\nthird line",
        )
        interactor.prompt(req)
        rendered = out.getvalue()
        assert "first line" in rendered
        assert "second line" not in rendered
        assert "third line" not in rendered
