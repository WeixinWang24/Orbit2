from __future__ import annotations

import base64
import time
from pathlib import Path

import httpx
import openai
import pytest

from src.core.runtime.models import (
    ConversationMessage,
    ExecutionPlan,
    Message,
    MessageRole,
    ProviderNormalizedResult,
    Session,
    SessionStatus,
    TurnRequest,
)
from src.core.runtime.session import SessionManager
from src.core.providers.base import ExecutionBackend
from src.core.providers.codex import (
    CodexBackend,
    CodexConfig,
    OAuthCredential,
    load_oauth_credential,
)
from src.core.providers.openai_compatible import OpenAICompatibleBackend, OpenAICompatibleConfig
from src.core.store.base import SessionStore
from src.core.store.sqlite import SQLiteSessionStore
from src.core.transports.codex_sse import CodexHttpError, CodexSSEEvent


# ---------------------------------------------------------------------------
# Test backends
# ---------------------------------------------------------------------------

class DummyBackend(ExecutionBackend):
    @property
    def backend_name(self) -> str:
        return "dummy"

    def plan_from_messages(self, request: TurnRequest, *, on_partial_text=None) -> ExecutionPlan:
        return ExecutionPlan(
            source_backend=self.backend_name,
            plan_label="dummy-final-text",
            final_text=f"echo: {request.messages[-1].content}",
            model="dummy-model",
            metadata={"message_count": len(request.messages)},
        )


class NormalizeProbeBackend(ExecutionBackend):
    @property
    def backend_name(self) -> str:
        return "probe"

    def plan_from_messages(self, request: TurnRequest, *, on_partial_text=None) -> ExecutionPlan:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Fake OpenAI SDK objects
# ---------------------------------------------------------------------------

class FakeMessage:
    def __init__(self, content: str | None):
        self.content = content
        self.tool_calls = None


class FakeChoice:
    def __init__(self, content: str | None, finish_reason: str = "stop"):
        self.message = FakeMessage(content) if content is not None else None
        self.finish_reason = finish_reason


class FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeResponse:
    def __init__(self, *, choices, model: str = "fake-model", usage=None):
        self.choices = choices
        self.model = model
        self.usage = usage


class FakeCompletions:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class RaisingCompletions:
    def create(self, **kwargs):
        request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
        raise openai.APIConnectionError(request=request)


class FakeChat:
    def __init__(self, response):
        self.completions = FakeCompletions(response)


class RaisingChat:
    def __init__(self):
        self.completions = RaisingCompletions()


class FakeOpenAIClient:
    def __init__(self, response):
        self.chat = FakeChat(response)


class RaisingOpenAIClient:
    def __init__(self):
        self.chat = RaisingChat()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_request() -> TurnRequest:
    return TurnRequest(
        system="You are concise.",
        messages=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
            Message(role="user", content="continue"),
        ],
    )


@pytest.fixture
def sqlite_store(tmp_path: Path) -> SQLiteSessionStore:
    return SQLiteSessionStore(tmp_path / "test.db")


@pytest.fixture
def session_manager(sqlite_store: SQLiteSessionStore) -> SessionManager:
    return SessionManager(backend=DummyBackend(), store=sqlite_store)


# ---------------------------------------------------------------------------
# Provider tests (preserved from prior slice)
# ---------------------------------------------------------------------------

def test_normalize_to_plan_preserves_fields() -> None:
    backend = NormalizeProbeBackend()
    normalized = ProviderNormalizedResult(
        source_backend="probe",
        plan_label="probe-final-text",
        final_text="done",
        model="probe-model",
        metadata={"k": "v", "n": 1},
    )
    plan = backend._normalize_to_plan(normalized)
    assert plan == ExecutionPlan(
        source_backend="probe",
        plan_label="probe-final-text",
        final_text="done",
        model="probe-model",
        metadata={"k": "v", "n": 1},
    )


def test_openai_compatible_build_chat_messages(sample_request: TurnRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_response = FakeResponse(choices=[])
    fake_client = FakeOpenAIClient(fake_response)
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: fake_client)

    backend = OpenAICompatibleBackend(OpenAICompatibleConfig())
    messages = backend._build_chat_messages(sample_request)
    assert messages == [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "continue"},
    ]


def test_openai_compatible_plan_from_messages_returns_final_text(sample_request: TurnRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_response = FakeResponse(
        choices=[FakeChoice("orbit says hi")],
        model="demo-model",
        usage=FakeUsage(prompt_tokens=7, completion_tokens=4),
    )
    fake_client = FakeOpenAIClient(fake_response)
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: fake_client)

    backend = OpenAICompatibleBackend(
        OpenAICompatibleConfig(model="demo-model", base_url="http://localhost:8000/v1")
    )
    plan = backend.plan_from_messages(sample_request)
    assert plan.source_backend == "openai-compatible"
    assert plan.final_text == "orbit says hi"
    assert plan.model == "demo-model"


def test_openai_compatible_backend_accepts_basic_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_response = FakeResponse(choices=[])
    fake_client = FakeOpenAIClient(fake_response)
    captured_kwargs = {}
    monkeypatch.setattr(
        "openai.OpenAI",
        lambda **kwargs: captured_kwargs.update(kwargs) or fake_client,
    )

    OpenAICompatibleBackend(
        OpenAICompatibleConfig(
            model="demo-model",
            base_url="http://10.204.18.32:8080/v1",
            basic_auth_username="alice",
            basic_auth_password="secret",
        )
    )

    token = base64.b64encode(b"alice:secret").decode("ascii")
    assert captured_kwargs["base_url"] == "http://10.204.18.32:8080/v1"
    assert captured_kwargs["api_key"] == "EMPTY"
    assert captured_kwargs["default_headers"] == {"Authorization": f"Basic {token}"}


def test_openai_compatible_connection_error_returns_plan(
    sample_request: TurnRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: RaisingOpenAIClient())
    backend = OpenAICompatibleBackend(
        OpenAICompatibleConfig(model="demo-model", base_url="http://localhost:8000/v1")
    )

    plan = backend.plan_from_messages(sample_request)

    assert plan.plan_label == "openai-compatible-provider-error"
    assert plan.model == "demo-model"
    assert "APIConnectionError" in (plan.final_text or "")
    assert "base_url=http://localhost:8000/v1" in (plan.final_text or "")
    assert plan.metadata["base_url"] == "http://localhost:8000/v1"


def test_openai_compatible_normalize_response_handles_empty_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_response = FakeResponse(choices=[], model="fallback-model", usage=None)
    fake_client = FakeOpenAIClient(fake_response)
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: fake_client)

    backend = OpenAICompatibleBackend(OpenAICompatibleConfig(model="cfg-model"))
    plan = backend._normalize_response_to_plan(fake_response)
    assert plan.plan_label == "openai-compatible-empty-response"
    assert plan.final_text is None


# ---------------------------------------------------------------------------
# Credential loading tests (preserved)
# ---------------------------------------------------------------------------

def test_load_oauth_credential_reads_valid_file(tmp_path: Path) -> None:
    credential_path = tmp_path / "openai_oauth_credentials.json"
    credential_path.write_text(
        '{'
        '"access_token":"token-123",'
        '"refresh_token":"refresh-456",'
        f'"expires_at_epoch_ms":{int(time.time() * 1000) + 60000},'
        '"account_email":"user@example.com"'
        '}',
        encoding="utf-8",
    )
    credential = load_oauth_credential(credential_path)
    assert credential.access_token == "token-123"
    assert credential.account_email == "user@example.com"


def test_load_oauth_credential_rejects_expired_file(tmp_path: Path) -> None:
    credential_path = tmp_path / "openai_oauth_credentials.json"
    credential_path.write_text(
        '{'
        '"access_token":"token-123",'
        '"refresh_token":"refresh-456",'
        f'"expires_at_epoch_ms":{int(time.time() * 1000) - 1000}'
        '}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="expired"):
        load_oauth_credential(credential_path)


# ---------------------------------------------------------------------------
# Codex backend tests (preserved)
# ---------------------------------------------------------------------------

def test_codex_backend_builds_request_parts(sample_request: TurnRequest, tmp_path: Path) -> None:
    credential_path = tmp_path / "cred.json"
    credential_path.write_text(
        '{'
        '"access_token":"bearer-xyz",'
        '"refresh_token":"refresh-456",'
        f'"expires_at_epoch_ms":{int(time.time() * 1000) + 60000}'
        '}',
        encoding="utf-8",
    )
    backend = CodexBackend(
        CodexConfig(model="test-codex-model", credential_path=str(credential_path)),
        repo_root=tmp_path,
    )
    assert backend._build_request_url() == "https://chatgpt.com/backend-api/codex/responses"
    assert backend._build_request_headers()["Authorization"] == "Bearer bearer-xyz"


def test_codex_backend_normalizes_streamed_text(tmp_path: Path) -> None:
    credential_path = tmp_path / "cred.json"
    credential_path.write_text(
        '{'
        '"access_token":"bearer-xyz",'
        '"refresh_token":"refresh-456",'
        f'"expires_at_epoch_ms":{int(time.time() * 1000) + 60000}'
        '}',
        encoding="utf-8",
    )
    backend = CodexBackend(
        CodexConfig(model="test-codex-model", credential_path=str(credential_path)),
        repo_root=tmp_path,
    )
    events = [
        CodexSSEEvent(payload={"type": "response.output_text.delta", "delta": "Hello"}, raw_line="data: ..."),
        CodexSSEEvent(payload={"type": "response.output_text.delta", "delta": " world"}, raw_line="data: ..."),
        CodexSSEEvent(
            payload={
                "type": "response.completed",
                "response": {"id": "resp_123", "status": "completed", "model": "gpt-5.4", "usage": {"input_tokens": 3, "output_tokens": 2}},
            },
            raw_line="data: ...",
        ),
    ]
    plan = backend._normalize_events(events)
    assert plan.final_text == "Hello world"
    assert plan.model == "gpt-5.4"


def test_codex_backend_normalizes_error_event(tmp_path: Path) -> None:
    credential_path = tmp_path / "cred.json"
    credential_path.write_text(
        '{'
        '"access_token":"bearer-xyz",'
        '"refresh_token":"refresh-456",'
        f'"expires_at_epoch_ms":{int(time.time() * 1000) + 60000}'
        '}',
        encoding="utf-8",
    )
    backend = CodexBackend(
        CodexConfig(model="test-codex-model", credential_path=str(credential_path)),
        repo_root=tmp_path,
    )
    events = [CodexSSEEvent(payload={"type": "error", "message": "bad auth"}, raw_line="data: ...")]
    plan = backend._normalize_events(events)
    assert plan.plan_label == "openai-codex-error-event"
    assert plan.final_text == "openai-codex returned an error event: bad auth"


def test_codex_backend_normalizes_output_item_done_message_text(tmp_path: Path) -> None:
    credential_path = tmp_path / "cred.json"
    credential_path.write_text(
        '{'
        '"access_token":"bearer-xyz",'
        '"refresh_token":"refresh-456",'
        f'"expires_at_epoch_ms":{int(time.time() * 1000) + 60000}'
        '}',
        encoding="utf-8",
    )
    backend = CodexBackend(
        CodexConfig(model="test-codex-model", credential_path=str(credential_path)),
        repo_root=tmp_path,
    )
    events = [
        CodexSSEEvent(
            payload={
                "type": "response.output_item.done",
                "item": {
                    "id": "msg_1",
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello from item"}],
                },
            },
            raw_line="data: ...",
        ),
        CodexSSEEvent(
            payload={
                "type": "response.completed",
                "response": {"id": "resp_123", "status": "completed", "model": "gpt-5.4"},
            },
            raw_line="data: ...",
        ),
    ]
    plan = backend._normalize_events(events)
    assert plan.plan_label == "openai-codex-final-text"
    assert plan.final_text == "Hello from item"


def test_codex_backend_normalizes_completed_message_text(tmp_path: Path) -> None:
    credential_path = tmp_path / "cred.json"
    credential_path.write_text(
        '{'
        '"access_token":"bearer-xyz",'
        '"refresh_token":"refresh-456",'
        f'"expires_at_epoch_ms":{int(time.time() * 1000) + 60000}'
        '}',
        encoding="utf-8",
    )
    backend = CodexBackend(
        CodexConfig(model="test-codex-model", credential_path=str(credential_path)),
        repo_root=tmp_path,
    )
    events = [
        CodexSSEEvent(
            payload={
                "type": "response.completed",
                "response": {
                    "id": "resp_123",
                    "status": "completed",
                    "model": "gpt-5.4",
                    "output": [
                        {
                            "id": "msg_1",
                            "type": "message",
                            "content": [{"type": "output_text", "text": "Hello completed"}],
                        }
                    ],
                },
            },
            raw_line="data: ...",
        ),
    ]
    plan = backend._normalize_events(events)
    assert plan.plan_label == "openai-codex-final-text"
    assert plan.final_text == "Hello completed"


def test_codex_backend_plan_from_messages_handles_transport_failure(
    sample_request: TurnRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential_path = tmp_path / "cred.json"
    credential_path.write_text(
        '{'
        '"access_token":"bearer-xyz",'
        '"refresh_token":"refresh-456",'
        f'"expires_at_epoch_ms":{int(time.time() * 1000) + 60000}'
        '}',
        encoding="utf-8",
    )
    backend = CodexBackend(
        CodexConfig(model="test-codex-model", credential_path=str(credential_path)),
        repo_root=tmp_path,
    )

    def fake_stream_sse_events(**kwargs):
        raise CodexHttpError("Codex HTTP error 401: unauthorized")

    monkeypatch.setattr("src.core.providers.codex.stream_sse_events", fake_stream_sse_events)
    plan = backend.plan_from_messages(sample_request)
    assert plan.plan_label == "openai-codex-transport-failure"
    assert plan.final_text == "openai-codex request failed: Codex HTTP error 401: unauthorized"


# ---------------------------------------------------------------------------
# Session manager tests (new)
# ---------------------------------------------------------------------------

def test_create_session(session_manager: SessionManager) -> None:
    session = session_manager.create_session(system_prompt="Be concise.")
    assert session.session_id.startswith("session_")
    assert session.backend_name == "dummy"
    assert session.system_prompt == "Be concise."
    assert session.status == SessionStatus.ACTIVE


def test_get_session(session_manager: SessionManager) -> None:
    created = session_manager.create_session()
    retrieved = session_manager.get_session(created.session_id)
    assert retrieved.session_id == created.session_id


def test_get_session_not_found(session_manager: SessionManager) -> None:
    with pytest.raises(ValueError, match="not found"):
        session_manager.get_session("nonexistent")


def test_list_sessions(session_manager: SessionManager) -> None:
    session_manager.create_session()
    session_manager.create_session()
    sessions = session_manager.list_sessions()
    assert len(sessions) == 2


def test_run_turn_single(session_manager: SessionManager) -> None:
    session = session_manager.create_session(system_prompt="Be concise.")
    plan = session_manager.run_turn(session.session_id, "hello")

    assert isinstance(plan, ExecutionPlan)
    assert plan.source_backend == "dummy"
    assert plan.final_text == "echo: hello"

    messages = session_manager.list_messages(session.session_id)
    assert len(messages) == 2
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == "hello"
    assert messages[1].role == MessageRole.ASSISTANT
    assert messages[1].content == "echo: hello"


def test_run_turn_multi_turn(session_manager: SessionManager) -> None:
    session = session_manager.create_session(system_prompt="Be concise.")

    plan1 = session_manager.run_turn(session.session_id, "first")
    assert plan1.final_text == "echo: first"

    plan2 = session_manager.run_turn(session.session_id, "second")
    assert plan2.final_text == "echo: second"
    assert plan2.metadata["message_count"] == 3  # user, assistant, user

    messages = session_manager.list_messages(session.session_id)
    assert len(messages) == 4
    assert [m.role for m in messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert messages[2].content == "second"


def test_transcript_persists_across_manager_instances(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "persist.db")
    backend = DummyBackend()

    mgr1 = SessionManager(backend=backend, store=store)
    session = mgr1.create_session(system_prompt="test")
    mgr1.run_turn(session.session_id, "hello")

    mgr2 = SessionManager(backend=backend, store=store)
    plan = mgr2.run_turn(session.session_id, "world")
    assert plan.metadata["message_count"] == 3  # hello, echo:hello, world

    messages = mgr2.list_messages(session.session_id)
    assert len(messages) == 4
    assert messages[0].content == "hello"
    assert messages[2].content == "world"

    store.close()


# ---------------------------------------------------------------------------
# SQLite store tests (new)
# ---------------------------------------------------------------------------

def test_sqlite_store_roundtrip_session(sqlite_store: SQLiteSessionStore) -> None:
    from datetime import datetime, timezone
    session = Session(
        session_id="test_session",
        backend_name="dummy",
        system_prompt="hello",
        status=SessionStatus.ACTIVE,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    sqlite_store.save_session(session)
    retrieved = sqlite_store.get_session("test_session")
    assert retrieved is not None
    assert retrieved.session_id == "test_session"
    assert retrieved.system_prompt == "hello"


def test_sqlite_store_roundtrip_messages(sqlite_store: SQLiteSessionStore) -> None:
    from datetime import datetime, timezone
    session = Session(
        session_id="s1",
        backend_name="dummy",
        status=SessionStatus.ACTIVE,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    sqlite_store.save_session(session)

    msg = ConversationMessage(
        message_id="m1",
        session_id="s1",
        role=MessageRole.USER,
        content="hi",
        turn_index=1,
        created_at=datetime.now(timezone.utc),
    )
    sqlite_store.save_message(msg)

    messages = sqlite_store.list_messages("s1")
    assert len(messages) == 1
    assert messages[0].message_id == "m1"
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == "hi"


def test_sqlite_store_update_session(sqlite_store: SQLiteSessionStore) -> None:
    from datetime import datetime, timezone
    session = Session(
        session_id="s1",
        backend_name="dummy",
        status=SessionStatus.ACTIVE,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    sqlite_store.save_session(session)

    session.status = SessionStatus.COMPLETED
    session.updated_at = datetime.now(timezone.utc)
    sqlite_store.save_session(session)

    retrieved = sqlite_store.get_session("s1")
    assert retrieved is not None
    assert retrieved.status == SessionStatus.COMPLETED
