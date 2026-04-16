from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel

from src.runtime.models import TurnRequest, ProviderNormalizedResult, ExecutionPlan
from src.providers.base import BackendConfig, ExecutionBackend
from src.transports.codex_sse import CodexHttpError, CodexSSEEvent, stream_sse_events

CODEX_BASE_URL = "https://chatgpt.com/backend-api"
DEFAULT_CREDENTIAL_PATH = ".runtime/openai_oauth_credentials.json"


class OAuthCredential(BaseModel):
    access_token: str
    refresh_token: str
    expires_at_epoch_ms: int
    account_email: Optional[str] = None


class CodexConfig(BackendConfig):
    model: str = "gpt-5.4"
    api_base: str = CODEX_BASE_URL
    bearer_token: Optional[str] = None
    credential_path: Optional[str] = None
    timeout_seconds: int = 60

    model_config = {"json_schema_extra": {"properties": {"bearer_token": {"writeOnly": True}}}}

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude", set())
        if isinstance(kwargs["exclude"], set):
            kwargs["exclude"].add("bearer_token")
        return super().model_dump(**kwargs)


def load_oauth_credential(path: Path) -> OAuthCredential:
    if not path.exists():
        raise FileNotFoundError(f"OAuth credential file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    credential = OAuthCredential(**raw)
    now_ms = int(time.time() * 1000)
    if credential.expires_at_epoch_ms <= now_ms:
        raise RuntimeError(
            f"OAuth access token is expired "
            f"(expired at {credential.expires_at_epoch_ms}, now {now_ms}); "
            f"refresh the token in {path}"
        )
    if not credential.access_token.strip():
        raise RuntimeError(f"OAuth credential file has an empty access_token: {path}")
    return credential


class CodexBackend(ExecutionBackend):
    def __init__(self, config: Optional[CodexConfig] = None, repo_root: Optional[Path] = None) -> None:
        self._config = config or CodexConfig()
        self._repo_root = repo_root or Path.cwd()
        self._credential: Optional[OAuthCredential] = None
        self._bearer_token = self._resolve_bearer_token()

    def _resolve_bearer_token(self) -> str:
        if self._config.bearer_token:
            return self._config.bearer_token
        if self._config.credential_path:
            cred_path = self._repo_root / self._config.credential_path
        else:
            cred_path = self._repo_root / DEFAULT_CREDENTIAL_PATH
        self._credential = load_oauth_credential(cred_path)
        return self._credential.access_token

    def _check_token_expiry(self) -> None:
        if self._credential is None:
            return
        now_ms = int(time.time() * 1000)
        if self._credential.expires_at_epoch_ms <= now_ms:
            raise RuntimeError(
                f"OAuth access token expired during session "
                f"(expired at {self._credential.expires_at_epoch_ms}, now {now_ms}); "
                f"refresh the token and reconstruct the backend"
            )

    @property
    def backend_name(self) -> str:
        return "openai-codex"

    def plan_from_messages(
        self,
        request: TurnRequest,
        *,
        on_partial_text: Callable[[str], None] | None = None,
    ) -> ExecutionPlan:
        self._check_token_expiry()
        url = self._build_request_url()
        headers = self._build_request_headers()
        payload = self._build_request_payload(request)
        try:
            events: list[CodexSSEEvent] = []
            accumulated_text: list[str] = []
            for event in stream_sse_events(
                url=url,
                headers=headers,
                payload=payload,
                timeout_seconds=self._config.timeout_seconds,
            ):
                events.append(event)
                if on_partial_text and event.payload.get("type") == "response.output_text.delta":
                    delta = event.payload.get("delta")
                    if isinstance(delta, str):
                        accumulated_text.append(delta)
                        on_partial_text("".join(accumulated_text))
        except CodexHttpError as exc:
            normalized = ProviderNormalizedResult(
                source_backend=self.backend_name,
                plan_label=f"{self.backend_name}-transport-failure",
                final_text=None,
                model=self._config.model,
                metadata={"error": str(exc)},
            )
            return self._normalize_to_plan(normalized)
        return self._normalize_events(events)

    def _build_request_url(self) -> str:
        return self._config.api_base.rstrip("/") + "/codex/responses"

    def _build_request_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._bearer_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def _build_request_payload(self, request: TurnRequest) -> dict:
        codex_input = self._build_codex_input(request)
        payload: dict = {
            "model": self._config.model,
            "store": False,
            "stream": True,
            "input": codex_input,
            "text": {"verbosity": "medium"},
        }
        if request.system:
            payload["instructions"] = request.system
        return payload

    def _build_codex_input(self, request: TurnRequest) -> list[dict]:
        items: list[dict] = []
        for m in request.messages:
            if m.role == "assistant":
                items.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": m.content}],
                })
            else:
                items.append({
                    "role": m.role,
                    "content": [{"type": "input_text", "text": m.content}],
                })
        return items

    def _normalize_events(self, events: list[CodexSSEEvent]) -> ExecutionPlan:
        text_parts: list[str] = []
        response_id: str | None = None
        status: str | None = None
        usage: dict | None = None
        model: str | None = None

        for event in events:
            p = event.payload
            event_type = p.get("type")

            if response_id is None and isinstance(p.get("response_id"), str):
                response_id = p["response_id"]
            if status is None and isinstance(p.get("status"), str):
                status = p["status"]
            if model is None and isinstance(p.get("model"), str):
                model = p["model"]
            if usage is None and isinstance(p.get("usage"), dict):
                usage = p["usage"]

            if event_type == "response.output_text.delta":
                delta = p.get("delta")
                if isinstance(delta, str):
                    text_parts.append(delta)

            elif event_type in {
                "response.completed",
                "response.done",
                "response.incomplete",
            }:
                response = p.get("response")
                if isinstance(response, dict):
                    if isinstance(response.get("id"), str):
                        response_id = response["id"]
                    if isinstance(response.get("status"), str):
                        status = response["status"]
                    if isinstance(response.get("usage"), dict):
                        usage = response["usage"]
                    if isinstance(response.get("model"), str):
                        model = response["model"]

            elif event_type == "error":
                message = (
                    p.get("message")
                    if isinstance(p.get("message"), str)
                    else "Codex returned an error event"
                )
                normalized = ProviderNormalizedResult(
                    source_backend=self.backend_name,
                    plan_label=f"{self.backend_name}-error-event",
                    final_text=None,
                    model=model or self._config.model,
                    metadata={"error": message, "event_count": len(events)},
                )
                return self._normalize_to_plan(normalized)

        final_text = "".join(text_parts).strip()
        if final_text:
            normalized = ProviderNormalizedResult(
                source_backend=self.backend_name,
                plan_label=f"{self.backend_name}-final-text",
                final_text=final_text,
                model=model or self._config.model,
                metadata={
                    "response_id": response_id,
                    "status": status,
                    "event_count": len(events),
                    "usage": usage,
                },
            )
            return self._normalize_to_plan(normalized)

        normalized = ProviderNormalizedResult(
            source_backend=self.backend_name,
            plan_label=f"{self.backend_name}-empty-response",
            final_text=None,
            model=model or self._config.model,
            metadata={
                "response_id": response_id,
                "status": status,
                "event_count": len(events),
                "usage": usage,
            },
        )
        return self._normalize_to_plan(normalized)
