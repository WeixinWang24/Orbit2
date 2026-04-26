from __future__ import annotations

import base64
import json
from typing import Callable, Optional

import openai

from src.core.runtime.models import (
    TurnRequest,
    ProviderNormalizedResult,
    ExecutionPlan,
    ToolRequest,
)
from src.core.providers.base import BackendConfig, ExecutionBackend


class OpenAICompatibleConfig(BackendConfig):
    model: str = "default"
    base_url: str = "http://localhost:8000/v1"
    api_key: Optional[str] = None
    basic_auth_username: Optional[str] = None
    basic_auth_password: Optional[str] = None


class OpenAICompatibleBackend(ExecutionBackend):
    def __init__(self, config: Optional[OpenAICompatibleConfig] = None) -> None:
        self._config = config or OpenAICompatibleConfig()
        client_kwargs: dict = {
            "base_url": self._config.base_url,
            "api_key": self._config.api_key if self._config.api_key is not None else "EMPTY",
        }
        if self._config.basic_auth_username is not None:
            token = _basic_auth_token(
                self._config.basic_auth_username,
                self._config.basic_auth_password or "",
            )
            client_kwargs["default_headers"] = {"Authorization": f"Basic {token}"}
        self._client = openai.OpenAI(**client_kwargs)

    @property
    def backend_name(self) -> str:
        return "openai-compatible"

    def plan_from_messages(
        self,
        request: TurnRequest,
        *,
        on_partial_text: Callable[[str], None] | None = None,
    ) -> ExecutionPlan:
        messages = self._build_chat_messages(request)
        kwargs: dict = dict(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            messages=messages,
        )
        tools = self._build_tools(request)
        if tools:
            kwargs["tools"] = tools
        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.APIError as exc:
            return self._normalize_provider_error(exc)
        return self._normalize_response_to_plan(response)

    def _build_chat_messages(self, request: TurnRequest) -> list[dict]:
        messages: list[dict] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        for m in request.messages:
            msg: dict = {"role": m.role}
            if m.content is not None:
                msg["content"] = m.content
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc["tool_call_id"],
                        "type": "function",
                        "function": {
                            "name": tc["tool_name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in m.tool_calls
                ]
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            messages.append(msg)
        return messages

    def _build_tools(self, request: TurnRequest) -> list[dict] | None:
        if not request.tool_definitions:
            return None
        return [
            {"type": "function", "function": td}
            for td in request.tool_definitions
        ]

    def _normalize_response_to_plan(self, response) -> ExecutionPlan:
        if not response.choices:
            return ExecutionPlan(
                source_backend=self.backend_name,
                plan_label=f"{self.backend_name}-empty-response",
                final_text=None,
                model=response.model or self._config.model,
                metadata={},
            )
        choice = response.choices[0]
        message = choice.message

        tool_requests: list[ToolRequest] = []
        if message and message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw_arguments": args, "_parse_error": True}
                tool_requests.append(ToolRequest(
                    tool_call_id=tc.id,
                    tool_name=tc.function.name,
                    arguments=args if isinstance(args, dict) else {},
                ))

        plan_label = f"{self.backend_name}-tool-calls" if tool_requests else f"{self.backend_name}-final-text"

        return ExecutionPlan(
            source_backend=self.backend_name,
            plan_label=plan_label,
            final_text=message.content if message else None,
            model=response.model,
            tool_requests=tool_requests,
            metadata={
                "finish_reason": choice.finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                } if response.usage else {},
            },
        )

    def _normalize_provider_error(self, exc: openai.APIError) -> ExecutionPlan:
        error_type = type(exc).__name__
        text = (
            f"{self.backend_name} request failed: {error_type}: {exc}. "
            f"base_url={self._config.base_url}"
        )
        normalized = ProviderNormalizedResult(
            source_backend=self.backend_name,
            plan_label=f"{self.backend_name}-provider-error",
            final_text=text,
            model=self._config.model,
            metadata={
                "error_type": error_type,
                "error": str(exc),
                "base_url": self._config.base_url,
            },
        )
        return self._normalize_to_plan(normalized)


def _basic_auth_token(username: str, password: str) -> str:
    raw = f"{username}:{password}".encode("utf-8")
    return base64.b64encode(raw).decode("ascii")
