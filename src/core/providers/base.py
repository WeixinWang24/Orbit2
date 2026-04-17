from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from pydantic import BaseModel

from src.core.runtime.models import TurnRequest, ProviderNormalizedResult, ExecutionPlan


class BackendConfig(BaseModel):
    model: str
    max_tokens: int = 1024


class ExecutionBackend(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str: ...

    @abstractmethod
    def plan_from_messages(
        self,
        request: TurnRequest,
        *,
        on_partial_text: Callable[[str], None] | None = None,
    ) -> ExecutionPlan: ...

    def _normalize_to_plan(self, result: ProviderNormalizedResult) -> ExecutionPlan:
        return ExecutionPlan(
            source_backend=result.source_backend,
            plan_label=result.plan_label,
            final_text=result.final_text,
            model=result.model,
            metadata=result.metadata,
        )
