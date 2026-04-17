from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.knowledge.runtime_context import RuntimeContextSnapshot


_DEFAULT_SELF_LOCATION_FIELDS: frozenset[str] = frozenset({
    "cwd",
    "runtime_root_path",
    "runtime_root_source",
    "repo_root",
    "store_path",
})


@dataclass(frozen=True)
class RuntimeContextDisclosureDecision:
    disclose: bool
    visible_fields: frozenset[str]
    policy_name: str


class RuntimeContextDisclosurePolicy(ABC):
    @property
    @abstractmethod
    def policy_name(self) -> str: ...

    @abstractmethod
    def decide(
        self, snapshot: RuntimeContextSnapshot
    ) -> RuntimeContextDisclosureDecision: ...


class BasicSelfLocationDisclosurePolicy(RuntimeContextDisclosurePolicy):
    _POLICY_NAME = "basic_self_location"

    @property
    def policy_name(self) -> str:
        return self._POLICY_NAME

    def decide(
        self, snapshot: RuntimeContextSnapshot
    ) -> RuntimeContextDisclosureDecision:
        return RuntimeContextDisclosureDecision(
            disclose=True,
            visible_fields=_DEFAULT_SELF_LOCATION_FIELDS,
            policy_name=self._POLICY_NAME,
        )


DEFAULT_RUNTIME_CONTEXT_DISCLOSURE_POLICY: RuntimeContextDisclosurePolicy = (
    BasicSelfLocationDisclosurePolicy()
)
