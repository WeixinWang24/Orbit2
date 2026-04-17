from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.config.runtime import (
    DEFAULT_DB_NAME,
    REPO_ROOT,
    STORE_SUBDIR,
    RuntimeRoot,
)
from src.knowledge.models import ContextFragment

RUNTIME_CONTEXT_FRAGMENT_NAME = "runtime_context"
RUNTIME_CONTEXT_VISIBILITY_SCOPE = "runtime_awareness"
RUNTIME_CONTEXT_PRIORITY = 90

_OPEN_TAG = "<runtime-context>"
_CLOSE_TAG = "</runtime-context>"


@dataclass(frozen=True)
class RuntimeContextSnapshot:
    cwd: Path
    runtime_root_path: Path
    runtime_root_source: str
    repo_root: Path
    store_path: Path

    def to_visible_dict(self, visible_fields: frozenset[str]) -> dict[str, str]:
        full = {
            "cwd": str(self.cwd),
            "runtime_root_path": str(self.runtime_root_path),
            "runtime_root_source": self.runtime_root_source,
            "repo_root": str(self.repo_root),
            "store_path": str(self.store_path),
        }
        return {k: v for k, v in full.items() if k in visible_fields}


class RuntimeContextCollector:
    def __init__(
        self, runtime_root: RuntimeRoot, store_path: Path | None = None
    ) -> None:
        self._runtime_root = runtime_root
        self._store_path = (
            store_path
            if store_path is not None
            else runtime_root.path / STORE_SUBDIR / DEFAULT_DB_NAME
        )

    @property
    def runtime_root(self) -> RuntimeRoot:
        return self._runtime_root

    def collect(self) -> RuntimeContextSnapshot:
        return RuntimeContextSnapshot(
            cwd=Path(os.getcwd()).resolve(),
            runtime_root_path=self._runtime_root.path,
            runtime_root_source=self._runtime_root.source,
            repo_root=REPO_ROOT,
            store_path=self._store_path,
        )


def build_runtime_context_fragment(
    snapshot: RuntimeContextSnapshot,
    *,
    visible_fields: frozenset[str],
    policy_name: str,
) -> ContextFragment:
    visible = snapshot.to_visible_dict(visible_fields)
    lines = [f"{k}: {v}" for k, v in visible.items()]
    content = "\n".join([_OPEN_TAG, *lines, _CLOSE_TAG])
    return ContextFragment(
        fragment_name=RUNTIME_CONTEXT_FRAGMENT_NAME,
        visibility_scope=RUNTIME_CONTEXT_VISIBILITY_SCOPE,
        content=content,
        priority=RUNTIME_CONTEXT_PRIORITY,
        metadata={
            "origin": "runtime_context",
            "policy_name": policy_name,
            "visible_fields": sorted(visible_fields),
        },
    )
