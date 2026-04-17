from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

RUNTIME_ROOT_ENV = "ORBIT2_RUNTIME_ROOT"
STORE_SUBDIR = ".runtime"
DEFAULT_DB_NAME = "sessions.db"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass(frozen=True)
class RuntimeRoot:
    path: Path
    source: str  # "cli-flag" | "env:ORBIT2_RUNTIME_ROOT" | "repo-root"


def resolve_runtime_root(override: Path | str | None = None) -> RuntimeRoot:
    if override is not None:
        return RuntimeRoot(Path(override).expanduser().resolve(), "cli-flag")
    env_value = os.environ.get(RUNTIME_ROOT_ENV)
    if env_value:
        return RuntimeRoot(
            Path(env_value).expanduser().resolve(),
            f"env:{RUNTIME_ROOT_ENV}",
        )
    return RuntimeRoot(REPO_ROOT, "repo-root")


def default_db_path(runtime_root: Path | RuntimeRoot) -> Path:
    path = runtime_root.path if isinstance(runtime_root, RuntimeRoot) else runtime_root
    store_dir = path / STORE_SUBDIR
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir / DEFAULT_DB_NAME
