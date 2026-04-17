"""Tests for runtime/workspace/store root resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.runtime.paths import (
    DEFAULT_DB_NAME,
    REPO_ROOT,
    RUNTIME_ROOT_ENV,
    STORE_SUBDIR,
    RuntimeRoot,
    default_db_path,
    resolve_runtime_root,
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RUNTIME_ROOT_ENV, raising=False)


class TestRepoRootDefault:
    def test_repo_root_points_at_checkout_parent_of_src(self, clean_env) -> None:
        assert (REPO_ROOT / "src" / "core" / "runtime" / "paths.py").exists()

    def test_default_resolve_returns_repo_root(self, clean_env) -> None:
        resolved = resolve_runtime_root()
        assert isinstance(resolved, RuntimeRoot)
        assert resolved.path == REPO_ROOT
        assert resolved.source == "repo-root"


class TestEnvOverride:
    def test_env_var_wins_over_default(
        self, clean_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        target = tmp_path / "iso_env"
        target.mkdir()
        monkeypatch.setenv(RUNTIME_ROOT_ENV, str(target))
        resolved = resolve_runtime_root()
        assert resolved.path == target.resolve()
        assert resolved.source == f"env:{RUNTIME_ROOT_ENV}"

    def test_resolve_is_pure_does_not_create_root(
        self, clean_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Resolution should not materialize the root directory; callers opt in
        via default_db_path when they actually need persistence."""
        missing = tmp_path / "does_not_exist_yet"
        monkeypatch.setenv(RUNTIME_ROOT_ENV, str(missing))
        resolved = resolve_runtime_root()
        assert resolved.path == missing.resolve()
        assert not missing.exists()


class TestExplicitOverride:
    def test_explicit_wins_over_env(
        self, clean_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        env_target = tmp_path / "from_env"
        flag_target = tmp_path / "from_flag"
        monkeypatch.setenv(RUNTIME_ROOT_ENV, str(env_target))
        resolved = resolve_runtime_root(flag_target)
        assert resolved.path == flag_target.resolve()
        assert resolved.source == "cli-flag"

    def test_override_accepts_string(self, clean_env, tmp_path: Path) -> None:
        resolved = resolve_runtime_root(str(tmp_path))
        assert resolved.path == tmp_path.resolve()
        assert resolved.source == "cli-flag"

    def test_override_expands_user(
        self, clean_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        resolved = resolve_runtime_root("~/fake")
        assert resolved.path == (tmp_path / "fake").resolve()


class TestCwdIsIgnored:
    def test_cwd_never_influences_default(
        self, clean_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert resolve_runtime_root().path == REPO_ROOT


class TestDbPath:
    def test_db_path_layout_from_path(self, tmp_path: Path) -> None:
        db = default_db_path(tmp_path)
        assert db == tmp_path / STORE_SUBDIR / DEFAULT_DB_NAME
        assert db.parent.exists()

    def test_db_path_layout_from_runtime_root(self, tmp_path: Path) -> None:
        runtime_root = RuntimeRoot(tmp_path, "cli-flag")
        db = default_db_path(runtime_root)
        assert db == tmp_path / STORE_SUBDIR / DEFAULT_DB_NAME

    def test_db_path_creates_store_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "root"
        assert not target.exists()
        default_db_path(target)
        assert (target / STORE_SUBDIR).is_dir()
