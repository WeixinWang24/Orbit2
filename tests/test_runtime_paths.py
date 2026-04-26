"""Tests for runtime/workspace/store root resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.runtime import (
    AGENT_RUNTIME_CONFIG_NAME,
    DEFAULT_CODE_INTEL_DB_NAME,
    DEFAULT_PROVIDER_MODEL_ENV,
    DEFAULT_PROVIDER_MODEL,
    DEFAULT_VLLM_API_KEY_ENV,
    DEFAULT_VLLM_BASE_URL,
    DEFAULT_VLLM_BASE_URL_ENV,
    DEFAULT_VLLM_PASSWORD_ENV,
    DEFAULT_VLLM_USERNAME_ENV,
    DEFAULT_DB_NAME,
    REPO_ROOT,
    RUNTIME_ROOT_ENV,
    STORE_SUBDIR,
    RuntimeRoot,
    code_intel_db_path,
    default_db_path,
    resolve_obsidian_vault_root,
    resolve_provider_model,
    resolve_runtime_root,
    resolve_vllm_provider_settings,
    runtime_config_path,
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RUNTIME_ROOT_ENV, raising=False)
    monkeypatch.delenv(DEFAULT_PROVIDER_MODEL_ENV, raising=False)
    monkeypatch.delenv("ORBIT2_TEST_PROVIDER_MODEL", raising=False)
    monkeypatch.delenv(DEFAULT_VLLM_BASE_URL_ENV, raising=False)
    monkeypatch.delenv(DEFAULT_VLLM_API_KEY_ENV, raising=False)
    monkeypatch.delenv(DEFAULT_VLLM_USERNAME_ENV, raising=False)
    monkeypatch.delenv(DEFAULT_VLLM_PASSWORD_ENV, raising=False)
    monkeypatch.delenv("ORBIT2_TEST_VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("ORBIT2_TEST_VLLM_USERNAME", raising=False)
    monkeypatch.delenv("ORBIT2_TEST_VLLM_PASSWORD", raising=False)


class TestRepoRootDefault:
    def test_repo_root_points_at_checkout_parent_of_src(self, clean_env) -> None:
        assert (REPO_ROOT / "src" / "config" / "runtime.py").exists()

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


class TestConfigCentralization:
    def test_max_tool_turns_importable_from_config(self) -> None:
        from src.config.runtime import MAX_TOOL_TURNS
        assert isinstance(MAX_TOOL_TURNS, int) and MAX_TOOL_TURNS > 0

    def test_max_tool_turns_not_assigned_in_session_source(self) -> None:
        session_src = (REPO_ROOT / "src" / "core" / "runtime" / "session.py").read_text()
        assert "MAX_TOOL_TURNS = " not in session_src, (
            "MAX_TOOL_TURNS should be defined in src.config.runtime, not assigned in session.py"
        )

    def test_config_package_re_exports_runtime_symbols(self) -> None:
        import src.config as cfg
        for sym in (
            "REPO_ROOT",
            "RUNTIME_ROOT_ENV",
            "DEFAULT_PROVIDER_MODEL",
            "DEFAULT_PROVIDER_MODEL_ENV",
            "DEFAULT_VLLM_API_KEY_ENV",
            "DEFAULT_VLLM_BASE_URL",
            "DEFAULT_VLLM_BASE_URL_ENV",
            "DEFAULT_VLLM_PASSWORD_ENV",
            "DEFAULT_VLLM_USERNAME_ENV",
            "DEFAULT_CODE_INTEL_DB_NAME",
            "MAX_TOOL_TURNS",
            "RuntimeRoot",
            "RuntimeEnvSetting",
            "RuntimePathSetting",
            "VllmProviderSettings",
            "resolve_runtime_root",
            "resolve_obsidian_vault_root",
            "resolve_provider_model",
            "resolve_vllm_provider_settings",
            "runtime_config_path",
            "code_intel_db_path",
            "default_db_path",
        ):
            assert hasattr(cfg, sym), f"src.config missing re-export: {sym}"

    def test_paths_module_removed_from_core_runtime(self) -> None:
        from pathlib import Path
        assert not (REPO_ROOT / "src" / "core" / "runtime" / "paths.py").exists()


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

    def test_code_intel_db_path_uses_separate_runtime_db(self, tmp_path: Path) -> None:
        db = code_intel_db_path(tmp_path)
        assert db == tmp_path / STORE_SUBDIR / DEFAULT_CODE_INTEL_DB_NAME
        assert db != default_db_path(tmp_path)


class TestAgentRuntimeConfig:
    def test_runtime_config_path_layout(self, tmp_path: Path) -> None:
        assert runtime_config_path(tmp_path) == (
            tmp_path / STORE_SUBDIR / AGENT_RUNTIME_CONFIG_NAME
        )

    def test_obsidian_vault_root_missing_config_returns_none(
        self, tmp_path: Path
    ) -> None:
        assert resolve_obsidian_vault_root(tmp_path) is None

    def test_obsidian_vault_root_reads_runtime_config(
        self, tmp_path: Path
    ) -> None:
        vault = tmp_path / "vault"
        vault.mkdir()
        config_dir = tmp_path / STORE_SUBDIR
        config_dir.mkdir()
        (config_dir / AGENT_RUNTIME_CONFIG_NAME).write_text(
            f"[obsidian]\nvault_root = {str(vault)!r}\n",
            encoding="utf-8",
        )

        resolved = resolve_obsidian_vault_root(tmp_path)

        assert resolved is not None
        assert resolved.path == vault.resolve()
        assert resolved.source.startswith("config:")

    def test_obsidian_vault_root_cli_override_wins_over_config(
        self, tmp_path: Path
    ) -> None:
        from_config = tmp_path / "from_config"
        from_cli = tmp_path / "from_cli"
        from_config.mkdir()
        from_cli.mkdir()
        config_dir = tmp_path / STORE_SUBDIR
        config_dir.mkdir()
        (config_dir / AGENT_RUNTIME_CONFIG_NAME).write_text(
            f"[obsidian]\nvault_root = {str(from_config)!r}\n",
            encoding="utf-8",
        )

        resolved = resolve_obsidian_vault_root(tmp_path, from_cli)

        assert resolved is not None
        assert resolved.path == from_cli.resolve()
        assert resolved.source == "cli-flag"

    def test_provider_model_reads_default_env(
        self, clean_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(DEFAULT_PROVIDER_MODEL_ENV, "env-model")

        resolved = resolve_provider_model(tmp_path)

        assert resolved.env_name == DEFAULT_PROVIDER_MODEL_ENV
        assert resolved.value == "env-model"
        assert resolved.source == f"default:{DEFAULT_PROVIDER_MODEL_ENV}"

    def test_provider_model_env_name_reads_runtime_config(
        self, clean_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / STORE_SUBDIR
        config_dir.mkdir()
        (config_dir / AGENT_RUNTIME_CONFIG_NAME).write_text(
            "[provider]\nmodel_env = 'ORBIT2_TEST_PROVIDER_MODEL'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("ORBIT2_TEST_PROVIDER_MODEL", "configured-env-model")

        resolved = resolve_provider_model(tmp_path)

        assert resolved.env_name == "ORBIT2_TEST_PROVIDER_MODEL"
        assert resolved.value == "configured-env-model"
        assert resolved.source.startswith("config:")

    def test_provider_model_direct_value_reads_runtime_config(
        self, clean_env, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / STORE_SUBDIR
        config_dir.mkdir()
        (config_dir / AGENT_RUNTIME_CONFIG_NAME).write_text(
            "[provider]\nmodel_env = 'local-model'\n",
            encoding="utf-8",
        )

        resolved = resolve_provider_model(tmp_path)

        assert resolved.value == "local-model"
        assert resolved.source.endswith("[provider].model_env")

    def test_provider_model_missing_env_uses_default(
        self, clean_env, tmp_path: Path
    ) -> None:
        resolved = resolve_provider_model(tmp_path)

        assert resolved.env_name == DEFAULT_PROVIDER_MODEL_ENV
        assert resolved.value == DEFAULT_PROVIDER_MODEL
        assert resolved.source == f"default:{DEFAULT_PROVIDER_MODEL}"

    def test_vllm_settings_use_local_default(
        self, clean_env, tmp_path: Path
    ) -> None:
        resolved = resolve_vllm_provider_settings(tmp_path)

        assert resolved.base_url == DEFAULT_VLLM_BASE_URL
        assert resolved.base_url_source == f"default:{DEFAULT_VLLM_BASE_URL}"
        assert resolved.api_key is None
        assert resolved.basic_auth_username is None
        assert resolved.basic_auth_password is None

    def test_vllm_settings_read_default_env(
        self, clean_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(
            DEFAULT_VLLM_BASE_URL_ENV,
            "http://10.204.18.32:8080/v1/chat/completions",
        )
        monkeypatch.setenv(DEFAULT_VLLM_USERNAME_ENV, "alice")
        monkeypatch.setenv(DEFAULT_VLLM_PASSWORD_ENV, "secret")

        resolved = resolve_vllm_provider_settings(tmp_path)

        assert resolved.base_url == "http://10.204.18.32:8080/v1"
        assert resolved.base_url_source == f"env:{DEFAULT_VLLM_BASE_URL_ENV}"
        assert resolved.basic_auth_username == "alice"
        assert resolved.basic_auth_password == "secret"
        assert resolved.basic_auth_source == (
            f"env:{DEFAULT_VLLM_USERNAME_ENV},env:{DEFAULT_VLLM_PASSWORD_ENV}"
        )

    def test_vllm_settings_env_names_read_runtime_config(
        self, clean_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / STORE_SUBDIR
        config_dir.mkdir()
        (config_dir / AGENT_RUNTIME_CONFIG_NAME).write_text(
            "\n".join([
                "[vllm]",
                "base_url_env = 'ORBIT2_TEST_VLLM_BASE_URL'",
                "basic_auth_username_env = 'ORBIT2_TEST_VLLM_USERNAME'",
                "basic_auth_password_env = 'ORBIT2_TEST_VLLM_PASSWORD'",
            ]),
            encoding="utf-8",
        )
        monkeypatch.setenv("ORBIT2_TEST_VLLM_BASE_URL", "http://localhost:9000/v1")
        monkeypatch.setenv("ORBIT2_TEST_VLLM_USERNAME", "bob")
        monkeypatch.setenv("ORBIT2_TEST_VLLM_PASSWORD", "")

        resolved = resolve_vllm_provider_settings(tmp_path)

        assert resolved.base_url == "http://localhost:9000/v1"
        assert resolved.base_url_source == "env:ORBIT2_TEST_VLLM_BASE_URL"
        assert resolved.basic_auth_username == "bob"
        assert resolved.basic_auth_password == ""

    def test_vllm_settings_direct_values_read_runtime_config(
        self, clean_env, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / STORE_SUBDIR
        config_dir.mkdir()
        (config_dir / AGENT_RUNTIME_CONFIG_NAME).write_text(
            "\n".join([
                "[vllm]",
                "base_url_env = 'http://10.204.18.32:8080/v1/chat/completions'",
                "basic_auth_username_env = 'alice'",
                "basic_auth_password_env = 'secret'",
            ]),
            encoding="utf-8",
        )

        resolved = resolve_vllm_provider_settings(tmp_path)

        assert resolved.base_url == "http://10.204.18.32:8080/v1"
        assert resolved.base_url_source.endswith("[vllm].base_url_env")
        assert resolved.basic_auth_username == "alice"
        assert resolved.basic_auth_password == "secret"
        assert "[vllm].basic_auth_username_env" in (resolved.basic_auth_source or "")
