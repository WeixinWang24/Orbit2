from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUNTIME_ROOT_ENV = "ORBIT2_RUNTIME_ROOT"
DEFAULT_PROVIDER_MODEL_ENV = "ORBIT2_PROVIDER_MODEL"
DEFAULT_PROVIDER_MODEL = "gpt-5.5"
DEFAULT_VLLM_BASE_URL_ENV = "ORBIT2_VLLM_BASE_URL"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_VLLM_API_KEY_ENV = "ORBIT2_VLLM_API_KEY"
DEFAULT_VLLM_USERNAME_ENV = "ORBIT2_VLLM_USERNAME"
DEFAULT_VLLM_PASSWORD_ENV = "ORBIT2_VLLM_PASSWORD"
STORE_SUBDIR = ".runtime"
DEFAULT_DB_NAME = "sessions.db"
DEFAULT_CODE_INTEL_DB_NAME = "code_intel.db"
AGENT_RUNTIME_CONFIG_NAME = "agent_runtime.toml"
MAX_TOOL_TURNS = 10

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class RuntimeRoot:
    path: Path
    source: str  # "cli-flag" | "env:ORBIT2_RUNTIME_ROOT" | "repo-root"


@dataclass(frozen=True)
class RuntimePathSetting:
    path: Path
    source: str


@dataclass(frozen=True)
class RuntimeEnvSetting:
    env_name: str
    value: str
    source: str


@dataclass(frozen=True)
class VllmProviderSettings:
    base_url: str
    base_url_source: str
    api_key: str | None = None
    api_key_source: str | None = None
    basic_auth_username: str | None = None
    basic_auth_password: str | None = None
    basic_auth_source: str | None = None


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


def code_intel_db_path(runtime_root: Path | RuntimeRoot) -> Path:
    path = runtime_root.path if isinstance(runtime_root, RuntimeRoot) else runtime_root
    store_dir = path / STORE_SUBDIR
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir / DEFAULT_CODE_INTEL_DB_NAME


def runtime_config_path(runtime_root: Path | RuntimeRoot) -> Path:
    path = runtime_root.path if isinstance(runtime_root, RuntimeRoot) else runtime_root
    return path / STORE_SUBDIR / AGENT_RUNTIME_CONFIG_NAME


def resolve_obsidian_vault_root(
    runtime_root: Path | RuntimeRoot,
    override: Path | str | None = None,
) -> RuntimePathSetting | None:
    if override is not None and str(override).strip():
        return RuntimePathSetting(
            Path(override).expanduser().resolve(),
            "cli-flag",
        )

    config_path = runtime_config_path(runtime_root)
    if not config_path.exists():
        return None

    data = _read_runtime_config(config_path)
    obsidian = data.get("obsidian")
    if not isinstance(obsidian, dict):
        return None
    raw = obsidian.get("vault_root")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return RuntimePathSetting(
        Path(raw).expanduser().resolve(),
        f"config:{config_path}",
    )


def resolve_provider_model(runtime_root: Path | RuntimeRoot) -> RuntimeEnvSetting:
    config_path = runtime_config_path(runtime_root)
    env_name = DEFAULT_PROVIDER_MODEL_ENV
    source = f"default:{DEFAULT_PROVIDER_MODEL_ENV}"
    if config_path.exists():
        data = _read_runtime_config(config_path)
        provider = data.get("provider")
        if isinstance(provider, dict):
            direct_model = _config_direct_value(provider, "model")
            if direct_model is not None:
                return RuntimeEnvSetting(
                    env_name=env_name,
                    value=direct_model,
                    source=f"config:{config_path} [provider].model",
                )
            model_env_or_value = _config_direct_value(provider, "model_env")
            if model_env_or_value is not None:
                if not _looks_like_env_name(model_env_or_value):
                    return RuntimeEnvSetting(
                        env_name=env_name,
                        value=model_env_or_value,
                        source=f"config:{config_path} [provider].model_env",
                    )
                env_name = model_env_or_value
                source = f"config:{config_path}"

    value = os.environ.get(env_name, "").strip()
    if value:
        return RuntimeEnvSetting(env_name=env_name, value=value, source=source)
    return RuntimeEnvSetting(
        env_name=env_name,
        value=DEFAULT_PROVIDER_MODEL,
        source=f"default:{DEFAULT_PROVIDER_MODEL}",
    )


def resolve_vllm_provider_settings(
    runtime_root: Path | RuntimeRoot,
) -> VllmProviderSettings:
    config_path = runtime_config_path(runtime_root)
    vllm_config = _runtime_config_section(config_path, "vllm")

    base_url_value = _config_direct_value(vllm_config, "base_url")
    base_url_env_or_value = _config_env_name(
        vllm_config,
        "base_url_env",
        DEFAULT_VLLM_BASE_URL_ENV,
    )
    if base_url_value is not None:
        base_url = base_url_value
        base_url_source = f"config:{config_path} [vllm].base_url"
    elif not _looks_like_env_name(base_url_env_or_value):
        base_url = base_url_env_or_value
        base_url_source = f"config:{config_path} [vllm].base_url_env"
    else:
        base_url = os.environ.get(base_url_env_or_value, "").strip()
        base_url_source = f"env:{base_url_env_or_value}"
    if not base_url:
        base_url = DEFAULT_VLLM_BASE_URL
        base_url_source = f"default:{DEFAULT_VLLM_BASE_URL}"

    api_key_value = _config_direct_value(vllm_config, "api_key")
    api_key_env_or_value = _config_env_name(
        vllm_config,
        "api_key_env",
        DEFAULT_VLLM_API_KEY_ENV,
    )
    if api_key_value is not None:
        api_key = api_key_value
        api_key_source = f"config:{config_path} [vllm].api_key"
    elif not _looks_like_env_name(api_key_env_or_value):
        api_key = api_key_env_or_value
        api_key_source = f"config:{config_path} [vllm].api_key_env"
    else:
        api_key = os.environ.get(api_key_env_or_value, "").strip() or None
        api_key_source = f"env:{api_key_env_or_value}" if api_key else None

    username_value = _config_direct_value(vllm_config, "basic_auth_username")
    password_value = _config_direct_value(vllm_config, "basic_auth_password")
    username_env_or_value = _config_env_name(
        vllm_config,
        "basic_auth_username_env",
        DEFAULT_VLLM_USERNAME_ENV,
    )
    password_env_or_value = _config_env_name(
        vllm_config,
        "basic_auth_password_env",
        DEFAULT_VLLM_PASSWORD_ENV,
    )
    if username_value is not None:
        username = username_value
        username_source = f"config:{config_path} [vllm].basic_auth_username"
    elif not _looks_like_env_name(username_env_or_value):
        username = username_env_or_value
        username_source = f"config:{config_path} [vllm].basic_auth_username_env"
    else:
        username = os.environ.get(username_env_or_value, "").strip() or None
        username_source = f"env:{username_env_or_value}" if username else None

    if password_value is not None:
        password = password_value
        password_source = f"config:{config_path} [vllm].basic_auth_password"
    elif not _looks_like_env_name(password_env_or_value):
        password = password_env_or_value
        password_source = f"config:{config_path} [vllm].basic_auth_password_env"
    else:
        password = os.environ.get(password_env_or_value, "")
        password_source = f"env:{password_env_or_value}" if username is not None else None

    basic_auth_source = None
    if username is not None:
        basic_auth_source = ",".join(
            part for part in (username_source, password_source) if part
        )

    return VllmProviderSettings(
        base_url=_normalize_openai_base_url(base_url),
        base_url_source=base_url_source,
        api_key=api_key,
        api_key_source=api_key_source,
        basic_auth_username=username,
        basic_auth_password=password if username is not None else None,
        basic_auth_source=basic_auth_source,
    )


def _runtime_config_section(path: Path, name: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = _read_runtime_config(path)
    section = data.get(name)
    return section if isinstance(section, dict) else {}


def _config_env_name(
    section: dict[str, Any],
    key: str,
    default: str,
) -> str:
    raw = section.get(key)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return default


def _config_direct_value(section: dict[str, Any], key: str) -> str | None:
    raw = section.get(key)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _looks_like_env_name(value: str) -> bool:
    if not value:
        return False
    first = value[0]
    return (
        (first.isalpha() or first == "_")
        and all(c.isupper() or c.isdigit() or c == "_" for c in value)
    )


def _normalize_openai_base_url(value: str) -> str:
    url = value.strip().rstrip("/")
    suffix = "/chat/completions"
    if url.endswith(suffix):
        url = url[: -len(suffix)]
    return url


def _read_runtime_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return data if isinstance(data, dict) else {}
