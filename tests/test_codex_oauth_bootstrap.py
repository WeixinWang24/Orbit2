from __future__ import annotations

import json
import stat
import time
import base64
from pathlib import Path

import pytest

from src.core.providers.codex import OAuthCredential
from src.core.providers import codex_oauth
from src.core.providers.codex_oauth import (
    CodexOAuthBootstrapError,
    codex_oauth_credential_status,
    create_codex_oauth_pkce_session,
    exchange_callback_input,
    generate_pkce_challenge,
    import_codex_cli_auth_credential,
    parse_codex_oauth_callback_input,
    refresh_codex_oauth_credential,
    save_codex_oauth_credential,
)
from src.operation.cli import codex_auth


def test_generate_pkce_challenge_uses_s256_base64url() -> None:
    verifier = "test-verifier"
    assert generate_pkce_challenge(verifier) == "JBbiqONGWPaAmwXk_8bT6UnlPfrn65D32eZlJS-zGG0"


def test_create_codex_oauth_pkce_session_contains_openai_parameters() -> None:
    session = create_codex_oauth_pkce_session(originator="orbit2-test")
    assert "https://auth.openai.com/oauth/authorize?" in session.authorize_url
    assert "client_id=app_EMoamEEZ73f0CkXaXp7hrann" in session.authorize_url
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A1455%2Fauth%2Fcallback" in session.authorize_url
    assert "code_challenge_method=S256" in session.authorize_url
    assert "originator=orbit2-test" in session.authorize_url
    assert session.code_challenge == generate_pkce_challenge(session.code_verifier)


def test_parse_codex_oauth_callback_input_extracts_code_and_state() -> None:
    parsed = parse_codex_oauth_callback_input(
        "http://localhost:1455/auth/callback?code=code-123&state=state-456"
    )
    assert parsed.code == "code-123"
    assert parsed.state == "state-456"


def test_exchange_callback_input_rejects_wrong_state() -> None:
    with pytest.raises(CodexOAuthBootstrapError, match="state"):
        exchange_callback_input(
            callback_input="http://localhost:1455/auth/callback?code=code-123&state=wrong",
            code_verifier="verifier",
            expected_state="expected",
        )


def test_exchange_callback_input_posts_form_and_normalizes_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post_form_json(url: str, form: dict[str, str], *, timeout_seconds: int) -> dict:
        assert url == codex_oauth.OPENAI_OAUTH_TOKEN_URL
        assert form["grant_type"] == "authorization_code"
        assert form["code"] == "code-123"
        assert form["code_verifier"] == "verifier"
        assert form["state"] == "state-456"
        assert timeout_seconds == 7
        return {
            "access_token": "access-new",
            "refresh_token": "refresh-new",
            "expires_in": 120,
            "email": "user@example.com",
        }

    monkeypatch.setattr(codex_oauth, "_post_form_json", fake_post_form_json)
    before_ms = int(time.time() * 1000)
    credential = exchange_callback_input(
        callback_input="http://localhost:1455/auth/callback?code=code-123&state=state-456",
        code_verifier="verifier",
        expected_state="state-456",
        timeout_seconds=7,
    )
    assert credential.access_token == "access-new"
    assert credential.refresh_token == "refresh-new"
    assert credential.account_email == "user@example.com"
    assert credential.expires_at_epoch_ms >= before_ms + 119_000


def test_refresh_codex_oauth_credential_preserves_refresh_token(monkeypatch: pytest.MonkeyPatch) -> None:
    original = OAuthCredential(
        access_token="access-old",
        refresh_token="refresh-old",
        expires_at_epoch_ms=1,
        account_email="user@example.com",
    )

    def fake_post_form_json(url: str, form: dict[str, str], *, timeout_seconds: int) -> dict:
        assert form == {
            "grant_type": "refresh_token",
            "client_id": codex_oauth.OPENAI_OAUTH_CLIENT_ID,
            "refresh_token": "refresh-old",
            "scope": codex_oauth.OPENAI_OAUTH_SCOPE,
        }
        return {"access_token": "access-new", "expires_at": 2_000}

    monkeypatch.setattr(codex_oauth, "_post_form_json", fake_post_form_json)
    refreshed = refresh_codex_oauth_credential(original)
    assert refreshed.access_token == "access-new"
    assert refreshed.refresh_token == "refresh-old"
    assert refreshed.expires_at_epoch_ms == 2_000_000
    assert refreshed.account_email == "user@example.com"


def test_save_codex_oauth_credential_writes_private_file_and_status(tmp_path: Path) -> None:
    path = tmp_path / ".runtime" / "openai_oauth_credentials.json"
    credential = OAuthCredential(
        access_token="access",
        refresh_token="refresh",
        expires_at_epoch_ms=int(time.time() * 1000) + 60_000,
        account_email=None,
    )
    save_codex_oauth_credential(path, credential)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["access_token"] == "access"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    status = codex_oauth_credential_status(path)
    assert status.exists is True
    assert status.expired is False


def test_import_codex_cli_auth_credential_reads_nested_tokens_and_jwt_expiry(tmp_path: Path) -> None:
    exp = int(time.time()) + 600
    access_token = _unsigned_jwt({"exp": exp})
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": "refresh-from-codex",
                    "account_id": "account-123",
                },
            }
        ),
        encoding="utf-8",
    )
    existing = OAuthCredential(
        access_token="old-access",
        refresh_token="old-refresh",
        expires_at_epoch_ms=1,
        account_email="user@example.com",
    )

    credential = import_codex_cli_auth_credential(auth_path, existing_credential=existing)

    assert credential.access_token == access_token
    assert credential.refresh_token == "refresh-from-codex"
    assert credential.expires_at_epoch_ms == exp * 1000
    assert credential.account_email == "user@example.com"


def test_import_codex_cli_auth_credential_accepts_root_tokens(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps({"access_token": "access-root", "refresh_token": "refresh-root"}),
        encoding="utf-8",
    )
    credential = import_codex_cli_auth_credential(auth_path)
    assert credential.access_token == "access-root"
    assert credential.refresh_token == "refresh-root"
    assert credential.expires_at_epoch_ms > int(time.time() * 1000)


def test_codex_auth_refresh_command_updates_default_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_root = tmp_path
    path = runtime_root / ".runtime" / "openai_oauth_credentials.json"
    save_codex_oauth_credential(
        path,
        OAuthCredential(
            access_token="access-old",
            refresh_token="refresh-old",
            expires_at_epoch_ms=1,
        ),
    )

    def fake_post_form_json(url: str, form: dict[str, str], *, timeout_seconds: int) -> dict:
        return {
            "access_token": "access-new",
            "refresh_token": "refresh-new",
            "expires_at": 2_000,
        }

    monkeypatch.setattr(codex_oauth, "_post_form_json", fake_post_form_json)
    exit_code = codex_auth.main(["--runtime-root", str(runtime_root), "refresh", "--json"])
    assert exit_code == 0
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["access_token"] == "access-new"
    assert stored["refresh_token"] == "refresh-new"
    output = json.loads(capsys.readouterr().out)
    assert output["source"] == "refresh"
    assert output["credential_path"] == str(path)


def test_codex_auth_import_codex_auth_command_updates_default_credential(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_root = tmp_path / "runtime"
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": _unsigned_jwt({"exp": 2_000}),
                    "refresh_token": "refresh-from-codex",
                }
            }
        ),
        encoding="utf-8",
    )
    exit_code = codex_auth.main(
        [
            "--runtime-root",
            str(runtime_root),
            "import-codex-auth",
            "--auth-path",
            str(auth_path),
            "--json",
        ]
    )
    assert exit_code == 0
    credential_path = runtime_root / ".runtime" / "openai_oauth_credentials.json"
    stored = json.loads(credential_path.read_text(encoding="utf-8"))
    assert stored["refresh_token"] == "refresh-from-codex"
    output = json.loads(capsys.readouterr().out)
    assert output["source"] == "import-codex-auth"
    assert "refresh-from-codex" not in json.dumps(output)


def _unsigned_jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}
    return ".".join(
        [
            _base64url_json(header),
            _base64url_json(payload),
            "",
        ]
    )


def _base64url_json(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")
