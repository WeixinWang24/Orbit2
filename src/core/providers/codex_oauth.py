from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from src.core.providers.codex import DEFAULT_CREDENTIAL_PATH, OAuthCredential

OPENAI_OAUTH_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
OPENAI_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_OAUTH_REDIRECT_URI = "http://localhost:1455/auth/callback"
OPENAI_OAUTH_SCOPE = "openid profile email offline_access"
DEFAULT_CODEX_AUTH_PATH = "~/.codex/auth.json"


class CodexOAuthBootstrapError(RuntimeError):
    pass


@dataclass(frozen=True)
class CodexOAuthPkceSession:
    originator: str
    state: str
    code_verifier: str
    code_challenge: str
    authorize_url: str
    redirect_uri: str = OPENAI_OAUTH_REDIRECT_URI
    client_id: str = OPENAI_OAUTH_CLIENT_ID
    scope: str = OPENAI_OAUTH_SCOPE
    authorize_url_base: str = OPENAI_OAUTH_AUTHORIZE_URL
    token_url: str = OPENAI_OAUTH_TOKEN_URL


@dataclass(frozen=True)
class ParsedCodexOAuthCallback:
    code: str
    state: str | None


@dataclass(frozen=True)
class CodexOAuthCredentialStatus:
    credential_path: Path
    exists: bool
    expired: bool | None = None
    account_email: str | None = None
    expires_at_epoch_ms: int | None = None


def credential_path_for_repo(repo_root: Path, credential_path: str | Path | None = None) -> Path:
    path = Path(credential_path or DEFAULT_CREDENTIAL_PATH).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


def generate_pkce_verifier(length_bytes: int = 48) -> str:
    return secrets.token_urlsafe(length_bytes)


def generate_pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def create_codex_oauth_pkce_session(originator: str = "orbit2") -> CodexOAuthPkceSession:
    state = secrets.token_urlsafe(24)
    code_verifier = generate_pkce_verifier()
    code_challenge = generate_pkce_challenge(code_verifier)
    query = urlencode(
        {
            "response_type": "code",
            "client_id": OPENAI_OAUTH_CLIENT_ID,
            "redirect_uri": OPENAI_OAUTH_REDIRECT_URI,
            "scope": OPENAI_OAUTH_SCOPE,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "originator": originator,
        }
    )
    return CodexOAuthPkceSession(
        originator=originator,
        state=state,
        code_verifier=code_verifier,
        code_challenge=code_challenge,
        authorize_url=f"{OPENAI_OAUTH_AUTHORIZE_URL}?{query}",
    )


def parse_codex_oauth_callback_input(callback_input: str) -> ParsedCodexOAuthCallback:
    text = callback_input.strip()
    if not text:
        raise CodexOAuthBootstrapError("callback input is empty")
    if text.startswith("http://") or text.startswith("https://"):
        parsed = urlparse(text)
        query = parse_qs(parsed.query)
    elif "code=" in text:
        query = parse_qs(text.lstrip("?"))
    else:
        return ParsedCodexOAuthCallback(code=text, state=None)
    code = query.get("code", [None])[0]
    state = query.get("state", [None])[0]
    error = query.get("error", [None])[0]
    error_description = query.get("error_description", [None])[0]
    if error:
        detail = f": {error_description}" if error_description else ""
        raise CodexOAuthBootstrapError(f"callback returned OAuth error {error}{detail}")
    if not code:
        raise CodexOAuthBootstrapError("could not extract authorization code from callback input")
    return ParsedCodexOAuthCallback(code=code, state=state)


def exchange_codex_authorization_code(
    *,
    code: str,
    code_verifier: str,
    state: str | None = None,
    timeout_seconds: int = 60,
) -> OAuthCredential:
    form = {
        "grant_type": "authorization_code",
        "client_id": OPENAI_OAUTH_CLIENT_ID,
        "redirect_uri": OPENAI_OAUTH_REDIRECT_URI,
        "code": code,
        "code_verifier": code_verifier,
    }
    if state:
        form["state"] = state
    payload = _post_form_json(OPENAI_OAUTH_TOKEN_URL, form, timeout_seconds=timeout_seconds)
    return credential_from_token_payload(payload)


def exchange_callback_input(
    *,
    callback_input: str,
    code_verifier: str,
    expected_state: str | None = None,
    timeout_seconds: int = 60,
) -> OAuthCredential:
    parsed = parse_codex_oauth_callback_input(callback_input)
    if expected_state and parsed.state and parsed.state != expected_state:
        raise CodexOAuthBootstrapError("callback state did not match the PKCE session state")
    return exchange_codex_authorization_code(
        code=parsed.code,
        code_verifier=code_verifier,
        state=parsed.state,
        timeout_seconds=timeout_seconds,
    )


def refresh_codex_oauth_credential(
    credential: OAuthCredential,
    *,
    timeout_seconds: int = 30,
) -> OAuthCredential:
    if not credential.refresh_token.strip():
        raise CodexOAuthBootstrapError("credential has no refresh token")
    payload = _post_form_json(
        OPENAI_OAUTH_TOKEN_URL,
        {
            "grant_type": "refresh_token",
            "client_id": OPENAI_OAUTH_CLIENT_ID,
            "refresh_token": credential.refresh_token,
            "scope": OPENAI_OAUTH_SCOPE,
        },
        timeout_seconds=timeout_seconds,
    )
    return credential_from_token_payload(
        payload,
        previous_refresh_token=credential.refresh_token,
        account_email=credential.account_email,
    )


def import_codex_cli_auth_credential(
    auth_path: Path,
    *,
    existing_credential: OAuthCredential | None = None,
) -> OAuthCredential:
    if not auth_path.exists():
        raise CodexOAuthBootstrapError(f"Codex auth file not found: {auth_path}")
    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CodexOAuthBootstrapError(f"Codex auth file is not valid JSON: {exc}") from exc
    token_payload = payload.get("tokens") if isinstance(payload, dict) else None
    if not isinstance(token_payload, dict):
        token_payload = payload
    if not isinstance(token_payload, dict):
        raise CodexOAuthBootstrapError("Codex auth file must contain a JSON object")

    access_token = token_payload.get("access_token")
    refresh_token = token_payload.get("refresh_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise CodexOAuthBootstrapError("Codex auth file is missing access_token")
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise CodexOAuthBootstrapError("Codex auth file is missing refresh_token")

    account_email = existing_credential.account_email if existing_credential else None
    expires_at_epoch_ms = _jwt_expiry_epoch_ms(access_token)
    if expires_at_epoch_ms is None and existing_credential is not None:
        expires_at_epoch_ms = existing_credential.expires_at_epoch_ms
    if expires_at_epoch_ms is None:
        expires_at_epoch_ms = int((time.time() + 3600) * 1000)

    return OAuthCredential(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at_epoch_ms=expires_at_epoch_ms,
        account_email=account_email,
    )


def credential_from_token_payload(
    payload: dict,
    *,
    previous_refresh_token: str | None = None,
    account_email: str | None = None,
) -> OAuthCredential:
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise CodexOAuthBootstrapError("token response missing access_token")
    refresh_token = payload.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        refresh_token = previous_refresh_token
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise CodexOAuthBootstrapError("token response missing refresh_token")
    expires_at_epoch_ms = _expires_at_epoch_ms_from_payload(payload)
    payload_email = payload.get("email") or payload.get("account_email")
    if isinstance(payload_email, str) and payload_email.strip():
        account_email = payload_email
    return OAuthCredential(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at_epoch_ms=expires_at_epoch_ms,
        account_email=account_email,
    )


def load_saved_codex_oauth_credential(path: Path) -> OAuthCredential:
    if not path.exists():
        raise CodexOAuthBootstrapError(f"OAuth credential file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CodexOAuthBootstrapError(f"OAuth credential file is not valid JSON: {exc}") from exc
    try:
        return OAuthCredential(**data)
    except Exception as exc:
        raise CodexOAuthBootstrapError(f"OAuth credential file is missing required fields: {exc}") from exc


def save_codex_oauth_credential(path: Path, credential: OAuthCredential) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(credential.model_dump(), indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def codex_oauth_credential_status(path: Path) -> CodexOAuthCredentialStatus:
    if not path.exists():
        return CodexOAuthCredentialStatus(credential_path=path, exists=False)
    credential = load_saved_codex_oauth_credential(path)
    now_ms = int(time.time() * 1000)
    return CodexOAuthCredentialStatus(
        credential_path=path,
        exists=True,
        expired=credential.expires_at_epoch_ms <= now_ms,
        account_email=credential.account_email,
        expires_at_epoch_ms=credential.expires_at_epoch_ms,
    )


def _expires_at_epoch_ms_from_payload(payload: dict) -> int:
    expires_at = payload.get("expires_at")
    if expires_at is not None:
        return int(expires_at) * 1000
    expires_in = payload.get("expires_in")
    if expires_in is not None:
        return int((time.time() + int(expires_in)) * 1000)
    return int((time.time() + 3600) * 1000)


def _jwt_expiry_epoch_ms(token: str) -> int | None:
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload_segment = parts[1]
    padding = "=" * (-len(payload_segment) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_segment + padding))
    except Exception:
        return None
    exp = payload.get("exp")
    if isinstance(exp, int | float):
        return int(exp * 1000)
    return None


def _post_form_json(url: str, form: dict[str, str], *, timeout_seconds: int) -> dict:
    body = urlencode(form).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise CodexOAuthBootstrapError(f"OAuth token request failed ({exc.code}): {body_text}") from exc
    except Exception as exc:
        raise CodexOAuthBootstrapError(f"OAuth token request failed: {exc}") from exc
