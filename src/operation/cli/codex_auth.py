from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src.config.runtime import resolve_runtime_root
from src.core.providers.codex_oauth import (
    CodexOAuthBootstrapError,
    DEFAULT_CODEX_AUTH_PATH,
    codex_oauth_credential_status,
    create_codex_oauth_pkce_session,
    credential_path_for_repo,
    exchange_callback_input,
    import_codex_cli_auth_credential,
    load_saved_codex_oauth_credential,
    refresh_codex_oauth_credential,
    save_codex_oauth_credential,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    runtime_root = resolve_runtime_root(args.runtime_root)
    credential_path = credential_path_for_repo(runtime_root.path, args.credential_path)

    try:
        if args.command == "status":
            _print_status(credential_path, json_output=args.json)
            return 0
        if args.command == "login-url":
            session = create_codex_oauth_pkce_session(originator=args.originator)
            payload = {
                "authorize_url": session.authorize_url,
                "state": session.state,
                "code_verifier": session.code_verifier,
                "credential_path": str(credential_path),
            }
            _print_payload(payload, json_output=args.json)
            return 0
        if args.command == "exchange":
            credential = exchange_callback_input(
                callback_input=args.callback,
                code_verifier=args.code_verifier,
                expected_state=args.state,
                timeout_seconds=args.timeout_seconds,
            )
            save_codex_oauth_credential(credential_path, credential)
            _print_bootstrap_result("exchange", credential_path, credential, json_output=args.json)
            return 0
        if args.command == "refresh":
            credential = load_saved_codex_oauth_credential(credential_path)
            refreshed = refresh_codex_oauth_credential(
                credential,
                timeout_seconds=args.timeout_seconds,
            )
            save_codex_oauth_credential(credential_path, refreshed)
            _print_bootstrap_result("refresh", credential_path, refreshed, json_output=args.json)
            return 0
        if args.command == "import-codex-auth":
            existing = None
            if credential_path.exists():
                existing = load_saved_codex_oauth_credential(credential_path)
            credential = import_codex_cli_auth_credential(
                Path(args.auth_path).expanduser(),
                existing_credential=existing,
            )
            save_codex_oauth_credential(credential_path, credential)
            _print_bootstrap_result("import-codex-auth", credential_path, credential, json_output=args.json)
            return 0
    except CodexOAuthBootstrapError as exc:
        parser.exit(1, f"codex credential bootstrap failed: {exc}\n")

    parser.error(f"unknown command: {args.command}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap Orbit2 OpenAI Codex OAuth credentials")
    parser.add_argument(
        "--runtime-root",
        default=None,
        help="Runtime root containing .runtime/openai_oauth_credentials.json",
    )
    parser.add_argument(
        "--credential-path",
        default=None,
        help="Credential file path. Relative paths are resolved under runtime root.",
    )
    parser.set_defaults(command="status")

    subparsers = parser.add_subparsers(dest="command")
    status = subparsers.add_parser("status", help="Show credential status without printing tokens")
    status.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    login_url = subparsers.add_parser("login-url", help="Create an OpenAI OAuth PKCE login URL")
    login_url.add_argument("--originator", default="orbit2", help="OAuth originator parameter")
    login_url.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    exchange = subparsers.add_parser("exchange", help="Persist credentials from a pasted OAuth callback URL")
    exchange.add_argument("--callback", required=True, help="Callback URL, callback query, or raw authorization code")
    exchange.add_argument("--code-verifier", required=True, help="PKCE code verifier emitted by login-url")
    exchange.add_argument("--state", default=None, help="Expected state emitted by login-url")
    exchange.add_argument("--timeout-seconds", type=int, default=60)
    exchange.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    refresh = subparsers.add_parser("refresh", help="Refresh the persisted credential using its refresh token")
    refresh.add_argument("--timeout-seconds", type=int, default=30)
    refresh.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    import_codex_auth = subparsers.add_parser(
        "import-codex-auth",
        help="Copy access_token and refresh_token from the current Codex CLI auth file",
    )
    import_codex_auth.add_argument(
        "--auth-path",
        default=DEFAULT_CODEX_AUTH_PATH,
        help="Codex auth JSON path (default: ~/.codex/auth.json)",
    )
    import_codex_auth.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser


def _print_status(path: Path, *, json_output: bool) -> None:
    status = codex_oauth_credential_status(path)
    payload = {
        "credential_path": str(status.credential_path),
        "exists": status.exists,
        "expired": status.expired,
        "account_email": status.account_email,
        "expires_at": _format_epoch_ms(status.expires_at_epoch_ms),
    }
    _print_payload(payload, json_output=json_output)


def _print_bootstrap_result(
    source: str,
    path: Path,
    credential,
    *,
    json_output: bool,
) -> None:
    payload = {
        "source": source,
        "credential_path": str(path),
        "account_email": credential.account_email,
        "expires_at": _format_epoch_ms(credential.expires_at_epoch_ms),
    }
    _print_payload(payload, json_output=json_output)


def _print_payload(payload: dict, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


def _format_epoch_ms(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
