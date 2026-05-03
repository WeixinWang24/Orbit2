from __future__ import annotations

import main as main_module


def test_main_dispatches_codex_auth() -> None:
    called: dict[str, list[str]] = {}

    def fake_codex_auth(argv):
        called["argv"] = list(argv)
        return 17

    original = main_module.codex_auth_main
    try:
        main_module.codex_auth_main = fake_codex_auth
        result = main_module.main(["codex-auth", "import-codex-auth", "--json"])
    finally:
        main_module.codex_auth_main = original

    assert result == 17
    assert called["argv"] == ["import-codex-auth", "--json"]


def test_main_dispatches_auth_alias() -> None:
    called: dict[str, list[str]] = {}

    def fake_codex_auth(argv):
        called["argv"] = list(argv)
        return 0

    original = main_module.codex_auth_main
    try:
        main_module.codex_auth_main = fake_codex_auth
        result = main_module.main(["auth", "status"])
    finally:
        main_module.codex_auth_main = original

    assert result == 0
    assert called["argv"] == ["status"]


def test_main_defaults_to_chat_harness() -> None:
    called: dict[str, list[str]] = {}

    def fake_harness(argv):
        called["argv"] = list(argv)

    original = main_module.harness_main
    try:
        main_module.harness_main = fake_harness
        result = main_module.main(["--list-sessions"])
    finally:
        main_module.harness_main = original

    assert result == 0
    assert called["argv"] == ["--list-sessions"]
