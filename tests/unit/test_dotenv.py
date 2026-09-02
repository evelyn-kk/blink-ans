"""仓库 `.env` 加载的行为回归（T-107）。"""

from __future__ import annotations

from pathlib import Path

from packages.config.env import load_dotenv, missing_credential_message


def write_env(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_loads_values_and_parses_comments_blank_lines_and_quotes(tmp_path):
    path = tmp_path / ".env"
    write_env(path, "# comment\n\nA=plain\nB='quoted value'\nC=\"double quoted\"\n")
    env: dict[str, str] = {}

    state = load_dotenv(path, env)

    assert env == {"A": "plain", "B": "quoted value", "C": "double quoted"}
    assert state.loaded == frozenset({"A", "B", "C"})


def test_exported_value_is_not_overwritten_even_when_the_file_has_a_value(tmp_path):
    path = tmp_path / ".env"
    write_env(path, "A=from-file\n")
    env = {"A": "from-terminal"}

    state = load_dotenv(path, env)

    assert env["A"] == "from-terminal"
    assert not state.loaded


def test_missing_file_is_silent_and_explains_the_actual_reason(tmp_path):
    state = load_dotenv(tmp_path / ".env", {})

    assert not state.exists
    assert missing_credential_message("A", state, {}) == "未找到 .env，且环境变量 A 未设置"


def test_distinguishes_missing_variable_from_empty_value(tmp_path):
    path = tmp_path / ".env"
    write_env(path, "PRESENT=value\nEMPTY=''\n")
    env: dict[str, str] = {}
    state = load_dotenv(path, env)

    assert missing_credential_message("MISSING", state, env) == ".env 未声明 MISSING，且环境变量未设置"
    assert missing_credential_message("EMPTY", state, env) == ".env 中的 EMPTY 值为空"


def test_explicitly_empty_export_is_not_overwritten_or_misreported(tmp_path):
    path = tmp_path / ".env"
    write_env(path, "A=from-file\n")
    env = {"A": ""}

    state = load_dotenv(path, env)

    assert env["A"] == ""
    assert missing_credential_message("A", state, env) == "环境变量 A 已设置但值为空"


def test_diagnostic_never_contains_secret_value(tmp_path):
    path = tmp_path / ".env"
    secret = "this-must-never-appear"
    write_env(path, f"A={secret}\n")
    state = load_dotenv(path, {})

    message = missing_credential_message("MISSING", state, {})
    assert secret not in message
