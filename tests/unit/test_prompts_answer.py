"""双语提示词模板测试（T-022）。

覆盖三件事：语言选择接口的行为、拒答标记在两版提示词里都存在、
`template_version()` 对四份提示词内容都敏感（换任一语言的任一份都要能看出差异）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.prompts import answer as ap  # noqa: E402


# ---------- 语言选择 ----------

def test_system_prompt_returns_distinct_text_per_language():
    zh = ap.system_prompt("zh")
    en = ap.system_prompt("en")
    assert zh != en
    assert isinstance(zh, str) and isinstance(en, str)


def test_insufficient_prompt_returns_distinct_text_per_language():
    zh = ap.insufficient_prompt("zh")
    en = ap.insufficient_prompt("en")
    assert zh != en


def test_system_prompt_defaults_to_zh():
    assert ap.system_prompt() == ap.system_prompt("zh")


def test_unsupported_language_raises():
    with pytest.raises(ValueError, match="fr"):
        ap.system_prompt("fr")
    with pytest.raises(ValueError, match="fr"):
        ap.insufficient_prompt("fr")


def test_supported_languages_are_zh_and_en():
    assert ap.SUPPORTED_LANGUAGES == ("zh", "en")


# ---------- 拒答标记 ----------

@pytest.mark.parametrize("language", ap.SUPPORTED_LANGUAGES)
def test_system_prompt_instructs_decline_token(language):
    """两版系统提示词都必须要求模型在证据不支撑结论时输出这个固定标记——
    否则 `answering.declined()` 在对应语言下永远判不出拒答。
    """
    assert ap.DECLINE_TOKEN in ap.system_prompt(language)


@pytest.mark.parametrize("language", ap.SUPPORTED_LANGUAGES)
def test_insufficient_prompt_instructs_decline_token(language):
    assert ap.DECLINE_TOKEN in ap.insufficient_prompt(language)


def test_decline_token_is_language_agnostic_ascii():
    """标记本身不随语言变化——这是它能被 `str.startswith` 判定的前提。"""
    assert ap.DECLINE_TOKEN.isascii()
    assert ap.DECLINE_TOKEN == "NO_EVIDENCE"


# ---------- 引用要求：六段式作废，但引用格式约束保留 ----------

@pytest.mark.parametrize("language", ap.SUPPORTED_LANGUAGES)
def test_system_prompt_still_requires_citations(language):
    """scope.md §3 废的是强制六段模板，不是证据可追溯承诺——引用要求必须还在。"""
    text = ap.system_prompt(language)
    assert "[1]" in text  # 提示词自带的引用格式示例


def test_system_prompt_zh_does_not_force_six_fixed_section_headers():
    """旧模板把这六个词当**固定分节标题**（后接全角冒号，如"适用前提："），
    新提示词允许在正文里自然提到"失败模式"这类概念（例如作为风险的举例），
    但不能再要求模型逐题输出这六个标题——检查的是标题形态，不是禁用这些词本身。
    """
    text = ap.system_prompt("zh")
    for banned_heading in ("适用前提：", "实施步骤：", "失败模式：", "监控与验证：", "来源："):
        assert banned_heading not in text


def test_system_prompt_en_does_not_force_six_fixed_section_headers():
    text = ap.system_prompt("en")
    for banned_heading in (
        "Applicable precondition:", "Implementation steps:",
        "Failure mode:", "Monitoring and verification:", "Sources:",
    ):
        assert banned_heading not in text


# ---------- render_user_message ----------

def test_render_user_message_uses_chinese_labels_by_default():
    msg = ap.render_user_message("问题文本", [])
    assert msg.startswith("【证据】")
    assert "【问题】问题文本" in msg


def test_render_user_message_uses_english_labels():
    msg = ap.render_user_message("question text", [], "en")
    assert msg.startswith("[Evidence]")
    assert "[Question]question text" in msg
    assert "【" not in msg


def test_render_user_message_rejects_unsupported_language():
    with pytest.raises(ValueError):
        ap.render_user_message("q", [], "fr")


def test_render_evidence_numbering_unaffected_by_language():
    """引用编号是语言无关的锚点——中英文渲染的证据编号必须一致。"""
    items = [
        ap.Evidence(1, "text one", "citation one", "https://x/1"),
        ap.Evidence(2, "text two", "citation two", "https://x/2"),
    ]
    zh = ap.render_user_message("q", items, "zh")
    en = ap.render_user_message("q", items, "en")
    for rendered in (zh, en):
        assert "[1] citation one" in rendered
        assert "[2] citation two" in rendered


# ---------- template_version ----------

def test_template_version_is_stable_hex_digest():
    v = ap.template_version()
    assert len(v) == 12
    int(v, 16)  # 必须是合法十六进制
    assert v == ap.template_version()  # 确定性


def test_template_version_changes_with_any_single_prompt(monkeypatch):
    """换任一语言的任一份提示词都要能在 template_version() 上看出差异——
    这是它存在的唯一目的：评测报告要能追溯"是哪次提示词改动导致质量变化"。
    """
    baseline = ap.template_version()
    for attr in (
        "_SYSTEM_PROMPT_ZH", "_SYSTEM_PROMPT_EN",
        "_INSUFFICIENT_PROMPT_ZH", "_INSUFFICIENT_PROMPT_EN",
    ):
        monkeypatch.setattr(ap, attr, getattr(ap, attr) + " 改动")
        assert ap.template_version() != baseline, f"{attr} 改动未反映到 template_version()"
        monkeypatch.undo()
