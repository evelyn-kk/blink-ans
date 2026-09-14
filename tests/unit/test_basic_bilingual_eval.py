"""T-023：基础回归的双语题面、事实/来源断言与分语言报告。"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages" / "evaltools"))

import run_basic as basic  # noqa: E402


class _FakeOrchestrator:
    def __init__(self, answer: str, citation: str = "kafka 4.0 · test"):
        self.answer_text = answer
        self.citation = citation
        self.requests = []

    def answer_events(self, request):
        self.requests.append(request)
        yield {"type": "retrieval", "sufficiency": "sufficient"}
        yield {"type": "sources", "items": [{"url": "https://example.test", "citation": self.citation}]}
        yield {"type": "answer_delta", "text": self.answer_text}
        yield {"type": "done", "cited_evidence": [1], "ttft_s": 0.1, "total_s": 0.2,
               "prompt_tokens": 10, "evidence_count": 1, "served_by": "local"}

    def answer(self, request):
        return self.answer_events(request)


_SPEC = {
    "id": "fact-and-source",
    "q_zh": "中文题面",
    "q_en": "An independently authored English question.",
    "expect": "answered",
    "expect_keypoints": [r"(?i)idempotent"],
    "expect_sources": ["kafka"],
}


def _legacy_structural_failures(answer: str, citation: str) -> list[str]:
    """T-022 旧行为的最小复现：有来源、有引用便会放行，不检查事实。"""
    return [] if answer and citation.split(" ", 1)[0] == "kafka" else ["structural failure"]


def test_fact_assertion_rejects_a_cited_answer_that_old_structural_gate_accepted():
    """判别性：旧实现会因结构正确而放行；新实现因缺事实而失败。"""
    answer = "A Kafka client can use retries. [1]"
    assert _legacy_structural_failures(answer, "kafka 4.0 · test") == []

    case = basic.run_case(_FakeOrchestrator(answer), _SPEC, "en")

    assert case.question == _SPEC["q_en"]
    assert case.keypoints_missed == [r"(?i)idempotent"]
    assert any("未命中关键事实" in failure for failure in case.failures)


def test_source_assertion_rejects_the_wrong_cited_project():
    case = basic.run_case(
        _FakeOrchestrator("The producer must be idempotent. [1]", "postgresql 18 · test"),
        _SPEC, "en",
    )
    assert case.sources_missed == ["kafka"]
    assert any("缺少期望来源" in failure for failure in case.failures)


def test_language_summary_keeps_results_separate():
    zh = basic.Case(id="one", language="zh", question="中文", expect="answered")
    en = basic.Case(id="one", language="en", question="English", expect="answered")
    en.failures.append("fact missing")

    grouped = basic.summarize_by_language([zh, en])

    assert grouped["zh"]["passed"] == 1
    assert grouped["en"]["passed"] == 0
    assert grouped["zh"]["cases"][0]["question"] == "中文"
    assert grouped["en"]["cases"][0]["question"] == "English"


def test_basic_dataset_preserves_the_original_chinese_baseline_and_has_real_english_fields():
    questions = yaml.safe_load((ROOT / "knowledge/eval/basic_questions.yaml").read_text(encoding="utf-8"))["questions"]
    joined_zh = "\n".join(case["q_zh"] for case in questions)

    # This pins the exact 50 I2 Chinese strings, not merely their count. It
    # would fail if migration silently rewrote a Chinese baseline question.
    assert hashlib.sha256(joined_zh.encode()).hexdigest() == (
        "bfc2cb81b9953c3d061db2f772de40ae1b2521ece1a87237f0b7e0d1d36b3dcd"
    )
    assert len(questions) == 50
    for case in questions:
        assert case["q_en"] and case["q_en"] != case["q_zh"]
        assert any(char.isascii() and char.isalpha() for char in case["q_en"])
    basic.validate_specs(questions)


def test_schema_does_not_fallback_to_chinese_when_the_english_question_is_missing():
    broken = dict(_SPEC)
    broken.pop("q_en")
    try:
        basic.validate_specs([broken])
    except ValueError as exc:
        assert "q_en" in str(exc)
    else:
        raise AssertionError("missing q_en must fail closed")
