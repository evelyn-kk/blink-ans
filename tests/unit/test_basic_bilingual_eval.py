"""T-023：基础回归的双语题面、事实/来源断言与分语言报告。"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

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
        yield {"type": "sources", "items": [{
            "index": 1, "chunk_id": 101, "url": "https://example.test", "citation": self.citation,
            "text_sha256": hashlib.sha256(b"fake source evidence").hexdigest(),
        }]}
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

_HPA_KEYPOINT = r"(?i)(metric|指标).{0,40}(target|目标)|(target|目标).{0,40}(metric|指标)"


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


def test_keypoint_scoring_treats_markdown_newlines_as_single_spaces():
    """CR-148：R153 等价的“指标 … 利用率目标”不应因 Markdown 换行漏判。"""
    answer = "HPA 使用 CPU 或内存指标来扩缩容。\n\n- CPU 利用率目标（如 80%）作为触发条件。"

    # 固定 2f83ce9 的旧判据直接对 raw text 做 regex，`.` 不跨换行，行为应失败。
    legacy_hit = [p for p in [_HPA_KEYPOINT] if re.search(p, answer)]
    assert legacy_hit == []

    hit, missed = basic._score_keypoints(answer, [_HPA_KEYPOINT])

    assert hit == [_HPA_KEYPOINT]
    assert missed == []
    assert answer == "HPA 使用 CPU 或内存指标来扩缩容。\n\n- CPU 利用率目标（如 80%）作为触发条件。"


def test_keypoint_scoring_keeps_already_inline_match_and_rejects_missing_target():
    inline_hit, inline_missed = basic._score_keypoints("metric 的 target 是 80%", [_HPA_KEYPOINT])
    missing_hit, missing_missed = basic._score_keypoints("HPA 指标如下：\n\n- 仅记录当前 CPU 使用率。", [_HPA_KEYPOINT])

    assert inline_hit == [_HPA_KEYPOINT]
    assert inline_missed == []
    assert missing_hit == []
    assert missing_missed == [_HPA_KEYPOINT]


def test_case_marks_a_sources_event_without_chunk_identity_as_incomplete():
    class MissingChunkIdOrchestrator:
        def answer(self, _request):
            yield {"type": "retrieval", "sufficiency": "sufficient"}
            yield {"type": "sources", "items": [{
                "index": 1, "url": "https://example.test", "citation": "kafka 4.0 · test",
                "text_sha256": hashlib.sha256(b"fake source evidence").hexdigest(),
            }]}
            yield {"type": "answer_delta", "text": "The producer is idempotent. [1]"}
            yield {"type": "done", "cited_evidence": [1], "ttft_s": 0.1, "total_s": 0.2,
                   "prompt_tokens": 10, "evidence_count": 1, "served_by": "local"}

    case = basic.run_case(MissingChunkIdOrchestrator(), _SPEC, "en")

    assert case.selected_evidence == []
    assert "来源事件缺少审计字段 'chunk_id'" in case.failures


def test_case_marks_a_sources_event_without_text_checksum_as_incomplete():
    class MissingChecksumOrchestrator:
        def answer(self, _request):
            yield {"type": "retrieval", "sufficiency": "sufficient"}
            yield {"type": "sources", "items": [{
                "index": 1, "chunk_id": 101, "url": "https://example.test",
                "citation": "kafka 4.0 · test",
            }]}
            yield {"type": "answer_delta", "text": "The producer is idempotent. [1]"}
            yield {"type": "done", "cited_evidence": [1], "ttft_s": 0.1, "total_s": 0.2,
                   "prompt_tokens": 10, "evidence_count": 1, "served_by": "local"}

    case = basic.run_case(MissingChecksumOrchestrator(), _SPEC, "en")

    assert case.selected_evidence == []
    assert "来源事件缺少审计字段 'text_sha256'" in case.failures


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


def test_main_writes_the_language_grouped_json_report(monkeypatch, tmp_path):
    """CR-124：断言实际 main() 写出的交付物，而非仅测内存汇总函数。"""
    questions = tmp_path / "questions.yaml"
    questions.write_text(yaml.safe_dump({"questions": [_SPEC]}, allow_unicode=True), encoding="utf-8")

    class FakeEngine:
        def __init__(self, _model):
            self.status = SimpleNamespace(loaded=True, error=None)

        def load(self, _prompt):
            # 模拟评测开始后工作树 HEAD 前移；报告仍必须保留启动时已钉住的身份。
            head["current"] = late_commit

    class FakeEmbedder:
        def load(self):
            pass

    class FakeStore:
        meta = {"dictionary_version": "test-dictionary"}

        def count(self):
            return 7

        def close(self):
            pass

    class FakeRouter:
        def __init__(self, *_args, **_kwargs):
            pass

    class FakeMainOrchestrator:
        def __init__(self, *_args, **_kwargs):
            pass

        def answer(self, _request):
            yield {"type": "retrieval", "sufficiency": "sufficient"}
            # URL 刻意按非字母序排列，两个 rowid 共享同 URL；报告须逐项保留
            # event 原顺序、引用编号、rowid 与完整 citation，不能只从 URL 复建。
            yield {"type": "sources", "items": [
                {"index": 1, "chunk_id": 73, "url": "https://z.example.test/page",
                 "citation": "kafka 4.0 · first selected chunk",
                 "text_sha256": hashlib.sha256(b"first selected body").hexdigest()},
                {"index": 2, "chunk_id": 72, "url": "https://z.example.test/page",
                 "citation": "kafka 4.0 · second selected chunk",
                 "text_sha256": hashlib.sha256(b"second selected body").hexdigest()},
                # 合成同 rowid/URL 的重建漂移：报告必须记录运行时正文身份，不能
                # 在写盘后从当前 store 反查并把同一 rowid 误作同一正文。
                {"index": 3, "chunk_id": 73, "url": "https://z.example.test/page",
                 "citation": "kafka 4.0 · changed body at same identity",
                 "text_sha256": hashlib.sha256(b"changed body at same identity").hexdigest()},
            ]}
            yield {"type": "answer_delta", "text": "The producer is idempotent. [1]"}
            yield {"type": "done", "cited_evidence": [1], "ttft_s": 0.1, "total_s": 0.2,
                   "prompt_tokens": 10, "evidence_count": 1, "served_by": "local"}

    monkeypatch.setattr(basic, "QUESTIONS", questions)
    monkeypatch.setattr(basic, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(basic, "InferenceEngine", FakeEngine)
    monkeypatch.setattr(basic, "Embedder", FakeEmbedder)
    monkeypatch.setattr(basic, "ChunkStore", FakeStore)
    monkeypatch.setattr(basic, "Router", FakeRouter)
    monkeypatch.setattr(basic, "LocalBackend", lambda engine: engine)
    monkeypatch.setattr(basic, "ClaudeBackend", lambda prompt: prompt)
    monkeypatch.setattr(basic, "Orchestrator", FakeMainOrchestrator)
    monkeypatch.setattr(basic, "load_dotenv", lambda _path: None)
    monkeypatch.setattr(basic, "system_prompt", lambda language: f"prompt-{language}")
    monkeypatch.setattr(basic, "template_version", lambda: "test-template")
    start_commit = "1" * 40
    late_commit = "2" * 40
    head = {"current": start_commit}
    git_calls = []

    def fake_git_run(args, **kwargs):
        git_calls.append((args, kwargs.get("cwd"), head["current"]))
        return SimpleNamespace(returncode=0, stdout=f"{head['current']}\n", stderr="")

    monkeypatch.setattr(basic.subprocess, "run", fake_git_run)
    monkeypatch.setattr(sys, "argv", ["run_basic.py", "--offline", "--language", "en"])

    assert basic.main() == 0

    report_files = list((tmp_path / "reports").glob("eval-basic-*.json"))
    assert len(report_files) == 1
    report = json.loads(report_files[0].read_text(encoding="utf-8"))
    group = report["by_language"]
    assert set(group) == {"en"}
    assert report["schema_version"] == 2
    # 这同时锁住“运行开始时取一次”而非写盘时重读：FakeEngine.load 后 HEAD 已变为 late。
    assert report["implementation_commit"] == start_commit
    assert git_calls == [
        (["git", "rev-parse", "--verify", "HEAD^{commit}"], basic.ROOT, start_commit),
    ]
    assert report["language"] == "en"
    assert (report["passed"], report["total"]) == (1, 1)
    assert (group["en"]["passed"], group["en"]["total"]) == (1, 1)
    assert (group["en"]["keypoints_hit"], group["en"]["keypoints_total"]) == (1, 1)
    assert group["en"]["sources_missed"] == 0
    assert group["en"]["cases"][0]["question"] == _SPEC["q_en"]
    assert group["en"]["cases"][0]["selected_evidence"] == [
        {"index": 1, "chunk_id": 73, "url": "https://z.example.test/page",
         "citation": "kafka 4.0 · first selected chunk",
         "text_sha256": hashlib.sha256(b"first selected body").hexdigest()},
        {"index": 2, "chunk_id": 72, "url": "https://z.example.test/page",
         "citation": "kafka 4.0 · second selected chunk",
         "text_sha256": hashlib.sha256(b"second selected body").hexdigest()},
        {"index": 3, "chunk_id": 73, "url": "https://z.example.test/page",
         "citation": "kafka 4.0 · changed body at same identity",
         "text_sha256": hashlib.sha256(b"changed body at same identity").hexdigest()},
    ]


def test_main_fails_closed_before_model_load_when_runtime_git_identity_is_not_a_full_commit(
    monkeypatch, tmp_path, capsys,
):
    """无可审计运行期身份时不构造模型、更不得写出无 commit 的报告。"""
    model_started = False

    class UnexpectedEngine:
        def __init__(self, _model):
            nonlocal model_started
            model_started = True

    monkeypatch.setattr(basic, "InferenceEngine", UnexpectedEngine)
    monkeypatch.setattr(
        basic.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="short\n", stderr=""),
    )
    monkeypatch.setattr(basic, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(sys, "argv", ["run_basic.py", "--offline"])

    assert basic.main() == 2
    assert not model_started
    assert not (tmp_path / "reports").exists()
    assert "评测身份验证失败" in capsys.readouterr().err
