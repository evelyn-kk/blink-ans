"""T-023：基础回归的双语题面、事实/来源断言与分语言报告。"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages" / "evaltools"))

import run_basic as basic  # noqa: E402

# CR-160：替身索引的两块内容。用 dict 行即可——`index_fingerprint` 只按
# ["source_url"] / ["checksum"] 取值，与 sqlite3.Row 的取值方式一致。
_FAKE_CHUNKS = (
    {"source_url": "https://a.example.test/one", "checksum": "aaa"},
    {"source_url": "https://b.example.test/two", "checksum": "bbb"},
)


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
        # CR-160：身份字段齐全（含 embedding_model），且 execute() 返回真实的
        # (source_url, checksum) 行，使报告里的 index_fingerprint 可被独立复算。
        # R189：证据侧统计会按 `chunk_id` 回查正文，因此这个替身必须分得清两种查询；
        # 一律返回同一批行会让"回查到的正文"变成指纹用的那几行，测不出真实行为。
        meta = {"dictionary_version": "test-dictionary",
                "embedding_model": "test-embedder"}
        rows = _FAKE_CHUNKS
        bodies = {
            72: ("https://z.example.test/page", "second selected body"),
            73: ("https://z.example.test/page", "first selected body"),
        }

        def execute(self, sql, params=()):
            if "WHERE id = ?" in sql:
                found = self.bodies.get(params[0])
                return [{"source_url": found[0], "text": found[1]}] if found else []
            if "SELECT text FROM chunks" in sql:
                return [{"text": body} for _url, body in self.bodies.values()]
            return list(self.rows)

        def count(self):
            return len(self.rows)

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
    # R189 把证据侧统计写进报告，schema 随之到 3。
    assert report["schema_version"] == 3
    # 这同时锁住“运行开始时取一次”而非写盘时重读：FakeEngine.load 后 HEAD 已变为 late。
    assert report["implementation_commit"] == start_commit
    assert git_calls == [
        (["git", "rev-parse", "--verify", "HEAD^{commit}"], basic.ROOT, start_commit),
    ]
    assert report["language"] == "en"
    # CR-160：报告必须带可独立复算的索引内容身份，而不是只有块数。
    expected_fp = hashlib.sha256(
        b"https://a.example.test/one\taaa\nhttps://b.example.test/two\tbbb\n"
    ).hexdigest()
    assert report["index_fingerprint"] == expected_fp
    assert report["index_chunks"] == 2
    assert report["embedding_model"] == "test-embedder"
    assert report["dictionary_version"] == "test-dictionary"
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


# ---------- CR-160：评测报告的索引内容身份 ----------

def _identity_store(rows, *, meta=None, count=None):
    class _Store:
        def __init__(self):
            self.meta = {"dictionary_version": "test-dictionary",
                         "embedding_model": "test-embedder"} if meta is None else meta

        def execute(self, _sql, _params=()):
            return list(rows)

        def count(self):
            return len(rows) if count is None else count

        def close(self):
            pass
    return _Store()


def test_same_chunk_count_with_different_checksums_gives_a_different_fingerprint():
    """CR-160 的核心：块数相同不代表索引相同。

    这条正是 R184 三份报告缺失的判据——它们只存 `index_chunks: 17080`，
    于是两份内容不同、块数相同的索引会被当成同一实验条件。
    """
    a = _identity_store(_FAKE_CHUNKS)
    b = _identity_store((
        {"source_url": "https://a.example.test/one", "checksum": "aaa"},
        {"source_url": "https://b.example.test/two", "checksum": "CHANGED"},
    ))
    assert a.count() == b.count() == 2
    assert basic.index_identity(a)["index_fingerprint"] \
        != basic.index_identity(b)["index_fingerprint"]


def test_fingerprint_ignores_row_order_so_a_rebuilt_index_stays_comparable():
    """同样的内容换个遍历顺序必须得到同一指纹，否则重建索引后就没法比对。"""
    forward = _identity_store(_FAKE_CHUNKS)
    reversed_rows = _identity_store(tuple(reversed(_FAKE_CHUNKS)))
    assert basic.index_identity(forward)["index_fingerprint"] \
        == basic.index_identity(reversed_rows)["index_fingerprint"]


@pytest.mark.parametrize("meta, why", [
    ({"dictionary_version": "d"}, "缺 embedding_model"),
    ({"embedding_model": "e"}, "缺 dictionary_version"),
    ({}, "两者都缺"),
])
def test_missing_index_identity_fields_refuse_to_produce_a_report(meta, why):
    """身份不全时失败关闭，而不是记个 None 就把报告写出去。"""
    store = _identity_store(_FAKE_CHUNKS, meta=meta)
    with pytest.raises(RuntimeError, match="索引缺少身份字段"):
        basic.index_identity(store)


def test_fingerprint_row_count_mismatch_fails_closed():
    """遍历到的块数与 count() 不符 → 指纹覆盖不全，必须报错而不是返回半份指纹。"""
    store = _identity_store(_FAKE_CHUNKS, count=99)
    with pytest.raises(ValueError, match="与 count\\(\\) 的 99 不一致"):
        basic.index_identity(store)


def test_eval_and_bench_share_one_fingerprint_implementation():
    """CR-160 要求“复用”而不是各抄一份：两边必须是同一个函数对象。

    各抄一份的话，任何一侧改了取值顺序或分隔符都会算出不同指纹，
    而指纹的全部价值就在于跨工具可比。
    """
    sys.path.insert(0, str(ROOT / "bench"))
    import bench_speculative_retrieval as spec  # noqa: PLC0415
    from services.retrieval import store as store_mod  # noqa: PLC0415

    assert basic.index_fingerprint is store_mod.index_fingerprint
    assert spec._index_fingerprint is store_mod.index_fingerprint
    # bench 侧只包一层失败关闭语义，算法仍是同一份
    with pytest.raises(SystemExit, match="拒绝产出报告"):
        spec.index_fingerprint(_identity_store(_FAKE_CHUNKS, count=99))


# ---------- R189：登记正则 × 证据文本（测量，不是门禁） ----------

class _EvidenceStore:
    """按 `chunk_id` 回查正文的最小替身；不解析 SQL 之外的条件。"""

    def __init__(self, bodies: dict, corpus: list[str] | None = None):
        self.bodies = bodies
        self.corpus = corpus if corpus is not None else [b for _u, b in bodies.values()]

    def execute(self, sql, params=()):
        if "WHERE id = ?" in sql:
            found = self.bodies.get(params[0])
            return [{"source_url": found[0], "text": found[1]}] if found else []
        return [{"text": text} for text in self.corpus]


def _evidence_case(pattern, *, answer_hit, chunk_id=1, url="u", body="body", sha=None,
                   answer_text=None):
    case = basic.Case(id="x", language="en", question="q", expect="answered",
                      expect_keypoints=[pattern])
    case.keypoints_hit = [pattern] if answer_hit else []
    # 答案文本必须带上那句话：实现若把正则拿去匹配答案而不是证据，
    # 只有这样才会被下面的断言逮住（变异实验第一版就是因为 answer_text 为空而漏网）。
    case.answer_text = answer_text if answer_text is not None else (
        f"answer mentioning {pattern}" if answer_hit else "answer")
    case.selected_evidence = [{
        "index": 1, "chunk_id": chunk_id, "citation": "c", "url": url,
        "text_sha256": sha or hashlib.sha256(body.encode()).hexdigest(),
    }]
    # CR-162：`run_case()` 会同时记下这两个计数，替身必须照做——
    # 只填 `selected_evidence` 的替身构造不出"本该有几行"，而未核验数正是按它算的。
    case.sources = len(case.selected_evidence)
    case.evidence_count = len(case.selected_evidence)
    case.keypoints_scored = True
    return case


def test_keypoint_found_in_the_evidence_is_reported_with_its_chunk_id():
    case = _evidence_case(r"(?i)idempotent", answer_hit=True,
                          body="the producer is idempotent")
    store = _EvidenceStore({1: ("u", "the producer is idempotent")})

    rows, unverified = basic.keypoint_evidence(store, case)

    assert rows == [{"pattern": r"(?i)idempotent", "in_answer": True,
                     "in_evidence_chunk_ids": [1]}]
    assert unverified == 0


def test_a_keypoint_that_only_the_answer_has_is_not_credited_to_the_evidence():
    """R187/R188 反复撞到的那一格：判据命中、而它命中的话不在这道题引的块里。

    实现若把这条正则拿去匹配答案（而不是证据正文），这里会返回 `[1]`，
    `in_answer_only` 这一格就永远是 0——那正是本轮要量的东西。
    """
    case = _evidence_case(r"(?i)commons-pool2", answer_hit=True,
                          answer_text="pooling needs commons-pool2 on the classpath [1]",
                          body="connect to redis through a connection factory")
    store = _EvidenceStore({1: ("u", "connect to redis through a connection factory")})

    rows, _ = basic.keypoint_evidence(store, case)

    assert rows[0]["in_answer"] is True
    assert rows[0]["in_evidence_chunk_ids"] == []


def test_evidence_text_is_newline_normalized_like_the_answer_is():
    """邻近判据不能因为块正文里的换行而判断开——两侧共用同一个归一化函数。"""
    pattern = r"(?i)(pressure).{0,10}(memory)"
    case = _evidence_case(pattern, answer_hit=False, body="pressure\n  memory")
    store = _EvidenceStore({1: ("u", "pressure\n  memory")})

    rows, _ = basic.keypoint_evidence(store, case)

    assert rows[0]["in_evidence_chunk_ids"] == [1]


def test_url_or_hash_mismatch_counts_as_unverified_and_the_text_is_not_used():
    pattern = r"(?i)idempotent"
    wrong_url = _evidence_case(pattern, answer_hit=True, url="other", body="idempotent")
    wrong_hash = _evidence_case(pattern, answer_hit=True, body="idempotent",
                                sha="0" * 64)
    store = _EvidenceStore({1: ("u", "idempotent")})

    for case in (wrong_url, wrong_hash):
        rows, unverified = basic.keypoint_evidence(store, case)

        assert unverified == 1
        assert rows[0]["in_evidence_chunk_ids"] == []


def test_missing_chunk_counts_as_unverified():
    case = _evidence_case(r"(?i)idempotent", answer_hit=True, chunk_id=999)

    _rows, unverified = basic.keypoint_evidence(_EvidenceStore({}), case)

    assert unverified == 1


def test_the_evidence_statistic_never_touches_the_pass_or_fail_verdict():
    """它是测量不是门禁：算完之后 `failures` 必须一个字都没多。"""
    case = _evidence_case(r"(?i)commons-pool2", answer_hit=True, body="unrelated")
    store = _EvidenceStore({1: ("u", "unrelated")})

    basic.keypoint_evidence(store, case)

    assert case.failures == []
    assert case.ok is True


def test_counts_split_the_four_cells_and_exclude_unverified_cases():
    def case_with(rows, unverified=0):
        case = basic.Case(id="x", language="en", question="q", expect="answered")
        case.keypoint_evidence = rows
        case.evidence_rows_unverified = unverified
        return case

    counts = basic.keypoint_evidence_counts([
        case_with([{"pattern": "a", "in_answer": True, "in_evidence_chunk_ids": [1]},
                   {"pattern": "b", "in_answer": True, "in_evidence_chunk_ids": []}]),
        case_with([{"pattern": "c", "in_answer": False, "in_evidence_chunk_ids": [2]},
                   {"pattern": "d", "in_answer": False, "in_evidence_chunk_ids": []}]),
        # 身份核不上的题整道排除：把它算进 in_answer_only 会凭空造出一条"无据通过"
        case_with([{"pattern": "e", "in_answer": True, "in_evidence_chunk_ids": []}],
                  unverified=1),
    ])

    assert counts == {"in_answer_and_evidence": 1, "in_answer_only": 1,
                      "in_evidence_only": 1, "in_neither": 1,
                      "cases_counted": 2,
                      "cases_excluded_for_unverified_evidence": 1}


def test_patterns_that_match_nothing_in_the_whole_index_are_listed_once():
    """`spring-graceful-shutdown` 那一类：判据在整份语料里一块都匹配不到。"""
    # 两条语料都**不以**待匹配的词开头：按行首匹配的实现会把两条都报成"匹配不到"。
    store = _EvidenceStore({}, corpus=["the server performs graceful shutdown",
                                       "then the application context is closed"])

    unmatched = basic.keypoints_without_corpus_match(store, [
        r"(?i)(graceful shutdown).{0,40}(application context)",   # 跨块，匹配不到
        r"(?i)graceful shutdown",                                  # 匹配得到
        r"(?i)(graceful shutdown).{0,40}(application context)",   # 重复登记只报一次
    ])

    assert unmatched == [r"(?i)(graceful shutdown).{0,40}(application context)"]


# ---------- CR-162：缺审计字段的来源事件必须整题退出四格统计 ----------

class _MissingFieldOrchestrator:
    """`sources` 事件少一个审计字段——`run_case()` 的列表推导会整体中止。"""

    def __init__(self, dropped: str, answer: str = "The producer is idempotent. [1]"):
        self.dropped = dropped
        self.answer_text = answer   # 不能叫 answer：会盖掉下面的生成器方法

    def answer(self, _request):
        item = {"index": 1, "chunk_id": 101, "url": "https://example.test",
                "citation": "kafka 4.0 · test",
                "text_sha256": hashlib.sha256(b"body").hexdigest()}
        item.pop(self.dropped)
        yield {"type": "retrieval", "sufficiency": "sufficient"}
        yield {"type": "sources", "items": [item]}
        yield {"type": "answer_delta", "text": self.answer_text}
        yield {"type": "done", "cited_evidence": [1], "ttft_s": 0.1, "total_s": 0.2,
               "prompt_tokens": 10, "evidence_count": 1, "served_by": "local"}


@pytest.mark.parametrize("dropped", ["chunk_id", "url", "text_sha256"])
def test_missing_source_audit_fields_keep_the_case_out_of_the_four_cells(dropped):
    """CR-162：缺字段是最根本的"核不上"，走的却曾是相反的路径。

    复现（R189 实现）：缺 `chunk_id` 时 `selected_evidence` 为空，按"遍历失败次数"
    算出 `unverified=0`，于是这道题被当成已核验，命中的 keypoint 记进 `in_answer_only`——
    正是这份统计里最需要可信的那一格被凭空加了一条。

    现在按"本该核对几行 − 真正核对通过几行"算，缺字段的那一行必然落在差里。
    """
    case = basic.run_case(_MissingFieldOrchestrator(dropped), _SPEC, "en")
    assert any("缺少审计字段" in f for f in case.failures)   # 原有诊断不受影响

    rows, unverified = basic.keypoint_evidence(_EvidenceStore({101: ("https://example.test", "body")}), case)
    case.keypoint_evidence, case.evidence_rows_unverified = rows, unverified
    counts = basic.keypoint_evidence_counts([case])

    assert unverified == 1
    assert counts["cases_excluded_for_unverified_evidence"] == 1
    assert counts["cases_counted"] == 0
    assert counts["in_answer_only"] == 0


def test_a_sources_event_that_never_arrives_also_stays_out_of_the_four_cells():
    """另一个同形的洞：`done` 自报送了证据，`sources` 事件却没来。

    只数 `selected_evidence` 的实现在这里同样算出 0，把"没法核验"说成"证据里没有"。
    """
    class _NoSources:
        def answer(self, _request):
            yield {"type": "retrieval", "sufficiency": "sufficient"}
            yield {"type": "answer_delta", "text": "The producer is idempotent. [1]"}
            yield {"type": "done", "cited_evidence": [1], "ttft_s": 0.1, "total_s": 0.2,
                   "prompt_tokens": 10, "evidence_count": 3, "served_by": "local"}

    case = basic.run_case(_NoSources(), _SPEC, "en")

    _rows, unverified = basic.keypoint_evidence(_EvidenceStore({}), case)

    assert unverified == 3


def test_a_fully_verified_case_reports_no_unverified_rows():
    """反向：字段齐全、身份核得上时必须是 0，否则这条规则会把好题也排除掉。"""
    case = _evidence_case(r"(?i)idempotent", answer_hit=True, body="idempotent")

    _rows, unverified = basic.keypoint_evidence(
        _EvidenceStore({1: ("u", "idempotent")}), case)

    assert unverified == 0


def test_a_refused_question_is_not_reported_as_unverifiable_evidence():
    """拒答题按契约不返回来源，`done` 却仍自报 evidence_count——

    照"本该核对几行"硬算，这类题会被整片记成"核不上的证据"，
    而它根本没有登记正则、无从测起。实测 `refuse-react-useeffect` 就是
    `sources=0` / `evidence_count=5` 这个形状（见
    `bench/audits/t023-r190-evidence-row-accounting.json`）。
    """
    case = basic.Case(id="refuse", language="en", question="q", expect="refused")
    case.sources, case.evidence_count, case.declined = 0, 5, True

    rows, unverified = basic.keypoint_evidence(_EvidenceStore({}), case)

    assert (rows, unverified) == ([], 0)
    assert basic.keypoint_evidence_counts([case])["cases_excluded_for_unverified_evidence"] == 0


def test_an_answered_question_the_model_declined_is_also_left_out():
    """模型自己拒答的应作答题同样没走到 `_score_keypoints()`，统计对它无话可说。"""
    case = basic.Case(id="x", language="en", question="q", expect="answered",
                      expect_keypoints=[r"(?i)idempotent"])
    case.sources, case.evidence_count, case.declined = 0, 5, True

    assert basic.keypoint_evidence(_EvidenceStore({}), case) == ([], 0)


@pytest.mark.parametrize("answer, why", [
    ("Spring Boot exposes metrics at /actuator/prometheus.", "有技术内容但一个编号都没标"),
    ("", "整条回答是空的"),
])
def test_an_answer_with_no_citation_markers_is_not_a_keypoint_scoring_subject(answer, why):
    """CR-163：`cited=0` 的题在 `run_case()` 里就没评过分，四格不能替它评。

    这正是 R190 收窄分母时**没有配回归**的那个形状，也是 R189 英文报告里
    一条**假**的 `in_evidence_only`：`spring-prometheus` 的答案确实写了
    `/actuator/prometheus`，只因没标编号而未被评分，却被记成"证据里有、答案没写"。
    `in_answer=False` 在这种题上表示"从未评分"，不表示"答案里没有"——
    两者混在同一格里，那一格就不能用了。
    """
    class _NoCitations:
        def answer(self, _request):
            yield {"type": "retrieval", "sufficiency": "sufficient"}
            yield {"type": "sources", "items": [{
                "index": 1, "chunk_id": 101, "url": "https://example.test",
                "citation": "kafka 4.0 · test",
                "text_sha256": hashlib.sha256(b"idempotent producer").hexdigest()}]}
            if answer:
                yield {"type": "answer_delta", "text": answer}
            yield {"type": "done", "cited_evidence": [], "ttft_s": 0.1, "total_s": 0.2,
                   "prompt_tokens": 10, "evidence_count": 1, "served_by": "local"}

    case = basic.run_case(_NoCitations(), _SPEC, "en")
    assert case.cited == 0 and case.keypoints_scored is False, why

    rows, unverified = basic.keypoint_evidence(
        _EvidenceStore({101: ("https://example.test", "idempotent producer")}), case)
    case.keypoint_evidence, case.evidence_rows_unverified = rows, unverified

    assert (rows, unverified) == ([], 0)
    counts = basic.keypoint_evidence_counts([case])
    assert counts["cases_counted"] == 0
    assert counts["in_evidence_only"] == 0
    assert counts["cases_excluded_for_unverified_evidence"] == 0
