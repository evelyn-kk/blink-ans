"""T-023 R187：两臂失败归因审计的行为回归。

这份审计的价值全在"它测到的就是它名字说的那件事"：`in_answer` 只说答案文本，
`in_selected_evidence` 只说实际进入 prompt 的那几块，`min_proximity_window_chars`
把"隔得远"和"根本没写"分开。下面每条都钉一种会让这三件事被混为一谈的写法。

身份与证据核对（`shared_identity` / `verify_evidence`）同样是判据而非防御性检查：
身份不同的两份报告之间，"英文独有失败 7 条"里有多少来自语言无从分开；
证据哈希对不上时读到的正文也不是当时那条证据。

**模块按 `ATTRIBUTION_MODULE` 指定的路径加载**（默认为仓库里的那一份）：
变异实验据此把同一组断言跑在被改坏的副本上，见 `wait-for-review.md` R187 的判别性一节。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "bench" / "bench_language_failure_attribution.py"
AUDIT = ROOT / "bench" / "audits" / "t023-r187-english-only-failure-attribution.json"


def _load():
    path = Path(os.environ.get("ATTRIBUTION_MODULE", DEFAULT))
    spec = importlib.util.spec_from_file_location("_attribution_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


attr = _load()

PROXIMITY = r"(?i)(pressure|压力).{0,40}(memory|内存|disk|磁盘)|(memory|内存|disk|磁盘).{0,40}(pressure|压力)"
PLAIN = r"(?i)(max_connections|最大连接)"


def _report(language, **over):
    report = {
        "language": language,
        "index_fingerprint": "f" * 64,
        "index_chunks": 17080,
        "embedding_model": "bge-m3",
        "dictionary_version": "eba42d53742f",
        "template_version": "07e2d41f1aeb",
        "model": "qwen3-4b",
    }
    report.update(over)
    return report


def _case(case_id="x", *, question="q", answer="a", failures=(), keypoints=(),
          missed=(), sources_missed=(), expect="answered", declined=False, cited=2,
          evidence=(), sufficiency="sufficient"):
    return {
        "id": case_id, "question": question, "answer_text": answer,
        "failures": list(failures), "expect_keypoints": list(keypoints),
        "keypoints_missed": list(missed), "sources_missed": list(sources_missed),
        "expect": expect, "declined": declined, "cited": cited,
        "sufficiency": sufficiency, "served_by": "local",
        "evidence_count": len(evidence), "expect_sources": [], "projects": [],
        "selected_evidence": list(evidence),
    }


# ---------- 身份：不同身份的两份报告不得互相比较 ----------

@pytest.mark.parametrize("field", attr.IDENTITY_FIELDS)
def test_any_differing_identity_field_refuses_the_comparison(field):
    """六个字段逐个试。少钉一个，那一项就能悄悄变化而比较照做。"""
    with pytest.raises(SystemExit) as err:
        attr.shared_identity(_report("zh"), _report("en", **{field: "other"}))

    assert field in str(err.value)


def test_identical_identity_returns_the_shared_values():
    identity = attr.shared_identity(_report("zh"), _report("en"))

    assert identity["index_fingerprint"] == "f" * 64
    assert set(identity) == set(attr.IDENTITY_FIELDS)


def test_missing_identity_field_is_refused_rather_than_treated_as_equal():
    """两边都缺同一个字段时 `None == None` 会"相等"——那正是不能放行的输入。"""
    with pytest.raises(SystemExit) as err:
        attr.shared_identity(_report("zh", index_fingerprint=None),
                             _report("en", index_fingerprint=None))

    assert "index_fingerprint" in str(err.value)


def test_arms_must_actually_be_the_languages_they_claim():
    with pytest.raises(SystemExit):
        attr.shared_identity(_report("en"), _report("en"))


# ---------- 失败集合的三分 ----------

def test_partition_splits_by_which_arm_failed():
    zh = {"a": _case("a", failures=["x"]), "b": _case("b"), "c": _case("c", failures=["x"])}
    en = {"a": _case("a", failures=["y"]), "b": _case("b", failures=["y"]), "c": _case("c")}

    got = attr.failure_partition(zh, en)

    assert got == {"both": ["a"], "zh_only": ["c"], "en_only": ["b"]}


def test_different_question_sets_are_refused():
    with pytest.raises(SystemExit):
        attr.failure_partition({"a": _case("a")}, {"b": _case("b")})


# ---------- 邻近窗口：把"隔得远"和"根本没写"分开 ----------

def test_registered_window_reads_the_declared_number():
    assert attr.registered_window(PROXIMITY) == 40
    assert attr.registered_window(PLAIN) is None


def test_mixed_windows_are_refused_instead_of_silently_taking_one():
    with pytest.raises(SystemExit):
        attr.registered_window(r"a.{0,40}b|b.{0,20}a")


def test_terms_that_never_co_occur_report_none_not_a_large_window():
    """英文答案里只有 memory、没有 pressure——放宽到多少都救不回来。

    若这里返回某个数字，"放宽窗口即可命中"的计数就会把"事实缺失"也算进去。
    """
    assert attr.min_proximity_window(PROXIMITY, "memory and disk usage only") is None


def test_adjacent_terms_report_window_zero():
    """窗口是两词之间的字符数：紧挨着是 0，中间一个空格就是 1。"""
    assert attr.min_proximity_window(PROXIMITY, "pressurememory") == 0
    assert attr.min_proximity_window(PROXIMITY, "pressure memory") == 1


def test_the_window_needed_is_the_real_gap_not_the_registered_one():
    text = "pressure" + "." * 46 + "memory"

    assert attr.min_proximity_window(PROXIMITY, text) == 46
    assert attr.min_proximity_window(PROXIMITY, "pressure" + "." * 39 + "memory") == 39


def test_non_proximity_keypoints_have_no_window():
    assert attr.min_proximity_window(PLAIN, "max_connections") is None


# ---------- 失败分类按字段，不按失败文案 ----------

def test_keypoint_and_source_failures_are_read_from_their_own_fields():
    case = _case(missed=["kp"], sources_missed=["spring-data-redis"], failures=["…"])

    assert attr.failure_kinds(case) == ["keypoint", "source"]


def test_an_answered_refusal_question_counts_as_a_refusal_failure():
    assert attr.failure_kinds(_case(expect="refused", declined=False)) == ["refusal"]
    assert attr.failure_kinds(_case(expect="refused", declined=True)) == []


def test_zero_citations_counts_only_for_answers_that_were_actually_given():
    """拒答本来就没有编号，把它算成"未标引用"会凭空多出一类失败。"""
    assert attr.failure_kinds(_case(cited=0)) == ["citation"]
    assert attr.failure_kinds(_case(cited=0, declined=True)) == []


# ---------- 证据回查：URL 与正文哈希任一对不上就中止 ----------

class _FakeStore:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, sql, params=()):
        row = self.rows.get(params[0])
        return [row] if row else []


def _evidence_case(text="hello", *, url="u", digest=None):
    import hashlib
    digest = digest or hashlib.sha256(text.encode("utf-8")).hexdigest()
    return _case(evidence=[{"index": 1, "chunk_id": 7, "url": url, "text_sha256": digest}])


def test_matching_chunk_is_returned_with_its_text():
    store = _FakeStore({7: {"source_url": "u", "text": "hello"}})

    assert attr.verify_evidence(store, _evidence_case()) == {7: "hello"}


def test_url_mismatch_stops_before_the_text_is_used():
    store = _FakeStore({7: {"source_url": "other", "text": "hello"}})

    with pytest.raises(SystemExit) as err:
        attr.verify_evidence(store, _evidence_case())

    assert "source_url" in str(err.value)


def test_text_hash_mismatch_stops():
    """同一个 URL 下正文改过——块号还在，但它已经不是当时那条证据。"""
    store = _FakeStore({7: {"source_url": "u", "text": "changed"}})

    with pytest.raises(SystemExit) as err:
        attr.verify_evidence(store, _evidence_case())

    assert "哈希" in str(err.value)


def test_missing_chunk_stops():
    with pytest.raises(SystemExit):
        attr.verify_evidence(_FakeStore({}), _evidence_case())


# ---------- 语料覆盖：分开"语料里就没有"与"没进这次证据" ----------

class _CorpusStore:
    def __init__(self, rows):
        self.rows = rows
        self.scans = 0

    def execute(self, sql, params=()):
        self.scans += 1
        return [{"id": i, "text": t} for i, t in self.rows]


def test_corpus_match_is_found_anywhere_in_the_chunk_not_only_at_its_start():
    """判据要的词几乎从不出现在块首；按行首匹配会把覆盖统计压成 0。"""
    store = _CorpusStore([(1, "set max_connections in postgresql.conf"), (2, "unrelated")])

    assert attr.corpus_matches(store, PLAIN, {}) == [1]


def test_corpus_scan_is_cached_per_pattern():
    store = _CorpusStore([(1, "max_connections")])
    cache: dict = {}

    attr.corpus_matches(store, PLAIN, cache)
    attr.corpus_matches(store, PLAIN, cache)

    assert store.scans == 1


# ---------- 三个"在哪里出现"是三件事 ----------

def test_answer_evidence_and_question_are_reported_separately():
    zh = _case(question="分区表怎么建", answer="用 PARTITION BY 建 [1]", keypoints=[PLAIN])
    en = _case(question="how to partition", answer="use CREATE TABLE")
    texts = {"zh": {1: "max_connections is a parameter"}, "en": {2: "nothing here"}}

    got = attr.keypoint_signals(PLAIN, zh, en, texts)

    assert got["in_answer"] == {"zh": False, "en": False}
    assert got["in_selected_evidence"] == {"zh": [1], "en": []}
    assert got["in_question"] == {"zh": False, "en": False}


def test_a_keypoint_that_matches_the_question_itself_is_flagged():
    """复述题面即可通过的判据——`postgres-partitioning` 中文侧就是这一形状。"""
    kp = r"(?i)(PARTITION BY|分区键|分区表)"
    zh = _case(question="PostgreSQL 分区表怎么创建和使用", answer="…")
    en = _case(question="How do I create partitioned tables?", answer="…")

    got = attr.keypoint_signals(kp, zh, en, {"zh": {}, "en": {}})

    assert got["in_question"] == {"zh": True, "en": False}


# ---------- 整臂统计 ----------

def test_arm_totals_count_keypoints_over_answered_questions_only():
    cases = {
        "a": _case("a", keypoints=["k"], missed=["k"], failures=["x"]),
        "b": _case("b", keypoints=["k"]),
        "r": _case("r", expect="refused", declined=True, cited=0),
    }

    totals = attr.arm_totals(cases)

    assert totals["keypoints_registered"] == 2
    assert totals["keypoints_missed"] == 1
    assert totals["failed_cases"] == 1
    assert totals["answered_with_zero_citations"] == 0


# ---------- 已记录产物：不判别新旧，钉的是这份审计的数据 ----------

@pytest.fixture(scope="module")
def audit():
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def test_recorded_partition_sizes(audit):
    assert len(audit["failure_partition"]["en_only"]) == 7
    assert len(audit["failure_partition"]["zh_only"]) == 4
    assert len(audit["failure_partition"]["both"]) == 2


def test_recorded_keypoint_miss_counts_are_equal_across_arms(audit):
    """两臂各漏 6 条 keypoint——44 vs 41 的差额全部来自另外三类失败。"""
    assert audit["arm_totals"]["zh"]["keypoints_missed"] == 6
    assert audit["arm_totals"]["en"]["keypoints_missed"] == 6
    assert audit["arm_totals"]["zh"]["failure_kind_counts"] == {"keypoint": 6}
    assert audit["arm_totals"]["en"]["failure_kind_counts"] == {
        "keypoint": 6, "source": 1, "refusal": 1, "citation": 1,
    }


def test_recorded_term_expansion_is_chinese_only(audit):
    assert audit["arm_totals"]["zh"]["expanded_terms_nonempty"] == 28
    assert audit["arm_totals"]["en"]["expanded_terms_nonempty"] == 0


def test_recorded_window_binding_case_is_only_k8s_eviction(audit):
    assert audit["wider_window_would_match"] == [{
        "id": "k8s-eviction", "arm": "en", "pattern": PROXIMITY,
        "registered_window_chars": 40, "min_proximity_window_chars": 46,
    }]


def test_recorded_question_echo_passes_are_both_chinese(audit):
    assert [(q["id"], q["arm"]) for q in audit["question_echo_passes"]] == [
        ("postgres-partitioning", "zh"), ("redis-cache-annotations", "zh"),
    ]


def test_recorded_identity_matches_the_pinned_index(audit):
    assert audit["index_fingerprint"] == (
        "fe11b6b41ded3d60f422768727c67a9aefe4dac8ebc83dac625efc6b1d176305"
    )
    assert audit["evidence_rows_verified"] == 124


def test_recorded_two_keypoints_match_nothing_in_the_whole_corpus(audit):
    """这两道题的判据在 17080 块里一块都匹配不到——从证据出发永远通不过。

    它们不是语言问题：一道是两臂共有失败，另一道是中文独有失败而英文靠模型
    自己写出来才过的。钉住 0 这个数，是为了让"补语料/改题面"之后这条必须重写。
    """
    zeros = {
        c["id"] for c in audit["cases"]
        for kp in c["keypoints"] if kp["in_corpus_chunk_count"] == 0
    }

    assert zeros == {"spring-graceful-shutdown", "spring-auto-configuration"}


def test_recorded_btree_index_keypoint_ranks_first_in_zh_and_fourteenth_in_en(audit):
    """英文独有失败里唯一一道"证据块确实存在、只是英文侧排不进来"的题。"""
    row = next(r for r in audit["retrieval_probe"]["keypoint_rank"]
               if r["id"] == "postgres-btree-index")

    assert row["best_rank"] == {"zh": 1, "en": 14}


def test_recorded_probe_shows_the_expansion_terms_do_not_explain_the_redis_case(audit):
    """本轮**否定**的那个解释：把中文侧展开词拼进英文题面并没有把 spring-data-redis 拉进来。

    钉住它是为了不让"展开词缺失导致英文取错来源"这句话在没有新证据时回来。
    """
    redis = next(p for p in audit["retrieval_probe"]["source_mix"]
                 if p["id"] == "redis-cache-annotations")
    counts = {name: projects.count("spring-data-redis")
              for name, projects in redis["top_source_projects"].items()}

    assert counts == {"zh_question": 6, "en_question": 1,
                      "en_question_plus_zh_expansion_terms": 0}


def test_recorded_counts_are_recomputable_from_the_per_case_rows(audit):
    """只断言总数的话，一处多算一处少算可以互相抵消。"""
    recomputed = [w for case in audit["cases"] for w in attr.wider_window_would_match(case)]
    echoes = [q for case in audit["cases"] for q in attr.question_echo_passes(case)]

    assert recomputed == audit["wider_window_would_match"]
    assert echoes == audit["question_echo_passes"]
