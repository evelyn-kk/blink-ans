"""T-112 候选审计自身的回归。

CR-121/122 的要求不是“JSON 看起来更丰富”，而是候选参数、共同输入身份和
相对基线的退步计数必须由同一套代码产生，不能靠文档手抄。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from packages.evaltools.t112_candidates import CANDIDATES, _audit, compare_ranks  # noqa: E402


def _rows(**ranks):
    return [{"q": question, "rank": rank} for question, rank in ranks.items()]


def test_comparison_counts_only_actual_rank_regressions_not_improvements():
    """CR-122 的同型反例：None→整数与 14→13 都是改善，不能算退步。

    旧 R105 的手工统计把 keyword-tail 的这两项也算进去，写成 5 而非 4；
    这里用同一个比较程序锁住正确关系，而非再手写一张候选专用数字表。
    """
    baseline = _rows(btree=1, delivery=7, dlq=7, kafka_dlq=14, covering=3, self=None)
    candidate = _rows(btree=8, delivery=15, dlq=9, kafka_dlq=13, covering=4, self=12)

    comparison = compare_ranks(baseline, candidate)

    assert comparison["regression_count"] == 4
    assert [r["q"] for r in comparison["regressions"]] == [
        "btree", "delivery", "dlq", "covering",
    ]
    assert comparison["improvement_count"] == 2
    assert {r["q"] for r in comparison["improvements"]} == {"kafka_dlq", "self"}


def test_comparison_cannot_silently_drop_a_question_from_baseline():
    with pytest.raises(ValueError, match="题集不一致"):
        compare_ranks(_rows(a=1, b=2), _rows(a=1))


def test_candidates_declare_all_temporary_rules_in_data_not_monkeypatches():
    by_name = {candidate.name: candidate for candidate in CANDIDATES}
    assert set(by_name) == {
        "baseline", "single-path-credit", "vector-only-credit",
        "keyword-tail", "vector-keyword-rescue",
    }
    assert by_name["single-path-credit"].experiment.imputed_keyword_rank == 31
    assert by_name["single-path-credit"].experiment.imputed_vector_rank == 31
    assert by_name["vector-only-credit"].experiment.imputed_keyword_rank == 31
    assert by_name["vector-only-credit"].experiment.imputed_vector_rank is None
    tail = by_name["keyword-tail"].experiment
    assert (tail.keyword_candidate_depth, tail.keyword_score_depth) == (150, 150)
    rescue = by_name["vector-keyword-rescue"].experiment
    assert (rescue.keyword_candidate_depth, rescue.keyword_score_depth) == (150, 30)
    assert (rescue.vector_candidate_depth, rescue.vector_score_depth) == (150, 30)
    assert (rescue.keyword_rescue_depth, rescue.vector_rescue_max_rank) == (150, 15)
    assert rescue.rescue_credit_rank == "vector_zero"


def test_audit_embeds_provenance_candidate_and_computed_comparison():
    audit = _audit(
        kind="probe",
        common={"index": {"sha256": "index"}, "implementation": {"git_commit": "commit"}},
        candidate=CANDIDATES[1], results=_rows(q=3),
        comparison=compare_ranks(_rows(q=1), _rows(q=3)),
    )

    assert audit["schema"] == "t112-candidate-audit/v1"
    assert audit["candidate"]["experiment"]["imputed_keyword_rank"] == 31
    assert audit["provenance"]["index"]["sha256"] == "index"
    assert audit["comparison_to_baseline"]["regression_count"] == 1
