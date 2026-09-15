"""T-112 候选审计自身的回归。

CR-121/122 的要求不是“JSON 看起来更丰富”，而是候选参数、共同输入身份和
相对基线的退步计数必须由同一套代码产生，不能靠文档手抄。
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from packages.evaltools import t112_candidates as runner  # noqa: E402
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
        "vector-distance-credit-0.75-0.10", "vector-distance-credit-0.75-0.20",
        "relative-vector-credit-0.002", "relative-vector-credit-0.005",
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
    assert {
        name: (by_name[name].experiment.vector_distance_max, by_name[name].experiment.vector_distance_credit)
        for name in ("vector-distance-credit-0.75-0.10", "vector-distance-credit-0.75-0.20")
    } == {
        "vector-distance-credit-0.75-0.10": (0.75, 0.10),
        "vector-distance-credit-0.75-0.20": (0.75, 0.20),
    }
    assert {
        name: by_name[name].experiment.relative_vector_credit
        for name in ("relative-vector-credit-0.002", "relative-vector-credit-0.005")
    } == {
        "relative-vector-credit-0.002": 0.002,
        "relative-vector-credit-0.005": 0.005,
    }


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


class _FakeStore:
    """可控 index 身份；值在候选执行中变化时应被 run_all 发现。"""

    def __init__(self, path: Path):
        self.path = path
        self.meta = {"embedding_model": "fake-embed", "dictionary_version": "dict-v1"}
        self._count = 3
        self.closed = False

    def count(self):
        return self._count

    def close(self):
        self.closed = True


class _FakeEmbedder:
    model_id = "fake-embed"
    error = None

    def load(self):
        return None


def _install_fake_run_all(monkeypatch, tmp_path, *, mutate: str | None = None):
    """让真正的 run_all/main 走完，仅替换模型/索引和两个评测执行器。"""
    index = tmp_path / "current.db"
    index.write_bytes(b"index-v1")
    probes = tmp_path / "probe.yaml"
    validation = tmp_path / "validation.yaml"
    probes.write_text(yaml.safe_dump({"probes": []}), encoding="utf-8")
    validation.write_text(yaml.safe_dump({"limit": 5, "cases": [{"q": "held-out"}]}), encoding="utf-8")
    store = _FakeStore(index)
    calls = []
    mutated = False

    monkeypatch.setattr(runner, "PROBES", probes)
    monkeypatch.setattr(runner, "VALIDATION", validation)
    monkeypatch.setattr(runner, "ChunkStore", lambda: store)
    monkeypatch.setattr(runner, "Embedder", _FakeEmbedder)
    monkeypatch.setattr(runner.validation, "unresolvable_golds", lambda spec, active: [])

    def fake_ranking(spec, active, embedder, *, limit, candidates, experiment):
        nonlocal mutated
        calls.append(("probe", candidates, experiment))
        # baseline 已经测完之后才让身份变化，逼出 run_all 的收尾门而非开头门。
        if experiment is not None and not mutated and mutate:
            mutated = True
            if mutate == "sha":
                index.write_bytes(b"index-v2")
            elif mutate == "count":
                store._count += 1
            elif mutate == "meta":
                store.meta["dictionary_version"] = "dict-v2"
        rank = 1 if experiment is None else 2
        return [SimpleNamespace(question="probe", gold="gold", rank=rank, passed=True,
                                top=[f"rank-{rank}"], fts_query="fake")]

    def fake_validation(case, active, embedder, limit, *, candidates, experiment):
        calls.append(("validation", candidates, experiment))
        return 1 if experiment is None else 2

    monkeypatch.setattr(runner.ranking, "run", fake_ranking)
    monkeypatch.setattr(runner.validation, "run", fake_validation)
    return store, index, calls


def test_run_all_collects_runtime_provenance_and_forwards_every_candidate(monkeypatch, tmp_path):
    """不能靠构造 common 字典假装采集过：真实 run_all 必须产生身份与候选结果。"""
    store, index, calls = _install_fake_run_all(monkeypatch, tmp_path)

    audits = runner.run_all("fake-command --prefix audit")

    assert store.closed is True
    baseline_probe = audits["baseline"][0]
    provenance = baseline_probe["provenance"]
    assert provenance["command"] == "fake-command --prefix audit"
    assert provenance["index"]["path"] == str(index)
    assert provenance["index"]["sha256"] == runner._sha256(index)
    assert provenance["index"]["chunk_count"] == 3
    assert provenance["index"]["meta"] == store.meta
    assert provenance["embedding_model"] == "fake-embed"
    assert provenance["implementation"]["git_commit"] == runner._git_commit()
    assert set(provenance["implementation"]["files"]) == {
        str(path.relative_to(ROOT)) for path in runner.IMPLEMENTATION_FILES
    }
    # baseline 明确传 None；每个声明的实验都实际抵达两种 runner，且 rank 改变。
    assert calls[0] == ("probe", 30, None)
    passed = [call[2] for call in calls if call[0] == "probe"]
    assert passed == [candidate.experiment for candidate in CANDIDATES]
    assert audits["single-path-credit"][0]["results"][0]["rank"] == 2
    assert audits["single-path-credit"][0]["comparison_to_baseline"]["regression_count"] == 1


@pytest.mark.parametrize("mutate", ["sha", "count", "meta"])
def test_identity_change_fails_closed_before_main_overwrites_any_audit(monkeypatch, tmp_path, mutate):
    """CR-123：三种身份字段在候选中变化都必须拒绝，旧产物一个字节也不能覆盖。"""
    _, _, _ = _install_fake_run_all(monkeypatch, tmp_path, mutate=mutate)
    output = tmp_path / "audits"
    output.mkdir()
    sentinel = output / "t112-baseline-probe.json"
    sentinel.write_text("keep-me", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "t112_candidates.py", "--output-dir", str(output), "--prefix", "t112",
    ])

    with pytest.raises(RuntimeError, match="current.db"):
        runner.main()

    assert sentinel.read_text(encoding="utf-8") == "keep-me"


def test_fusion_experiment_changes_real_hybrid_order_not_just_runner_arguments(monkeypatch):
    """删掉 experiment 融合分支会让同一组候选的排序不再变化。"""
    from services.retrieval import search as search_mod

    rows = {
        1: {"id": 1, "text": "keyword only", "title_path": "keyword", "source_url": "u1",
            "source_project": "p", "version_or_commit": "v", "retrieved_at": "2026-01-01",
            "technology": "t", "content_type": "prose", "token_estimate": 1},
        2: {"id": 2, "text": "vector only", "title_path": "vector", "source_url": "u2",
            "source_project": "p", "version_or_commit": "v", "retrieved_at": "2026-01-01",
            "technology": "t", "content_type": "prose", "token_estimate": 1},
    }

    class Store:
        def execute(self, sql, params=()):
            return [rows[rid] for rid in params]

    monkeypatch.setattr(search_mod, "keyword_search", lambda *args, **kwargs: [(1, -1.0)])
    monkeypatch.setattr(search_mod, "vector_search", lambda *args, **kwargs: [(2, 0.1)])
    baseline = search_mod.hybrid_search(Store(), "q", [0.0], limit=2)
    candidate = search_mod.hybrid_search(
        Store(), "q", [0.0], limit=2, experiment=CANDIDATES[2].experiment,
    )

    assert [hit.rowid for hit in baseline] == [1, 2]
    assert [hit.rowid for hit in candidate] == [2, 1]


def test_vector_distance_credit_uses_actual_distance_without_imputing_a_missing_rank(monkeypatch):
    """距离信用只作用于关键词候选外、距离低于阈值的向量块。"""
    from services.retrieval import search as search_mod

    rows = {
        1: {"id": 1, "text": "keyword only", "title_path": "keyword", "source_url": "u1",
            "source_project": "p", "version_or_commit": "v", "retrieved_at": "2026-01-01",
            "technology": "t", "content_type": "prose", "token_estimate": 1},
        2: {"id": 2, "text": "vector only", "title_path": "vector", "source_url": "u2",
            "source_project": "p", "version_or_commit": "v", "retrieved_at": "2026-01-01",
            "technology": "t", "content_type": "prose", "token_estimate": 1},
    }

    class Store:
        def execute(self, sql, params=()):
            return [rows[rid] for rid in params]

    monkeypatch.setattr(search_mod, "keyword_search", lambda *args, **kwargs: [(1, -1.0)])
    monkeypatch.setattr(search_mod, "vector_search", lambda *args, **kwargs: [(2, 0.60)])
    credit = search_mod.FusionExperiment(
        "distance", vector_distance_max=0.75, vector_distance_credit=0.10,
    )
    baseline = search_mod.hybrid_search(Store(), "q", [0.0], limit=2)
    boosted = search_mod.hybrid_search(Store(), "q", [0.0], limit=2, experiment=credit)
    assert [hit.rowid for hit in baseline] == [1, 2]
    assert [hit.rowid for hit in boosted] == [2, 1]

    monkeypatch.setattr(search_mod, "vector_search", lambda *args, **kwargs: [(2, 0.80)])
    threshold_miss = search_mod.hybrid_search(Store(), "q", [0.0], limit=2, experiment=credit)
    assert [hit.rowid for hit in threshold_miss] == [1, 2]


@pytest.mark.parametrize("field, invalid", [
    ("vector_distance_max", float("nan")),
    ("vector_distance_max", float("inf")),
    ("vector_distance_max", float("-inf")),
    ("vector_distance_credit", float("nan")),
    ("vector_distance_credit", float("inf")),
    ("vector_distance_credit", float("-inf")),
    ("vector_distance_max", 0.0),
    ("vector_distance_credit", -0.10),
])
def test_vector_distance_credit_rejects_nonfinite_parameters_before_search(monkeypatch, field, invalid):
    """CR-127：非有限距离参数不能进入候选查询或产生不可排序融合分。"""
    from services.retrieval import search as search_mod

    called = []
    monkeypatch.setattr(search_mod, "keyword_search", lambda *args, **kwargs: called.append("keyword"))
    monkeypatch.setattr(search_mod, "vector_search", lambda *args, **kwargs: called.append("vector"))
    params = {"vector_distance_max": 0.75, "vector_distance_credit": 0.10}
    params[field] = invalid

    with pytest.raises(ValueError, match="向量距离"):
        search_mod.hybrid_search(
            object(), "q", [0.0], experiment=search_mod.FusionExperiment("bad", **params),
        )

    assert called == []


def test_relative_vector_credit_uses_within_query_distance_range(monkeypatch):
    """相对信用只由同一 query 的向量距离范围计算，不引用绝对距离阈值。"""
    from services.retrieval import search as search_mod

    rows = {
        rid: {"id": rid, "text": str(rid), "title_path": str(rid), "source_url": f"u{rid}",
              "source_project": "p", "version_or_commit": "v", "retrieved_at": "2026-01-01",
              "technology": "t", "content_type": "prose", "token_estimate": 1}
        for rid in (1, 2, 3)
    }

    class Store:
        def execute(self, sql, params=()):
            return [rows[rid] for rid in params]

    monkeypatch.setattr(search_mod, "keyword_search", lambda *args, **kwargs: [(1, -1.0)])
    monkeypatch.setattr(search_mod, "vector_search", lambda *args, **kwargs: [(2, 0.60), (3, 0.80)])
    relative = search_mod.FusionExperiment("relative", relative_vector_credit=0.01)
    baseline = search_mod.hybrid_search(Store(), "q", [0.0], limit=3)
    boosted = search_mod.hybrid_search(Store(), "q", [0.0], limit=3, experiment=relative)
    assert [hit.rowid for hit in baseline] == [1, 2, 3]
    assert [hit.rowid for hit in boosted] == [2, 1, 3]

    monkeypatch.setattr(search_mod, "vector_search", lambda *args, **kwargs: [(2, 0.80), (3, 0.80)])
    no_spread = search_mod.hybrid_search(Store(), "q", [0.0], limit=3, experiment=relative)
    assert [hit.rowid for hit in no_spread] == [1, 2, 3]
