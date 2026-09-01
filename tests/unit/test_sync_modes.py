"""三种同步模式的语义边界（CR-004）。

旧实现只有一个隐含模式：`--only` 建局部索引却跑全量回归，必然失败。
这些用例把「全量重建 / 局部验证 / 合并更新」各自的约束固定下来，
特别是"局部索引永不激活"和"合并前必须先清掉旧来源"这两条——
两者出错都不报错，只是索引悄悄不对。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.schemas.chunk import Chunk, utc_now  # noqa: E402
from services.retrieval import store as store_mod  # noqa: E402
from services.retrieval.embed import DIM  # noqa: E402
from services.retrieval.store import ChunkStore, IndexBuilder, IndexError_  # noqa: E402
from services.sync.pipeline import _resolve_sources, run_regression  # noqa: E402


# ---------- 模式与 --only 的组合约束 ----------

def test_full_mode_takes_all_ingestible_sources():
    got = {s.id for s in _resolve_sources(None, "full")}
    assert "redis-official" not in got, "许可受限的来源不得进入任何同步模式"
    assert {"kafka", "postgresql", "kubernetes"} <= got


def test_full_mode_rejects_only():
    """全量重建会覆盖整个索引，配 --only 就是在要求"用一个来源覆盖全部"。"""
    with pytest.raises(ValueError, match="不接受 --only"):
        _resolve_sources(["kafka"], "full")


@pytest.mark.parametrize("mode", ["verify", "merge"])
def test_partial_modes_require_only(mode):
    with pytest.raises(ValueError, match="必须配合 --only"):
        _resolve_sources(None, mode)


@pytest.mark.parametrize("mode", ["verify", "merge"])
def test_typo_in_source_id_is_an_error_not_a_silent_skip(mode):
    """打错来源 id 若只是少同步一个来源，合并模式下会静默保留旧数据。"""
    with pytest.raises(ValueError, match="kafkka"):
        _resolve_sources(["kafka", "kafkka"], mode)


def test_link_only_source_cannot_be_synced():
    with pytest.raises(ValueError, match="redis-official"):
        _resolve_sources(["redis-official"], "verify")


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="未知同步模式"):
        _resolve_sources(None, "incremental")


# ---------- 合并更新：搬运底座 ----------

def _chunk(proj: str, tech: str, n: int, text: str) -> Chunk:
    return Chunk(
        source_url=f"https://example.com/{proj}/{n}.html#s",
        source_project=proj, version_or_commit="v1", license="Apache-2.0",
        retrieved_at=utc_now(), title_path=[proj.title(), f"节 {n}"],
        technology=tech, content_type="prose", locale="zh", text=text,
    )


def _vec(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


@pytest.fixture
def base_index(tmp_path, monkeypatch) -> Path:
    """一个两来源的底座索引：kafka 2 块、postgresql 1 块。"""
    monkeypatch.setattr(store_mod, "INDEX_DIR", tmp_path)
    b = IndexBuilder("base")
    chunks = [
        _chunk("kafka", "kafka", 1, "消费者组重平衡导致重复消费。"),
        _chunk("kafka", "kafka", 2, "事务型生产者提供精确一次语义。"),
        _chunk("postgresql", "postgresql", 1, "统计信息过期会让执行计划退化为顺序扫描。"),
    ]
    for c in chunks:
        c.validate()
    b.add(chunks, [_vec(0), _vec(1), _vec(2)])
    b.finalize({"kafka": "aaa", "postgresql": "bbb"}, "synthetic")
    b.activate()
    return tmp_path / "base.db"


def test_carry_over_keeps_untouched_sources_and_drops_merged_one(base_index, tmp_path):
    b = IndexBuilder("merged")
    moved = b.carry_over(base_index, {"kafka"}, "synthetic")
    assert moved == 1, "只应搬来 postgresql 那一块"

    new = _chunk("kafka", "kafka", 9, "重平衡协议改为增量协作式后停顿显著缩短。")
    new.validate()
    b.add([new], [_vec(3)])
    b.finalize({"kafka": "ccc", "postgresql": "bbb"}, "synthetic")
    b.activate()

    s = ChunkStore(tmp_path / "merged.db")
    try:
        assert s.stats() == {"postgresql": 1, "kafka": 1}
        assert s.meta["chunk_count"] == "2", "chunk_count 须是索引实际行数，不是本次新写入数"
        rows = s.execute("SELECT text FROM chunks WHERE source_project='kafka'")
        assert "增量协作式" in rows[0]["text"], "kafka 的旧块必须被换掉而不是共存"
    finally:
        s.close()


def test_carried_rows_stay_searchable(base_index, tmp_path):
    """搬运时 FTS 是按当前词典重算的，搬过来的来源必须仍能被关键词检索到。"""
    from services.retrieval.search import keyword_search

    b = IndexBuilder("merged2")
    b.carry_over(base_index, {"kafka"}, "synthetic")
    b.finalize({"postgresql": "bbb"}, "synthetic")
    b.activate()

    s = ChunkStore(tmp_path / "merged2.db")
    try:
        assert keyword_search(s, "执行计划", limit=5), "搬运后关键词索引丢失"
    finally:
        s.close()


def test_merge_refuses_when_embedding_model_differs(base_index):
    """向量不重算，模型不一致就是在混用语义不同的向量——必须直接拒绝。"""
    b = IndexBuilder("merged3")
    with pytest.raises(IndexError_, match="不能混用"):
        b.carry_over(base_index, {"kafka"}, "another-embedding-model")


def test_merge_requires_existing_base(tmp_path):
    b = IndexBuilder("merged4")
    with pytest.raises(IndexError_, match="不存在"):
        b.carry_over(tmp_path / "nope.db", {"kafka"}, "synthetic")


def test_existing_versions_are_preserved_for_untouched_sources(base_index):
    b = IndexBuilder("merged5")
    assert b.existing_versions(base_index) == {"kafka": "aaa", "postgresql": "bbb"}


# ---------- 局部验证：回归范围 ----------

def test_regression_skips_queries_for_absent_sources(base_index, monkeypatch):
    """局部索引里没有的来源，其回归查询必须跳过而非判失败。"""
    monkeypatch.setattr(
        "services.sync.pipeline.Embedder", lambda *a, **k: _StubEmbedder()
    )
    ok, failures, skipped = run_regression(base_index, lambda *_: None, {"kafka"})
    assert ok, failures
    assert any("Kubernetes" in q or "Spring" in q or "Redis" in q for q in skipped)
    assert not any("Kafka" in q for q in skipped), "选中来源的查询不得被跳过"


def test_regression_without_scope_runs_everything_and_fails_here(base_index, monkeypatch):
    """同一个局部索引，按全量回归跑就会失败——这正是旧实现的症状。"""
    monkeypatch.setattr(
        "services.sync.pipeline.Embedder", lambda *a, **k: _StubEmbedder()
    )
    ok, failures, skipped = run_regression(base_index, lambda *_: None, None)
    assert not ok and not skipped
    assert any("kubernetes" in f for f in failures)


class _StubEmbedder:
    """回归的查询向量在这里无关紧要：断言的是"哪些查询被跑了"，
    命中与否由关键词路决定，不需要真实嵌入模型。"""

    def encode_one(self, text: str) -> list[float]:
        return [0.0] * DIM


# ---------- 来源失败时不得激活 ----------

def _fake_sync_env(monkeypatch, tmp_path, failing: set[str]):
    """把 sync() 的网络与模型依赖换掉，只保留控制流。"""
    from services.sync import pipeline as pl

    monkeypatch.setattr(store_mod, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(pl, "Embedder", lambda *a, **k: _StubEmbedder())
    monkeypatch.setattr(pl, "run_regression", lambda *a, **k: (True, [], []))
    monkeypatch.setattr(pl, "_embed_with_cache",
                        lambda chunks, *a: [_vec(i % DIM) for i in range(len(chunks))])

    def fake_collect(src, log):
        res = pl.SourceResult(source_id=src.id)
        if src.id in failing:
            res.error = "FetchError: 登记路径不存在于仓库中"
            return [], res
        res.commit = "aaa"
        c = _chunk(src.project, src.technology, 1, f"{src.project} 的一段说明文字。")
        c.validate()
        return [c], res

    monkeypatch.setattr(pl, "collect_chunks", fake_collect)
    return pl


def test_source_failure_blocks_activation(monkeypatch, tmp_path):
    """回归只有 6 条烟雾查询，少掉一整个来源它照样可能通过——
    因此"某来源拉取失败"必须自己成为一道拦截，不能只靠退出码。"""
    pl = _fake_sync_env(monkeypatch, tmp_path, failing={"kafka"})
    rep = pl.sync(log=lambda *_: None)
    assert rep.regression_passed and rep.incomplete
    assert not rep.activated
    assert not (tmp_path / "current.db").exists(), "不完整的索引不得成为当前索引"


def test_allow_partial_opts_into_activation(monkeypatch, tmp_path):
    pl = _fake_sync_env(monkeypatch, tmp_path, failing={"kafka"})
    rep = pl.sync(allow_partial=True, log=lambda *_: None)
    assert rep.activated and not rep.incomplete


def test_clean_full_sync_activates(monkeypatch, tmp_path):
    pl = _fake_sync_env(monkeypatch, tmp_path, failing=set())
    rep = pl.sync(log=lambda *_: None)
    assert rep.activated and not rep.incomplete
    assert (tmp_path / "current.db").exists()


def test_merge_failure_does_not_delete_the_source(monkeypatch, tmp_path):
    """merge 模式下最危险的一幕：旧块已在 carry_over 时排除，
    新块又没拉下来，激活即等于把这个来源从索引里静默删掉。"""
    pl = _fake_sync_env(monkeypatch, tmp_path, failing=set())
    pl.sync(log=lambda *_: None)                      # 先建一个完整索引
    before = ChunkStore(tmp_path / "current.db", check_dictionary=False)
    kafka_before = before.stats().get("kafka")
    before.close()
    assert kafka_before

    monkeypatch.setattr(store_mod, "CURRENT", tmp_path / "current.db")
    monkeypatch.setattr(pl, "CURRENT", tmp_path / "current.db")
    pl = _fake_sync_env(monkeypatch, tmp_path, failing={"kafka"})
    monkeypatch.setattr(pl, "CURRENT", tmp_path / "current.db")
    rep = pl.sync(["kafka"], mode="merge", log=lambda *_: None)

    assert rep.incomplete and not rep.activated
    after = ChunkStore(tmp_path / "current.db", check_dictionary=False)
    try:
        assert after.stats().get("kafka") == kafka_before, "当前索引不得被残缺的合并结果覆盖"
    finally:
        after.close()
