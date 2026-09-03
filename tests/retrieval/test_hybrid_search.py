"""混合检索的行为回归。

用合成向量建一个小索引，避免依赖模型加载——这些断言检验的是融合、
过滤和预算逻辑，与嵌入质量无关。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.schemas.chunk import Chunk, utc_now  # noqa: E402
from services.retrieval import store as store_mod  # noqa: E402
from services.retrieval.embed import DIM  # noqa: E402
from services.retrieval.search import hybrid_search, keyword_search, rrf_fuse  # noqa: E402
from services.retrieval.store import ChunkStore, IndexBuilder  # noqa: E402

DOCS = [
    ("kafka", "kafka", ["Kafka", "消费者"], "消费者组重平衡会导致未提交的 offset 被重新拉取，产生重复消费。"),
    ("kafka", "kafka", ["Kafka", "事务"], "Kafka 事务型生产者通过 transactional.id 实现精确一次语义。"),
    ("redis", "spring-data-redis", ["Redis", "库存"], "使用 Lua 脚本做预扣库存可以保证原子性，但主从切换时可能丢失写入导致超卖。"),
    ("postgresql", "postgresql", ["PostgreSQL", "慢查询"], "慢查询通常来自统计信息过期，执行计划从 Index Scan 退化为 Seq Scan。"),
    ("kubernetes", "kubernetes", ["Kubernetes", "探针"], "liveness probe 配置过于激进会在 GC 停顿期间误杀 Pod。"),
]


def _vec(i: int) -> list[float]:
    """确定性的合成向量：第 i 维为 1，其余为 0。查询取哪一维就命中哪一条。"""
    v = [0.0] * DIM
    v[i] = 1.0
    return v


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> ChunkStore:
    d = tmp_path_factory.mktemp("index")
    orig = store_mod.INDEX_DIR
    store_mod.INDEX_DIR = d
    try:
        b = IndexBuilder("test")
        chunks, vecs = [], []
        for i, (tech, proj, path, text) in enumerate(DOCS):
            chunks.append(Chunk(
                source_url=f"https://example.com/{proj}/{i}.html#s",
                source_project=proj, version_or_commit="abc123", license="Apache-2.0",
                retrieved_at=utc_now(), title_path=path, technology=tech,
                content_type="prose", locale="zh", text=text,
            ))
            vecs.append(_vec(i))
        chunks.append(Chunk(
            source_url="project://orders-prod/services/orders/checkout.py#reserve_stock",
            source_project="project:orders-prod", version_or_commit="v1", license="Proprietary",
            retrieved_at=utc_now(), title_path=["orders-prod", "orders", "reserve_stock"],
            technology="java", content_type="code", locale="en",
            text="void reserve_stock() { inventory.reserve(); }", source_path="services/orders/checkout.py",
            project_id="orders-prod", module="orders", symbol="reserve_stock",
            cloud_generation_allowed=False,
        ))
        vecs.append(_vec(5))
        for c in chunks:
            c.validate()
        assert b.add(chunks, vecs) == len(chunks)
        b.finalize({"t": "abc123"}, "synthetic")
        b.activate()
        s = ChunkStore(d / "test.db")
        yield s
        s.close()
    finally:
        store_mod.INDEX_DIR = orig


# ---------- 关键词检索 ----------

def test_chinese_keyword_search_works(store):
    """FTS5 默认分词器对中文命中为 0；预分词后必须能检索到。"""
    hits = keyword_search(store, "重复消费")
    assert hits, "中文关键词检索无结果，分词管线可能坏了"


def test_tech_term_not_split_in_search(store):
    assert keyword_search(store, "预扣库存")
    assert keyword_search(store, "慢查询")


def test_keyword_tolerates_partial_query(store):
    """查询用 OR 融合：转写写错一个词不应毁掉整次检索。"""
    assert keyword_search(store, "Kafka 消费者 CAFCA 乱码词")


# ---------- 融合 ----------

def test_rrf_prefers_item_ranked_by_both(store):
    """两路都命中的条目应排在只有一路命中的前面。"""
    kw = [(1, -2.0), (2, -1.0)]
    vec = [(2, 0.1), (3, 0.2)]
    fused = rrf_fuse(kw, vec)
    assert fused[2][0] > fused[1][0] and fused[2][0] > fused[3][0]


def test_hybrid_records_which_path_matched(store):
    hits = hybrid_search(store, "重复消费", _vec(0), limit=5)
    top = hits[0]
    assert top.keyword_rank is not None or top.vector_rank is not None


def test_vector_only_query_returns_expected_doc(store):
    hits = hybrid_search(store, "无关词汇xyzzy", _vec(3), limit=1)
    assert hits and "慢查询" in hits[0].text


# ---------- 过滤与预算 ----------

def test_technology_filter_applies_to_both_paths(store):
    hits = hybrid_search(store, "配置", _vec(0), limit=10, technology="kubernetes")
    assert hits and all(h.technology == "kubernetes" for h in hits)


def test_project_filter(store):
    hits = hybrid_search(store, "Kafka", _vec(0), limit=10, project="kafka")
    assert all(h.source_project == "kafka" for h in hits)


def test_project_id_filters_both_paths_and_exposes_boundary_metadata(store):
    """T-103：用户项目必须与外部来源按 project_id 严格隔离。"""
    hits = hybrid_search(
        store, "reserve_stock inventory", _vec(5), limit=5,
        project_id="orders-prod", module="orders", symbol="reserve_stock",
    )
    assert hits
    assert all(h.project_id == "orders-prod" for h in hits)
    assert all(h.module == "orders" and h.symbol == "reserve_stock" for h in hits)
    assert all(h.cloud_generation_allowed is False for h in hits)


def test_project_metadata_survives_index_write_and_merge(tmp_path, monkeypatch):
    """T-103：项目隔离字段必须真实入库，且 merge 不能静默抹掉它们。"""
    monkeypatch.setattr(store_mod, "INDEX_DIR", tmp_path)
    project_chunk = Chunk(
        source_url="https://example.com/orders/checkout.py#reserve",
        source_project="user-project", version_or_commit="v1.2.3", license="MIT",
        retrieved_at=utc_now(), title_path=["orders", "checkout", "reserve"],
        technology="spring", content_type="code", locale="en", text="def reserve_stock(): pass",
        source_path="services/orders/checkout.py", project_id="orders-prod", module="orders",
        symbol="reserve_stock", cloud_generation_allowed=False,
    )
    project_chunk.validate()
    source = IndexBuilder("source")
    source.add([project_chunk], [_vec(0)])
    source.finalize({"user-project": "v1.2.3"}, "synthetic")
    source.activate()

    merged = IndexBuilder("merged")
    assert merged.carry_over(tmp_path / "source.db", set(), "synthetic") == 1
    merged.finalize({"user-project": "v1.2.3"}, "synthetic")
    merged.activate()
    check = ChunkStore(tmp_path / "merged.db")
    try:
        row = check.execute(
            "SELECT project_id, module, symbol, cloud_generation_allowed, source_path "
            "FROM chunks"
        )[0]
        assert dict(row) == {
            "project_id": "orders-prod", "module": "orders", "symbol": "reserve_stock",
            "cloud_generation_allowed": 0, "source_path": "services/orders/checkout.py",
        }
    finally:
        check.close()


def test_merge_can_carry_legacy_index_without_project_metadata_columns(tmp_path, monkeypatch):
    """T-103：首次项目导入必须能以扩列前的 current.db 作为增量底座。"""
    monkeypatch.setattr(store_mod, "INDEX_DIR", tmp_path)
    chunk = Chunk(
        source_url="https://example.com/legacy", source_project="official", version_or_commit="v1",
        license="Apache-2.0", retrieved_at=utc_now(), title_path=["Legacy"], technology="java",
        content_type="prose", locale="en", text="legacy official evidence",
    )
    chunk.validate()
    source = IndexBuilder("legacy")
    source.add([chunk], [_vec(0)])
    source.finalize({"official": "v1"}, "synthetic")
    source.activate()

    # 真实 current.db 可能来自 T-103 扩列前；用旧表形状复现它，而非只测新 schema。
    db = store_mod._connect(tmp_path / "legacy.db")
    try:
        db.execute("DROP INDEX idx_chunks_project_id")
        db.execute("DROP INDEX idx_chunks_project_symbol")
        for column in ("project_id", "module", "symbol", "cloud_generation_allowed"):
            db.execute(f"ALTER TABLE chunks DROP COLUMN {column}")
        db.commit()
    finally:
        db.close()

    legacy = ChunkStore(tmp_path / "legacy.db")
    try:
        hits = hybrid_search(legacy, "legacy official evidence", _vec(0), limit=1)
        assert hits and hits[0].project_id is None and hits[0].cloud_generation_allowed is None
    finally:
        legacy.close()

    merged = IndexBuilder("merged")
    assert merged.carry_over(tmp_path / "legacy.db", set(), "synthetic") == 1
    merged.finalize({"official": "v1"}, "synthetic")
    merged.activate()
    check = ChunkStore(tmp_path / "merged.db")
    try:
        row = check.execute("SELECT project_id, module, symbol, cloud_generation_allowed FROM chunks")[0]
        assert tuple(row) == (None, None, None, None)
    finally:
        check.close()


def test_token_budget_is_respected(store):
    """上下文预算是 I0 实测的硬约束：超出即首 token 时延超标。"""
    budget = 30
    hits = hybrid_search(store, "消费 库存 查询 探针", _vec(0), limit=10, token_budget=budget)
    assert sum(h.token_estimate for h in hits) <= budget


def test_no_results_returns_empty_not_error(store):
    """完全无共同 token 时返回空列表而非报错。"""
    assert hybrid_search(store, "zzzqqq xyzzy plugh", None, limit=5) == []


def test_or_query_gives_high_recall_low_precision(store):
    """OR 查询会让几乎任何中文提问都命中——这是有意为之，但有代价。

    用 OR 是为了容忍语音转写的错字（I0 实测 Kafka→CAFCA）：
    要求全部词命中会让一个错字毁掉整次检索。
    代价是召回极高而精度全靠 bm25 与 RRF 排序兜底。

    **对 I2 的直接影响：判定"证据不足"不能靠空结果，必须靠相关性阈值。**
    这条断言就是把该行为固化下来，避免后续误以为空结果可用作判据。
    """
    hits = hybrid_search(store, "完全无关的中文句子内容", None, limit=5)
    assert hits, "含常见中文词的查询预期仍会命中，判定证据不足需另设阈值"


# ---------- 引用 ----------

def test_hit_citation_contains_version_and_date(store):
    hits = hybrid_search(store, "重复消费", _vec(0), limit=1)
    c = hits[0].citation
    assert "abc123" in c and hits[0].retrieved_at[:4] in c


def test_source_url_is_clickable(store):
    hits = hybrid_search(store, "重复消费", _vec(0), limit=1)
    assert hits[0].source_url.startswith("https://")


# ---------- 词典版本绑定 ----------

def test_dictionary_mismatch_refuses_to_open(store, monkeypatch):
    """词典变更后必须拒绝打开旧索引，否则召回静默劣化。"""
    monkeypatch.setattr(store_mod, "dictionary_version", lambda: "deadbeef0000")
    with pytest.raises(store_mod.IndexError_, match="词典"):
        ChunkStore(store.path, check_dictionary=True)


# ---------- 向量复用的安全性 ----------

def test_embedding_cache_rejects_different_model(store):
    """换嵌入模型后必须拒绝复用：校验和只覆盖正文，向量语义已变。

    不比对模型名会静默复用错向量——不报错，只是检索悄悄变差，
    是最难排查的一类问题。
    """
    from services.retrieval.store import EmbeddingCache

    c = EmbeddingCache(store.path, embedding_model="some-other-model")
    assert not c.available
    assert c.rejected_reason and "不可复用" in c.rejected_reason


def test_embedding_cache_reuses_when_model_matches(store):
    from services.retrieval.store import EmbeddingCache

    c = EmbeddingCache(store.path, embedding_model="synthetic")
    assert c.available
    row = store.db.execute("SELECT checksum FROM chunks LIMIT 1").fetchone()
    v = c.get(row["checksum"])
    assert v is not None and len(v) == DIM
    assert c.hits == 1
    c.close()


# ---------- 唯一键必须带来源身份（CR-003）----------

def _same_text_chunks(url_a: str, url_b: str) -> list[Chunk]:
    """同一段正文出现在两个页面上——官方文档里很常见（共用的注意事项、
    同一段说明被两章引用）。两条都必须留下，否则其中一个来源就引用不到。"""
    text = "Set spring.datasource.hikari.maximum-pool-size to bound connection usage."
    out = []
    for url, proj in ((url_a, "spring-boot"), (url_b, "spring-data-redis")):
        out.append(Chunk(
            source_url=url, source_project=proj, version_or_commit="abc123",
            license="Apache-2.0", retrieved_at=utc_now(), title_path=["Pool", "Sizing"],
            technology="spring", content_type="prose", locale="en", text=text,
        ))
    return out


def test_same_text_from_different_urls_both_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "INDEX_DIR", tmp_path)
    b = IndexBuilder("dup")
    chunks = _same_text_chunks("https://example.com/a.html#p", "https://example.com/b.html#p")
    for c in chunks:
        c.validate()
    assert chunks[0].checksum == chunks[1].checksum, "前提：正文相同则校验和相同"
    assert b.add(chunks, [_vec(0), _vec(1)]) == 2

    b.finalize({"t": "abc123"}, "synthetic")
    b.activate()
    s = ChunkStore(tmp_path / "dup.db")
    try:
        rows = s.execute("SELECT source_url, source_project FROM chunks ORDER BY source_url")
        assert [r["source_url"] for r in rows] == [
            "https://example.com/a.html#p", "https://example.com/b.html#p",
        ]
        assert {r["source_project"] for r in rows} == {"spring-boot", "spring-data-redis"}
    finally:
        s.close()


def test_same_text_same_url_is_deduplicated(tmp_path, monkeypatch):
    """同一页内的完全重复仍然只留一条——去重本身没有被取消，只是加了来源维度。"""
    monkeypatch.setattr(store_mod, "INDEX_DIR", tmp_path)
    b = IndexBuilder("dup2")
    chunks = _same_text_chunks("https://example.com/a.html#p", "https://example.com/a.html#p")
    for c in chunks:
        c.validate()
    assert b.add(chunks, [_vec(0), _vec(1)]) == 1

    b.finalize({"t": "abc123"}, "synthetic")
    b.activate()
    s = ChunkStore(tmp_path / "dup2.db")
    try:
        assert s.count() == 1
    finally:
        s.close()


def test_embedding_cache_serves_shared_checksum_once_per_lookup(tmp_path, monkeypatch):
    """同一 checksum 现在对应多行，复用查询必须仍返回单个向量而不是报错。"""
    from services.retrieval.store import EmbeddingCache

    monkeypatch.setattr(store_mod, "INDEX_DIR", tmp_path)
    b = IndexBuilder("dup3")
    chunks = _same_text_chunks("https://example.com/a.html#p", "https://example.com/b.html#p")
    for c in chunks:
        c.validate()
    b.add(chunks, [_vec(0), _vec(0)])
    b.finalize({"t": "abc123"}, "synthetic")
    b.activate()

    c = EmbeddingCache(tmp_path / "dup3.db", embedding_model="synthetic")
    try:
        v = c.get(chunks[0].checksum)
        assert v is not None and len(v) == DIM
    finally:
        c.close()
