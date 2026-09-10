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

REPO_ROOT = Path(__file__).resolve().parents[2]


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

def _chunk(proj: str, tech: str, n: int, text: str, source_path: str | None = None) -> Chunk:
    return Chunk(
        source_url=f"https://example.com/{proj}/{n}.html#s",
        source_project=proj, version_or_commit="v1", license="Apache-2.0",
        retrieved_at=utc_now(), title_path=[proj.title(), f"节 {n}"],
        technology=tech, content_type="prose", locale="zh", text=text,
        source_path=source_path,
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


def test_carry_over_excludes_authored_chunks_by_source_path_not_project(tmp_path, monkeypatch):
    """场景卡片的块 source_project 是它引用的真实来源（如 kafka），不是
    "scenario-cards"——单靠 exclude_projects 认不出旧卡片块，必须靠
    source_path 前缀。这里直接复现过一次真实的合入前 bug：只传
    exclude_projects={"scenario-cards"} 时，一条 source_project="kafka"
    的旧卡片块会被误当"未参与本次同步"搬运下去，与新写入的同 URL 新块
    共存成两条——加上 exclude_source_path_prefixes 才会被正确排除。
    """
    monkeypatch.setattr(store_mod, "INDEX_DIR", tmp_path)
    b = IndexBuilder("cardbase")
    old_card_chunk = _chunk(
        "kafka", "kafka", 1, "旧版卡片正文。",
        source_path="knowledge/scenarios/outbox-pattern.md",
    )
    native_kafka_chunk = _chunk("kafka", "kafka", 2, "kafka 官方文档正文。")
    for c in (old_card_chunk, native_kafka_chunk):
        c.validate()
    b.add([old_card_chunk, native_kafka_chunk], [_vec(0), _vec(1)])
    b.finalize({"kafka": "aaa", "scenario-cards": "old-fingerprint"}, "synthetic")
    b.activate()
    base = tmp_path / "cardbase.db"

    # 不传 source_path 排除：旧卡片块被误当"其它来源"搬运下来（复现 bug）。
    buggy = IndexBuilder("buggy")
    moved = buggy.carry_over(base, {"scenario-cards"}, "synthetic")
    assert moved == 2, "缺少 source_path 排除时，旧卡片块会被误搬运"

    # 传了 source_path 前缀排除：旧卡片块被正确排除，kafka 原生块仍保留。
    fixed = IndexBuilder("fixed")
    moved = fixed.carry_over(
        base, {"scenario-cards"}, "synthetic",
        exclude_source_path_prefixes=("knowledge/scenarios",),
    )
    assert moved == 1, "只应搬运 kafka 官方文档那一块，旧卡片块必须被排除"
    fixed.finalize({"kafka": "aaa"}, "synthetic")
    fixed.activate()
    s = ChunkStore(tmp_path / "fixed.db")
    try:
        rows = s.execute("SELECT text FROM chunks")
        assert [r["text"] for r in rows] == ["kafka 官方文档正文。"]
    finally:
        s.close()


# ---------- T-118：合并某个来源会静默带走引用它的卡片小节 ----------

def test_carry_over_drops_card_chunks_whose_cited_project_is_being_synced(tmp_path, monkeypatch):
    """先把丢失的**机制**钉住（这条在修复前后都通过，是机制说明不是判别性
    测试）：卡片小节的 `source_project` 是被引用来源，于是
    `carry_over(..., exclude_projects={"spring-kafka"})` 会把引用
    spring-kafka 的卡片小节和 spring-kafka 官方块一起排除掉。

    这本身没有错——排除是为了让本轮重建的新块写进来。真正的缺陷在于
    "卡片来源不在本轮同步清单里，于是没有任何东西把它重建回来"，见下面
    两条判别性测试。
    """
    monkeypatch.setattr(store_mod, "INDEX_DIR", tmp_path)
    b = IndexBuilder("t118base")
    card = _chunk("spring-kafka", "spring-kafka", 1, "卡片小节：非阻塞重试的等待发生在重试主题上。",
                  source_path="knowledge/scenarios/kafka-consumer-retry-dlq.md")
    native = _chunk("spring-kafka", "spring-kafka", 2, "spring-kafka 官方文档正文。")
    other = _chunk("postgresql", "postgresql", 3, "postgresql 官方文档正文。")
    for c in (card, native, other):
        c.validate()
    b.add([card, native, other], [_vec(0), _vec(1), _vec(2)])
    b.finalize({"spring-kafka": "aaa", "postgresql": "bbb"}, "synthetic")
    b.activate()

    merged = IndexBuilder("t118merged")
    moved = merged.carry_over(tmp_path / "t118base.db", {"spring-kafka"}, "synthetic")
    assert moved == 1, "卡片小节随被引用来源一起被排除——这正是它必须被重建的原因"


def test_merge_always_syncs_authored_sources_even_when_only_names_another():
    """判别性：修复前 `_resolve_sources(["spring-kafka"], "merge")` 只返回
    spring-kafka，卡片来源不在清单里 → 被 carry_over 排除掉的卡片小节没有
    任何东西重建它们 → 静默消失。R75 就是这样丢了 12 块。
    """
    got = [s.id for s in _resolve_sources(["spring-kafka"], "merge")]
    assert "scenario-cards" in got, "merge 模式必须连带重建卡片，否则引用该来源的小节会丢"
    assert got[-1] == "scenario-cards", (
        "卡片必须排在最后同步：它要引用本轮刚同步来源的新版本号与新块地址"
    )
    assert "spring-kafka" in got


def test_verify_mode_does_not_pull_in_authored_sources():
    """反向边界：verify 只建被点名来源的局部索引、永不激活，
    不该顺手把卡片也拉进来（卡片的引用校验要看整份语料）。"""
    got = [s.id for s in _resolve_sources(["spring-kafka"], "verify")]
    assert got == ["spring-kafka"]


def test_a_card_file_with_no_chunks_blocks_activation(monkeypatch, tmp_path):
    """判别性：即使同步清单里每个来源都"成功"了，只要某张卡片文件在即将
    激活的索引里一块都没有，就必须拦下激活。

    这道门与"为什么会丢"解耦——`failing`/`rejected` 都是空的，来源全部
    报成功，旧实现在这里直接激活。
    """
    missing = "outbox-pattern.md"
    pl = _fake_sync_env(monkeypatch, tmp_path, failing=set(), skip_cards={missing})
    lines: list[str] = []
    rep = pl.sync(log=lines.append)

    assert rep.regression_passed, "回归本身是过的——拦下它的必须是卡片覆盖检查"
    assert rep.incomplete and not rep.activated
    assert not (tmp_path / "current.db").exists()
    assert any(missing in line for line in lines), "必须点名是哪份卡片缺了"


def test_merging_a_cited_source_keeps_every_card_file(monkeypatch, tmp_path):
    """R78 实测那次丢失的端到端复现：合并一个**被卡片引用**的来源，
    合并完之后每份卡片文件都必须还在索引里。

    修复前这里会掉到 0（卡片块按 source_project 被排除，又没被重建）。
    """
    pl = _fake_sync_env(monkeypatch, tmp_path, failing=set())
    pl.sync(log=lambda *_: None)
    before = _count_card_chunks(tmp_path / "current.db")
    assert before

    monkeypatch.setattr(store_mod, "CURRENT", tmp_path / "current.db")
    monkeypatch.setattr(pl, "CURRENT", tmp_path / "current.db")
    rep = pl.sync(["kafka"], mode="merge", log=lambda *_: None)

    assert rep.activated and not rep.incomplete
    assert _count_card_chunks(tmp_path / "current.db") == before, (
        "合并一个被卡片引用的来源，不得让卡片小节静默出局"
    )


def test_merge_refuses_when_embedding_model_differs(base_index):
    """向量不重算，模型不一致就是在混用语义不同的向量——必须直接拒绝。"""
    b = IndexBuilder("merged3")
    with pytest.raises(IndexError_, match="不能混用"):
        b.carry_over(base_index, {"kafka"}, "another-embedding-model")


def test_merge_requires_existing_base(tmp_path, monkeypatch):
    # 本轮（T-114）顺带修掉：这条用例原来不取 base_index fixture，于是
    # store_mod.INDEX_DIR 没被改写，IndexBuilder 直接在**生产索引目录**
    # data/index/ 里建 merged4.building.db，每跑一次测试就留一份 84 KB
    # 的垃圾在真实数据目录里（实测确认该文件由本用例产生）。
    monkeypatch.setattr(store_mod, "INDEX_DIR", tmp_path)
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


def _count_card_chunks(index_path: Path) -> int:
    """按 `source_path` 前缀数卡片块。

    **不能按 `source_project` 数**（这条测试原来就是这么写的）：真实卡片块的
    `source_project` 是它引用的来源（debezium / spring-kafka / ...），
    "scenario-cards" 这个项目名在索引里一条都不存在。按项目数在旧的仿真
    环境里恰好成立，因此那条断言从来没有真正验证过卡片是否还在（T-118）。
    """
    s = ChunkStore(index_path, check_dictionary=False)
    try:
        rows = s.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE source_path LIKE 'knowledge/scenarios%'"
        )
        return rows[0]["n"]
    finally:
        s.close()


# ---------- 来源失败时不得激活 ----------

def _fake_sync_env(monkeypatch, tmp_path, failing: set[str], rejected: set[str] = frozenset(),
                   skip_cards: set[str] = frozenset()):
    """把 sync() 的网络与模型依赖换掉，只保留控制流。

    `skip_cards`：让 authored 来源在本轮"漏掉"指定的卡片文件，用来构造
    T-118 那种"某张卡片一块都没进索引"的局面。

    `rejected`：模拟 CR-046 场景——来源本轮返回 0 块且 `res.rejected>0`，
    但 `res.error` 仍是 `None`（authored 来源格式/引用错误就是这样报告的，
    好让同一来源里其它合法文件/小节仍能入库，见 services/sync/cards.py）。
    与 `failing`（`res.error` 非空，模拟拉取/许可失败）是两种不同的失败
    形状，CR-046 之前只有 `failing` 这种形状会被"有来源失败不得激活"拦住。
    """
    from services.sync import pipeline as pl

    monkeypatch.setattr(store_mod, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(pl, "Embedder", lambda *a, **k: _StubEmbedder())
    monkeypatch.setattr(pl, "run_regression", lambda *a, **k: (True, [], []))
    monkeypatch.setattr(pl, "_embed_with_cache",
                        lambda chunks, *a: [_vec(i % DIM) for i in range(len(chunks))])

    def fake_collect(src, log, versions=None, known_urls=None):
        res = pl.SourceResult(source_id=src.id)
        if src.id in failing:
            res.error = "FetchError: 登记路径不存在于仓库中"
            return [], res
        if src.id in rejected:
            res.commit = "aaa"
            res.rejected = 1
            return [], res
        res.commit = "aaa"
        if src.format == "authored":
            # authored 来源必须仿真到两个关键属性上，否则用它做的实验证明不了
            # 任何关于卡片的事（T-118）：块的 `source_project` 是**被引用的
            # 真实来源**（这里统一用 kafka），`source_path` 是卡片文件自身的
            # 相对路径。前者正是 carry_over 按项目排除时误伤卡片的原因，
            # 后者是逐文件覆盖检查的依据。
            chunks = []
            for rel in src.paths:
                for i, f in enumerate(sorted((REPO_ROOT / rel).glob("*.md"))):
                    if f.name in skip_cards:
                        continue
                    c = _chunk("kafka", "kafka", 100 + i, f"{f.name} 的卡片小节正文。",
                               source_path=f"{rel.rstrip('/')}/{f.name}")
                    c.validate()
                    chunks.append(c)
            return chunks, res
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


# ---------- CR-046：authored 来源的"拒绝"必须和硬错误一样拦下激活 ----------

def test_authored_rejection_without_error_blocks_activation(monkeypatch, tmp_path):
    """卡片格式/引用错误只累加 res.rejected、不设 res.error（好让同一
    来源里其它合法文件/小节仍能入库）。修复前只有 `error` 会被"有来源
    失败不得激活"这道门拦下，`rejected` 会被放过——一张格式错误的卡片
    本轮零产出，仍会被当成"来源成功"直接激活。"""
    pl = _fake_sync_env(monkeypatch, tmp_path, failing=set(), rejected={"scenario-cards"})
    rep = pl.sync(log=lambda *_: None)
    assert rep.regression_passed and rep.incomplete
    assert not rep.activated
    assert not (tmp_path / "current.db").exists(), "authored 来源有拒绝时不得激活"


def test_authored_rejection_allow_partial_still_opts_in(monkeypatch, tmp_path):
    pl = _fake_sync_env(monkeypatch, tmp_path, failing=set(), rejected={"scenario-cards"})
    rep = pl.sync(allow_partial=True, log=lambda *_: None)
    assert rep.activated and not rep.incomplete


def test_merge_authored_rejection_does_not_wipe_existing_cards(monkeypatch, tmp_path):
    """merge 模式下最危险的一幕（CR-046 复现的具体案例）：carry_over()
    已经按路径前缀把旧卡片块整批排除，若新解析零产出的"拒绝"不算失败，
    激活就等于把这张卡片从索引里静默删掉——回归的 6 条烟雾查询几乎不
    可能覆盖到具体某张卡片，拦不住这种情况。"""
    pl = _fake_sync_env(monkeypatch, tmp_path, failing=set())
    pl.sync(log=lambda *_: None)  # 先建一个包含 scenario-cards 的完整索引
    cards_before = _count_card_chunks(tmp_path / "current.db")
    assert cards_before

    monkeypatch.setattr(store_mod, "CURRENT", tmp_path / "current.db")
    monkeypatch.setattr(pl, "CURRENT", tmp_path / "current.db")
    pl = _fake_sync_env(monkeypatch, tmp_path, failing=set(), rejected={"scenario-cards"})
    monkeypatch.setattr(pl, "CURRENT", tmp_path / "current.db")
    rep = pl.sync(["scenario-cards"], mode="merge", log=lambda *_: None)

    assert rep.incomplete and not rep.activated
    after = ChunkStore(tmp_path / "current.db", check_dictionary=False)
    try:
        assert _count_card_chunks(tmp_path / "current.db") == cards_before, (
            "当前索引不得被清空了卡片内容的合并结果覆盖"
        )
    finally:
        after.close()
