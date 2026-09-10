"""场景卡片（authored 来源）的解析、引用解析与注册表校验。

卡片本身不产生新的证据实体：每个 `## ` 小节的 source_url/source_project/
version_or_commit/license/technology 全部照抄它引用的真实来源。这里锁住的是
那条"照抄"链路本身的判别性行为——尤其是"一个坏小节不该拖垮同一张卡片里
其它小节"和"不能引用未登记/受限来源"这两条，都是本来很容易被静默放过的地方。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.schemas.chunk import Chunk, utc_now  # noqa: E402
from services.retrieval import store as store_mod  # noqa: E402
from services.retrieval.embed import DIM  # noqa: E402
from services.retrieval.store import ChunkStore, IndexBuilder  # noqa: E402
from services.sync import cards  # noqa: E402
from services.sync.registry import RegistryError, Source, load_registry  # noqa: E402


def _vec(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


def _authored_src(tmp_path: Path, **over) -> Source:
    base = dict(
        id="scenario-cards", project="scenario-cards", technology="scenario-cards",
        format="authored", locale="en", paths=(str(tmp_path),),
    )
    base.update(over)
    return Source(**base)


def _fake_registry() -> list[Source]:
    return [
        Source(
            id="widgetdocs", project="widgetdocs", technology="widgets",
            format="markdown", locale="en", paths=("docs",),
            repo="https://github.com/example/widgetdocs", ref="v1",
            license="Apache-2.0", license_file="LICENSE", base_url="https://example.com/",
        ),
        Source(
            id="blockeddocs", project="blockeddocs", technology="blocked",
            format="markdown", locale="en", paths=("docs",),
            repo="https://github.com/example/blockeddocs", ref="v1",
            # 特意给一个在 ALLOWED_LICENSES 里的许可，只为了证明拦截它的是
            # ingest=False 这道门，而不是"恰好许可也不合规"这种复合原因。
            license="CC-BY-4.0", license_file="LICENSE", base_url="https://example.com/",
            ingest=False, ingest_blocked_reason="仅供测试的受限来源",
        ),
        Source(
            # 模拟"注册表本身登记了一个不在 ALLOWED_LICENSES 白名单里的许可"——
            # registry._one() 不检查许可白名单（那是 Chunk.validate() 的职责），
            # 所以这条路径在真实系统里是可达的，不是纯假设。
            id="mplsource", project="mplsource", technology="mpl",
            format="markdown", locale="en", paths=("docs",),
            repo="https://github.com/example/mplsource", ref="v1",
            license="MPL-2.0", license_file="LICENSE", base_url="https://example.com/",
        ),
        # 镜像生产注册表里真实的 scenario-cards 条目——CR-047 的测试需要
        # `_authored_path_prefixes()` 能从注册表里认出"knowledge/scenarios"
        # 是一个 authored 路径前缀，才能验证回退查询确实排除了这个前缀下
        # 历史遗留的卡片块。
        Source(
            id="scenario-cards", project="scenario-cards", technology="scenario-cards",
            format="authored", locale="en", paths=("knowledge/scenarios",),
        ),
    ]


def _write_card(tmp_path: Path, name: str, text: str) -> None:
    (tmp_path / name).write_text(text, encoding="utf-8")


CARD_OK = """\
---
title: Widget lifecycle
---

## Why widgets need draining before shutdown
source: widgetdocs https://example.com/docs/lifecycle.html#draining

Widgets hold in-flight work that must finish before the process exits.

## How the drain hook is wired up
source: widgetdocs https://example.com/docs/lifecycle.html#hooks

Register a shutdown hook that calls drain() and waits for it to return.
"""

# CR-045：URL 必须逐字匹配语料里已存在的真实块地址，不能凭空写一个。
# 这里模拟"widgetdocs 语料实际已入库的两条真实地址"——CARD_OK 引用的
# 就是这两条，供大多数测试传入证明"合法引用能通过"；专门测 CR-045 的
# 用例会引用不在这个集合里的地址。
KNOWN_URLS = {
    "widgetdocs": {
        "https://example.com/docs/lifecycle.html#draining",
        "https://example.com/docs/lifecycle.html#hooks",
    },
}


@pytest.fixture(autouse=True)
def _stub_registry(monkeypatch):
    monkeypatch.setattr(cards, "load_registry", _fake_registry)


# ---------- 正常路径 ----------

def test_two_sections_produce_two_chunks_with_inherited_metadata(tmp_path):
    _write_card(tmp_path, "widget.md", CARD_OK)
    src = _authored_src(tmp_path)

    chunks, res = cards.collect_chunks(
        src, lambda *_: None, versions={"widgetdocs": "abc123"}, known_urls=KNOWN_URLS
    )

    assert res.error is None
    assert res.rejected == 0
    assert len(chunks) == 2
    first = chunks[0]
    assert first.source_project == "widgetdocs"
    assert first.version_or_commit == "abc123"
    assert first.license == "Apache-2.0"          # 照抄被引用来源，不是卡片自己发明的
    assert first.technology == "widgets"           # 同上
    assert first.locale == "en"
    assert first.content_type == "prose"
    assert first.title_path == ["Widget lifecycle", "Why widgets need draining before shutdown"]
    assert first.source_url == "https://example.com/docs/lifecycle.html#draining"
    first.validate()  # 不抛异常


def test_version_prefers_this_run_over_current_index(tmp_path, monkeypatch):
    """versions 参数（本轮同步内刚拿到的）优先于查当前已激活索引。"""
    _write_card(tmp_path, "widget.md", CARD_OK)
    monkeypatch.setattr(cards, "_lookup_current_version", lambda project, prefixes: "stale-from-disk")

    chunks, _res = cards.collect_chunks(
        _authored_src(tmp_path), lambda *_: None,
        versions={"widgetdocs": "fresh-this-run"}, known_urls=KNOWN_URLS,
    )
    assert all(c.version_or_commit == "fresh-this-run" for c in chunks)


def test_version_falls_back_to_current_index_when_absent_from_this_run(tmp_path, monkeypatch):
    _write_card(tmp_path, "widget.md", CARD_OK)
    monkeypatch.setattr(cards, "_lookup_current_version", lambda project, prefixes: "from-disk")

    chunks, _res = cards.collect_chunks(
        _authored_src(tmp_path), lambda *_: None, versions={}, known_urls=KNOWN_URLS
    )
    assert all(c.version_or_commit == "from-disk" for c in chunks)


# ---------- 引用解析的判别性拒绝 ----------

def test_section_citing_unregistered_project_is_rejected_without_killing_the_file(tmp_path):
    """一个小节引用了没登记的项目：只拒那一节，同文件其它合法小节仍然入库。"""
    text = CARD_OK.replace(
        "source: widgetdocs https://example.com/docs/lifecycle.html#hooks",
        "source: nonexistent-project https://example.com/docs/lifecycle.html#hooks",
    )
    _write_card(tmp_path, "widget.md", text)

    chunks, res = cards.collect_chunks(
        _authored_src(tmp_path), lambda *_: None,
        versions={"widgetdocs": "abc123"}, known_urls=KNOWN_URLS,
    )
    assert len(chunks) == 1, "合法的第一节不该被第二节的错误拖累"
    assert res.rejected == 1
    assert any("未登记" in k for k in res.reject_reasons)


def test_section_citing_link_only_source_is_rejected_for_being_blocked(tmp_path):
    """引用 ingest=False 的来源必须被拒绝，且理由是"不入库"而非许可格式问题。"""
    text = CARD_OK.replace(
        "source: widgetdocs https://example.com/docs/lifecycle.html#hooks",
        "source: blockeddocs https://example.com/docs/lifecycle.html#hooks",
    )
    _write_card(tmp_path, "widget.md", text)

    chunks, res = cards.collect_chunks(
        _authored_src(tmp_path), lambda *_: None,
        versions={"widgetdocs": "abc123"}, known_urls=KNOWN_URLS,
    )
    assert len(chunks) == 1
    assert res.rejected == 1
    assert any("不入库来源" in k for k in res.reject_reasons)


def test_section_citing_a_disallowed_license_is_rejected_by_chunk_validate(tmp_path):
    """registry 不检查许可白名单——这道门完全靠 Chunk.validate() 兜底，
    这里直接验证那道门确实拦住了它，而不是假设它会拦。"""
    text = CARD_OK.replace(
        "source: widgetdocs https://example.com/docs/lifecycle.html#hooks",
        "source: mplsource https://example.com/docs/lifecycle.html#hooks",
    )
    _write_card(tmp_path, "widget.md", text)

    known_urls = {**KNOWN_URLS, "mplsource": {"https://example.com/docs/lifecycle.html#hooks"}}
    chunks, res = cards.collect_chunks(
        _authored_src(tmp_path), lambda *_: None,
        versions={"widgetdocs": "abc123", "mplsource": "v1"}, known_urls=known_urls,
    )
    assert len(chunks) == 1
    assert res.rejected == 1
    assert any("MetadataError" in k for k in res.reject_reasons)


def test_section_without_resolvable_version_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(cards, "_lookup_current_version", lambda project, prefixes: None)
    _write_card(tmp_path, "widget.md", CARD_OK)

    chunks, res = cards.collect_chunks(
        _authored_src(tmp_path), lambda *_: None, versions={}, known_urls=KNOWN_URLS
    )
    assert chunks == []
    assert res.rejected == 2
    assert any("无法确定引用版本" in k for k in res.reject_reasons)


# ---------- CR-045：引用的 URL 必须是语料里已存在的真实块地址 ----------

def test_section_citing_a_url_not_known_to_the_project_is_rejected(tmp_path):
    """项目已登记、可入库，不代表这个具体 URL 是真的——独立复现 codex 的
    构造：跨域 URL 必须被拒绝，同域但编造的路径同样必须被拒绝（证明这道
    校验是"精确成员校验"而不是弱得多的"域名前缀匹配"）。"""
    for bad_url in (
        "https://attacker.invalid/not-widget",
        "https://example.com/docs/totally-made-up-page.html",
    ):
        text = CARD_OK.replace(
            "source: widgetdocs https://example.com/docs/lifecycle.html#hooks",
            f"source: widgetdocs {bad_url}",
        )
        _write_card(tmp_path, "widget.md", text)

        chunks, res = cards.collect_chunks(
            _authored_src(tmp_path), lambda *_: None,
            versions={"widgetdocs": "abc123"}, known_urls=KNOWN_URLS,
        )
        assert len(chunks) == 1, f"合法的第一节不该被 {bad_url!r} 这一节拖累"
        assert res.rejected == 1, bad_url
        assert any("已存在的真实块地址" in k for k in res.reject_reasons), bad_url


def test_known_urls_prefers_this_run_over_current_index(tmp_path, monkeypatch):
    """known_urls 参数（本轮同步内刚拿到的）优先于查当前已激活索引——
    与 version 解析的优先级规则对称。"""
    _write_card(tmp_path, "widget.md", CARD_OK)
    monkeypatch.setattr(cards, "_lookup_current_urls", lambda project, prefixes: {"https://stale.example/x"})

    chunks, res = cards.collect_chunks(
        _authored_src(tmp_path), lambda *_: None,
        versions={"widgetdocs": "abc123"}, known_urls=KNOWN_URLS,
    )
    assert res.rejected == 0
    assert len(chunks) == 2


def test_known_urls_falls_back_to_current_index_when_absent_from_this_run(tmp_path, monkeypatch):
    _write_card(tmp_path, "widget.md", CARD_OK)
    monkeypatch.setattr(cards, "_lookup_current_urls", lambda project, prefixes: set(KNOWN_URLS["widgetdocs"]))

    chunks, res = cards.collect_chunks(
        _authored_src(tmp_path), lambda *_: None, versions={"widgetdocs": "abc123"}, known_urls={}
    )
    assert res.rejected == 0
    assert len(chunks) == 2


# ---------- CR-047：回退查询不得把历史 authored 块当成官方真实块 ----------

def test_current_index_fallback_excludes_authored_blocks_from_self_endorsing(tmp_path, monkeypatch):
    """独立复现 codex 的构造：CR-045 修复前跑过的旧版卡片同步，会把一个
    编造/跨域 URL 写成 source_project='widgetdocs'（因为卡片小节照抄的
    是被引用来源的 source_project，和该来源真正的原生块无法用这个字段
    区分）。如果 _lookup_current_urls()/_lookup_current_version() 的回退
    查询不排除 authored 来源自己产生的块，这条历史伪造记录会被当成
    "widgetdocs 语料里已有的真实地址"，让新卡片引用同一个 URL 时
    自我背书通过——这正是 CR-047 指出的洞。这里用真实的 IndexBuilder/
    ChunkStore 建一个真实 current.db（不是纯 mock），同时放一条这样的
    历史 authored 块和一条真正的原生块，验证回退查询只认后者。
    """
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(store_mod, "INDEX_DIR", index_dir)
    monkeypatch.setattr(cards, "CURRENT", index_dir / "current.db")

    poisoned = Chunk(
        source_url="https://attacker.invalid/old-card-url",
        source_project="widgetdocs", version_or_commit="v0-poisoned",
        license="Apache-2.0", retrieved_at=utc_now(),
        title_path=["Old card", "Bad section"], technology="widgets",
        content_type="prose", locale="en", text="旧版卡片留下的伪造引用。",
        # 落在 _fake_registry() 里 scenario-cards 的 authored 路径前缀下，
        # 模拟这条块真的是历史 authored 同步产生的。
        source_path="knowledge/scenarios/old-card.md",
    )
    legit = Chunk(
        source_url="https://example.com/docs/real-page.html",
        source_project="widgetdocs", version_or_commit="v1-real",
        license="Apache-2.0", retrieved_at=utc_now(),
        title_path=["Real Docs", "Real section"], technology="widgets",
        content_type="prose", locale="en", text="widgetdocs 官方文档的真实正文。",
        source_path="docs/real-page.md",  # 非 authored 路径，真正的原生块
    )
    poisoned.validate()
    legit.validate()

    b = IndexBuilder("current")
    b.add([poisoned, legit], [_vec(0), _vec(1)])
    b.finalize({"widgetdocs": "v1-real"}, "synthetic")
    b.activate()

    text = CARD_OK.replace(
        "source: widgetdocs https://example.com/docs/lifecycle.html#draining",
        "source: widgetdocs https://attacker.invalid/old-card-url",
    ).replace(
        "source: widgetdocs https://example.com/docs/lifecycle.html#hooks",
        "source: widgetdocs https://example.com/docs/real-page.html",
    )
    _write_card(tmp_path, "widget.md", text)

    chunks, res = cards.collect_chunks(
        _authored_src(tmp_path), lambda *_: None, versions={}, known_urls={}
    )

    assert res.rejected == 1, "伪造 URL 即便已经在 current.db 里也不该被接受"
    assert any("已存在的真实块地址" in k for k in res.reject_reasons)
    assert len(chunks) == 1
    assert chunks[0].source_url == "https://example.com/docs/real-page.html", (
        "真实的原生块 URL 仍应能通过回退校验"
    )
    assert chunks[0].version_or_commit == "v1-real", (
        "版本回退同样必须排除 authored 块，不能取到被伪造的 v0-poisoned"
    )


# ---------- 卡片格式本身的拒绝（整份文件级别） ----------

@pytest.mark.parametrize("bad_text", [
    "no frontmatter at all\n\n## Heading\nsource: widgetdocs https://e.com/x\n\nbody\n",
    "---\ntitle: X\n---\n\nstray text before any heading\n\n## Heading\nsource: widgetdocs https://e.com/x\n\nbody\n",
    "---\ntitle: X\n---\n\n## Heading\nnot a source line\n\nbody\n",
    "---\ntitle: X\n---\n\n## Heading\nsource: widgetdocs https://e.com/x\n",  # source 之后没有正文
])
def test_malformed_card_file_is_rejected_wholesale(tmp_path, bad_text):
    """格式性错误（缺 frontmatter/标题前有正文/缺 source 行/source 后无正文）
    拒绝整份文件——这类错误意味着作者格式没写对，不是"部分内容可信"的情况。
    """
    _write_card(tmp_path, "bad.md", bad_text)
    chunks, res = cards.collect_chunks(
        _authored_src(tmp_path), lambda *_: None, versions={"widgetdocs": "abc123"}
    )
    assert chunks == []
    assert res.rejected == 1
    assert any("卡片格式错误" in k for k in res.reject_reasons)


def test_no_markdown_files_is_a_source_level_error(tmp_path):
    chunks, res = cards.collect_chunks(_authored_src(tmp_path), lambda *_: None)
    assert chunks == []
    assert res.error is not None


def test_missing_paths_directory_is_a_source_level_error(tmp_path):
    src = _authored_src(tmp_path, paths=(str(tmp_path / "does-not-exist"),))
    chunks, res = cards.collect_chunks(src, lambda *_: None)
    assert chunks == []
    assert res.error is not None


# ---------- 注册表对 authored 格式的校验 ----------

def _registry_yaml(tmp_path: Path, source: dict) -> Path:
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "sources": [source]}), encoding="utf-8")
    return p


def test_authored_source_loads_without_fetch_fields(tmp_path):
    p = _registry_yaml(tmp_path, {
        "id": "scenario-cards", "project": "scenario-cards", "technology": "scenario-cards",
        "format": "authored", "locale": "en", "paths": ["knowledge/scenarios"],
    })
    [s] = load_registry(p)
    assert s.format == "authored"
    assert s.repo is None and s.license is None and s.ref is None


def test_authored_source_rejects_fetch_only_fields(tmp_path):
    """写了 repo/license 这类拉取式字段就是伪造"有独立仓库/许可"的元数据。"""
    p = _registry_yaml(tmp_path, {
        "id": "scenario-cards", "project": "scenario-cards", "technology": "scenario-cards",
        "format": "authored", "locale": "en", "paths": ["knowledge/scenarios"],
        "repo": "https://github.com/example/should-not-be-here",
    })
    with pytest.raises(RegistryError, match="不适用"):
        load_registry(p)


def test_fetched_format_still_requires_fetch_fields(tmp_path):
    """确认拆分必填字段没有连带放松了拉取式来源本来的校验。"""
    p = _registry_yaml(tmp_path, {
        "id": "x", "project": "p", "technology": "t",
        "format": "markdown", "locale": "en", "paths": ["docs"],
        # 故意漏掉 repo/ref/license/license_file/base_url
    })
    with pytest.raises(RegistryError, match="缺少必填字段"):
        load_registry(p)


# ---- R79：卡片引文必须真的出自它引用的那一页（CR-089 的邻居） ----
#
# `_build_chunk()` 的 CR-045 校验只保证**引用的 URL** 在语料里真实存在，
# 完全不看引号里的内容是不是抄自那一页。R78 写第六张卡片时是用一个一次性
# 脚本核对的 43 段引文，当时就在自陈里登记了"没有留成回归"——CR-089 恰好
# 指出的是同一小节里另一种失真（标题把结论说过头），说明卡片正文这一层
# 确实需要机械判据。这一条把当时那个脚本固化下来。
#
# 判据的边界（§5.3，不作全称承诺）：它验的是**双引号里的每一段都能在被引用
# 那一页的正文里逐字找到**（归一化掉大小写/标点/空白/反引号，`…` 处切开），
# 因此卡片正文里的双引号被约定为"只用于引文"——想写强调或反问句，用单引号。
# 它**不**验证引文有没有被断章取义，也不验证卡片自己的推断部分是否正确。
_CARD_DIR = Path(__file__).resolve().parents[2] / "knowledge" / "scenarios"


def _quoted_spans(body: str) -> list[str]:
    """按双引号切开，取**奇数段**当引文。

    不用"配对正则"去匹配一整段引文：官方正文自己就带引号（`"check-and-set"`、
    `"undo the last change"`），配对正则要么把相邻两段引文连同中间的过渡语
    并成一段，要么在嵌套处错位——第一版就是这么误报的。改成按引号切分，
    偶数段是卡片作者的过渡语、奇数段是引文，嵌套与相邻两种情形都自然成立，
    前提正是本判据要求的那条约定：**卡片正文里的双引号只用于引文**。
    """
    parts = body.split('"')
    if len(parts) % 2 == 0:          # 引号不成对，本身就是格式错误
        raise AssertionError(f"卡片正文里的双引号不成对（共 {len(parts) - 1} 个）")
    return parts[1::2]


# 官方正文里的行内链接 `[restart policy](/docs/...)`，引用时按惯例只保留可读的
# 那部分（"its restart policy."）。归一化必须先把链接目标去掉，否则一段完全
# 忠实的引文会因为 URL 里的字母数字对不上而误报——第一版就误报了 k8s 卡片
# 里两段引文。
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# 同理，官方正文里的行内 LaTeX（k8s 那句判定式写成 `\\( a + b \\times c \\)`）
# 在卡片里按渲染后的样子引用（`a + b × c`）。这里把两种写法都折成同一个词，
# **只折这一组数学记号**，不做通用的 LaTeX 解析——列清楚它等同了什么，
# 免得日后有人以为这条判据能容忍任意改写（§5.3）。
_MATH_EQUIV = ((r"\times", "times"), ("×", "times"), (r"\\(", " "), (r"\\)", " "))


def _normalize_for_quote_match(text: str) -> str:
    text = _MD_LINK.sub(r"\1", text)
    for src, dst in _MATH_EQUIV:
        text = text.replace(src, dst)
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _card_sections(path: Path):
    body = path.read_text(encoding="utf-8")
    for sec in re.split(r"^## ", body, flags=re.M)[1:]:
        lines = sec.splitlines()
        m = re.match(r"source:\s+(\S+)\s+(\S+)", lines[1])
        assert m, f"{path.name} § {lines[0][:40]}: 第二行必须是 source 指令"
        yield lines[0], m.group(1), m.group(2), "\n".join(lines[2:])


@pytest.mark.skipif(not store_mod.CURRENT.exists(),
                    reason="需要本机已建好的 current.db 才能取到被引用页的正文（语料与索引都不入库）")
@pytest.mark.parametrize("card_path", sorted(_CARD_DIR.glob("*.md")), ids=lambda p: p.name)
def test_every_quoted_span_in_a_card_appears_in_the_cited_page(card_path):
    """判别性验证（写这条时实际做过）：把第六张卡片里一段引文改掉一个词，
    这条立刻失败；把它改回来即通过。它在 R78 那个一次性脚本下也确实抓出过
    两处会写错的地方（引文跨块、标点走样）。
    """
    store = ChunkStore(store_mod.CURRENT, check_dictionary=False)
    try:
        checked = 0
        for heading, project, url, body in _card_sections(card_path):
            # 比对范围是**整页**而不是"URL 逐字相同的那一块"：一页会被切成
            # 多块，很多来源还把锚点写进 source_url，于是被引用的那一条锚点
            # 地址常常只对应页面里的一小段，而引文来自同一页的另一块。第一版
            # 按 URL 逐字取块，六张卡片里有四张因此误报——判据自己搞错了范围。
            page_url = url.split("#")[0]
            rows = store.execute(
                "SELECT text FROM chunks WHERE source_project = ? "
                "AND (source_url = ? OR source_url LIKE ?) "
                "AND (source_path IS NULL OR source_path NOT LIKE 'knowledge/scenarios%') "
                # 必须按 id 排序再拼：一段引文可能横跨相邻两块，乱序拼接会把
                # 本来连续的正文接断，判据就会误报"引文找不到"（第一版如此）。
                "ORDER BY id",
                (project, page_url, page_url + "#%"),
            )
            assert rows, f"{card_path.name} § {heading[:40]}: 引用的页面在语料里没有对应的原生块"
            page = _normalize_for_quote_match(" ".join(r["text"] for r in rows))
            for quoted in _quoted_spans(body):
                for part in re.split(r"\u2026|\s\.\.\.\s", quoted):
                    part = part.strip(" ,.;")
                    if len(part) < 12:      # 太短的片段（字段名、单词）不足以定位，跳过
                        continue
                    checked += 1
                    assert _normalize_for_quote_match(part) in page, (
                        f"{card_path.name} § {heading[:40]}: 这段引文在被引用页里找不到"
                        f"（若本意是强调而非引用，请改用单引号）: {part[:80]!r}"
                    )
        # 这里**不**断言"每张卡片至少要有一段引文"。第一版这么写，结果把
        # `kafka-consumer-retry-dlq.md` 判失败——那张卡片通篇是转述（只在
        # `source:` 指令里给出处），而"卡片必须逐字引用"从来不是卡片契约
        # （见 architecture.md §5.3 与 services/sync/cards.py），是我顺手加的
        # 新规矩。**代价要写清楚**（§5.4：哪些输入会让它什么都不判）：全篇
        # 转述的卡片在这条判据下 checked=0，等于不受任何检查——它验的是
        # "引号里的话是不是原话"，验不了"转述得对不对"。
        if not checked:
            pytest.skip(f"{card_path.name} 通篇转述、没有双引号引文，这条判据对它不判任何东西")
    finally:
        store.close()
