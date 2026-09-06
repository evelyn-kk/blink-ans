"""场景卡片（authored 来源）的解析、引用解析与注册表校验。

卡片本身不产生新的证据实体：每个 `## ` 小节的 source_url/source_project/
version_or_commit/license/technology 全部照抄它引用的真实来源。这里锁住的是
那条"照抄"链路本身的判别性行为——尤其是"一个坏小节不该拖垮同一张卡片里
其它小节"和"不能引用未登记/受限来源"这两条，都是本来很容易被静默放过的地方。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.sync import cards  # noqa: E402
from services.sync.registry import RegistryError, Source, load_registry  # noqa: E402


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
    monkeypatch.setattr(cards, "_lookup_current_version", lambda project: "stale-from-disk")

    chunks, _res = cards.collect_chunks(
        _authored_src(tmp_path), lambda *_: None,
        versions={"widgetdocs": "fresh-this-run"}, known_urls=KNOWN_URLS,
    )
    assert all(c.version_or_commit == "fresh-this-run" for c in chunks)


def test_version_falls_back_to_current_index_when_absent_from_this_run(tmp_path, monkeypatch):
    _write_card(tmp_path, "widget.md", CARD_OK)
    monkeypatch.setattr(cards, "_lookup_current_version", lambda project: "from-disk")

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
    monkeypatch.setattr(cards, "_lookup_current_version", lambda project: None)
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
    monkeypatch.setattr(cards, "_lookup_current_urls", lambda project: {"https://stale.example/x"})

    chunks, res = cards.collect_chunks(
        _authored_src(tmp_path), lambda *_: None,
        versions={"widgetdocs": "abc123"}, known_urls=KNOWN_URLS,
    )
    assert res.rejected == 0
    assert len(chunks) == 2


def test_known_urls_falls_back_to_current_index_when_absent_from_this_run(tmp_path, monkeypatch):
    _write_card(tmp_path, "widget.md", CARD_OK)
    monkeypatch.setattr(cards, "_lookup_current_urls", lambda project: set(KNOWN_URLS["widgetdocs"]))

    chunks, res = cards.collect_chunks(
        _authored_src(tmp_path), lambda *_: None, versions={"widgetdocs": "abc123"}, known_urls={}
    )
    assert res.rejected == 0
    assert len(chunks) == 2


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
