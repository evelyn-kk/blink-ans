"""锚点推导规则（实测于 2026-09-01 的官方站点）。

页面返回 200 不代表引用正确：锚点错了链接照样 200，只是停在页面顶部。
实测发现四种各不相同的规则，靠"生成一个 slug"一律处理会全错——
修复前抽样 60 条锚点只有 1 条真实存在。

每条规则都附实测出处，改动时须用 `kb verify-links --check-anchors` 重新核对。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.sync.parse import parse_asciidoc, parse_markdown  # noqa: E402


def _anchors(sections) -> list[str | None]:
    return [s.anchor for s in sections]


# ---------- AsciiDoc / Antora（Spring）----------

def test_adoc_explicit_id_wins_over_title_slug():
    """Spring 的文档几乎每个标题都写了 [[id]]，且是点号命名，
    slug 化标题永远推不出来。实测 docs.spring.io 上是
    <h2 id="documentation.first-steps">。"""
    src = "= Documentation Overview\n\n引言\n\n[[documentation.first-steps]]\n== First Steps\n\n正文内容足够长以免被丢弃。\n"
    secs = parse_asciidoc(src, "Documentation")
    assert "documentation.first-steps" in _anchors(secs)
    assert "first-steps" not in _anchors(secs)


def test_adoc_page_title_gets_no_anchor():
    """一级 `=` 是页标题，Antora 渲染成 <h1 id="page-title">：
    源里的 [[using.build-systems]] 在页面上并不存在，引用应只给页地址。"""
    src = "[[using.build-systems]]\n= Build Systems\n\n这里是页首正文，需要足够长才会成块。\n"
    secs = parse_asciidoc(src, "Build Systems")
    assert _anchors(secs) == [None] * len(secs)


def test_adoc_without_explicit_id_uses_asciidoctor_auto_id():
    """没写 [[id]] 时 Asciidoctor 用 `_` 而非 `-` 生成，且带下划线前缀。
    实测 spring-data/redis/reference/4.1/redis.html 上是
    <h2 id="_why_spring_data_redis">。"""
    src = "= Redis\n\n引言部分。\n\n== Why Spring Data Redis?\n\n正文内容足够长以免被丢弃。\n"
    secs = parse_asciidoc(src, "Redis")
    assert "_why_spring_data_redis" in _anchors(secs)


def test_adoc_anchor_line_is_not_left_in_body_text():
    """[[id]] 是标记不是正文，混进证据里会被当成内容送给模型。"""
    src = "= T\n\n引言。\n\n[[a.b]]\n== 标题\n\n这是正文，长度足够成块。\n"
    secs = parse_asciidoc(src, "T")
    assert not any("[[a.b]]" in s.body for s in secs)


def test_adoc_block_attribute_is_not_mistaken_for_anchor():
    """[NOTE] 这类块属性行不是锚点，不能被当成 id。"""
    src = "= T\n\n引言。\n\n[NOTE]\n== 标题\n\n这是正文，长度足够成块。\n"
    secs = parse_asciidoc(src, "T")
    assert "NOTE" not in _anchors(secs)


# ---------- Markdown / Hugo（Kubernetes、Kafka）----------

def test_md_explicit_id_wins_and_leaves_title_clean():
    """Hugo 把显式锚点写在标题末尾：`## Increase the load {#increase-load}`。
    Kubernetes 语料里 650 处标题这么写；不识别的话锚点和标题路径同时被污染。"""
    src = "## Increase the load {#increase-load}\n\n正文内容足够长以免被丢弃。\n"
    secs = parse_markdown(src, "Doc")
    assert _anchors(secs) == ["increase-load"]
    assert all("{#" not in " ".join(s.title_path) for s in secs)


def test_md_without_explicit_id_falls_back_to_title_slug():
    """实测 kafka.apache.org/43/operations/monitoring 上
    `## Producer Monitoring` 就是 <h2 id=producer-monitoring>。"""
    src = "## Producer Monitoring\n\n正文内容足够长以免被丢弃。\n"
    secs = parse_markdown(src, "Doc")
    assert _anchors(secs) == ["producer-monitoring"]


def test_md_slug_drops_slashes_without_leaving_separators():
    """`## Common monitoring metrics for producer/consumer/connect/streams`
    在页面上是 id=common-monitoring-metrics-for-producerconsumerconnectstreams。"""
    src = "## Common monitoring metrics for producer/consumer/connect/streams\n\n正文内容足够长以免被丢弃。\n"
    secs = parse_markdown(src, "Doc")
    assert _anchors(secs) == ["common-monitoring-metrics-for-producerconsumerconnectstreams"]
