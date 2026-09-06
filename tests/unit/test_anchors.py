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


def test_adoc_id_attribute_form_is_also_recognized_as_explicit_anchor():
    """T-018：Debezium 全篇用 `[id="..."]` 而非 `[[id]]` 声明显式锚点（两者语义等价）。

    实测 debezium.io/.../transformations/outbox-event-router.html 上，源里的
    `[id="options-for-applying-the-transformation-selectively"]` 确实原样发布为
    <h2 id="options-for-applying-the-transformation-selectively">；这一形式此前
    被 `_ADOC_BLOCK_ATTR` 当成普通块属性（如 [NOTE]）整行删除，锚点信息永久丢失，
    退化为 Asciidoctor 默认自动 id（`_options_for_applying_the_transformation_
    selectively`），实测该 id 在发布页面上不存在。
    """
    src = (
        '= T\n\n引言。\n\n[id="options-for-applying-the-transformation-selectively"]\n'
        "== Options for applying the transformation selectively\n\n"
        "这是正文，长度足够成块，不会被丢弃掉。\n"
    )
    secs = parse_asciidoc(src, "T")
    assert "options-for-applying-the-transformation-selectively" in _anchors(secs)
    assert "_options_for_applying_the_transformation_selectively" not in _anchors(secs)


def test_adoc_id_attribute_line_is_not_left_in_body_text():
    src = '= T\n\n引言。\n\n[id="a-b"]\n== 标题\n\n这是正文，长度足够成块。\n'
    secs = parse_asciidoc(src, "T")
    assert not any('[id="a-b"]' in s.body for s in secs)


def test_adoc_id_attribute_with_dollar_sign_matches_real_debezium_usage():
    """实测 debezium.io 上 MongoDB `$unset` 一节确实是 <h2 id="mongodb-$unset-handling">。"""
    src = '= T\n\n引言。\n\n[id="mongodb-$unset-handling"]\n== MongoDB `$unset` handling\n\n正文足够长成块。\n'
    secs = parse_asciidoc(src, "T")
    assert "mongodb-$unset-handling" in _anchors(secs)


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


def test_kafka_docsy_slug_preserves_underscores_and_emphasis_markers():
    """官网 4.3 的 config 与 ACL 页分别实测为这两个 id。"""
    src = (
        "### rack.aware.assignment.non_overlap_cost\n\n正文足够长。\n\n"
        "### _Behavior Without ACLs:_\n\n正文足够长。\n"
    )
    assert _anchors(parse_markdown(src, "Doc", "kafka")) == [
        "rackawareassignmentnon_overlap_cost", "_behavior-without-acls_",
    ]


def test_kubernetes_blackfriday_slug_decodes_entities_before_separating():
    """官网 v1.36 的 add-ons 页使用 id=visualization-control。"""
    secs = parse_markdown("## Visualization &amp; Control\n\n正文足够长。\n", "Doc", "kubernetes")
    assert _anchors(secs) == ["visualization-control"]
