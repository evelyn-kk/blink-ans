"""解析与切块的回归测试。

重点锁两类曾经真实出错的行为：
- 标题正则的 \\s 跨换行，会把 AsciiDoc 分隔符下一行误当标题（已修）。
- 代码块被切断——切断后的代码既不能执行也无法理解，是最没用的一类证据。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.sync.chunk import (  # noqa: E402
    MAX_TOKENS, _dedupe_path, _split_body, build_url, sections_to_chunks,
)
from services.sync.parse import Section, parse_asciidoc, parse_markdown  # noqa: E402
from services.sync.registry import Source  # noqa: E402


def src(**over) -> Source:
    base = dict(
        id="x", project="spring-data-redis", technology="redis",
        repo="https://github.com/spring-projects/spring-data-redis", ref="main",
        license="Apache-2.0", license_file="LICENSE.txt", format="asciidoc",
        locale="en", base_url="https://docs.spring.io/spring-data-redis/reference/",
        paths=("src",),
    )
    base.update(over)
    return Source(**base)


# ---------- 标题解析 ----------

def test_asciidoc_delimiter_not_parsed_as_heading():
    """==== 是示例块分隔符，其下一行不是标题。

    此前正则用 \\s+ 匹配标题与文本之间的空白，而 \\s 含换行，
    导致 "====\\n[source,java]" 被解析成标题 "[source,java]"。
    """
    text = "== 正文标题\n\n说明文字。\n\n====\n[source,java]\n----\nint x = 1;\n----\n====\n"
    secs = parse_asciidoc(text, "Doc")
    titles = [t for s in secs for t in s.title_path]
    assert "[source,java]" not in titles
    assert "正文标题" in titles


def test_markdown_heading_stack_builds_full_path():
    text = "# 一级\n\n引言\n\n## 二级\n\n内容 A\n\n### 三级\n\n内容 B\n\n## 另一个二级\n\n内容 C\n"
    secs = parse_markdown(text, "Doc")
    paths = [" › ".join(s.title_path) for s in secs]
    assert any(p.endswith("一级 › 二级 › 三级") for p in paths)
    # 回到二级时必须弹出三级，不能把不相关的上级标题拼进路径
    assert any(p.endswith("一级 › 另一个二级") for p in paths)
    assert not any("三级 › 另一个二级" in p for p in paths)


def test_heading_skip_level_does_not_leak_unrelated_parent():
    text = "# 一级\n\n引言\n\n### 跳级三级\n\n内容\n"
    secs = parse_markdown(text, "Doc")
    assert any(s.title_path[-1] == "跳级三级" for s in secs)


# ---------- 切块 ----------

def test_code_block_is_never_split():
    code = "```\n" + "\n".join(f"line_{i} = value_{i}" for i in range(200)) + "\n```"
    pieces = _split_body(f"前言段落。\n\n{code}\n\n结尾段落。")
    holding = [p for p in pieces if "line_0" in p]
    assert len(holding) == 1, "代码块被切散了"
    assert "line_199" in holding[0], "代码块被截断"


def test_long_prose_is_split_near_target():
    body = "\n\n".join("这是一段中文技术说明文字，用于验证切块长度控制。" * 3 for _ in range(30))
    pieces = _split_body(body)
    assert len(pieces) > 1
    from packages.schemas.chunk import estimate_tokens
    # 纯散文不应出现远超硬上限的块
    assert all(estimate_tokens(p) <= MAX_TOKENS * 1.5 for p in pieces)


def test_dedupe_adjacent_titles():
    """文件名派生的标题常与文首 H1 重复，会产生 "Appendix › Appendix"。"""
    assert _dedupe_path(["Appendix", "Appendix", "Schema"]) == ["Appendix", "Schema"]
    assert _dedupe_path(["Web", "Servlet"]) == ["Web", "Servlet"]


def _chunks(secs, s=None):
    return sections_to_chunks(
        secs, s or src(), Path("src/main/antora/modules/ROOT/pages/x.adoc"),
        "abc123", "2026-08-31T00:00:00+00:00",
    )


def test_stub_section_dropped_not_merged_forward():
    """整节内容不足下限时直接丢弃，绝不并入下一小节。

    代码评审发现的缺陷：早期实现把上一小节的尾巴并进下一小节的首块，
    产出了正文来自 A 节、却标注 #sec-b 的块。
    对一个以可追溯引用为卖点的产品，引用错位比丢一小段内容严重得多。
    """
    secs = [Section(["Doc", "A"], "[[anchor]]"), Section(["Doc", "B"], "这是一段足够长的正文内容。" * 6)]
    chunks = _chunks(secs)
    assert chunks
    assert all(c.token_estimate >= 20 for c in chunks)


def test_chunk_text_never_spans_two_sections():
    """不变量：每个块的正文完整来自单一小节，引用才不会张冠李戴。"""
    secs = [
        Section(["Doc", "A 节"], "A 节独有的尾巴内容。" * 3, anchor="sec-a"),
        Section(["Doc", "B 节"], "B 节独有的正文内容。" * 8, anchor="sec-b"),
    ]
    for c in _chunks(secs):
        if "#sec-b" in c.source_url:
            assert "A 节独有" not in c.text, "B 节的引用里混进了 A 节的正文"
        if "#sec-a" in c.source_url:
            assert "B 节独有" not in c.text


def test_trailing_short_content_not_silently_dropped():
    """小节末尾的短片段应并入同节前一块，而不是消失。"""
    body = "第一段足够长的正文内容用于形成独立块。" * 12 + "\n\n收尾。"
    chunks = _chunks([Section(["Doc", "A"], body, anchor="a")])
    assert any("收尾。" in c.text for c in chunks), "尾部内容被丢弃"


def test_oversized_prose_is_split_to_cap():
    """超上限的散文必须继续下切，否则永远进不了上下文预算。

    实测索引里曾出现 70203 token 的单块，而 2.5 秒预算只有约 879 token，
    这类块无论如何都不可能被检索选中。
    """
    body = "\n".join(f"* 第 {i} 条变更说明，描述某个配置项的行为调整。" for i in range(400))
    chunks = _chunks([Section(["Doc", "Notable changes"], body, anchor="nc")])
    assert len(chunks) > 1
    assert max(c.token_estimate for c in chunks) <= MAX_TOKENS * 1.5


# ---------- URL 回链 ----------

def test_url_asciidoc_root_module_omits_segment():
    """ROOT 模块在 Antora 的 URL 中不出现。"""
    u = build_url(src(), Path("src/main/antora/modules/ROOT/pages/redis/redis-cache.adoc"), "ttl")
    assert u == "https://docs.spring.io/spring-data-redis/reference/redis/redis-cache.html#ttl"


def test_url_asciidoc_named_module_is_kept():
    """非 ROOT 模块名必须保留在 URL 中。

    实测教训: Spring Boot 有 reference / how-to / api 等模块，
    早期实现只取 /pages/ 之后的部分，导致抽样 8 条链接全部 404。
    """
    s = src(project="spring-boot", base_url="https://docs.spring.io/spring-boot/")
    u = build_url(s, Path("documentation/spring-boot-docs/src/docs/antora/modules/reference/pages/using/build-systems.adoc"), "build-systems")
    assert u == "https://docs.spring.io/spring-boot/reference/using/build-systems.html#build-systems"

    u2 = build_url(s, Path("documentation/spring-boot-docs/src/docs/antora/modules/how-to/pages/logging.adoc"), None)
    assert u2 == "https://docs.spring.io/spring-boot/how-to/logging.html"


def test_url_markdown_hugo_layout():
    s = src(format="markdown", base_url="https://kubernetes.io/docs/")
    u = build_url(s, Path("content/en/docs/concepts/workloads/pods.md"), "lifecycle")
    assert u == "https://kubernetes.io/docs/concepts/workloads/pods/#lifecycle"


def test_url_docbook_sect1_becomes_page():
    s = src(format="docbook", base_url="https://www.postgresql.org/docs/17/")
    u = build_url(s, Path("doc/src/sgml/config.sgml"), "runtime-config-query", "runtime-config-query")
    assert u == "https://www.postgresql.org/docs/17/runtime-config-query.html"


def test_url_docbook_deep_sect_becomes_anchor_not_page():
    """PostgreSQL 只在 chapter/sect1 级别分页；深层 sect 当页名会 404。

    实测: /docs/current/collation-managing-create-libc.html 返回 404，
    而 /docs/17/collation.html 返回 200。
    """
    s = src(format="docbook", base_url="https://www.postgresql.org/docs/17/")
    u = build_url(s, Path("doc/src/sgml/charset.sgml"), "collation-managing-create-libc", "collation")
    assert u == "https://www.postgresql.org/docs/17/collation.html#collation-managing-create-libc"


def test_url_without_anchor_has_no_fragment():
    assert "#" not in build_url(src(), Path("src/main/antora/modules/ROOT/pages/x.adoc"), None)


# ---------- 代码评审发现的解析缺陷 ----------

def test_markdown_hash_comment_in_code_block_is_not_heading():
    """代码块内的 `# 注释` 是 shell 注释，不是标题。

    评审发现：不遮罩围栏会把一个 bash 块切成三段，
    并伪造出 `下载-tarball` 这类锚点写进 source_url——
    引用会指向官方页面上根本不存在的位置。影响 kubernetes 与 kafka。
    """
    md = "# 安装\n\n说明。\n\n```bash\n# 下载 tarball\ncurl -LO x\n# 解压\ntar xzf x\n```\n\n后续。\n"
    secs = parse_markdown(md, "Doc")
    assert len(secs) == 1
    assert secs[0].title_path[-1] == "安装"
    assert "下载 tarball" in secs[0].body


def test_asciidoc_attributes_inside_listing_are_preserved():
    """`[main]`、`:mode: fast` 在正文里是标记，在 listing 块里是配置内容。

    一律删除会静默损坏被当作证据引用的配置示例。
    """
    adoc = "== 配置\n\n说明文字。\n\n----\n[main]\n:mode: fast\nkey=value\n----\n\n结尾。\n"
    body = parse_asciidoc(adoc, "Doc")[-1].body
    assert "[main]" in body and ":mode: fast" in body


def test_navigation_page_is_dropped():
    """纯 xref 导航页没有技术结论，只会挤占索引与上下文预算。

    实测索引中最大的一块是 70203 token 的 Antora 重定向页。
    """
    from services.sync.parse import _is_navigation

    nav = " ".join(f"xref:ROOT:page{i}.adoc#a{i}[Page {i}]" for i in range(40))
    assert _is_navigation("Redirect", nav)
    assert _is_navigation("Acknowledgments", "Abhijit Menon-Sen Adnan Dautovic " * 20)
    assert not _is_navigation("Using EXPLAIN", "EXPLAIN ANALYZE shows the actual run time. " * 10)


def test_navigation_files_are_skipped():
    """Antora 的导航文件在站点上没有对应页面，入库只会产生 404 引用。

    实测残留: https://docs.spring.io/spring-boot/nav-reference.html 返回 404。
    """
    from services.sync.parse import parse_file

    for name in ("nav.adoc", "nav-reference.adoc", "nav_extra.md"):
        p = Path("/tmp") / name
        p.write_text("= 导航\n\n* xref:a.adoc[A]\n", encoding="utf-8")
        assert parse_file(p, src()) == [], f"{name} 未被跳过"
        p.unlink()
