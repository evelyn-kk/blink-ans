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


def test_tiny_section_merged_not_dropped():
    """过短的小节并入相邻块，而不是丢弃——丢弃会让锚点断链。"""
    secs = [Section(["Doc", "A"], "[[anchor]]"), Section(["Doc", "B"], "这是一段足够长的正文内容。" * 6)]
    chunks = sections_to_chunks(secs, src(), Path("src/main/antora/modules/ROOT/pages/x.adoc"), "abc123", "2026-08-31T00:00:00+00:00")
    assert len(chunks) >= 1
    assert all(c.token_estimate >= 20 for c in chunks)


# ---------- URL 回链 ----------

def test_url_asciidoc_antora_layout():
    u = build_url(src(), Path("src/main/antora/modules/ROOT/pages/redis/redis-cache.adoc"), "ttl")
    assert u == "https://docs.spring.io/spring-data-redis/reference/redis/redis-cache.html#ttl"


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
