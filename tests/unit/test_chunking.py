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


def test_fenced_example_is_not_merged_with_neighboring_prose():
    """完整示例要保留，但不应携带相邻说明一起越过上下文预算。"""
    prose = "说明文字。" * 100
    code = "```\n" + "line = value\n" * 30 + "```"
    pieces = _split_body(f"{prose}\n\n{code}\n\n{prose}")
    fenced = [p for p in pieces if "line = value" in p]
    assert len(fenced) == 1
    assert fenced[0].strip().startswith("```") and fenced[0].strip().endswith("```")


def test_definition_list_with_inline_snippets_is_not_over_fragmented():
    """CR-019：定义列表里逐条内联的短示例（单块内开合的完整围栏）不应被强制
    独立成块。旧逻辑对任何含围栏的块一律先封口、再单独切出，把这类小节拆成
    "引言 1 块 + 每条定义 2 块"（本例会拆成 12 块）；修复后应随常规大小控制
    合并成远少于这个数字的证据块。
    """
    intro = "以下字段用于配置存活探针。" * 3
    items = "\n\n".join(
        f"`field{i}` (int)：说明字段 {i} 的用途和取值范围，用于控制探测行为。"
        f"\n\n```\nfield{i}: 1\n```"
        for i in range(6)
    )
    pieces = _split_body(f"{intro}\n\n{items}")
    assert len(pieces) <= 3, f"定义列表被过度碎片化: {len(pieces)} 块"


def test_long_prose_is_split_near_target():
    body = "\n\n".join("这是一段中文技术说明文字，用于验证切块长度控制。" * 3 for _ in range(30))
    pieces = _split_body(body)
    assert len(pieces) > 1
    from packages.schemas.chunk import estimate_tokens
    # 纯散文不应出现远超硬上限的块
    assert all(estimate_tokens(p) <= MAX_TOKENS * 1.5 for p in pieces)


def test_single_unpunctuated_long_line_still_respects_hard_limit():
    """日志和压缩配置常没有换行或句号，不能借此绕过硬上限。"""
    from packages.schemas.chunk import estimate_tokens
    pieces = _split_body("setting=value " * 1_000)
    assert len(pieces) > 1
    assert all(estimate_tokens(p) <= MAX_TOKENS for p in pieces)


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


def test_url_markdown_drops_source_configured_path_characters():
    """Kafka 的 Geo-Replication 源文件带圆括号，官网 URL 不带。"""
    s = src(format="markdown", base_url="https://kafka.apache.org/43/", url_strip_prefix="docs/", url_path_drop_chars="()")
    u = build_url(s, Path("docs/operations/geo-replication-(cross-cluster-data-mirroring).md"), None)
    assert u == "https://kafka.apache.org/43/operations/geo-replication-cross-cluster-data-mirroring/"


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


def test_asciidoc_include_directive_is_not_evidence():
    """CR-084：`include-code::` / `include::` 引的正文不在这个文件里。

    我们不解析这两个指令，原样入库等于把一行指令当成证据。实测危害：
    spring-framework 的这一块（121 字符）在
    `keyword_search("Spring C3P0 configuration", technology="spring")`
    里排**第 1**，而真正的配置代码根本不在块内。
    """
    adoc = (
        "== Using `DataSource`\n\n"
        "The following example shows C3P0 configuration:\n\n"
        "include-code::./ComboPooledDataSourceConfiguration[tag=snippet,indent=0]\n"
    )
    body = parse_asciidoc(adoc, "Doc")[-1].body
    assert "include-code::" not in body
    assert "ComboPooledDataSourceConfiguration" not in body
    assert "C3P0 configuration" in body, "散文本身必须留下，删的只是指令行"


def test_asciidoc_plain_include_directive_is_also_dropped():
    """裸 `include::` 是同一个毛病，且更严重。

    实测合入前的库里有 6 块**整块只有 include 指令**（spring-data-redis 5 块、
    spring-kafka 1 块），去掉指令后正文为空。CR-084 只点名了 include-code::，
    这一条是同一次一并修掉的。
    """
    adoc = (
        "== Custom Implementations\n\n"
        "include::{commons}@data-commons::page$repositories/custom-implementations.adoc[]\n"
    )
    # 正文清空后整节被既有的空正文规则跳过，连 Section 都不产出。
    secs = [x for x in parse_asciidoc(adoc, "Doc")
            if x.title_path[-1] == "Custom Implementations"]
    assert secs == [], "整节只有一行 include 指令时不应产出任何小节"


def test_include_only_section_falls_below_min_tokens_and_is_dropped():
    """去掉指令行后，这类块靠既有的 MIN_TOKENS 门槛被丢弃——不需要新阈值。

    判别性：这条测的是"块数为 0"，而修复前同一段 AsciiDoc 会产出 1 块
    （30 token，指令行本身贡献了 19 个）。
    """
    from services.sync.chunk import MIN_TOKENS, estimate_tokens

    raw = ("The following example shows C3P0 configuration:\n\n"
           "include-code::./ComboPooledDataSourceConfiguration[tag=snippet,indent=0]")
    assert estimate_tokens(raw) >= MIN_TOKENS, "修复前它够长，所以当初进了索引"

    adoc = f"== Using `DataSource`\n\n{raw}\n"
    secs = parse_asciidoc(adoc, "Doc")
    chunks = sections_to_chunks(secs, src(), Path("modules/ROOT/pages/x.adoc"), "c", "2026-09-10T00:00:00Z")
    assert chunks == [], "只剩一句引子的块不构成证据"


def test_fenced_include_does_not_count_toward_evidence_length():
    """围栏内的 include 删不得（删了留下空代码块），但不得计入证据长度。

    实测两块 spring-kafka 内容（estimate 55 / 130 token）扣掉指令后只剩
    16 / 14 token——**整块只有 Antora 标签页脚手架和 include 行，一个字
    正文都没有**。与 CR-084 点名的 C3P0 块同一形状，只是样例包在围栏里。

    判别性说明（§5.2）：把 `chunk.py` 换回旧版跑这条用例，它是因为
    `evidence_tokens` 不存在而 ImportError——那属于弱判别性。真正的对照
    写在用例体内且不依赖新函数：同一段正文 `estimate_tokens >= MIN_TOKENS`
    （旧判据放它进来）而 `evidence_tokens < MIN_TOKENS`（新判据拦下）。
    """
    from services.sync.chunk import MIN_TOKENS, estimate_tokens, evidence_tokens

    # 取自合入前索引里的真实块（spring-kafka，estimate 130 / evidence 14）：
    # 整块只有 Antora 的标签页脚手架和 6 行 include，一个字正文都没有。
    body = (
        "======\nJava::\n+\n```\n"
        "include::{java-examples}/started/noboot/Sender.java[tag=startedNoBootSender]\n\n"
        "include::{java-examples}/started/noboot/Listener.java[tag=startedNoBootListener]\n"
        "```\nKotlin::\n+\n```\n"
        "include::{kotlin-examples}/started/noboot/Sender.kt[tag=startedNoBootSender]\n\n"
        "include::{kotlin-examples}/started/noboot/Config.kt[tag=startedNoBootConfig]\n"
        "```\n======"
    )
    assert estimate_tokens(body) >= MIN_TOKENS, "算上指令行它够长——旧判据就是这样放它进来的"
    assert evidence_tokens(body) < MIN_TOKENS, "只按正文算就不够格"

    adoc = f"== Testing\n\n{body}\n"
    chunks = sections_to_chunks(parse_asciidoc(adoc, "Doc"), src(),
                                Path("modules/ROOT/pages/x.adoc"), "c", "2026-09-10T00:00:00Z")
    assert chunks == []


def test_fenced_include_with_enough_prose_survives_intact():
    """正文充足的块必须原样留下，包括围栏里的 include 行——不能顺手删成空代码块。"""
    prose = ("If you define a `KafkaAdmin` bean in your application context, it can "
             "automatically add topics to the broker. To do so, you can add a "
             "`NewTopic` bean for each topic to the application context. " * 3)
    adoc = f"== Configuring Topics\n\n{prose}\n\n```\ninclude::{{java-examples}}/topics/Config.java[tag=bean]\n```\n"
    chunks = sections_to_chunks(parse_asciidoc(adoc, "Doc"), src(),
                                Path("modules/ROOT/pages/x.adoc"), "c", "2026-09-10T00:00:00Z")
    assert len(chunks) >= 1
    assert any("include::" in c.text for c in chunks), "围栏内的指令行必须留着，删了就是个空代码块"


def test_include_directive_inside_listing_is_preserved():
    """listing 块里的 include:: 是被展示的语法本身，删掉就损坏了示例。

    与 `test_asciidoc_attributes_inside_listing_are_preserved` 同一条理由。
    """
    adoc = ("== 怎么写 include\n\n说明文字足够长，用来保证这一节不会因为太短被丢弃，"
            "这里再补一些正文让它稳稳超过最小 token 门槛。\n\n"
            "----\ninclude::partial$foo.adoc[]\n----\n")
    body = parse_asciidoc(adoc, "Doc")[-1].body
    assert "include::partial$foo.adoc[]" in body


def test_include_directive_removal_keeps_surrounding_prose():
    """绝大多数含指令的块正文充足，只应少掉那一行，不得被整块丢弃。

    实测：合入前 277 个含 `include-code::` 的块里，去掉指令后跌破门槛的
    只有 3 块，其余 274 块正文完整保留。
    """
    adoc = (
        "== Connections\n\n"
        "The following section uses Spring's `DriverManagerDataSource` implementation.\n"
        "Several other `DataSource` variants are covered later.\n\n"
        "include-code::./DriverManagerDataSourceConfiguration[tag=snippet,indent=0]\n\n"
        "The next two examples show the basic connectivity and configuration for DBCP and C3P0.\n"
    )
    body = parse_asciidoc(adoc, "Doc")[-1].body
    assert "include-code::" not in body
    assert "DriverManagerDataSource` implementation" in body
    assert "DBCP and C3P0" in body


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
