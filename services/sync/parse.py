"""把各来源的文档格式解析为带标题层级的小节。

只解析技术正文，过滤导航、营销和无关页面（development.md 第 4 节第 3 条）。
输出统一为 Section，交给 chunk.py 做长度切分。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .registry import Source


@dataclass
class Section:
    title_path: list[str]
    body: str
    anchor: str | None = None
    content_type: str = "prose"
    # DocBook 专用：HTML 分页依据的 sect1/chapter id。
    # 深层 sect 不是独立页面，只能作为该页的锚点。
    page_id: str | None = None


# 这些文件是导航、索引或贡献指南，不含技术结论
_SKIP_NAMES = {"_index.md", "index.adoc", "toc.html", "OWNERS"}
# Antora 的导航文件不止 nav.adoc，还有 nav-reference.adoc 等变体；
# 它们在站点上没有对应页面，入库只会产生 404 引用。
_SKIP_NAME_PATTERNS = (re.compile(r"^nav[-_.].*\.(adoc|md)$", re.I), re.compile(r"^nav\.(adoc|md)$", re.I))
_SKIP_PATTERNS = (
    re.compile(r"^\s*$"),
    re.compile(r"^(TODO|WIP)\b", re.I),
)

# 纯导航页与致谢名单没有技术结论，只会挤占索引和上下文预算。
# 实测最大的一块是 70203 token 的 Antora 重定向页，全是 xref 链接。
_NAV_TITLES = re.compile(
    r"^(redirect|acknowledg\w*|contributors?|index|table of contents|nav|附录目录)$", re.I
)
_LINK_MARKUP = re.compile(r"xref:[^\[]*\[[^\]]*\]|link:\S+\[[^\]]*\]|https?://\S+")


def _is_navigation(title: str, body: str) -> bool:
    if _NAV_TITLES.match(title.strip()):
        return True
    if len(body) < 200:
        return False
    # 链接标记占正文一半以上时，这是目录页而非技术正文
    return len(_LINK_MARKUP.findall(body)) * 20 > len(body) * 0.5


_FENCE_LINE = re.compile(r"^(```|~~~)", re.M)


def _mask_fenced(text: str) -> str:
    """把围栏代码块内的字符替换为占位符，仅用于标题匹配。

    换行保留，长度不变，因此匹配到的偏移量在原文里依然有效。

    存在的理由：Markdown 代码块里的 `# 下载 tarball` 是 shell 注释，不是标题。
    不遮罩会把一个 bash 块切成三段，并伪造出 `下载-tarball` 这类锚点写进 source_url——
    引用会指向一个官方页面上根本不存在的位置。
    """
    out = list(text)
    inside = False
    for m in _FENCE_LINE.finditer(text):
        line_end = text.find("\n", m.start())
        line_end = len(text) if line_end == -1 else line_end
        if inside:
            inside = False
            continue
        inside = True
        # 找到配对的收尾围栏
        nxt = _FENCE_LINE.search(text, line_end)
        stop = nxt.end() if nxt else len(text)
        for i in range(m.start(), stop):
            if out[i] != "\n":
                out[i] = "\x00"
        inside = False
    return "".join(out)


def _anchor(title: str) -> str:
    """由标题生成 URL 锚点，与主流静态站点生成器的规则保持一致。"""
    s = re.sub(r"[^\w\s-]", "", title.lower())
    return re.sub(r"[\s_]+", "-", s).strip("-")


def _classify(body: str) -> str:
    """判断正文以何种内容为主，供检索期按 content_type 过滤。"""
    fenced = len(re.findall(r"```|^----$|<programlisting", body, re.M))
    if fenced and len(re.findall(r"^\s*[\w.-]+\s*[:=]", body, re.M)) > 3:
        return "config"
    if fenced >= 2:
        return "code" if len(body) < 400 else "mixed"
    if body.count("|") > 8:
        return "table"
    return "prose"


# ---------- Markdown ----------

_MD_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
# 注意用 [ \t]+ 而非 \s+：\s 含换行，会把分隔符下一行的内容误当作标题文本
_MD_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.M)
# Hugo/Docsy 短代码与 HTML 注释不是技术正文
_MD_SHORTCODE = re.compile(r"\{\{[<%].*?[%>]\}\}", re.S)
_MD_COMMENT = re.compile(r"<!--.*?-->", re.S)


def parse_markdown(text: str, doc_title: str) -> list[Section]:
    title = doc_title
    if m := _MD_FRONTMATTER.search(text):
        if t := re.search(r"^title:\s*(.+?)\s*$", m.group(1), re.M):
            title = t.group(1).strip("\"'")
        text = text[m.end():]

    text = _MD_COMMENT.sub("", _MD_SHORTCODE.sub("", text))
    return _split_by_headings(text, _MD_HEADING, [title], _md_heading_parts)


# Hugo 的显式锚点写在标题末尾：`## Increase the load {#increase-load}`。
# 它既不是标题文字，也不能靠 slug 推出来——Kubernetes 语料里 650 处标题这么写，
# 按标题文字生成的锚点与官网实际 id 对不上，链接会落到页面顶部。
_MD_EXPLICIT_ID = re.compile(r"\s*\{#([^}\s]+)\}\s*$")


def _md_heading_parts(m) -> tuple[int, str, str | None]:
    title = m.group(2)
    if e := _MD_EXPLICIT_ID.search(title):
        return len(m.group(1)), title[: e.start()].rstrip(), e.group(1)
    return len(m.group(1)), title, _anchor(title)


# ---------- AsciiDoc ----------

# 标题前一行的 [[id]] 是 AsciiDoc 的显式锚点，Antora 原样发布为 <h2 id="...">。
# Spring 的文档几乎每个标题都写了（spring-boot 1028 个标题对 1032 个锚点），
# 而它们是 `documentation.first-steps` 这种点号命名，slug 化标题文字永远推不出来。
_ADOC_HEADING = re.compile(
    r"^(?:\[\[([\w.:$-]+)\]\][ \t]*\r?\n(?:[ \t]*\r?\n)*)?(={1,6})[ \t]+(.+?)[ \t]*$", re.M
)
_ADOC_ATTR = re.compile(r"^:[\w-]+:.*$", re.M)
# 块属性行（[source,java] / [NOTE] 等）不是正文，但要保留语言信息给代码块
_ADOC_SRC_ATTR = re.compile(r"^\[source[^\]]*\]\s*$", re.M)
_ADOC_BLOCK_ATTR = re.compile(r"^\[[A-Za-z][^\]]*\]\s*$", re.M)
_ADOC_FENCE = re.compile(r"^----+\s*$", re.M)


def parse_asciidoc(text: str, doc_title: str) -> list[Section]:
    # 先统一围栏，再逐段清理——属性行清理必须跳过代码块内部。
    # `[main]`、`:mode: fast` 在正文里是 AsciiDoc 标记，在 listing 块里却是
    # 配置文件的真实内容；一律删除会静默损坏被当作证据引用的配置示例。
    text = _ADOC_FENCE.sub("```", text)

    parts = text.split("```")
    for i in range(0, len(parts), 2):      # 偶数段在代码块之外
        parts[i] = _ADOC_ATTR.sub("", parts[i])
        parts[i] = _ADOC_SRC_ATTR.sub("", parts[i])
        parts[i] = _ADOC_BLOCK_ATTR.sub("", parts[i])
    text = "```".join(parts)
    def parts(m) -> tuple[int, str, str | None]:
        level, title = len(m.group(2)), m.group(3)
        # 一级 `=` 是页标题，Antora 渲染成 <h1 id="page-title">：
        # 源里写的 [[using.build-systems]] 也好、标题 slug 也好，页面上都不存在。
        # 带锚点的链接照样 200，只是停在页面顶部——正确的引用就是不带锚点的页地址。
        if level == 1:
            return level, title, None
        return level, title, m.group(1) or _adoc_auto_id(title)

    return _split_by_headings(text, _ADOC_HEADING, [doc_title], parts)


def _adoc_auto_id(title: str) -> str:
    """标题没写显式锚点时，Asciidoctor 自动生成的 id。

    规则与通用 slug 不同：非字母数字一律换成 `_`（不是 `-`）、合并连续分隔符、
    去掉首尾分隔符，再冠以 `_` 前缀（Asciidoctor 的 idprefix/idseparator 默认值）。
    `== Why Spring Data Redis?` 生成的是 `_why_spring_data_redis`，
    按通用 slug 猜成 `why-spring-data-redis` 就对不上。
    """
    body = re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]", "_", title)).strip("_").lower()
    return f"_{body}"


# ---------- HTML ----------

_TAG = re.compile(r"<[^>]+>")
_HTML_HEADING = re.compile(r"<h([1-6])[^>]*>(.*?)</h\1>", re.I | re.S)
_HTML_DROP = re.compile(r"<(script|style|nav|header|footer)\b.*?</\1>", re.I | re.S)


def parse_html(text: str, doc_title: str) -> list[Section]:
    text = _HTML_DROP.sub("", text)
    parts = _HTML_HEADING.split(text)
    sections: list[Section] = []
    stack = [doc_title]
    # split 后结构为 [前言, level, title, body, level, title, body, ...]
    if parts[0].strip():
        sections.append(Section(list(stack), _clean_html(parts[0])))
    for i in range(1, len(parts) - 2, 3):
        level, raw_title, body = int(parts[i]), _clean_html(parts[i + 1]), parts[i + 2]
        stack = stack[:level] + [raw_title]
        cleaned = _clean_html(body)
        if cleaned.strip():
            sections.append(Section(list(stack), cleaned, _anchor(raw_title)))
    return sections


def _clean_html(s: str) -> str:
    s = re.sub(r"<pre[^>]*>(.*?)</pre>", lambda m: "\n```\n" + _TAG.sub("", m.group(1)) + "\n```\n", s, flags=re.S | re.I)
    s = _TAG.sub(" ", s)
    for ent, ch in (("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"), ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")):
        s = s.replace(ent, ch)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", s)).strip()


# ---------- DocBook (PostgreSQL) ----------

_DB_SECT = re.compile(r"<(sect[1-5]|chapter)\b[^>]*\bid=[\"']([^\"']+)[\"'][^>]*>", re.I)
_DB_TITLE = re.compile(r"<title>(.*?)</title>", re.I | re.S)


def parse_docbook(text: str, doc_title: str) -> list[Section]:
    """DocBook SGML 的宽松解析。

    PostgreSQL 的文档源含 SGML 实体，标准 XML 解析器会直接报错，
    因此按 sect 标签切分后做标签剥离，而不是构建 DOM。
    """
    sections: list[Section] = []
    marks = list(_DB_SECT.finditer(text))
    page_id: str | None = None      # 最近的 sect1/chapter id —— PostgreSQL 的 HTML 按此分页
    stack: list[str] = []

    for i, m in enumerate(marks):
        tag, sect_id = m.group(1).lower(), m.group(2)
        # chapter 与 sect1 各自成页；sect2 及更深只是页内锚点
        level = 1 if tag == "chapter" else int(tag[-1]) + 1
        if level <= 2:
            page_id = sect_id

        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        block = text[m.end():end]
        title_m = _DB_TITLE.search(block)
        title = _clean_html(title_m.group(1)) if title_m else sect_id
        stack = stack[: level - 1] + [""] * max(0, level - 1 - len(stack)) + [title]

        body = _DB_TITLE.sub("", block, count=1)
        body = re.sub(r"<programlisting[^>]*>(.*?)</programlisting>",
                      lambda x: "\n```\n" + _TAG.sub("", x.group(1)) + "\n```\n",
                      body, flags=re.S | re.I)
        cleaned = _clean_html(body)
        if cleaned.strip():
            # PostgreSQL 发布的 HTML 把 SGML 源里的小写 id 全部转成大写
            # （源 runtime-config-query-constants → 页面 RUNTIME-CONFIG-QUERY-CONSTANTS）。
            # 片段标识符区分大小写，照抄源 id 的链接点开只会停在页面顶部。
            sec = Section([doc_title, *[t for t in stack if t]], cleaned, sect_id.upper())
            # 页面 id 与锚点分开存放，供 build_url 还原 <page>.html#<anchor>
            sec.page_id = page_id or sect_id
            sections.append(sec)
    return sections


# ---------- 通用 ----------

def _split_by_headings(text, pattern, root_path, extract) -> list[Section]:
    """按标题层级切分，并维护标题栈以还原完整定位路径。

    `extract(m)` 返回 (层级, 标题, 锚点)；锚点为 None 表示该标题在发布页面上
    没有对应的 id，引用应当只给页地址。

    title_path 是引用可读性的关键——回答里显示 "Spring Boot › Web › Servlet › 嵌入式容器"
    远比只给一个 URL 有用，用户能立刻判断这条证据是否对得上自己的问题。
    """
    sections: list[Section] = []
    # 标题只在代码块之外匹配；正文仍从原文切取
    marks = list(pattern.finditer(_mask_fenced(text)))

    lead = text[: marks[0].start()] if marks else text
    if lead.strip():
        sections.append(Section(list(root_path), lead.strip(), content_type=_classify(lead)))

    stack: list[str] = []
    for i, m in enumerate(marks):
        level, title, anchor = extract(m)
        title = title.strip()
        # 标题栈：level 1 位于栈底。跳级（如 h1 直接到 h3）时用空位补齐，
        # 避免把不相关的上级标题拼进路径。
        stack = stack[: level - 1] + [""] * max(0, level - 1 - len(stack)) + [title]

        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end].strip()
        if not body or any(p.match(body) for p in _SKIP_PATTERNS):
            continue
        if _is_navigation(title, body):
            continue

        path = list(root_path) + [t for t in stack if t]
        # 锚点由 extract 决定：显式锚点优先、没写时回退 slug，
        # 而"这个标题在发布页面上根本没有 id"必须能表达为 None——
        # 在这里统一兜底会把它又变回一个不存在的 slug。
        sections.append(Section(path, body, anchor, _classify(body)))

    return sections


_PARSERS = {
    "markdown": parse_markdown,
    "asciidoc": parse_asciidoc,
    "html": parse_html,
    "docbook": parse_docbook,
}


def parse_file(path: Path, src: Source) -> list[Section]:
    """解析单个文件。无法识别或全是导航内容时返回空列表。"""
    if path.name in _SKIP_NAMES or any(p.match(path.name) for p in _SKIP_NAME_PATTERNS):
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []
    doc_title = path.stem.replace("-", " ").replace("_", " ").title()
    return _PARSERS[src.format](text, doc_title)
