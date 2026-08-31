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


# 这些文件是导航、索引或贡献指南，不含技术结论
_SKIP_NAMES = {"nav.adoc", "_index.md", "index.adoc", "toc.html", "OWNERS"}
_SKIP_PATTERNS = (
    re.compile(r"^\s*$"),
    re.compile(r"^(TODO|WIP)\b", re.I),
)


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
    return _split_by_headings(text, _MD_HEADING, [title],
                             lambda m: (len(m.group(1)), m.group(2)))


# ---------- AsciiDoc ----------

_ADOC_HEADING = re.compile(r"^(={1,6})[ \t]+(.+?)[ \t]*$", re.M)
_ADOC_ATTR = re.compile(r"^:[\w-]+:.*$", re.M)
# 块属性行（[source,java] / [NOTE] 等）不是正文，但要保留语言信息给代码块
_ADOC_SRC_ATTR = re.compile(r"^\[source[^\]]*\]\s*$", re.M)
_ADOC_BLOCK_ATTR = re.compile(r"^\[[A-Za-z][^\]]*\]\s*$", re.M)
_ADOC_FENCE = re.compile(r"^----+\s*$", re.M)


def parse_asciidoc(text: str, doc_title: str) -> list[Section]:
    text = _ADOC_ATTR.sub("", text)
    text = _ADOC_SRC_ATTR.sub("", text)
    text = _ADOC_BLOCK_ATTR.sub("", text)
    # 统一代码块标记，让下游的代码块检测只需认一种围栏
    text = _ADOC_FENCE.sub("```", text)
    return _split_by_headings(text, _ADOC_HEADING, [doc_title],
                              lambda m: (len(m.group(1)), m.group(2)))


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
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        block = text[m.end():end]
        title_m = _DB_TITLE.search(block)
        title = _clean_html(title_m.group(1)) if title_m else m.group(2)
        body = _DB_TITLE.sub("", block, count=1)
        body = re.sub(r"<programlisting[^>]*>(.*?)</programlisting>",
                      lambda x: "\n```\n" + _TAG.sub("", x.group(1)) + "\n```\n",
                      body, flags=re.S | re.I)
        cleaned = _clean_html(body)
        if cleaned.strip():
            sections.append(Section([doc_title, title], cleaned, m.group(2)))
    return sections


# ---------- 通用 ----------

def _split_by_headings(text, pattern, root_path, extract) -> list[Section]:
    """按标题层级切分，并维护标题栈以还原完整定位路径。

    title_path 是引用可读性的关键——回答里显示 "Spring Boot › Web › Servlet › 嵌入式容器"
    远比只给一个 URL 有用，用户能立刻判断这条证据是否对得上自己的问题。
    """
    sections: list[Section] = []
    marks = list(pattern.finditer(text))

    lead = text[: marks[0].start()] if marks else text
    if lead.strip():
        sections.append(Section(list(root_path), lead.strip(), content_type=_classify(lead)))

    stack: list[str] = []
    for i, m in enumerate(marks):
        level, title = extract(m)
        title = title.strip()
        # 标题栈：level 1 位于栈底。跳级（如 h1 直接到 h3）时用空位补齐，
        # 避免把不相关的上级标题拼进路径。
        stack = stack[: level - 1] + [""] * max(0, level - 1 - len(stack)) + [title]

        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end].strip()
        if not body or any(p.match(body) for p in _SKIP_PATTERNS):
            continue

        path = list(root_path) + [t for t in stack if t]
        sections.append(Section(path, body, _anchor(title), _classify(body)))

    return sections


_PARSERS = {
    "markdown": parse_markdown,
    "asciidoc": parse_asciidoc,
    "html": parse_html,
    "docbook": parse_docbook,
}


def parse_file(path: Path, src: Source) -> list[Section]:
    """解析单个文件。无法识别或全是导航内容时返回空列表。"""
    if path.name in _SKIP_NAMES:
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []
    doc_title = path.stem.replace("-", " ").replace("_", " ").title()
    return _PARSERS[src.format](text, doc_title)
