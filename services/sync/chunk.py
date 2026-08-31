"""把解析出的小节切成可检索、可引用的块。

块长度目标来自 I0 实测，不是拍脑袋定的：
prefill 352 tok/s，2.5 秒预算对应约 879 token 上下文。
要在预算内塞进 3–5 段证据加提示词模板，每块须控制在 250 token 量级。
块过大会直接吃掉首 token 时延，过小则证据不成句、失去上下文。
"""

from __future__ import annotations

import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.schemas.chunk import Chunk, estimate_tokens, utc_now  # noqa: E402

from .parse import Section  # noqa: E402
from .registry import Source  # noqa: E402

TARGET_TOKENS = 250      # 单块目标长度
MAX_TOKENS = 400         # 硬上限，超过必须切
MIN_TOKENS = 20          # 低于此长度不成为独立证据，并入相邻块或丢弃

_CODE_FENCE = re.compile(r"^```", re.M)
_ADOC_ANCHOR = re.compile(r"^\[\[([\w.-]+)\]\]\s*$", re.M)


def build_url(src: Source, rel_path: Path, anchor: str | None, page_id: str | None = None) -> str:
    """由仓库内路径还原官方站点的可点击地址。

    引用必须回链到 source_url 并显示版本或抓取日期（architecture.md 5.2），
    因此这里的映射规则错了会让整条引用链失效——每个来源的规则单独写、单独测。
    """
    p = rel_path.as_posix()
    frag = f"#{anchor}" if anchor else ""

    if src.url_strip_prefix and p.startswith(src.url_strip_prefix):
        p = p[len(src.url_strip_prefix):]

    # 显式模板优先：上游站点结构与仓库目录结构不一定对应
    # （Kafka 的仓库已迁到 Hugo，但官网仍是单页锚点形式）
    if src.url_template:
        stem = p[: p.rfind(".")] if "." in Path(p).name else p
        return src.url_template.format(path=stem, anchor=anchor or "", base=src.base_url.rstrip("/"))

    if src.format == "asciidoc":
        # Antora 布局: .../modules/<module>/pages/<name>.adoc -> <name>.html
        if "/pages/" in p:
            p = p.split("/pages/", 1)[1]
        else:
            p = Path(p).name
        return f"{src.base_url.rstrip('/')}/{p[:-5] if p.endswith('.adoc') else p}.html{frag}"

    if src.format == "markdown":
        # Hugo 布局: content/en/docs/<rest>.md -> <rest>/
        for prefix in ("content/en/docs/", "content/en/", "content/"):
            if p.startswith(prefix):
                p = p[len(prefix):]
                break
        p = p[:-3] if p.endswith(".md") else p
        return f"{src.base_url.rstrip('/')}/{p}/{frag}"

    if src.format == "docbook":
        # PostgreSQL 的 HTML 只在 chapter / sect1 级别分页；
        # 更深的 sect 是页内锚点，直接当页名会产生 404。
        page = page_id or anchor or Path(p).stem
        tail = f"#{anchor}" if anchor and anchor != page else ""
        return f"{src.base_url.rstrip('/')}/{page}.html{tail}"

    # html: 保留原文件名
    return f"{src.base_url.rstrip('/')}/{Path(p).name}{frag}"


def _dedupe_path(path: list[str]) -> list[str]:
    """去掉相邻重复的标题层级。

    文件名派生的文档标题常与文首 H1 重复，会产生 "Appendix › Appendix" 这类冗余路径。
    """
    out: list[str] = []
    for t in path:
        if not out or out[-1].lower() != t.lower():
            out.append(t)
    return out


def _split_body(body: str) -> list[str]:
    """按段落切分长正文，且不切开代码块。

    代码块被切断后既不能执行也无法理解，是检索结果里最没用的一类证据。
    """
    blocks = re.split(r"\n\s*\n", body)
    out: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    in_code = False

    for b in blocks:
        if not b.strip():
            continue
        fences = len(_CODE_FENCE.findall(b))
        t = estimate_tokens(b)

        # 代码块内部一律不切，哪怕超过硬上限
        if in_code or (buf_tokens + t <= MAX_TOKENS) or not buf:
            buf.append(b)
            buf_tokens += t
        else:
            out.append("\n\n".join(buf))
            buf, buf_tokens = [b], t

        if fences % 2 == 1:
            in_code = not in_code

        if not in_code and buf_tokens >= TARGET_TOKENS:
            out.append("\n\n".join(buf))
            buf, buf_tokens = [], 0

    if buf:
        out.append("\n\n".join(buf))
    return out


def sections_to_chunks(
    sections: list[Section], src: Source, rel_path: Path, commit: str, retrieved_at: str
) -> list[Chunk]:
    chunks: list[Chunk] = []
    carry: str = ""          # 过短的小节并入下一块，而不是直接丢弃

    for sec in sections:
        body = _ADOC_ANCHOR.sub("", sec.body).strip()
        if not body:
            continue
        if carry:
            body = f"{carry}\n\n{body}"
            carry = ""
        if estimate_tokens(body) < MIN_TOKENS:
            carry = body
            continue

        path = _dedupe_path(sec.title_path)
        for piece in _split_body(body):
            if estimate_tokens(piece) < MIN_TOKENS:
                carry = piece
                continue
            chunks.append(
                Chunk(
                    source_url=build_url(src, rel_path, sec.anchor, getattr(sec, "page_id", None)),
                    source_project=src.project,
                    version_or_commit=commit,
                    license=src.license,
                    retrieved_at=retrieved_at,
                    title_path=path,
                    technology=src.technology,
                    content_type=sec.content_type,
                    locale=src.locale,
                    text=piece,
                    anchor=sec.anchor,
                    source_path=rel_path.as_posix(),
                )
            )
    return chunks
