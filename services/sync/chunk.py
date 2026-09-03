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

    if src.url_path_drop_chars:
        p = p.translate(str.maketrans("", "", src.url_path_drop_chars))

    # 显式模板优先：上游站点结构与仓库目录结构不一定对应
    # （Kafka 的仓库已迁到 Hugo，但官网仍是单页锚点形式）
    if src.url_template:
        stem = p[: p.rfind(".")] if "." in Path(p).name else p
        return src.url_template.format(path=stem, anchor=anchor or "", base=src.base_url.rstrip("/"))

    if src.format == "asciidoc":
        # Antora 布局: .../modules/<module>/pages/<rest>.adoc -> <base>/<module>/<rest>.html
        # 模块名必须保留：Spring Boot 有 reference / how-to / api 等多个模块，
        # 丢掉它会让所有链接 404（实测 8/8 失败）。
        # 例外是 ROOT 模块，Antora 在 URL 中省略该段（spring-data-redis 即是此例）。
        module = ""
        if "/modules/" in p and "/pages/" in p:
            module = p.split("/modules/", 1)[1].split("/pages/", 1)[0]
            p = p.split("/pages/", 1)[1]
        elif "/pages/" in p:
            p = p.split("/pages/", 1)[1]
        else:
            p = Path(p).name

        stem = p[:-5] if p.endswith(".adoc") else p
        prefix = f"{module}/" if module and module != "ROOT" else ""
        return f"{src.base_url.rstrip('/')}/{prefix}{stem}.html{frag}"

    if src.format == "markdown":
        # Hugo 布局: content/en/docs/<rest>.md -> <rest>/
        for prefix in ("content/en/docs/", "content/en/", "content/"):
            if p.startswith(prefix):
                p = p[len(prefix):]
                break
        p = p[:-3] if p.endswith(".md") else p
        # Hugo 的 _index.md 是所在目录的首页，URL 里不出现该文件名
        if p.endswith("/_index"):
            p = p[: -len("/_index")]
        elif p == "_index":
            p = ""
        base = f"{src.base_url.rstrip('/')}/{p}".rstrip("/")
        return f"{base}/{frag}"

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


_SENT_END = re.compile(r"(?<=[。！？.!?])\s+")


def _force_split(block: str) -> list[str]:
    """把超过硬上限的非代码块继续切小。

    先按行切（列表型内容，如 Kafka 的 Notable changes 常常整段没有空行），
    行仍过长时再按句号切。代码块不走这里——切断的代码既不能执行也无法理解。
    """
    if estimate_tokens(block) <= MAX_TOKENS:
        return [block]

    units = block.split("\n")
    if len(units) == 1:
        units = _SENT_END.split(block)

    out, buf, n = [], [], 0
    for u in units:
        t = estimate_tokens(u)
        if buf and n + t > MAX_TOKENS:
            out.append("\n".join(buf) if "\n" in block else " ".join(buf))
            buf, n = [], 0
        buf.append(u)
        n += t
    if buf:
        out.append("\n".join(buf) if "\n" in block else " ".join(buf))

    # 单行、没有句末标点的日志或代码说明会在上面的两级切分中仍是一个单位。
    # 它们不能因为"没有自然边界"就绕过硬上限；尽量在空白处分割，实在没有
    # 空白才按字符切。代码块已经由调用方排除，不会落到这里。
    wrapped: list[str] = []
    for piece in out:
        remaining = piece
        while estimate_tokens(remaining) > MAX_TOKENS:
            lo, hi = 1, len(remaining)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if estimate_tokens(remaining[:mid]) <= MAX_TOKENS:
                    lo = mid
                else:
                    hi = mid - 1
            cut = lo
            whitespace = max(remaining.rfind(" ", max(1, cut // 2), cut + 1),
                             remaining.rfind("\n", max(1, cut // 2), cut + 1))
            if whitespace > 0:
                cut = whitespace
            wrapped.append(remaining[:cut].strip())
            remaining = remaining[cut:].lstrip()
        if remaining:
            wrapped.append(remaining)
    return [o for o in wrapped if o.strip()]


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

        # 围栏中的示例（尤其是 ASCII 数据流图）与相邻散文合并后很容易跨过
        # 上下文硬上限。代码/图本身不切开，但在边界处先把已有散文封口；这样既
        # 保留完整示例，又不会让它拖着前一段一起成为无法选入 prompt 的大块。
        if fences and not in_code and buf:
            out.append("\n\n".join(buf))
            buf, buf_tokens = [], 0

        # 代码块内部一律不切，哪怕超过硬上限
        if in_code or (buf_tokens + t <= MAX_TOKENS) or not buf:
            buf.append(b)
            buf_tokens += t
        else:
            out.append("\n\n".join(buf))
            buf, buf_tokens = [b], t

        if fences % 2 == 1:
            in_code = not in_code

        # 一个完整围栏也独立成块，后续解释不要和它再次拼接。
        if fences and not in_code:
            out.append("\n\n".join(buf))
            buf, buf_tokens = [], 0
        elif not in_code and buf_tokens >= TARGET_TOKENS:
            out.append("\n\n".join(buf))
            buf, buf_tokens = [], 0

    if buf:
        out.append("\n\n".join(buf))

    # 代码块整体保留；其余超上限的继续下切，否则这些块永远进不了上下文预算
    final: list[str] = []
    for piece in out:
        if _CODE_FENCE.search(piece):
            final.append(piece)
        else:
            final.extend(_force_split(piece))
    return final


def _merge_small(pieces: list[str]) -> list[str]:
    """把过短的片段并入同一小节的相邻片段。

    只在小节**内部**合并。跨小节合并会让引用张冠李戴——
    早期实现把上一小节的尾巴并进下一小节的首块，于是出现了
    正文来自 A 节、却标注 #sec-b 的块。对一个以可追溯引用为卖点的产品，
    这比丢一小段内容严重得多。
    """
    out: list[str] = []
    for p in pieces:
        if out and estimate_tokens(p) < MIN_TOKENS:
            out[-1] = f"{out[-1]}\n\n{p}"
        else:
            out.append(p)
    # 首片过短时向后并
    while len(out) > 1 and estimate_tokens(out[0]) < MIN_TOKENS:
        out[1] = f"{out[0]}\n\n{out[1]}"
        out.pop(0)
    return out


def sections_to_chunks(
    sections: list[Section], src: Source, rel_path: Path, commit: str, retrieved_at: str
) -> list[Chunk]:
    """把小节切成块。

    不变量：**每个块的正文完整来自单一小节**，因此其 source_url、锚点和
    标题路径必然与正文对应。整节内容不足 MIN_TOKENS 时直接丢弃——
    十来个 token 的残片作为证据没有价值，不值得为它牺牲引用正确性。
    """
    chunks: list[Chunk] = []

    for sec in sections:
        body = _ADOC_ANCHOR.sub("", sec.body).strip()
        if not body:
            continue

        path = _dedupe_path(sec.title_path)
        url = build_url(src, rel_path, sec.anchor, getattr(sec, "page_id", None))

        for piece in _merge_small(_split_body(body)):
            if estimate_tokens(piece) < MIN_TOKENS:
                continue
            chunks.append(
                Chunk(
                    source_url=url,
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
