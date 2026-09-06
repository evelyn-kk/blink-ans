"""场景卡片（authored 来源）的解析与切块。

卡片是人工撰写的英文 Markdown，用来把分散在多份官方文档里的结论组织成一篇
连贯的说明（例如 Outbox 模式横跨 debezium 与 kafka 两个来源）。但卡片本身
**不产生新的证据实体**：每个 `## ` 小节必须显式声明它引用哪个已登记来源的
哪个真实 URL，落库后这条块的 `source_url`/`source_project`/`version_or_commit`/
`license`/`technology` 全部照抄被引用来源的真实值——卡片只重新组织表达，
不替代来源（architecture.md §5.2），也不需要为它发明新的技术域标签。

因此这里不走 `services/sync/parse.py` 那套面向官方发布器（Hugo/Antora/DocBook）
的通用解析器：格式完全由我们自己定义，足够简单，没有必要复用锚点推导那类逻辑。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from packages.schemas.chunk import Chunk, MetadataError, utc_now
from services.retrieval.store import CURRENT, ChunkStore

from .models import SourceResult
from .registry import Source, load_registry

ROOT = Path(__file__).resolve().parents[2]

_FRONTMATTER = re.compile(r"\A---\n(.*?\n)---\n", re.DOTALL)
_SECTION_SPLIT = re.compile(r"^## (.+)$\n", re.MULTILINE)
_SOURCE_LINE = re.compile(r"^source:\s+(\S+)\s+(https?://\S+)\s*$")


class CardFormatError(ValueError):
    """卡片文件不满足约定格式，或引用了无法作为出处的来源。"""


@dataclass
class _Section:
    heading: str
    project: str
    url: str
    body: str


def _parse_card(path: Path) -> tuple[str, list[_Section]]:
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER.match(text)
    if not m:
        raise CardFormatError(f"{path.name}: 缺少 YAML frontmatter（--- title: ... ---）")

    meta = yaml.safe_load(m.group(1)) or {}
    title = str(meta.get("title") or "").strip()
    if not title:
        raise CardFormatError(f"{path.name}: frontmatter 缺少 title")

    body = text[m.end():]
    parts = _SECTION_SPLIT.split(body)
    preamble, rest = parts[0], parts[1:]
    if preamble.strip():
        raise CardFormatError(
            f"{path.name}: 第一个 '## ' 小节之前不得有正文（发现 {preamble.strip()[:50]!r}）"
        )
    if not rest:
        raise CardFormatError(f"{path.name}: 没有任何 '## ' 小节")

    sections: list[_Section] = []
    for i in range(0, len(rest), 2):
        heading = rest[i].strip()
        lines = rest[i + 1].splitlines()
        idx = 0
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        if idx >= len(lines):
            raise CardFormatError(f"{path.name} § {heading}: 小节为空")

        line_match = _SOURCE_LINE.match(lines[idx].strip())
        if not line_match:
            raise CardFormatError(
                f"{path.name} § {heading}: 首行必须是 'source: <project> <url>'，"
                f"实际为 {lines[idx].strip()!r}"
            )
        prose = "\n".join(lines[idx + 1:]).strip()
        if not prose:
            raise CardFormatError(f"{path.name} § {heading}: source 指令之后没有正文")

        sections.append(_Section(
            heading=heading, project=line_match.group(1), url=line_match.group(2), body=prose,
        ))

    return title, sections


def _lookup_current_version(project: str) -> str | None:
    """从当前已激活索引里找该来源既有块的版本号，作为兜底。"""
    if not CURRENT.exists():
        return None
    store = ChunkStore(CURRENT)
    try:
        rows = store.execute(
            "SELECT version_or_commit FROM chunks WHERE source_project = ? LIMIT 1", (project,)
        )
        return rows[0]["version_or_commit"] if rows else None
    finally:
        store.close()


def _build_chunk(
    card_title: str,
    sec: _Section,
    by_project: dict[str, Source],
    version_by_project: dict[str, str],
    now: str,
    source_path: str,
) -> Chunk:
    cited = by_project.get(sec.project)
    if cited is None:
        raise CardFormatError(
            f"§ {sec.heading}: 未登记的来源项目 {sec.project!r}，"
            f"检查拼写，或需要先在 knowledge/sources.yaml 登记该来源"
        )
    if not cited.ingest:
        raise CardFormatError(
            f"§ {sec.heading}: {sec.project!r} 是不入库来源"
            f"（{cited.ingest_blocked_reason or '许可受限'}），不能作为卡片引用的出处"
        )

    version = version_by_project.get(sec.project) or _lookup_current_version(sec.project)
    if not version:
        raise CardFormatError(
            f"§ {sec.heading}: {sec.project!r} 尚未有任何已入库的块，无法确定引用版本，"
            f"请先同步该来源"
        )

    return Chunk(
        source_url=sec.url,
        source_project=sec.project,
        version_or_commit=version,
        license=cited.license,
        retrieved_at=now,
        title_path=[card_title, sec.heading],
        technology=cited.technology,
        content_type="prose",
        locale="en",
        text=sec.body,
        # 卡片小节的 source_project 是它引用的真实来源（如 debezium），不是
        # "scenario-cards"——所以 carry_over() 不能靠 source_project 认出
        # 这条块属于哪个 authored 来源。source_path 记录卡片文件自身的相对
        # 路径，供 carry_over() 按路径前缀识别、合并卡片来源时正确排除旧块，
        # 否则编辑/删除卡片小节后，旧版本会作为"未参与本次同步"被永久搬运下去。
        source_path=source_path,
    )


def _fingerprint(files: list[Path]) -> str:
    """authored 来源没有 git commit；用文件内容摘要做一个可比较的版本标识，
    仅用于本轮同步报告里的 `versions[src.id]` 记录，不进入任何 Chunk。"""
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def collect_chunks(
    src: Source,
    log: Callable[[str], None],
    versions: dict[str, str] | None = None,
) -> tuple[list[Chunk], SourceResult]:
    """authored 来源的采集入口，供 pipeline.collect_chunks() 按 format 分发调用。

    与拉取式来源的 collect_chunks 保持相同的返回形状（chunks, SourceResult），
    但完全不经过 fetch()/verify_license()——这两步对没有上游仓库的手写文件不适用。

    `versions` 是 `pipeline.sync()` 里按 **来源 id** 记录的、本轮已同步来源的
    提交号（合并模式下还包含从旧索引搬运来的既有版本）。当前注册表里所有
    来源恰好 id == project，但这不是保证——这里显式按注册表把它转成按
    **project** 索引，不依赖这个巧合。
    """
    versions = versions or {}
    res = SourceResult(source_id=src.id)

    # (相对路径前缀, 文件) 配对——carry_over() 要靠这个相对路径前缀识别
    # "这条块属于哪个 authored 来源"（它的 source_project 是被引用来源，
    # 不是 scenario-cards，见 _build_chunk 里的说明），而不是文件名本身。
    files: list[tuple[str, Path]] = []
    for rel in src.paths:
        base = ROOT / rel
        if not base.exists():
            res.error = f"{src.id}: 登记路径 {rel} 不存在"
            return [], res
        files.extend((rel, f) for f in sorted(base.glob("*.md")))
    if not files:
        res.error = f"{src.id}: {', '.join(src.paths)} 下没有任何 .md 卡片文件"
        return [], res
    res.files = len(files)
    res.commit = _fingerprint([f for _, f in files])

    registry = load_registry()
    by_project = {s.project: s for s in registry}
    version_by_project = {s.project: versions[s.id] for s in registry if s.id in versions}
    now = utc_now()
    chunks: list[Chunk] = []

    for rel, f in files:
        try:
            title, sections = _parse_card(f)
        except CardFormatError as exc:
            key = f"卡片格式错误: {exc}"
            res.reject_reasons[key] = res.reject_reasons.get(key, 0) + 1
            res.rejected += 1
            continue

        rel_path = f"{rel.rstrip('/')}/{f.name}"
        for sec in sections:
            try:
                chunk = _build_chunk(title, sec, by_project, version_by_project, now, rel_path)
                chunk.validate()
            except (CardFormatError, MetadataError) as exc:
                # CardFormatError 来自引用解析（未登记/受限来源、缺版本）；
                # MetadataError 来自 Chunk.validate()。两者都归为"这个小节拒绝
                # 入库"，不让一个坏小节拖垮同一张卡片里的其它小节。
                key = f"{type(exc).__name__}: {exc}"
                res.reject_reasons[key] = res.reject_reasons.get(key, 0) + 1
                res.rejected += 1
                continue
            chunks.append(chunk)

    res.chunks = len(chunks)
    log(
        f"  {src.id}: {res.files} 文件 -> {res.chunks} 块"
        + (f"，拒绝 {res.rejected}" if res.rejected else "")
    )
    return chunks, res
