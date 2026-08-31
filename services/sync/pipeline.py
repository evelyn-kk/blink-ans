"""同步管线：拉取 → 校验许可 → 解析 → 切块 → 校验元数据 → 嵌入 → 建索引 → 回归 → 激活。

顺序不可调换的两处：
- 许可校验必须在解析之前——不该入库的资料连解析都不做。
- 回归检索必须在激活之前——索引失败不得覆盖当前可用索引。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.schemas.chunk import Chunk, MetadataError, utc_now  # noqa: E402
from services.retrieval.embed import DEFAULT_MODEL, Embedder  # noqa: E402
from services.retrieval.search import hybrid_search  # noqa: E402
from services.retrieval.store import ChunkStore, EmbeddingCache, IndexBuilder  # noqa: E402

from .chunk import sections_to_chunks  # noqa: E402
from .fetch import LicenseError, collect_files, fetch, head_commit  # noqa: E402
from .parse import parse_file  # noqa: E402
from .registry import Source, ingestible, load_registry  # noqa: E402

REGRESSION_PATH = Path(__file__).resolve().parents[2] / "knowledge" / "regression_queries.yaml"
EMBED_BATCH = 16


@dataclass
class SourceResult:
    source_id: str
    commit: str = ""
    files: int = 0
    chunks: int = 0
    rejected: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@dataclass
class SyncReport:
    sources: list[SourceResult] = field(default_factory=list)
    total_chunks: int = 0
    regression_passed: bool = False
    regression_failures: list[str] = field(default_factory=list)
    activated: bool = False
    index_path: Path | None = None


def collect_chunks(src: Source, log: Callable[[str], None]) -> tuple[list[Chunk], SourceResult]:
    res = SourceResult(source_id=src.id)
    try:
        fetched = fetch(src)
    except LicenseError as exc:
        res.error = f"许可校验失败: {exc}"
        return [], res
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
        return [], res

    res.commit = fetched.commit
    files = collect_files(src, fetched.root)
    res.files = len(files)
    now = utc_now()

    chunks: list[Chunk] = []
    for f in files:
        try:
            sections = parse_file(f, src)
        except Exception as exc:
            res.reject_reasons[f"解析失败:{type(exc).__name__}"] = (
                res.reject_reasons.get(f"解析失败:{type(exc).__name__}", 0) + 1
            )
            continue
        for c in sections_to_chunks(sections, src, f.relative_to(fetched.root), fetched.commit, now):
            try:
                c.validate()
                chunks.append(c)
            except MetadataError as exc:
                res.rejected += 1
                res.reject_reasons[exc.field_name] = res.reject_reasons.get(exc.field_name, 0) + 1

    res.chunks = len(chunks)
    log(f"  {src.id}: {res.files} 文件 -> {res.chunks} 块" + (f"，拒绝 {res.rejected}" if res.rejected else ""))
    return chunks, res


def run_regression(index_path: Path, log: Callable[[str], None]) -> tuple[bool, list[str]]:
    """对暂存索引跑固定查询。任一条无结果即判定失败，不激活。"""
    spec = yaml.safe_load(REGRESSION_PATH.read_text(encoding="utf-8"))
    store = ChunkStore(index_path)
    embedder = Embedder()
    failures: list[str] = []

    try:
        for q in spec["queries"]:
            vec = embedder.encode_one(q["text"])
            hits = hybrid_search(store, q["text"], vec, limit=5)
            if not hits:
                failures.append(f"{q['text']!r}: 无任何检索结果")
                continue
            want = q.get("expect_project")
            if want and not any(h.source_project == want for h in hits):
                got = ", ".join(sorted({h.source_project for h in hits}))
                failures.append(f"{q['text']!r}: 期望命中 {want}，实际只有 {got}")
    finally:
        store.close()

    ok = not failures
    log(f"  回归检索: {'通过' if ok else f'失败 {len(failures)} 条'}")
    for f in failures:
        log(f"    ✗ {f}")
    return ok, failures


def _embed_with_cache(
    chunks: list[Chunk], embedder: Embedder, cache: EmbeddingCache
) -> list[list[float]]:
    """能复用的复用，剩下的按批嵌入。"""
    vectors: list[list[float] | None] = [cache.get(c.checksum) for c in chunks]
    todo = [i for i, v in enumerate(vectors) if v is None]
    for i in range(0, len(todo), EMBED_BATCH):
        idx = todo[i : i + EMBED_BATCH]
        for j, v in zip(idx, embedder.encode([chunks[j].text for j in idx])):
            vectors[j] = v
    return [v for v in vectors if v is not None]


def sync(
    only: list[str] | None = None,
    *,
    activate: bool = True,
    reuse_embeddings: bool = True,
    log: Callable[[str], None] = print,
) -> SyncReport:
    report = SyncReport()
    sources = ingestible(load_registry())
    if only:
        sources = [s for s in sources if s.id in only]
        if not sources:
            raise ValueError(f"没有匹配的可入库来源: {only}")

    log(f"同步 {len(sources)} 个来源")
    builder = IndexBuilder()
    embedder = Embedder()
    cache = EmbeddingCache() if reuse_embeddings else EmbeddingCache(Path("/nonexistent"))
    if cache.available:
        log("  复用当前索引中未变更正文的向量")
    versions: dict[str, str] = {}

    try:
        for src in sources:
            chunks, res = collect_chunks(src, log)
            report.sources.append(res)
            if res.error:
                log(f"  {src.id}: 跳过 —— {res.error}")
                continue
            versions[src.id] = res.commit

            builder.add(chunks, _embed_with_cache(chunks, embedder, cache))
            report.total_chunks += len(chunks)

        if cache.available:
            log(f"  向量复用 {cache.hits} 条，新算 {cache.misses} 条")
        cache.close()
        stats = builder.finalize(versions, DEFAULT_MODEL)
        log(f"暂存索引: {stats.chunks} 块 -> {stats.path.name}")
        for p, n in stats.projects.items():
            log(f"    {p}: {n}")

        ok, failures = run_regression(stats.path, log)
        report.regression_passed, report.regression_failures = ok, failures

        if ok and activate:
            report.index_path = builder.activate()
            report.activated = True
            log(f"已激活: {report.index_path}")
        elif not ok:
            log("回归未通过，保留当前索引不变；暂存索引留在磁盘上供排查")
    except Exception:
        builder.discard()
        raise

    return report
