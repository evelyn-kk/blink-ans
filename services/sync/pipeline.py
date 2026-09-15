"""同步管线：拉取 → 校验许可 → 解析 → 切块 → 校验元数据 → 嵌入 → 建索引 → 回归 → 激活。

顺序不可调换的两处：
- 许可校验必须在解析之前——不该入库的资料连解析都不做。
- 回归检索必须在激活之前——索引失败不得覆盖当前可用索引。
"""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.schemas.chunk import Chunk, MetadataError, utc_now  # noqa: E402
from services.retrieval.embed import DEFAULT_MODEL, Embedder  # noqa: E402
from services.retrieval.search import hybrid_search  # noqa: E402
from services.retrieval.store import CURRENT, ChunkStore, EmbeddingCache, IndexBuilder  # noqa: E402

from . import cards  # noqa: E402
from .chunk import sections_to_chunks  # noqa: E402
from .fetch import CachedSource, LicenseError, collect_files, fetch, head_commit, cached_source_head, validate_cached_source  # noqa: E402
from .models import SourceResult, SyncReport  # noqa: E402
from .parse import parse_file  # noqa: E402
from .registry import Source, ingestible, load_registry  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REGRESSION_PATH = ROOT / "knowledge" / "regression_queries.yaml"
EMBED_BATCH = 16
SYNC_PROGRESS_BATCH = 1024


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _peak_rss() -> dict[str, int | str]:
    """返回本进程 RSS 高水位及其平台口径，避免猜测单位。"""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS 的 ru_maxrss 是 bytes；Linux/BSD 通常是 KiB。状态文件保留原始值与
    # 单位，诊断时不能把一个平台的数值套到另一个平台上。
    return {"value": value, "unit": "bytes" if platform.system() == "Darwin" else "KiB"}


class _SyncStatus:
    """同步进度的原子、低敏感度诊断记录。

    `current.building.db` 在进程被杀时只能说明 finalize 尚未发生；此文件把最后
    已完成阶段、来源 ID、块计数和进程 RSS 高水位留在索引旁。它刻意不写题面、
    块正文或堆栈，以免把知识库内容复制到诊断文件。
    """

    def __init__(self, path: Path, *, mode: str, sources: list[Source], activate: bool) -> None:
        self.path = path
        self.data: dict[str, object] = {
            "schema_version": 1,
            "started_at": _utc_now(),
            "pid": os.getpid(),
            "mode": mode,
            "requested_sources": [s.id for s in sources],
            "activate_requested": activate,
            "stage": "started",
            "peak_rss": _peak_rss(),
            "sources": [],
        }
        self.write()

    def update(self, stage: str, **fields: object) -> None:
        self.data.update(fields)
        self.data["stage"] = stage
        self.data["updated_at"] = _utc_now()
        self.data["peak_rss"] = _peak_rss()
        self.write()

    def source(self, result: SourceResult, stage: str) -> None:
        entries = self.data["sources"]
        assert isinstance(entries, list)
        entries.append({
            "id": result.source_id,
            "stage": stage,
            "files": result.files,
            "chunks": result.chunks,
            "rejected": result.rejected,
            "error": result.error,
        })
        self.update(stage, active_source=result.source_id)

    def write(self) -> None:
        temp = self.path.with_name(self.path.name + ".tmp")
        temp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, self.path)


def collect_chunks(
    src: Source,
    log: Callable[[str], None],
    versions: dict[str, str] | None = None,
    known_urls: dict[str, set[str]] | None = None,
    offline: bool = False,
    cached: CachedSource | None = None,
) -> tuple[list[Chunk], SourceResult]:
    if src.format == "authored":
        # 场景卡片没有上游仓库，完全不走下面的 fetch/许可校验/通用解析——
        # 见 services/sync/cards.py 顶部说明。
        return cards.collect_chunks(src, log, versions, known_urls)

    res = SourceResult(source_id=src.id)
    try:
        fetched = fetch(src, offline=offline, cached=cached)
    except LicenseError as exc:
        res.error = f"许可校验失败: {exc}"
        return [], res
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
        return [], res

    res.commit = fetched.commit
    try:
        files = collect_files(src, fetched.root)
    except Exception as exc:
        # 必须留在 try 内：整个流程按来源隔离设计，
        # 上游改一个目录名不应让其余四个来源的索引一起作废。
        res.error = f"{type(exc).__name__}: {exc}"
        return [], res
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


def run_regression(
    index_path: Path,
    log: Callable[[str], None],
    projects: set[str] | None = None,
) -> tuple[bool, list[str], list[str]]:
    """对暂存索引跑固定查询。任一条无结果即判定失败，不激活。

    `projects` 不为 None 时只跑与这些来源相关的查询——局部验证模式下
    索引里本来就没有其他来源，拿全量回归去卡它必然失败，而那是配置错误
    不是索引质量问题（CR-004）。期望来源为空的查询在任何模式下都跑，
    它们检验的是"中文检索整体可用"，与来源无关。
    """
    spec = yaml.safe_load(REGRESSION_PATH.read_text(encoding="utf-8"))
    store = ChunkStore(index_path)
    embedder = Embedder()
    failures: list[str] = []
    skipped: list[str] = []

    try:
        for q in spec["queries"]:
            want = q.get("expect_project")
            if projects is not None and want is not None and want not in projects:
                skipped.append(q["text"])
                continue
            vec = embedder.encode_one(q["text"])
            hits = hybrid_search(store, q["text"], vec, limit=5)
            if not hits:
                failures.append(f"{q['text']!r}: 无任何检索结果")
                continue
            if want and not any(h.source_project == want for h in hits):
                got = ", ".join(sorted({h.source_project for h in hits}))
                failures.append(f"{q['text']!r}: 期望命中 {want}，实际只有 {got}")
    finally:
        store.close()

    ok = not failures
    ran = len(spec["queries"]) - len(skipped)
    log(f"  回归检索: {'通过' if ok else f'失败 {len(failures)} 条'}（跑了 {ran} 条）"
        + (f"，跳过 {len(skipped)} 条其他来源的查询" if skipped else ""))
    for f in failures:
        log(f"    ✗ {f}")
    return ok, failures, skipped


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


def _add_with_cache(
    builder: IndexBuilder,
    chunks: list[Chunk],
    embedder: Embedder,
    cache: EmbeddingCache,
    progress: Callable[[int], None] | None = None,
) -> int:
    """以小批向量写入，避免整个来源的 Python float 列表同时常驻内存。"""
    added = 0
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start:start + EMBED_BATCH]
        added += builder.add(batch, _embed_with_cache(batch, embedder, cache))
        processed = start + len(batch)
        if progress and (processed % SYNC_PROGRESS_BATCH == 0 or processed == len(chunks)):
            progress(processed)
    return added


MODES = ("full", "verify", "merge")


def _resolve_sources(only: list[str] | None, mode: str) -> list[Source]:
    """按模式解析要同步的来源，并把模式与 --only 的组合约束在这里统一把关。"""
    if mode not in MODES:
        raise ValueError(f"未知同步模式 {mode!r}，可选: {', '.join(MODES)}")

    everything = ingestible(load_registry())
    if mode == "full":
        if only:
            raise ValueError(
                "全量重建会覆盖整个索引，不接受 --only。"
                "只想处理部分来源请用 --mode verify（只验证不激活）"
                "或 --mode merge（并入当前索引后激活）"
            )
        return everything

    if not only:
        raise ValueError(f"--mode {mode} 必须配合 --only 指定来源")
    picked = [s for s in everything if s.id in only]
    if not picked:
        raise ValueError(f"没有匹配的可入库来源: {only}")
    unknown = sorted(set(only) - {s.id for s in picked})
    if unknown:
        # 打错一个来源 id 就静默少同步一个来源，比直接失败糟得多
        raise ValueError(f"未登记或不入库的来源: {', '.join(unknown)}")

    if mode == "merge":
        # authored 来源（场景卡片）在 merge 模式下**永远参与同步**，哪怕
        # --only 没点名它。理由是一次真实的静默证据丢失（T-118，R78 实测
        # 复现）：卡片小节的 `source_project` 是它引用的真实来源（如
        # spring-kafka），而 `carry_over()` 按 `source_project` 排除本轮同步
        # 的来源——于是 `--only spring-kafka --mode merge` 会把引用
        # spring-kafka 的卡片小节一并排除，而卡片来源又不在本轮同步清单里、
        # 不会被重建，那些小节就**从索引里消失了**，且全程没有任何报错。
        # R75 那次重同步五个 asciidoc 来源，正是这样让 12 个卡片块（DLQ 5 +
        # Redis 5 + Outbox 2）静默出局，还顺带让两条排序探针"变好"了。
        #
        # 修法只能是重建而不是搬运：卡片块的 version_or_commit/license 抄自
        # 被引用来源，被引用来源刚换了版本，旧卡片块就是过期元数据；重建还会
        # 顺带重跑 CR-045 的"引用 URL 必须在语料里真实存在"校验。
        picked_ids = {s.id for s in picked}
        picked += [s for s in everything if s.format == "authored" and s.id not in picked_ids]
    return picked


def _uncovered_card_files(index_path: Path, sources: list[Source]) -> list[str]:
    """注册表里每一份卡片文件，在即将激活的索引里都必须至少有一块。

    这是 T-118 那次静默丢失的**门禁**，和"为什么会丢"这个具体原因解耦：
    不管是 carry_over 的排除条件写错、authored 来源没进同步清单、还是将来
    某条新路径，只要激活前的索引里少了某个卡片文件的全部小节，这里就会
    报出来并拦下激活。回归的 6 条烟雾查询做不到这件事——它们几乎不可能
    恰好覆盖某一张卡片。
    """
    store = ChunkStore(index_path, check_dictionary=False)
    try:
        present = {
            r["source_path"] for r in store.execute(
                "SELECT DISTINCT source_path FROM chunks WHERE source_path IS NOT NULL"
            )
        }
    finally:
        store.close()

    missing: list[str] = []
    for src in sources:
        if src.format != "authored":
            continue
        for rel in src.paths:
            base = ROOT / rel
            if not base.exists():
                continue
            for f in sorted(base.glob("*.md")):
                rel_path = f"{rel.rstrip('/')}/{f.name}"
                if rel_path not in present:
                    missing.append(rel_path)
    return missing


def sync(
    only: list[str] | None = None,
    *,
    mode: str = "full",
    activate: bool = True,
    allow_partial: bool = False,
    reuse_embeddings: bool = True,
    offline: bool = False,
    log: Callable[[str], None] = print,
) -> SyncReport:
    """三种模式（CR-004）：

    - **full**：重建全部来源，跑全量回归，通过后激活。默认。
    - **verify**：只建指定来源的局部索引，只跑相关回归，**永不激活**。
      用于改了解析规则后快速看一个来源的效果——旧实现在这里拿全量回归
      去卡一个局部索引，必然失败，掩盖了"没有局部模式"这个设计缺口。
    - **merge**：以当前索引为底座，把指定来源换成新的、其余原样搬过来，
      跑全量回归后激活。用于单个来源的增量更新。
    """
    sources = _resolve_sources(only, mode)
    report = SyncReport(mode=mode, offline=offline)
    projects = {s.project for s in sources}

    # CR-131: 缺缓存不是“一个来源本轮没拉全”，而是 offline 调用的入口前提。
    # 在任何 builder/model/status 创建前检查所有远程来源，避免空 staging 或模型加载。
    cached_sources: dict[str, CachedSource] = {}
    if offline:
        for src in sources:
            if src.format != "authored":
                cached_sources[src.id] = cached_source_head(src)

    log(f"同步模式 {mode}，{len(sources)} 个来源" + (f": {', '.join(s.id for s in sources)}" if only else ""))
    builder = IndexBuilder()
    status = _SyncStatus(
        builder.staging.with_suffix(".status.json"), mode=mode, sources=sources, activate=activate,
    )
    report.diagnostics_path = status.path
    status.update("started", offline=report.offline)
    versions: dict[str, str] = {}
    # 只在本次同步内有效，不写入索引 meta——供 authored 来源的 URL 归属
    # 校验用（CR-045），键与 versions 一样按来源 id。
    known_urls: dict[str, set[str]] = {}
    cache: EmbeddingCache | None = None
    try:
        # 状态文件已在 builder 之后落盘；任何本地模型/缓存初始化失败都必须
        # 走下面的失败关闭，不得伪装成“刚启动后被 SIGKILL”。
        embedder = Embedder()
        cache = (
            EmbeddingCache(embedding_model=DEFAULT_MODEL)
            if reuse_embeddings
            else EmbeddingCache(Path("/nonexistent"))
        )
        if cache.available:
            log("  复用当前索引中未变更正文的向量")
        elif cache.rejected_reason:
            log(f"  {cache.rejected_reason}")

        if mode == "merge":
            status.update("carry_over_running")
            # 先搬底座再写新来源：底座里若还留着这些来源的旧块，
            # (source_url, checksum) 唯一键会把新块当重复丢掉，
            # 结果是"更新了却没变"。exclude 掉本次同步的项目即可。
            #
            # authored 来源（场景卡片）的块 source_project 是它引用的真实
            # 来源，不是卡片自己的项目名，所以按 source_project 排除认不出
            # 旧卡片块——额外按卡片目录的路径前缀排除一遍（见
            # store.carry_over() 与 services/sync/cards.py 的说明）。
            exclude_paths = tuple(
                p for s in sources if s.format == "authored" for p in s.paths
            )
            versions = builder.existing_versions(CURRENT)
            moved = builder.carry_over(CURRENT, projects, DEFAULT_MODEL, exclude_paths)
            report.carried_chunks = moved
            log(f"  从当前索引搬运 {moved} 块（{', '.join(sorted(projects))} 之外的来源）")
            status.update("carry_over_complete", carried_chunks=moved)

        for src in sources:
            # source_chunks* 是 active_source 的局部计数，不能让下一来源在
            # collect 或首次 encode 抛错时继承前一来源的进度（CR-138）。
            status.update(
                "source_collecting", active_source=src.id,
                source_chunks=0, source_chunks_embedded=0,
            )
            # authored 来源（场景卡片）用它们把引用解析成"本轮/当前索引里
            # 该来源的真实版本/真实块地址"；拉取式来源忽略这两个参数。
            chunks, res = collect_chunks(src, log, versions, known_urls, offline=offline,
                                        cached=cached_sources.get(src.id))
            report.sources.append(res)
            if res.error:
                log(f"  {src.id}: 跳过 —— {res.error}")
                status.source(res, "source_failed")
                continue
            versions[src.id] = res.commit
            known_urls[src.id] = {c.source_url for c in chunks}

            # 以 add() 的实际写入数为准：重复的块会被跳过，
            # 用 len(chunks) 会让同一条命令打印出两个不一致的总数。
            status.update(
                "source_embedding", active_source=src.id,
                source_chunks=len(chunks), source_chunks_embedded=0,
            )
            res.chunks = _add_with_cache(
                builder, chunks, embedder, cache,
                lambda processed: status.update(
                    "source_embedding", active_source=src.id,
                    source_chunks=len(chunks), source_chunks_embedded=processed,
                ),
            )
            report.total_chunks += res.chunks
            status.source(res, "source_complete")

        if cache.available:
            log(f"  向量复用 {cache.hits} 条，新算 {cache.misses} 条")
        cache.close()
        status.update("finalize_running", total_chunks=report.total_chunks)
        stats = builder.finalize(versions, DEFAULT_MODEL)
        report.index_chunks = stats.chunks
        report.staging_path = stats.path
        log(f"暂存索引: {stats.chunks} 块 -> {stats.path.name}")
        for p, n in stats.projects.items():
            log(f"    {p}: {n}")
        status.update("finalize_complete", index_chunks=stats.chunks)

        # 局部索引里本来就没有其他来源，只跑相关回归；全量与合并都跑全量回归。
        scope = projects if mode == "verify" else None
        status.update("regression_running")
        ok, failures, skipped = run_regression(stats.path, log, scope)
        report.regression_passed = ok
        report.regression_failures = failures
        report.regression_skipped = skipped
        status.update("regression_complete", regression_passed=ok, regression_failures=failures)

        # 有来源拉取或许可校验失败时不得激活。回归只有 6 条烟雾查询，
        # 少掉一整个来源它照样可能通过——merge 模式下更危险：
        # 旧块已在 carry_over 时排除，激活等于把这个来源从索引里静默删掉。
        #
        # authored 来源（场景卡片）额外把"有小节/文件被拒绝"也算作失败
        # （CR-046）：卡片格式或引用错误只累加 res.rejected、不设
        # res.error（好让同一文件里其它合法小节仍能入库），但 merge 模式
        # 已经在采集前按路径前缀把这个来源的旧块整批排除了——如果新解析
        # 出来的块比旧的少，激活就等于用一个内容缺失的版本覆盖旧内容，
        # 且回归的 6 条烟雾查询几乎不可能覆盖到具体某张卡片。因此
        # authored 来源必须"要么完整成功，要么和硬错误一样拒绝激活"，
        # 不像拉取式来源那样容忍少量块级拒绝——它们规模小、是人工维护的
        # 精确资产，没有"大体能用就行"的容忍空间。
        authored_ids = {s.id for s in sources if s.format == "authored"}
        failed = [
            r.source_id for r in report.sources
            if r.error or (r.source_id in authored_ids and r.rejected)
        ]

        if failed and not allow_partial:
            report.incomplete = True

        # T-118：上面那道门只看"本轮同步的来源有没有失败"，看不见"某张卡片
        # 压根没进这个索引"。丢失可以发生在完全不涉及卡片来源的一次合并里
        # （见 _uncovered_card_files 的说明），因此按注册表逐个文件核对覆盖，
        # 而不是相信同步清单。
        # verify 模式的暂存索引本来就只含被点名的来源，缺卡片是这个模式的
        # 定义而不是缺陷；它也永不激活。只在会激活整份索引的 full/merge 上判。
        #
        # **这一项不进 `failed`，也不受 `--allow-partial` 放行**（CR-088）：
        # 第一版把它追加进 `failed`，于是通用豁免顺手也豁免了它——实测
        # `sync(allow_partial=True)` 在日志点名缺了 outbox-pattern.md 的同时
        # 照样 `activated=True` 并替换 current.db，正是这道门禁最该拦下的
        # 那一个输入（与 CR-015/CR-048 同形）。`--allow-partial` 的语义是
        # "我知道某个来源这次没拉全，仍然要上线"，那是一个可以由人承担的
        # 取舍；而"索引里整张卡片不见了"是**证据丢失**，且它恰好能让门禁
        # 变绿（T-118 已经发生过一次），没有"知情放行"的余地。
        uncovered = (
            _uncovered_card_files(stats.path, ingestible(load_registry()))
            if mode != "verify" else []
        )
        report.uncovered_cards = uncovered
        if uncovered:
            log(f"  索引里缺少这些卡片文件的全部小节: {', '.join(uncovered)}")
            report.incomplete = True

        if mode == "verify":
            log(f"局部验证模式不激活索引；暂存索引留在 {stats.path} 供检查与 kb search --index")
            status.update("completed_not_activated", reason="verify_mode")
        elif report.uncovered_cards:
            # 单独一条文案：这条路径拒绝激活的理由不是"某个来源没拉全"，
            # 而且 --allow-partial 对它无效，不能给出那句建议（CR-088）。
            log(f"待激活索引里缺少这些卡片文件的全部小节: "
                f"{', '.join(report.uncovered_cards)}；这是证据丢失，"
                f"**--allow-partial 不放行**。请修好卡片引用后重新同步，"
                f"或先把它从 knowledge/sources.yaml 的 authored 来源里正式去掉"
                + (f"（本次另有来源未完整同步: {', '.join(failed)}）" if failed else ""))
            status.update("completed_not_activated", reason="uncovered_cards")
        elif report.incomplete:
            log(f"{', '.join(failed)} 未能完整同步，索引不完整，拒绝激活；"
                f"确认要带着缺口上线请加 --allow-partial")
            status.update("completed_not_activated", reason="incomplete_sources")
        elif ok and activate:
            status.update("activation_running")
            report.index_path = builder.activate()
            report.activated = True
            log(f"已激活: {report.index_path}")
            status.update("activated", index_path=str(report.index_path))
        elif not ok:
            log("回归未通过，保留当前索引不变；暂存索引留在磁盘上供排查")
            status.update("completed_not_activated", reason="regression_failed")
        else:
            status.update("completed_not_activated", reason="activation_disabled")
    except BaseException as exc:
        # SIGKILL 不能执行这里；它至少能保留上一次原子状态，从而区分“杀在
        # 哪个阶段”与“Python 异常”。其它可捕获退出原因必须写入，不能只剩
        # 一个无 meta 的 SQLite 暂存文件。
        status.update(
            "interrupted" if not isinstance(exc, Exception) else "failed",
            error_type=type(exc).__name__, error_message=str(exc),
        )
        if isinstance(exc, Exception):
            builder.discard()
        raise
    finally:
        # 正常路径已经提前 close 以释放 current.db；异常初始化后/来源中断时
        # 仍确保已打开的缓存连接不滞留。
        if cache is not None:
            cache.close()

    return report
