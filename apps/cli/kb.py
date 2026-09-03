"""kb —— 知识库命令行。

    kb sources                                  列出注册表中的来源与许可状态
    kb sync                                     全量重建：拉全部来源、跑全量回归、激活
    kb sync --only id,... --mode verify         局部验证：只建这些来源的索引，跑相关回归，不激活
    kb sync --only id,... --mode merge          合并更新：以当前索引为底座换掉这些来源后激活
    kb search "问题" [--index 路径]              检索当前索引，或 verify 留下的暂存索引
    kb project-import --manifest ... --project ... --file ...  导入项目说明文档到独立索引
    kb verify-links                             抽样验证引用链接可达性
    kb stats                                    当前索引概况
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.sync.registry import load_registry  # noqa: E402


def cmd_sources(args) -> int:
    for s in load_registry():
        mark = "入库" if s.ingest else "仅链接"
        print(f"{s.id:<20} {s.license:<18} {s.format:<9} {mark:<7} {s.repo}")
        if not s.ingest:
            print(f"  └─ {' '.join(s.ingest_blocked_reason.split())}")
    return 0


def cmd_sync(args) -> int:
    from services.sync.pipeline import sync

    only = [x.strip() for x in args.only.split(",")] if args.only else None
    t0 = time.perf_counter()
    try:
        report = sync(
            only, mode=args.mode,
            activate=not args.no_activate, allow_partial=args.allow_partial,
            reuse_embeddings=not args.no_reuse,
        )
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return 2

    elapsed = time.perf_counter() - t0
    if report.mode == "merge":
        print(f"\n新写入 {report.total_chunks} 块 + 搬运 {report.carried_chunks} 块 "
              f"= 索引共 {report.index_chunks} 块，耗时 {elapsed:.1f}s")
    else:
        print(f"\n共 {report.total_chunks} 块，耗时 {elapsed:.1f}s")

    skipped = [r for r in report.sources if r.error]
    if skipped:
        # 静默跳过意味着某个技术域整个从语料里消失，而回归可能仍然通过。
        # 这类失败必须有非零退出码，否则自动化流程发现不了。
        print(f"\n{len(skipped)} 个来源未能同步:", file=sys.stderr)
        for r in skipped:
            print(f"  {r.source_id}: {r.error}", file=sys.stderr)
    if report.incomplete:
        print("索引不完整，未激活；当前索引保持不变", file=sys.stderr)
        return 1
    if not report.regression_passed:
        print("回归未通过，索引未激活", file=sys.stderr)
        return 1
    if report.mode == "verify":
        # 局部验证故意不激活，这是预期结果而非失败；
        # 满意后用 --mode merge 并入当前索引。
        print(f"局部验证通过，索引未激活（这是 verify 模式的预期行为）。"
              f"确认无误后用: kb sync --only {args.only} --mode merge")
    return 1 if skipped else 0


def cmd_search(args) -> int:
    from services.retrieval.embed import Embedder
    from services.retrieval.search import hybrid_search
    from services.retrieval.store import ChunkStore

    store = ChunkStore(Path(args.index) if args.index else None,
                       check_dictionary=not args.index)
    t0 = time.perf_counter()
    vec = Embedder().encode_one(args.query) if not args.keyword_only else None
    hits = hybrid_search(
        store, args.query, vec,
        limit=args.limit, token_budget=args.token_budget,
        technology=args.technology, project=args.project,
        project_id=args.project_id, module=args.module, symbol=args.symbol,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    if not hits:
        print("无结果")
        return 1

    total = sum(h.token_estimate for h in hits)
    print(f"{len(hits)} 条结果，合计 {total} token，耗时 {elapsed:.0f} ms\n")
    for i, h in enumerate(hits, 1):
        ranks = f"kw#{h.keyword_rank or '-'} vec#{h.vector_rank or '-'}"
        print(f"[{i}] {h.citation}")
        print(f"    {h.source_url}")
        print(f"    融合分 {h.score:.4f} ({ranks}) | {h.token_estimate} tok | {h.content_type}")
        body = " ".join(h.text.split())
        print(f"    {body[:180]}{'...' if len(body) > 180 else ''}\n")
    return 0


def cmd_project_import(args) -> int:
    """导入用户逐一指定的本地项目文件；默认不替换全局 current 索引。"""
    from services.projects.assets import build_assets
    from services.projects.importer import read_materials
    from services.projects.indexing import rebuild_project_index
    from services.projects.registry import load_projects
    from services.retrieval.embed import Embedder
    from services.retrieval.store import CURRENT, IndexError_

    try:
        projects = load_projects(Path(args.manifest))
        project = next((p for p in projects if p.id == args.project_id), None)
        if project is None:
            raise ValueError(f"注册表中不存在项目: {args.project_id}")
        materials = read_materials(project, args.files)
        # 单独的派生索引先发布，避免还没有项目级回归集时静默覆盖所有用户的 current。
        name = "current" if args.activate_current else (args.name or f"project-{project.id}")
        if "/" in name or "\\" in name or name in {"", ".", ".."}:
            raise ValueError("索引名只能是一个文件名片段，不能包含路径分隔符")
        stats = rebuild_project_index(
            project, materials, Embedder(),
            source_index=Path(args.from_index) if args.from_index else CURRENT,
            name=name, activate=True,
        )
    except (ValueError, IndexError_, OSError, UnicodeDecodeError, ImportError, RuntimeError) as exc:
        print(f"项目导入失败: {exc}", file=sys.stderr)
        return 2

    # 摘要和别名是从本次明确材料生成的，不让模型臆测项目结构。
    from services.projects.importer import build_material_chunks
    assets = build_assets(build_material_chunks(project, materials))
    destination = stats.path.with_name(f"{name}.db")
    print(f"项目 {project.id}：写入 {stats.added} 块，搬运 {stats.carried} 块")
    print(f"索引: {destination}")
    print(assets.summary)
    if assets.aliases:
        print(f"符号/文件别名: {', '.join(assets.aliases)}")
    if not args.activate_current:
        print(f"查询此项目索引：kb search <问题> --index {destination} --project-id {project.id}")
    return 0


def cmd_verify_links(args) -> int:
    from services.retrieval.store import ChunkStore
    from services.retrieval.verify_links import verify

    store = ChunkStore(Path(args.index) if args.index else None, check_dictionary=False)
    what = "带锚点的链接并核对锚点是否存在" if args.check_anchors else "链接"
    print(f"每个来源抽样 {args.sample} 条{what}...")
    rep = verify(store, args.sample, args.timeout, check_anchors=args.check_anchors)

    for proj, buckets in sorted(rep.by_project.items()):
        total = sum(buckets.values())
        ok = buckets.get("ok", 0)
        print(f"  {proj:<22} {ok}/{total} 可达" + (f"  异常: {dict(buckets)}" if ok < total else ""))

    if rep.failures:
        print(f"\n{len(rep.failures)} 条链接不可达:")
        for proj, url, code in rep.failures[:15]:
            print(f"  [{code}] {url}")
        return 1
    print("\n全部抽样链接可达")
    return 0


def cmd_stats(args) -> int:
    from services.retrieval.store import ChunkStore

    store = ChunkStore(check_dictionary=False)
    print(f"索引: {store.path}")
    print(f"切块总数: {store.count()}")
    for k in ("embedding_model", "embedding_dim", "dictionary_version"):
        print(f"{k}: {store.meta.get(k)}")
    print("\n按来源:")
    for p, n in store.stats().items():
        print(f"  {p:<22} {n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kb", description="知识库管理")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("sources", help="列出来源").set_defaults(fn=cmd_sources)

    p = sub.add_parser("sync", help="同步并重建索引")
    p.add_argument("--only", help="只同步指定来源，逗号分隔（须配 --mode verify 或 merge）")
    p.add_argument(
        "--mode", choices=("full", "verify", "merge"), default="full",
        help="full 全量重建并激活；verify 只建局部索引验证、不激活；"
             "merge 以当前索引为底座替换指定来源后激活",
    )
    p.add_argument("--no-activate", action="store_true", help="只建暂存索引，不激活")
    p.add_argument("--allow-partial", action="store_true",
                   help="有来源同步失败时仍然激活（默认拒绝，避免静默丢掉整个来源）")
    p.add_argument("--no-reuse", action="store_true", help="不复用已有向量，全部重新嵌入")
    p.set_defaults(fn=cmd_sync)

    p = sub.add_parser("search", help="检索")
    p.add_argument("query")
    p.add_argument("-n", "--limit", type=int, default=5)
    p.add_argument("--token-budget", type=int, help="按 token 预算截断，模拟真实上下文约束")
    p.add_argument("--technology")
    p.add_argument("--project", help="外部来源项目 ID（如 kafka）；与 --project-id 不同")
    p.add_argument("--project-id", help="用户项目 ID；严格只检索该项目的材料")
    p.add_argument("--module", help="项目模块精确过滤（须配合 --project-id 使用）")
    p.add_argument("--symbol", help="项目符号精确过滤（须配合 --project-id 使用）")
    p.add_argument("--keyword-only", action="store_true", help="跳过向量检索")
    p.add_argument("--index", help="改查指定索引文件，例如 verify 模式留下的暂存索引")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("project-import", help="显式导入项目说明文档和挑选代码片段")
    p.add_argument("--manifest", required=True, help="项目注册表 YAML（含 root、版本、云端许可）")
    p.add_argument("--project-id", required=True, help="注册表中的项目 ID")
    p.add_argument("--file", dest="files", action="append", required=True,
                   help="相对项目 root 的说明文档（.md/.adoc/.rst/.txt）；可重复，不会扫描目录")
    p.add_argument("--from-index", help="作为增量底座的索引，默认 current.db")
    p.add_argument("--name", help="独立输出索引名，默认 project-<项目 ID>")
    p.add_argument("--activate-current", action="store_true",
                   help="显式将结果发布为 current.db；默认只发布独立项目索引")
    p.set_defaults(fn=cmd_project_import)

    p = sub.add_parser("verify-links", help="抽样验证引用链接可达性")
    p.add_argument("--sample", type=int, default=5, help="每个来源抽样条数")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--check-anchors", action="store_true",
                   help="取回页面核对锚点是否真的存在——页面 200 不代表锚点对")
    p.add_argument("--index", help="改查指定索引文件")
    p.set_defaults(fn=cmd_verify_links)

    sub.add_parser("stats", help="索引概况").set_defaults(fn=cmd_stats)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
