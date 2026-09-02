"""检索排序探针 —— 只读、秒级、不加载生成模型。

为什么需要它（T-017 建立，T-025 落成可提交的形式）：

- 一次 50 题 LLM 回归 8–10 分钟，无法用来扫参数或做改动前后的对照。
- 44 题「来源护栏」只断言 top5 里有没有期望**来源**，粒度太粗：
  「来源对但块不对」它抓不到。本探针断言到**小节**。

T-017 时它是临时脚本，用完即弃，于是 T-025 想复现改动前的基线时无从谈起。
这次固化下来，金标准与判据一并入库。

**踩过的坑（勿重蹈）**：首版探针扫 `KEYWORD_WEIGHT` 时改模块全局量，五档结果完全相同——
权重是 `rrf_fuse` 的默认参数，定义时即绑定，改全局无效。要扫参数请显式传参。

用法:
    .venv/bin/python packages/evaltools/probe_ranking.py
    .venv/bin/python packages/evaltools/probe_ranking.py --verbose   # 附带 top5 明细
    .venv/bin/python packages/evaltools/probe_ranking.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import yaml  # noqa: E402

from services.retrieval.embed import Embedder  # noqa: E402
from services.retrieval.search import Hit, hybrid_search  # noqa: E402
from services.retrieval.store import ChunkStore  # noqa: E402
from services.retrieval.tokenize import detect_technology, to_fts_query  # noqa: E402

PROBES = Path(__file__).resolve().parents[2] / "knowledge" / "eval" / "ranking_probe.yaml"


@dataclass
class ProbeResult:
    question: str
    gold: str
    rank: int | None          # 正确块的最好名次，1 起；None 表示不在候选里
    passed: bool
    baseline: int | None = None   # 上次记录的名次，用于区分"真退步"与"既有缺口"
    known_open: bool = False      # 已知未解决，不计入退步
    top: list[str] = field(default_factory=list)
    fts_query: str = ""


def _matches_gold(hit: Hit, gold: str, project: str | None) -> bool:
    """gold 用 title_path 前缀匹配：同一小节被切成多块时，任一块命中都算。

    块 id 每次重建索引都会变，title_path 不会——判据必须挂在稳定的字段上
    （这正是 CR-008「判据不得耦合表面形态」的同一条教训）。
    """
    if project and hit.source_project != project:
        return False
    return hit.title_path == gold or hit.title_path.startswith(gold + " › ")


def run(spec: dict, store: ChunkStore, embedder: Embedder,
        *, limit: int = 20) -> list[ProbeResult]:
    top_k = int(spec.get("top_k", 5))
    out: list[ProbeResult] = []
    for p in spec["probes"]:
        q, gold = p["q"], p["gold"]
        project = p.get("project")
        vector = embedder.encode_one(q)
        # 与生产路径一致：技术域当过滤条件而非检索词（见 tokenize.PROJECT_TERMS）
        hits = hybrid_search(
            store, q, vector, limit=limit,
            technology=detect_technology(q), candidates=30,
        )
        rank = next(
            (i for i, h in enumerate(hits, 1) if _matches_gold(h, gold, project)),
            None,
        )
        out.append(ProbeResult(
            question=q, gold=gold, rank=rank,
            passed=rank is not None and rank <= top_k,
            baseline=p.get("baseline"), known_open=bool(p.get("known_open")),
            top=[f"{h.source_project} | {h.title_path}" for h in hits[:top_k]],
            fts_query=to_fts_query(q),
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="打印 top5 明细与 FTS 查询")
    ap.add_argument("--json", type=Path, help="结果写入 JSON，供改动前后对照")
    args = ap.parse_args()

    spec = yaml.safe_load(PROBES.read_text(encoding="utf-8"))
    top_k = int(spec.get("top_k", 5))

    store = ChunkStore()
    embedder = Embedder()
    embedder.load()
    t0 = time.perf_counter()
    results = run(spec, store, embedder)
    elapsed = time.perf_counter() - t0

    passed = sum(r.passed for r in results)
    # 退步 = 名次比记录的基线更差。既有缺口（known_open）只要不再变差就不算退步——
    # 判据要能区分"这次改坏了"和"本来就没解决"，否则每次改动都会被同一批红叉淹没。
    regressed = [
        r for r in results
        if r.baseline is not None
        and (r.rank is None or r.rank > r.baseline)
    ]
    for r in results:
        mark = "✓" if r.passed else ("○" if r.known_open else "✗")
        pos = f"第 {r.rank} 名" if r.rank else "未进候选"
        delta = ""
        if r.baseline is not None and r.rank != r.baseline:
            delta = f"  (基线 {r.baseline})"
        print(f"{mark} [{pos:>8}]{delta} {r.question}")
        print(f"      期望: {r.gold}")
        if args.verbose or not r.passed:
            print(f"      查询: {r.fts_query}")
            for i, t in enumerate(r.top, 1):
                print(f"        {i}. {t}")
    known = sum(r.known_open for r in results)
    print(f"\n正确块进前 {top_k}: {passed}/{len(results)}"
          f"（其中 {known} 条为已知缺口 ○，见 ranking_probe.yaml 的 note）"
          f"    ({elapsed:.2f}s)")
    if regressed:
        print("退步:")
        for r in regressed:
            print(f"  {r.question}: 基线第 {r.baseline} 名 -> "
                  + (f"第 {r.rank} 名" if r.rank else "未进候选"))

    if args.json:
        args.json.write_text(json.dumps(
            {"top_k": top_k, "passed": passed, "total": len(results),
             "results": [{"q": r.question, "gold": r.gold, "rank": r.rank,
                          "passed": r.passed, "fts_query": r.fts_query}
                         for r in results]},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {args.json}")

    store.close()
    # 只有真退步才失败：既有缺口不阻塞，但一旦变得更差立刻报出来
    return 1 if regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
