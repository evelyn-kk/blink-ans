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

# 基线的第三种取值（CR-085）。背景：本探针按 `limit=20` 检索，名次超出这个
# 窗口一律记为 `rank=None`。于是"测过、确实差到没进候选"这个真实状态，
# 原来只能写成 `baseline: null`——而那是 CR-015 明令禁止的组合
# （既不受退步判据约束、又被 top_k 判据豁免，两道闸同时打开）。
#
# `not_in_candidates` 补上这个缺口，语义是**地板**而不是豁免：
#   - 已经在地板上，退步判据无从再判，这不是"开后门"，是没有更差的状态了；
#   - 一旦它进了候选，就必须把基线改成那个具体名次，从此按数字受门禁约束；
#   - 因此它只能停在地板或往上走，不存在"用它吸收一次退步"的用法
#     （CR-048 那种把 baseline 改成退步后数值的操作，在这里改不动）。
# 使用要求：必须配 `known_open: true` 与说明测量过程的 `note`。
NOT_IN_CANDIDATES = "not_in_candidates"


@dataclass
class ProbeResult:
    question: str
    gold: str
    rank: int | None          # 正确块的最好名次，1 起；None 表示不在候选里
    passed: bool
    # 上次记录的名次，用于区分"真退步"与"既有缺口"。
    # `int` 是名次；`NOT_IN_CANDIDATES` 表示"测过，当时就没进候选"；
    # `None` 只表示"从未测过"（CR-015：它会让退步判据整个跳过这条）。
    baseline: int | str | None = None
    known_open: bool = False      # 已知未解决，不计入退步
    top: list[str] = field(default_factory=list)
    fts_query: str = ""


def _matches_gold(hit: Hit, gold: str, project: str | None,
                  exact: bool = False, contains: str | None = None) -> bool:
    """gold 用 title_path 前缀匹配：同一小节被切成多块时，任一块命中都算。

    块 id 每次重建索引都会变，title_path 不会——判据必须挂在稳定的字段上
    （这正是 CR-008「判据不得耦合表面形态」的同一条教训）。

    `exact=True` 只认这一层、不认子小节；`contains` 再往下一层，在同一小节的
    多个块之间定位（都是 CR-085 加的）。两者都来自同一次实测踩坑：

    问题「同一个类里的方法互相调用时 @Transactional 为什么不生效」，
    gold 写 `Annotations › Using `@Transactional`` 时——

    | 匹配方式 | 报出的名次 | 命中的其实是 |
    | --- | --- | --- |
    | 前缀 | 第 2 名 | 子小节 `… › Multiple Transaction Managers with …` |
    | exact | 第 4 名 | 同名小节的另一块，讲 `@EnableTransactionManagement` 扫描范围 |
    | exact + contains | **第 60 名** | 真正讲自调用的那块 |

    也就是说只按 title_path 判，这条探针会给出一个**假绿**——正好是本探针
    当初要消灭的那个毛病（「来源对但块不对」）下沉一层的形态。

    `contains` 的使用纪律：**只能与 title_path gold 合用**，内容是人工读过、
    从官方正文逐字抄出的短语，用来指认是哪一块；**不得**单独拿关键词圈 gold
    ——那是本文件开头明令禁止的做法（会把上百块全算对，指标失效）。
    """
    if project and hit.source_project != project:
        return False
    if contains and contains not in hit.text:
        return False
    if exact:
        return hit.title_path == gold
    return hit.title_path == gold or hit.title_path.startswith(gold + " › ")


def evaluate(results: list[ProbeResult], top_k: int) -> tuple[list[ProbeResult], list[ProbeResult]]:
    """把结果分成"退步"与"未达标"两类，并决定命令是否失败。

    纯函数，不碰索引与模型，因此可以单测——门禁自身也需要回归保护（CR-015）。

    两条判据缺一不可：

    - **退步**：名次比记录的基线更差。`known_open` 的既有缺口也适用，
      只是它们的基线本来就不是第 1 名。
    - **未达标**：非 `known_open` 却没进前 `top_k`。
      只看退步是不够的——`baseline is null` 的题（新加入、或修复前根本没进候选）
      从第 1 名跌到未进候选时 `r.baseline is None`，退步判据整个跳过它，
      于是**刚修好的题恰好失去门禁**。CR-015 指出的正是这个洞。

    基线为 `NOT_IN_CANDIDATES` 时不参与退步判据——**因为它已经在地板上**，
    没有更差的状态可退（详见该常量处的说明）。它仍受 `known_open` 约束：
    不写 `known_open` 就会落进 `below`、照样让命令失败。
    """
    regressed = [
        r for r in results
        if isinstance(r.baseline, int) and (r.rank is None or r.rank > r.baseline)
    ]
    below = [r for r in results if not r.passed and not r.known_open]
    return regressed, below


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
        gold_exact = bool(p.get("gold_exact", False))
        gold_contains = p.get("gold_contains")
        rank = next(
            (i for i, h in enumerate(hits, 1)
             if _matches_gold(h, gold, project, gold_exact, gold_contains)),
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
    regressed, below = evaluate(results, top_k)
    for r in results:
        mark = "✓" if r.passed else ("○" if r.known_open else "✗")
        pos = f"第 {r.rank} 名" if r.rank else "未进候选"
        delta = ""
        if r.baseline == NOT_IN_CANDIDATES:
            delta = "  (基线 未进候选)" if r.rank else ""
        elif r.baseline is not None and r.rank != r.baseline:
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
        print("退步（比基线更差）:")
        for r in regressed:
            print(f"  {r.question}: 基线第 {r.baseline} 名 -> "
                  + (f"第 {r.rank} 名" if r.rank else "未进候选"))
    if below:
        print(f"未达标（非已知缺口，正确块没进前 {top_k}）:")
        for r in below:
            print(f"  {r.question}: "
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
    # 退步或未达标都失败。既有缺口（known_open）不阻塞，但一旦比基线更差立刻报出来。
    return 1 if (regressed or below) else 0


if __name__ == "__main__":
    raise SystemExit(main())
