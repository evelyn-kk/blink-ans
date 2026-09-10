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
#   - 一旦它进了候选，`evaluate()` 会把它归进 **needs_baseline 并让命令
#     非零退出**，强制把基线改成实测名次、从此按数字受约束；
#   - 因此它只能停在地板，或者往上走一步就被门禁拦下来要求固化。
#
# **CR-086 的教训**：R75 第二条最初只写在注释里（"一旦进了候选就必须改成
# 具体名次"），没有落成判据。结果 `baseline=not_in_candidates,
# known_open=True` 的探针在 rank 为 None/20/6/5/1 时**一律静默通过**——
# known_open 关掉 top_k 那道闸，字符串基线又让退步判据跳过它。
# 一句写在注释里的承诺不是门禁；判据没写出来，它就不存在。
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


def evaluate(
    results: list[ProbeResult], top_k: int
) -> tuple[list[ProbeResult], list[ProbeResult], list[ProbeResult]]:
    """把结果分成"退步""未达标""基线待固化"三类，并决定命令是否失败。

    纯函数，不碰索引与模型，因此可以单测——门禁自身也需要回归保护（CR-015）。

    三条判据缺一不可：

    - **退步**：名次比记录的数字基线更差。`known_open` 的既有缺口也适用，
      只是它们的基线本来就不是第 1 名。
    - **未达标**：非 `known_open` 却没进前 `top_k`。
      只看退步是不够的——`baseline is null` 的题（新加入、或修复前根本没进候选）
      从第 1 名跌到未进候选时 `r.baseline is None`，退步判据整个跳过它，
      于是**刚修好的题恰好失去门禁**。CR-015 指出的正是这个洞。
    - **基线待固化**（CR-086）：实测比基线好——地板值这次**进了候选**，
      或数字基线这次排到了更前面。这不是退步，是"门禁描述的现实已经变了"，
      必须当场把基线改成实测名次。
      两种情形合成一类，因为漏掉它们的后果相同：基线停在一个比现实更宽松的
      值上，此后从新水平滑回旧基线**不会有任何信号**。YAML 头部那句
      "基线记录上次验收通过时的名次、改动被接受后要就地更新"由此成为判据，
      不再只是一句要求人手执行的话。
      **超出 CR-086 字面范围的部分**：审查方点名的是地板值那一半；数字基线
      变好的那一半是同一个洞（`baseline=5, rank=1` 旧实现同样静默通过），
      一并补上。若认为这会让正常改进变得吵闹，删掉 `_improved_numeric`
      那一个分支即可，其余逻辑不受影响。

    CR-086 记的就是漏掉第三条的后果：R75 我在文档里写下"一旦进了候选就必须
    改成具体名次"，却只写在注释里、没有落成判据。于是
    `baseline=not_in_candidates, known_open=True` 的探针在 rank 为
    `None/20/6/5/1` 时**一律静默通过**——`known_open` 关掉了 top_k 那道闸，
    字符串基线又让退步判据跳过它，两道闸同时打开，正是 CR-015 的同款空间。
    **承诺必须落成判据，写在注释里的承诺不是门禁。**
    """
    regressed = [
        r for r in results
        if isinstance(r.baseline, int) and (r.rank is None or r.rank > r.baseline)
    ]
    below = [r for r in results if not r.passed and not r.known_open]
    def _breached_floor(r: ProbeResult) -> bool:
        return r.baseline == NOT_IN_CANDIDATES and r.rank is not None

    def _improved_numeric(r: ProbeResult) -> bool:
        return isinstance(r.baseline, int) and r.rank is not None and r.rank < r.baseline

    needs_baseline = [r for r in results if _breached_floor(r) or _improved_numeric(r)]
    return regressed, below, needs_baseline


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
    regressed, below, needs_baseline = evaluate(results, top_k)
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
    if needs_baseline:
        print("基线待固化（实测比基线好，必须改成实测名次）:")
        for r in needs_baseline:
            was = "未进候选" if r.baseline == NOT_IN_CANDIDATES else f"第 {r.baseline} 名"
            print(f"  {r.question}: 基线 {was} -> 实测第 {r.rank} 名；"
                  f"请把 ranking_probe.yaml 里这条的 baseline 改成 {r.rank}")

    if args.json:
        args.json.write_text(json.dumps(
            {"top_k": top_k, "passed": passed, "total": len(results),
             "results": [{"q": r.question, "gold": r.gold, "rank": r.rank,
                          "passed": r.passed, "fts_query": r.fts_query}
                         for r in results]},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {args.json}")

    store.close()
    # 退步、未达标、基线待固化都失败。既有缺口（known_open）不阻塞，
    # 但一旦比基线更差、或地板值被突破，立刻报出来（CR-086：改善也要报，
    # 否则地板基线就成了一个什么都不判的后门）。
    return 1 if (regressed or below or needs_baseline) else 0


if __name__ == "__main__":
    raise SystemExit(main())
