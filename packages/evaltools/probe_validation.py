"""检索验证集（held-out）的运行器 —— 只报名次，不判成败。

为什么它与 `probe_ranking.py` 分开（CR-097）：那 13 条排序探针既用来**选**
排序参数、又用来宣称"零退步"，属自证。这份集合的题面全部来自 I2 时期的
`basic_questions.yaml`、gold 逐条人工读过，**不参与任何参数选择**，只在
改动前后各跑一次做对照。

它**刻意不是门禁**：没有 baseline、退出码恒为 0。理由写在
`knowledge/eval/ranking_validation.yaml` 顶部——一旦让它决定 CI 红绿，
下一次调参就会把它也当成靶子，独立性当场消失。

用法:
    .venv/bin/python packages/evaltools/probe_validation.py
    .venv/bin/python packages/evaltools/probe_validation.py --rrf-k 10
    .venv/bin/python packages/evaltools/probe_validation.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import yaml  # noqa: E402

from services.retrieval import search as search_mod  # noqa: E402
from services.retrieval.embed import Embedder  # noqa: E402
from services.retrieval.search import hybrid_search  # noqa: E402
from services.retrieval.store import ChunkStore  # noqa: E402
from services.retrieval.tokenize import detect_technology  # noqa: E402

CASES = Path(__file__).resolve().parents[2] / "knowledge" / "eval" / "ranking_validation.yaml"


def _matches(hit, gold: str, project: str | None, exact: bool) -> bool:
    if project and hit.source_project != project:
        return False
    return hit.title_path == gold if exact else hit.title_path.startswith(gold)


def run(spec: dict, store: ChunkStore, embedder: Embedder, limit: int) -> int | None:
    q = spec["q"]
    hits = hybrid_search(
        store, q, embedder.encode_one(q), limit=limit,
        technology=detect_technology(q), candidates=30,
    )
    for i, h in enumerate(hits, 1):
        if _matches(h, spec["gold"], spec.get("project"), spec.get("gold_exact", False)):
            return i
    return None


def unresolvable_golds(spec: dict, store: ChunkStore) -> list[str]:
    """gold 在索引里必须真的存在——否则这把尺子量什么都是"未进候选"。

    这是本文件唯一会失败的检查。理由：`rank=None` 有两种完全不同的含义，
    "检索没找到"和"gold 写错了、根本没有这一块"，而它们在输出里长得一模一样。
    前者是要观测的现象，后者是工具坏了（§5.4：哪些输入会让它什么都不判）。
    """
    missing = []
    for case in spec["cases"]:
        gold, project = case["gold"], case.get("project")
        if case.get("gold_exact"):
            sql = "SELECT 1 FROM chunks WHERE title_path = ?"
        else:
            sql = "SELECT 1 FROM chunks WHERE title_path LIKE ? || '%'"
        params = [gold]
        if project:
            sql += " AND source_project = ?"
            params.append(project)
        if not store.execute(sql + " LIMIT 1", params):
            missing.append(f'{case["q"][:34]} -> gold {gold!r}')
    return missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rrf-k", type=int, help="临时改用这个 RRF_K 跑（只影响本次进程）")
    ap.add_argument("--json", help="把逐题名次写到文件，便于两次运行做 diff")
    args = ap.parse_args()

    spec = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    if args.rrf_k:
        # hybrid_search 在调用时读模块全局量，所以这里改是生效的；
        # `rrf_fuse()` 的默认参数则是定义时绑定（T-017 踩过的坑），本文件不用它。
        search_mod.RRF_K = args.rrf_k

    store = ChunkStore()
    missing = unresolvable_golds(spec, store)
    if missing:
        print("gold 在当前索引里找不到对应的块，先修这个"
              "（否则名次恒为『未进候选』，看起来像检索差，其实是尺子坏了）：",
              file=sys.stderr)
        for m in missing:
            print("  -", m, file=sys.stderr)
        return 2

    embedder = Embedder()
    embedder.load()

    print(f"检索验证集（held-out，不参与调参）  RRF_K={search_mod.RRF_K}  "
          f"limit={spec['limit']}\n")
    out = []
    for case in spec["cases"]:
        rank = run(case, store, embedder, spec["limit"])
        ref = case.get("rank_at_k60")
        delta = ""
        if isinstance(ref, int) and rank != ref:
            delta = f"   （记录值 {ref} → 本次 {rank if rank else '未进候选'}）"
        print(f"  {str(rank) if rank else '未进候选':>6}   {case['q'][:44]}{delta}")
        out.append({"q": case["q"], "rank": rank, "rank_at_k60": ref})

    top = sum(1 for o in out if o["rank"] and o["rank"] <= spec["top_k"])
    print(f"\n进前 {spec['top_k']}: {top}/{len(out)}"
          f"    中位名次: {sorted(o['rank'] or 10**6 for o in out)[len(out)//2]}")
    print("（这是对照尺，不是门禁：退出码恒为 0）")
    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
