"""混合检索 —— 关键词 + 向量，RRF 融合。

上下文预算是这里的一等公民。I0 实测 prefill 352 tok/s，2.5 秒首 token 预算
只对应约 879 token 上下文，因此检索不能只按相关性返回 top-k，
必须在**给定 token 预算内**挑出信息量最大的证据组合。
"""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from typing import Sequence

from .store import ChunkStore

# RRF 的平滑常数。60 是文献中的常用取值：让前几名之间的差距不至于压倒性，
# 使两路检索都能对最终排序产生影响。
RRF_K = 60

# 两路等权（2026-09-02 / T-017 重新标定）。
#
# I1 定为 0.5 的理由是"语料全英文、提问全中文，bm25 只能靠 ASCII 词排序"，
# 那时 term_map 还不存在。中文查询现已展开为英文检索词，前提已经变了。
#
# 0.5 的代价被低估了：RRF 下 `0.5/(60+1) == 1.0/(60+62)`，
# **关键词路的第 1 名只等价于向量路的第 62 名**，向量路 top-62 里任何一条都能压过它。
# 而配置项名、类名、错误码这类精确串恰恰只有关键词路找得到。
#
# 标定方法：10 题排序探针（人工核对过的金标准块）+ 44 题来源护栏，均不跑 LLM。
#
# | KEYWORD_WEIGHT | 探针正确块进前 5 | 44 题 top5 含期望来源 |
# | --- | --- | --- |
# | 0.5（原） | 3/10 | 44/44 |
# | **1.0** | **5/10** | **44/44** |
# | 2.0 | 6/10 | 42/44 ← 护栏退步 |
#
# 2.0 更激进但护栏掉了两题——I1 担心的关键词噪声确实存在，只是 0.5 矫枉过正。
KEYWORD_WEIGHT = 1.0
VECTOR_WEIGHT = 1.0


@dataclass
class Hit:
    rowid: int
    text: str
    title_path: str
    source_url: str
    source_project: str
    version_or_commit: str
    retrieved_at: str
    technology: str
    content_type: str
    token_estimate: int
    score: float
    # 项目过滤不应只依赖 source_project（它是外部来源的历史字段）。
    # 以下字段让调用方能明确把一次查询限制在某个用户项目及其代码边界内。
    project_id: str | None = None
    module: str | None = None
    symbol: str | None = None
    cloud_generation_allowed: bool | None = None
    keyword_rank: int | None = None
    vector_rank: int | None = None
    # 向量距离是判定"证据是否充分"的主要信号（见 orchestrator/answering.py）。
    # RRF 分数只反映名次，无法区分"最相关的一条"和"矮子里拔将军"。
    vector_distance: float | None = None

    @property
    def citation(self) -> str:
        return (
            f"{self.source_project} {self.version_or_commit} · "
            f"{self.title_path} · 抓取于 {self.retrieved_at[:10]}"
        )


def _pack(vec: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _where(
    technology: str | None,
    project: str | None,
    project_id: str | None,
    module: str | None,
    symbol: str | None,
) -> tuple[str, list]:
    clauses, params = [], []
    if technology:
        clauses.append("c.technology = ?")
        params.append(technology)
    if project:
        clauses.append("c.source_project = ?")
        params.append(project)
    if project_id:
        clauses.append("c.project_id = ?")
        params.append(project_id)
    if module:
        clauses.append("c.module = ?")
        params.append(module)
    if symbol:
        clauses.append("c.symbol = ?")
        params.append(symbol)
    return (" AND " + " AND ".join(clauses) if clauses else ""), params


def keyword_search(
    store: ChunkStore, query: str, limit: int = 30,
    technology: str | None = None, project: str | None = None,
    *, project_id: str | None = None, module: str | None = None, symbol: str | None = None,
) -> list[tuple[int, float]]:
    from .tokenize import to_fts_query

    cond, params = _where(technology, project, project_id, module, symbol)
    sql = f"""
        SELECT c.id AS rowid, bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        WHERE chunks_fts MATCH ?{cond}
        ORDER BY score
        LIMIT ?
    """
    try:
        rows = store.execute(sql, [to_fts_query(query), *params, limit])
    except sqlite3.OperationalError as exc:
        # FTS5 的 MATCH 语法错误按"无关键词命中"处理是合理的；
        # 但其他故障（表缺失、索引损坏）必须抛出——
        # 一律吞掉会让中文关键词整路失效却与"确实没命中"无法区分，
        # 索引照样通过回归并被激活。
        if "fts5" in str(exc).lower() or "malformed MATCH" in str(exc):
            return []
        raise
    # bm25 返回负值，越小越相关
    return [(r["rowid"], r["score"]) for r in rows]


def vector_search(
    store: ChunkStore, vector: Sequence[float], limit: int = 30,
    technology: str | None = None, project: str | None = None,
    *, project_id: str | None = None, module: str | None = None, symbol: str | None = None,
) -> list[tuple[int, float]]:
    # vec0 的 KNN 不支持与业务表 JOIN 后再过滤，因此先取更多候选再在外层过滤
    has_filter = any((technology, project, project_id, module, symbol))
    over = limit * 4 if has_filter else limit
    rows = store.execute(
        "SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (_pack(vector), over),
    )
    if not rows:
        return []

    if has_filter:
        cond, params = _where(technology, project, project_id, module, symbol)
        ids = [r["rowid"] for r in rows]
        keep = {
            r["id"] for r in store.execute(
                f"SELECT c.id FROM chunks c WHERE c.id IN ({','.join('?' * len(ids))}){cond}",
                [*ids, *params],
            )
        }
        rows = [r for r in rows if r["rowid"] in keep]

    return [(r["rowid"], r["distance"]) for r in rows[:limit]]


def rrf_fuse(
    keyword: list[tuple[int, float]],
    vector: list[tuple[int, float]],
    k: int = RRF_K,
    keyword_weight: float = KEYWORD_WEIGHT,
    vector_weight: float = VECTOR_WEIGHT,
) -> dict[int, tuple[float, int | None, int | None]]:
    """倒数排名融合。

    用排名而非原始分数：bm25 与向量距离的量纲完全不同，直接加权需要
    per-query 归一化，既脆弱又难调。RRF 只看名次，对分数分布不敏感。
    """
    scores: dict[int, list] = {}
    for rank, (rid, _) in enumerate(keyword, 1):
        scores.setdefault(rid, [0.0, None, None])
        scores[rid][0] += keyword_weight / (k + rank)
        scores[rid][1] = rank
    for rank, (rid, _) in enumerate(vector, 1):
        scores.setdefault(rid, [0.0, None, None])
        scores[rid][0] += vector_weight / (k + rank)
        scores[rid][2] = rank
    return {rid: tuple(v) for rid, v in scores.items()}


def hybrid_search(
    store: ChunkStore,
    query: str,
    query_vector: Sequence[float] | None = None,
    *,
    limit: int = 8,
    token_budget: int | None = None,
    technology: str | None = None,
    project: str | None = None,
    project_id: str | None = None,
    module: str | None = None,
    symbol: str | None = None,
    candidates: int = 30,
) -> list[Hit]:
    """关键词与向量并行检索后 RRF 融合，可选按 token 预算截断。

    token_budget 不为空时，按融合得分依次取块直到预算耗尽——
    这是把 I0 的时延约束落到检索层的地方。
    """
    kw = keyword_search(
        store, query, candidates, technology, project,
        project_id=project_id, module=module, symbol=symbol,
    )
    vec = vector_search(
        store, query_vector, candidates, technology, project,
        project_id=project_id, module=module, symbol=symbol,
    ) if query_vector else []
    distances = dict(vec)
    fused = rrf_fuse(kw, vec)
    if not fused:
        return []

    ordered = sorted(fused.items(), key=lambda kv: -kv[1][0])
    ids = [rid for rid, _ in ordered]
    rows = {
        r["id"]: r for r in store.execute(
            f"SELECT * FROM chunks WHERE id IN ({','.join('?' * len(ids))})", ids
        )
    }

    hits: list[Hit] = []
    used = 0
    for rid, (score, krank, vrank) in ordered:
        r = rows.get(rid)
        if r is None:
            continue
        # 仍在服役的 current.db 可能由 T-103 扩列之前构建；查询官方知识时，
        # 缺失的项目字段应等价于 NULL，而不能让整个检索路径崩溃。
        fields = set(r.keys())
        if token_budget is not None:
            if used + r["token_estimate"] > token_budget:
                continue  # 跳过放不下的，继续找更小的块把预算填满
            used += r["token_estimate"]
        hits.append(
            Hit(
                rowid=rid, text=r["text"], title_path=r["title_path"],
                source_url=r["source_url"], source_project=r["source_project"],
                version_or_commit=r["version_or_commit"], retrieved_at=r["retrieved_at"],
                technology=r["technology"], content_type=r["content_type"],
                token_estimate=r["token_estimate"], score=score,
                project_id=r["project_id"] if "project_id" in fields else None,
                module=r["module"] if "module" in fields else None,
                symbol=r["symbol"] if "symbol" in fields else None,
                cloud_generation_allowed=(
                    bool(r["cloud_generation_allowed"])
                    if "cloud_generation_allowed" in fields
                    and r["cloud_generation_allowed"] is not None else None
                ),
                keyword_rank=krank, vector_rank=vrank,
                vector_distance=distances.get(rid),
            )
        )
        if len(hits) >= limit:
            break
    return hits
