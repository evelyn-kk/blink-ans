"""混合检索 —— 关键词 + 向量，RRF 融合。

上下文预算是这里的一等公民。I0 实测 prefill 352 tok/s，2.5 秒首 token 预算
只对应约 879 token 上下文，因此检索不能只按相关性返回 top-k，
必须在**给定 token 预算内**挑出信息量最大的证据组合。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Sequence

from .store import ChunkStore

# RRF 的平滑常数。60 是文献中的常用取值：让前几名之间的差距不至于压倒性，
# 使两路检索都能对最终排序产生影响。
RRF_K = 60


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
    keyword_rank: int | None = None
    vector_rank: int | None = None

    @property
    def citation(self) -> str:
        return (
            f"{self.source_project} {self.version_or_commit} · "
            f"{self.title_path} · 抓取于 {self.retrieved_at[:10]}"
        )


def _pack(vec: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _where(technology: str | None, project: str | None) -> tuple[str, list]:
    clauses, params = [], []
    if technology:
        clauses.append("c.technology = ?")
        params.append(technology)
    if project:
        clauses.append("c.source_project = ?")
        params.append(project)
    return (" AND " + " AND ".join(clauses) if clauses else ""), params


def keyword_search(
    store: ChunkStore, query: str, limit: int = 30,
    technology: str | None = None, project: str | None = None,
) -> list[tuple[int, float]]:
    from .tokenize import to_fts_query

    cond, params = _where(technology, project)
    sql = f"""
        SELECT c.id AS rowid, bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        WHERE chunks_fts MATCH ?{cond}
        ORDER BY score
        LIMIT ?
    """
    try:
        rows = store.db.execute(sql, [to_fts_query(query), *params, limit]).fetchall()
    except Exception:
        return []
    # bm25 返回负值，越小越相关
    return [(r["rowid"], r["score"]) for r in rows]


def vector_search(
    store: ChunkStore, vector: Sequence[float], limit: int = 30,
    technology: str | None = None, project: str | None = None,
) -> list[tuple[int, float]]:
    # vec0 的 KNN 不支持与业务表 JOIN 后再过滤，因此先取更多候选再在外层过滤
    over = limit * 4 if (technology or project) else limit
    rows = store.db.execute(
        "SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (_pack(vector), over),
    ).fetchall()
    if not rows:
        return []

    if technology or project:
        cond, params = _where(technology, project)
        ids = [r["rowid"] for r in rows]
        keep = {
            r["id"] for r in store.db.execute(
                f"SELECT c.id FROM chunks c WHERE c.id IN ({','.join('?' * len(ids))}){cond}",
                [*ids, *params],
            )
        }
        rows = [r for r in rows if r["rowid"] in keep]

    return [(r["rowid"], r["distance"]) for r in rows[:limit]]


def rrf_fuse(
    keyword: list[tuple[int, float]], vector: list[tuple[int, float]], k: int = RRF_K
) -> dict[int, tuple[float, int | None, int | None]]:
    """倒数排名融合。

    用排名而非原始分数：bm25 与向量距离的量纲完全不同，直接加权需要
    per-query 归一化，既脆弱又难调。RRF 只看名次，对分数分布不敏感。
    """
    scores: dict[int, list] = {}
    for rank, (rid, _) in enumerate(keyword, 1):
        scores.setdefault(rid, [0.0, None, None])
        scores[rid][0] += 1.0 / (k + rank)
        scores[rid][1] = rank
    for rank, (rid, _) in enumerate(vector, 1):
        scores.setdefault(rid, [0.0, None, None])
        scores[rid][0] += 1.0 / (k + rank)
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
    candidates: int = 30,
) -> list[Hit]:
    """关键词与向量并行检索后 RRF 融合，可选按 token 预算截断。

    token_budget 不为空时，按融合得分依次取块直到预算耗尽——
    这是把 I0 的时延约束落到检索层的地方。
    """
    kw = keyword_search(store, query, candidates, technology, project)
    vec = vector_search(store, query_vector, candidates, technology, project) if query_vector else []
    fused = rrf_fuse(kw, vec)
    if not fused:
        return []

    ordered = sorted(fused.items(), key=lambda kv: -kv[1][0])
    ids = [rid for rid, _ in ordered]
    rows = {
        r["id"]: r for r in store.db.execute(
            f"SELECT * FROM chunks WHERE id IN ({','.join('?' * len(ids))})", ids
        )
    }

    hits: list[Hit] = []
    used = 0
    for rid, (score, krank, vrank) in ordered:
        r = rows.get(rid)
        if r is None:
            continue
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
                keyword_rank=krank, vector_rank=vrank,
            )
        )
        if len(hits) >= limit:
            break
    return hits
