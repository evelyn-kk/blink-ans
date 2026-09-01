"""问答编排：检索 → 判定证据充分性 → 构建上下文 → 生成带引用的答案。

三条约束贯穿本模块，都来自前两个迭代的实测：

1. **预算约束在 prefill token 上，不是总 token**（I0：prefill 352 tok/s）。
   系统提示词的 KV 常驻复用（实测 190 token），因此它不计入 prefill。
   2.5 秒预算 ≈ 880 个待 prefill 的 token，扣除问题与模板收尾约 40 token，
   证据可用约 840 个真实 token。切块的 token_estimate 对英文平均高估约 11%
   （200 块抽样，实际/估算中位 0.888），故估算口径的预算取 700 留出余量。

2. **"证据不足"不能靠空结果判定**（I1）。FTS 用 OR 查询以容忍语音转写的错字，
   代价是几乎任何中文提问都会返回结果。必须用相关性阈值判定。

3. **固定前缀排在最前**（I0）。系统提示词的 KV 常驻复用，证据与问题随请求变化。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.prompts.answer import (  # noqa: E402
    INSUFFICIENT_PROMPT, Evidence, render_user_message, template_version,
)
from packages.schemas.chunk import estimate_tokens  # noqa: E402
from services.inference.engine import InferenceEngine  # noqa: E402
from services.retrieval.embed import Embedder  # noqa: E402
from services.retrieval.search import Hit, hybrid_search  # noqa: E402
from services.retrieval.store import ChunkStore  # noqa: E402
from services.retrieval.tokenize import detect_technology  # noqa: E402


_CITATION = re.compile(r"\[(\d{1,2})\]")


def citation_coverage(answer: str, evidence_count: int) -> list[int]:
    """答案中实际引用到的证据编号。

    引用覆盖率是 architecture.md 第 9 节要求记录的验收指标之一。
    没有引用的技术论断无法追溯出处，对本产品等同于不可用——
    因此这个数字要随每次回答一起返回，而不是等到评测时才统计。
    """
    if evidence_count <= 0:
        return []
    found = {int(m) for m in _CITATION.findall(answer)}
    return sorted(i for i in found if 1 <= i <= evidence_count)


class Sufficiency(str, Enum):
    SUFFICIENT = "sufficient"    # 证据充分，正常作答
    LIMITED = "limited"          # 证据相关但不紧密，作答并标注不确定
    INSUFFICIENT = "insufficient"  # 证据不足，不给技术结论


@dataclass
class AnswerConfig:
    """默认值来自实测，不是经验取值。

    距离阈值由 10 条对照查询测得：语料覆盖的问题 top1 距离 0.60–0.70，
    不覆盖的 0.75–0.83，最近一对为 0.6995 与 0.7480。
    **样本量小，属暂定值，应由 I3 评测集重新标定。**
    """

    evidence_budget: int = 700
    max_evidence: int = 5
    sufficient_distance: float = 0.72
    limited_distance: float = 0.78
    max_tokens: int = 700
    candidates: int = 30
    # 首 token 时延超过此值即记录告警，用于发现预算漂移
    ttft_budget_s: float = 2.5


@dataclass
class Assessment:
    level: Sufficiency
    top_distance: float | None
    keyword_hits: int
    reason: str


def assess(hits: list[Hit], cfg: AnswerConfig) -> Assessment:
    """判定证据是否足以支撑一个技术结论。

    主信号是最相关一条的向量距离；关键词命中作为辅助——
    两路都不沾边时几乎可以确定问题不在语料范围内。
    """
    if not hits:
        return Assessment(Sufficiency.INSUFFICIENT, None, 0, "检索无结果")

    dists = [h.vector_distance for h in hits if h.vector_distance is not None]
    top = min(dists) if dists else None
    kw = sum(1 for h in hits if h.keyword_rank is not None)

    if top is None:
        # 只有关键词命中而无向量结果，无法评估语义相关性，保守处理
        return Assessment(Sufficiency.LIMITED, None, kw, "缺少向量相关性信号")

    if top <= cfg.sufficient_distance:
        return Assessment(Sufficiency.SUFFICIENT, top, kw, f"最相关证据距离 {top:.3f}")
    if top <= cfg.limited_distance and kw > 0:
        return Assessment(
            Sufficiency.LIMITED, top, kw,
            f"最相关证据距离 {top:.3f}，相关但不紧密",
        )
    return Assessment(
        Sufficiency.INSUFFICIENT, top, kw,
        f"最相关证据距离 {top:.3f}，超出可信范围",
    )


def select_evidence(hits: list[Hit], cfg: AnswerConfig) -> list[Evidence]:
    """在 token 预算内挑选证据。

    按融合得分依次取，放不下的跳过继续找更小的——
    宁可多带一条短证据，也不要让预算空着。
    """
    out: list[Evidence] = []
    used = 0
    for h in hits:
        if len(out) >= cfg.max_evidence:
            break
        if used + h.token_estimate > cfg.evidence_budget:
            continue
        used += h.token_estimate
        out.append(
            Evidence(
                index=len(out) + 1,
                text=h.text,
                citation=h.citation,
                source_url=h.source_url,
            )
        )
    return out


@dataclass
class AnswerRequest:
    question: str
    technology: str | None = None
    project: str | None = None
    max_tokens: int | None = None


class Orchestrator:
    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        engine: InferenceEngine,
        config: AnswerConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.engine = engine
        self.cfg = config or AnswerConfig()

    def answer(self, req: AnswerRequest) -> Iterator[dict]:
        """产出事件流。事件类型在 I2 定死，后续迭代只加不改语义。"""
        t0 = time.perf_counter()

        try:
            vector = self.embedder.encode_one(req.question)
            # 提问里的技术名当过滤条件用，而不是当检索词——
            # 它们区分度为零，留在查询里只会稀释信号（见 tokenize.PROJECT_TERMS）
            tech = req.technology or detect_technology(req.question)
            hits = hybrid_search(
                self.store, req.question, vector,
                limit=self.cfg.max_evidence * 2,
                technology=tech, project=req.project,
                candidates=self.cfg.candidates,
            )
        except Exception as exc:
            yield {"type": "error", "stage": "retrieval", "message": f"{type(exc).__name__}: {exc}"}
            return

        verdict = assess(hits, self.cfg)
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)

        yield {
            "type": "retrieval",
            "hits": len(hits),
            "sufficiency": verdict.level.value,
            "top_distance": round(verdict.top_distance, 4) if verdict.top_distance else None,
            "keyword_hits": verdict.keyword_hits,
            "reason": verdict.reason,
            "elapsed_ms": retrieval_ms,
        }

        if verdict.level is Sufficiency.INSUFFICIENT:
            yield {"type": "status", "message": "本地知识库未覆盖该问题"}
            yield from self._generate(
                req.question, max_tokens=220, system_override=INSUFFICIENT_PROMPT,
                started=t0, sufficiency=verdict.level,
            )
            yield {"type": "sources", "items": []}
            return

        evidence = select_evidence(hits, self.cfg)
        if not evidence:
            # 命中了但每一条都超出预算——属于切块异常，不应静默降级为无证据作答
            yield {"type": "error", "stage": "context",
                   "message": "检索有结果但均超出上下文预算，无法构建证据"}
            return

        yield {
            "type": "status",
            "message": f"已选取 {len(evidence)} 条证据"
                       + ("（相关性一般，回答将标注不确定）" if verdict.level is Sufficiency.LIMITED else ""),
            "evidence_tokens": sum(estimate_tokens(e.text) for e in evidence),
        }

        user_msg = render_user_message(req.question, evidence)
        yield from self._generate(
            user_msg, max_tokens=req.max_tokens or self.cfg.max_tokens,
            started=t0, sufficiency=verdict.level, evidence_count=len(evidence),
        )

        yield {
            "type": "sources",
            "items": [
                {"index": e.index, "citation": e.citation, "url": e.source_url}
                for e in evidence
            ],
        }

    def _generate(
        self, content: str, *, max_tokens: int, started: float,
        sufficiency: Sufficiency, system_override: str | None = None,
        evidence_count: int = 0,
    ) -> Iterator[dict]:
        text = ""
        try:
            for ev in self.engine.stream(
                content, max_tokens=max_tokens, system_override=system_override
            ):
                if ev["type"] == "delta":
                    text += ev["text"]
                    yield {"type": "answer_delta", "text": ev["text"]}
                elif ev["type"] == "done":
                    cited = citation_coverage(text, evidence_count)
                    yield {
                        "type": "done",
                        "sufficiency": sufficiency.value,
                        "template_version": template_version(),
                        "ttft_s": ev["ttft_s"],
                        "ttft_over_budget": ev["ttft_s"] > self.cfg.ttft_budget_s,
                        "prompt_tokens": ev["prompt_tokens"],
                        "prefilled_tokens": ev["prefilled_tokens"],
                        "prefix_reused": ev["prefix_reused"],
                        "decode_tps": ev["decode_tps"],
                        # architecture.md 第 9 节要求记录引用覆盖率：
                        # 没有引用的技术结论无法追溯，等同于不可用
                        "cited_evidence": cited,
                        "evidence_count": evidence_count,
                        "total_s": round(time.perf_counter() - started, 3),
                    }
        except Exception as exc:
            yield {"type": "error", "stage": "generation", "message": f"{type(exc).__name__}: {exc}"}
