"""问答编排：检索 → 判定证据充分性 → 构建上下文 → 生成带引用的答案。

三条约束贯穿本模块，都来自前两个迭代的实测：

1. **预算约束在 prefill token 上，不是总 token**（I0：prefill 352 tok/s）。
   系统提示词的 KV 常驻复用（2026-09-02 实测 341 token，套 chat template 后 354），
   因此它不计入 prefill。I2 为提高引用依从率扩写过提示词，此处原记的 190 是扩写前的值。
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
# 模型明确表示证据不支撑作答的措辞
_DECLINED = re.compile(r"证据未涵盖|现有证据不足|证据不足以")


def declined(answer: str) -> bool:
    """模型是否明确表示证据不支撑作答。

    检索给错证据时，模型说"证据未涵盖"是正确行为。此时展示来源会误导用户——
    界面上一边写着"没有依据"，一边列出五条链接，读者会以为那些就是依据。
    """
    return bool(_DECLINED.search(answer)) and len(answer.strip()) < 120


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

    距离阈值经两轮标定：初版用 10 条跨领域对照（覆盖 0.60–0.70 / 不覆盖 0.75–0.83），
    50 题回归暴露出边界过紧——「Spring Boot 怎么配置日志级别」实测 0.7241 被误判为超范围。
    因此把中间档放宽到 0.76，并使其真正生效（见 assess）。

    代价是极少数超范围问题会落入中间档、带警示作答而非直接拒答
    （实测「怎么用 Rust 写词法分析器」为 0.7463）。这是有意的取舍：
    **误拒有效问题的代价高于多答一句带警示的话**，且模型自身的"证据未涵盖"
    是第二道防线——50 题回归中它 6 次正确拒绝了基于错误证据编造。

    **样本量仍小，属暂定值，应由 I3 评测集重新标定。**
    """

    # 预算以**真实 token** 计（见 select_evidence）。
    # 复用前缀后实测 prefill 约 342 tok/s，2.5 秒对应 855 token；
    # 扣除问题与模板收尾约 60 token，证据预算取 680 并留出余量。
    evidence_budget: int = 680
    max_evidence: int = 5
    sufficient_distance: float = 0.72
    limited_distance: float = 0.76
    max_tokens: int = 700
    candidates: int = 30
    # 首 token 时延超过此值即记录告警，用于发现预算漂移。
    # 3.0 秒来自 architecture.md 第 6.2 节按实测重新分配后的生成预算：
    # 检索实测 0.15 秒（原预算 0.8 秒），富余的时间划给了生成阶段。
    ttft_budget_s: float = 3.0


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
    if top <= cfg.limited_distance:
        # 不再要求关键词命中：技术名词被剔除后（见 tokenize.PROJECT_TERMS），
        # 中文提问打英文语料时关键词命中常为 0，该条件会让中间档永不生效，
        # 把边缘的有效问题一律误拒。
        return Assessment(
            Sufficiency.LIMITED, top, kw,
            f"最相关证据距离 {top:.3f}，相关但不紧密",
        )
    return Assessment(
        Sufficiency.INSUFFICIENT, top, kw,
        f"最相关证据距离 {top:.3f}，超出可信范围",
    )


def select_evidence(
    hits: list[Hit], cfg: AnswerConfig, count_tokens=None
) -> list[Evidence]:
    """在 token 预算内挑选证据。

    按融合得分依次取，放不下的跳过继续找更小的——
    宁可多带一条短证据，也不要让预算空着。

    count_tokens 传入真实分词器时按真实 token 计数；缺省退回切块自带的估算值。
    估算对英文 P90 低估约 16%，只用估算会让首 token 时延的尾部失控。
    """
    out: list[Evidence] = []
    used = 0
    for h in hits:
        if len(out) >= cfg.max_evidence:
            break
        cost = count_tokens(h.text) if count_tokens else h.token_estimate
        if used + cost > cfg.evidence_budget:
            continue
        used += cost
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

        evidence = select_evidence(hits, self.cfg, self.engine.count_tokens)
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
        answer = ""
        for ev in self._generate(
            user_msg, max_tokens=req.max_tokens or self.cfg.max_tokens,
            started=t0, sufficiency=verdict.level, evidence_count=len(evidence),
        ):
            if ev["type"] == "answer_delta":
                answer += ev["text"]
            yield ev

        if declined(answer):
            # 模型判定这些证据支撑不了结论，就不该把它们当作来源展示
            yield {"type": "status", "message": "检索到的资料未涵盖该问题，未采纳为来源"}
            yield {"type": "sources", "items": []}
            return

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
