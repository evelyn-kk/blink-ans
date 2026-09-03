"""问答编排：检索 → 判定证据充分性 → 构建上下文 → 生成带引用的答案。

四条约束贯穿本模块：

1. **预算约束在 prefill token 上，不是总 token**（I0：prefill 352 tok/s）。
   本地路径的系统提示词 KV 常驻复用，不计入 prefill；云端路径靠 provider 的
   prompt cache，同样不占用这次请求的"证据预算"额度，但云端 prefill 本身
   不受本机算力线性约束（architecture.md §6.6 第 1 条），因此本地/云端两条
   路径的证据预算不再共用一个数字——见 `AnswerConfig.evidence_budget` /
   `evidence_budget_cloud` 与 `Orchestrator._evidence_budget()`。

2. **"证据不足"不能靠空结果判定**（I1）。FTS 用 OR 查询以容忍语音转写的错字，
   代价是几乎任何中文提问都会返回结果。必须用相关性阈值判定。

3. **固定前缀排在最前**（I0）。系统提示词的 KV 常驻复用，证据与问题随请求变化。

4. **拒答判据是语言无关的固定标记，不是散文正则**（T-022，development-notes.md
   2026-09-02「双语输出的架构影响」）。中文散文字面量匹配在英文回答下必然失效——
   界面会一边说"无依据"一边列来源。`declined()` 只做一次 `startswith` 比较，
   不为每种语言各维护一套判据。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterator

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.prompts.answer import (  # noqa: E402
    DECLINE_TOKEN, SUPPORTED_LANGUAGES, Evidence, Language, insufficient_prompt,
    render_user_message, system_prompt, template_version,
)
from packages.schemas.chunk import estimate_tokens  # noqa: E402
from services.inference.router import Router  # noqa: E402
from services.retrieval.embed import Embedder  # noqa: E402
from services.retrieval.search import Hit, hybrid_search  # noqa: E402
from services.retrieval.store import ChunkStore  # noqa: E402
from services.retrieval.tokenize import detect_technology  # noqa: E402


_CITATION = re.compile(r"\[(\d{1,2})\]")


def declined(answer: str) -> bool:
    """模型是否用固定标记表明证据不支撑作答。

    判据是 `packages/prompts/answer.DECLINE_TOKEN`——一个语言无关的固定 token，
    两版提示词都要求模型在证据不支撑结论时把整个回答替换成这一行。
    不匹配散文内容：模型换一种语言、换一种措辞都不影响这个判据。

    检索给错证据时，模型输出这个标记是正确行为。此时展示来源会误导用户——
    界面上一边写着"没有依据"，一边列出五条链接，读者会以为那些就是依据。
    """
    return answer.strip().startswith(DECLINE_TOKEN)


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
    **误拒有效问题的代价高于多答一句带警示的话**，且模型自身输出
    `DECLINE_TOKEN`（见 `packages/prompts/answer.py`）是第二道防线——
    50 题回归中它 6 次正确拒绝了基于错误证据编造（彼时判据仍是散文正则，
    T-022 换成固定标记后语义不变，仍是同一道防线）。

    **样本量仍小，属暂定值，应由 I3 评测集重新标定。**
    """

    # 预算以**真实 token** 计（见 select_evidence），本地/云端两条路径分开算——
    # T-022 重写提示词后重算（旧值 680/700 是按六段式 341 token 系统提示词反推的，
    # 且未区分本地/云端）。
    #
    # **本地（严格档，`evidence_budget`）**：系统提示词是常驻前缀，复用时不占
    # prefill 预算，因此提示词变短本身不改变本地证据上限——真正的依据是
    # architecture.md §6.5 已验证的本地生成子预算：证据 ≤~879 token 时
    # 3.0 秒内出首字。实测 render_user_message 的模板收尾+问题开销约 25–30
    # token，而 select_evidence 按证据正文计数、不含 render_evidence 拼的
    # "[N] 引用串\n"前缀（citation 串实测约 31 token/条，max_evidence=5 条
    # 最多再吃掉约 155 token 未被计入）。879 减去两项开销上限（约 185）
    # 只剩约 694，取整并再留一点余量定为 650。
    evidence_budget: int = 650
    # **云端（宽松档，`evidence_budget_cloud`）**：云端 prefill 不随上下文
    # 线性增长（architecture.md §6.6 第 1 条）——T-026 实测 4096 token 证据
    # 上下文冷启动 TTFT 仍只有 2.9553s，在 3.6s 云端生成子预算内。
    # 3200 留出约 900 token 余量覆盖引用串开销与该实测点之上的方差，
    # 同时远高于 max_evidence=5 条证据在实践中通常达到的总量——对云端路径，
    # 这个数字的作用基本是"不再是约束"，而不是一个需要精确卡线的上限。
    evidence_budget_cloud: int = 3200
    max_evidence: int = 5
    sufficient_distance: float = 0.72
    limited_distance: float = 0.76
    # 输出长度上限，不是 prefill 预算，两条路径共用一个数字。
    # 简洁契约下实测（真实本地生成，4B 模型）：两条证据、需要结论+步骤+前提的
    # 回答 99–116 token；四段式（诊断+修复+验证+回滚风险）复杂回答 176 token。
    # 350 留约 2 倍余量，仍显著小于旧六段式模板的 700——旧值是按"六节都要写满"
    # 的最坏情况反推的，新契约下这不再是典型情况。
    max_tokens: int = 350
    candidates: int = 30
    # 首 token 时延超过此值即记录告警，用于发现预算漂移。
    # 3.0 秒来自 architecture.md 第 6.2 节按实测重新分配后的生成预算：
    # 检索实测 0.15 秒（原预算 0.8 秒），富余的时间划给了生成阶段。
    # T-028：路由定案后本地降级为断网/失败兜底，这个值就是 §6.5 表里
    # "本地兜底路径"那一列的生成子预算，只用于 served_by=local 的判断。
    ttft_budget_s: float = 3.0
    # 云端主路径的生成子预算（architecture.md §6.5："generation_started→
    # first_answer_text" 云端一列 3.6s，T-026 四档 P95 最差值 3.5427s 再留边际）。
    # 与 ttft_budget_s 分开是因为两条路径的预算不同——用同一个阈值判 claude 的
    # ttft_over_budget 会把落在 3.0–3.6s 之间、完全达标的云端响应误判为超预算。
    cloud_ttft_budget_s: float = 3.6
    # 本地常驻前缀预热用的语言（见 apps/gateway/main.py 的 engine.load() 调用），
    # 也是 ClaudeBackend 构造时默认携带的 system prompt 语言。请求语言与这个值
    # 一致时，Orchestrator 对 sufficient/limited 分支传 system_override=None，
    # 让本地引擎复用常驻前缀 KV（云端语义不受影响：ClaudeBackend 的默认值同样
    # 来自这个语言，None 与显式传入同一段文本效果相同）；不一致时才显式传入
    # 对应语言的提示词，此时本地按 T-027 的既定取舍走完整 prefill，不重新实现
    # 多槽常驻前缀（development-notes.md「双语输出的架构影响」：本地已降级为
    # 断网/失败兜底，多语言常驻前缀的收益不值当）。
    default_language: Language = "zh"


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


def _select_evidence_greedy(
    hits: list[Hit], cfg: AnswerConfig, count_tokens=None
) -> list[Evidence]:
    """按融合得分顺序贪心装填的退化路径（旧实现，见 select_evidence 的缺陷说明）。

    只在候选规模/预算大到 0/1 背包 DP 状态表会失控时使用——正常调用路径
    （见 select_evidence 的规模注释）永远走 DP，这个函数只是防御性回退，
    宁可选到次优解也不让内存/时延失控。
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


# 0/1 背包 DP 状态数上限（len(hits) * (max_evidence+1) * (evidence_budget+1)）。
# 实践中唯一调用方 Orchestrator.answer 传入的 hits 来自
# hybrid_search(limit=max_evidence*2)，即 len(hits) <= 10；evidence_budget
# 本地档 650、云端档 3200——真实状态数约 10*6*3201 ≈ 19 万，DP 是毫秒级。
# 这个上限是给未来配置改动（比如把 max_evidence 或 evidence_budget_cloud
# 调大很多）设的安全网：状态数一旦超过它就退化回旧的顺序贪心，避免 DP 表
# 本身占用过多内存或耗时——那种情况下选到次优解也好过卡死。
_DP_STATE_CAP = 3_000_000


def select_evidence(
    hits: list[Hit], cfg: AnswerConfig, count_tokens=None
) -> list[Evidence]:
    """在 token 预算与证据条数上限内挑选融合分之和最大的证据组合。

    旧实现按融合得分顺序贪心装填：放不下就跳过继续看下一条，直到条数或预算耗尽。
    这个策略能处理"当前这条放不下、找条更小的补上"，但处理不了"当前这条放得下、
    但装了它就会挤掉后面两条体积更小、总分更高的组合"——贪心只要装得下就装，
    不会为了给后面的证据腾地方而放弃眼前这条已经能装下的证据。

    构造反例（CR-030 排查记录，development-notes.md 2026-09-02「模型拒答」不等于
    「检索未命中」）：候选 A(score=10, cost=650)、B(score=9, cost=325)、
    C(score=9, cost=325)，budget=650。旧贪心先装 A（650<=650，装满），B、C 都
    放不下，选中总分=10。但 {B, C} 总分=18，同样刚好装满 650——旧算法选到的
    不是预算内总分最大的组合（tests/unit/test_answering.py 的
    test_greedy_evidence_selection_is_suboptimal 用真实的旧函数体复现这一点）。

    换成 0/1 背包动态规划（维度：证据条数上限 × token 预算），在约束内求"融合分
    之和最大"的子集，把优化目标从"顺序贪心"换成"总分最大化"这个更通用的表述。
    不显式识别"这几条是不是同一小节切出来的"——那类情况只是"总分最大化"目标下
    会被自动处理的一个特例，不需要单独的小节感知逻辑。

    规模与复杂度见 _DP_STATE_CAP 的注释：真实调用规模下状态数不到 20 万，
    不需要退化到 O(2^n) 暴力枚举；状态数超过上限时退回 _select_evidence_greedy。

    count_tokens 传入真实分词器时按真实 token 计数；缺省退回切块自带的估算值。
    估算对英文 P90 低估约 16%，只用估算会让首 token 时延的尾部失控。
    """
    n = len(hits)
    if n == 0 or cfg.max_evidence <= 0 or cfg.evidence_budget <= 0:
        return []

    k = cfg.max_evidence
    budget = cfg.evidence_budget

    if n * (k + 1) * (budget + 1) > _DP_STATE_CAP:
        return _select_evidence_greedy(hits, cfg, count_tokens)

    costs = [
        int(count_tokens(h.text)) if count_tokens else int(h.token_estimate)
        for h in hits
    ]

    # dp[i][c][b]：只看前 i 条候选、最多选 c 条、总花费不超过 b 时能拿到的
    # 最大融合分之和。i=0（空集）时任意 (c, b) 都可行——"最多"是两个约束都
    # 取"不超过"，空集花费 0、条数 0，天然满足任意 c>=0、b>=0，值为 0，
    # 不需要 -inf 哨兵（每个状态至少有"什么都不选"这一可行解）。
    dp = [[[0.0] * (budget + 1) for _ in range(k + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost = costs[i - 1]
        score = hits[i - 1].score
        prev_layer = dp[i - 1]
        layer = dp[i]
        for c in range(k + 1):
            prev_c = prev_layer[c]
            row = layer[c]
            if c > 0 and cost <= budget:
                prev_c1 = prev_layer[c - 1]
                for b in range(budget + 1):
                    best = prev_c[b]
                    if b >= cost:
                        cand = prev_c1[b - cost] + score
                        if cand > best:
                            best = cand
                    row[b] = best
            else:
                # 这一条单独就超预算，永远选不了，直接照抄"不选它"的状态。
                row[:] = prev_c

    # 回溯：从"最多 k 条、预算 budget 全部可用"出发，逐条候选往回推，
    # dp[i][c][b] 严格大于 dp[i-1][c][b] 说明第 i 条必须被选中才能达到这个值；
    # 相等则说明存在不含它的同分解，按不选处理（并列时接受任一最优解）。
    chosen: list[int] = []
    c, b = k, budget
    for i in range(n, 0, -1):
        if dp[i][c][b] != dp[i - 1][c][b]:
            cost = costs[i - 1]
            chosen.append(i - 1)
            c -= 1
            b -= cost
    chosen.reverse()  # 恢复为候选原始顺序（即融合分从高到低）

    out: list[Evidence] = []
    for idx in chosen:
        h = hits[idx]
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
    project: str | None = None       # 外部语料的 source_project（兼容已有 API）
    project_id: str | None = None    # 用户项目的稳定 ID，不能与 source_project 混用
    module: str | None = None
    symbol: str | None = None
    max_tokens: int | None = None
    # scope.md 开头：会话开始前选定，同一会话内界面、转写提示与回答语言一致，
    # 不在会话内自动猜测或切换——因此这里是逐请求字段而非进程级配置。
    # 类型用 packages.prompts.answer.Language 的字面量集合（"zh"/"en"），
    # 未知值在 answer() 里显式拒绝，不静默回退到某个默认语言。
    language: Language = "zh"


class Orchestrator:
    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        router: Router,
        config: AnswerConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        # T-028：编排层不再直接持有某个具体生成实现（之前是 InferenceEngine），
        # 只依赖 Router 这一份契约——本地/云端/降级逻辑对编排层不可见，
        # 换句话说编排层现在完全不知道 MLX 的存在。
        self.router = router
        self.cfg = config or AnswerConfig()

    def _evidence_budget(self, cloud_allowed: bool) -> int:
        """按这次请求大概率会走的路径选证据预算档位。

        与 `Router.generate()` 内部判断 `use_cloud` 的三个条件保持一致
        （见 `services/inference/router.py`）——这里只读属性，不产生副作用，
        为的是"这次多半走云端"这个预判要在选证据、构建 prompt 之前就做出来，
        证据集合定型之后才真正开始生成，不可能等生成开始了再回头改证据。

        真正走到本地兜底、却仍用了云端档预算的少数情形（比如判断时云端可用，
        真正发请求时网络在最后一刻失败）是接受的降级：本地兜底路径本就不再是
        产品主 SLA（architecture.md §6.6 第 4 条），慢一次不算错，只是不达标。
        """
        will_use_cloud = (
            not self.router.offline
            and cloud_allowed
            and self.router.cloud is not None
            and self.router.cloud.available()
        )
        return self.cfg.evidence_budget_cloud if will_use_cloud else self.cfg.evidence_budget

    def answer(self, req: AnswerRequest) -> Iterator[dict]:
        """产出事件流。事件类型在 I2 定死，后续迭代只加不改语义。"""
        t0 = time.perf_counter()

        if req.language not in SUPPORTED_LANGUAGES:
            yield {
                "type": "error", "stage": "request",
                "message": f"不支持的语言 {req.language!r}，仅支持 {SUPPORTED_LANGUAGES}",
            }
            return

        try:
            vector = self.embedder.encode_one(req.question)
            # 提问里的技术名当过滤条件用，而不是当检索词——
            # 它们区分度为零，留在查询里只会稀释信号（见 tokenize.PROJECT_TERMS）
            tech = req.technology or detect_technology(req.question)
            hits = hybrid_search(
                self.store, req.question, vector,
                limit=self.cfg.max_evidence * 2,
                technology=tech, project=req.project,
                project_id=req.project_id, module=req.module, symbol=req.symbol,
                candidates=self.cfg.candidates,
            )
        except Exception as exc:
            yield {"type": "error", "stage": "retrieval", "message": f"{type(exc).__name__}: {exc}"}
            return

        # 路由输入之一：本轮命中证据里只要有一条项目材料显式禁止云端，
        # 就必须强制走本地——检索候选里未被最终选为证据的条目也算数（保守判断，
        # 见 architecture.md §8 数据边界：项目禁止云端时任何项目材料不得出网，
        # 而候选阶段这些材料已经进了这次请求的处理过程）。
        cloud_allowed = not any(h.cloud_generation_allowed is False for h in hits)

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
            # 证据不足分支用的是另一套短提示词，占比很小，从不复用常驻前缀
            # （见 engine.stream：system_override 非 None 即放弃复用），
            # 因此这里始终显式传入对应语言版本，没有 None 优化可谈。
            yield from self._generate(
                req.question, max_tokens=220,
                system_override=insufficient_prompt(req.language),
                started=t0, sufficiency=verdict.level, cloud_allowed=cloud_allowed,
            )
            yield {"type": "sources", "items": []}
            return

        # 证据预算按这次请求实际会走哪条路径分档（本地严格 / 云端宽松，
        # 见 AnswerConfig.evidence_budget / evidence_budget_cloud 的注释）。
        budget_cfg = replace(self.cfg, evidence_budget=self._evidence_budget(cloud_allowed))
        evidence = select_evidence(hits, budget_cfg, self.router.count_tokens)
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

        user_msg = render_user_message(req.question, evidence, req.language)
        # 请求语言与本地常驻前缀预热语言一致时传 None，让本地引擎（若被选中）
        # 复用常驻前缀；不一致时才显式传入——见 AnswerConfig.default_language。
        override = (
            None if req.language == self.cfg.default_language
            else system_prompt(req.language)
        )
        answer = ""
        for ev in self._generate(
            user_msg, max_tokens=req.max_tokens or self.cfg.max_tokens,
            started=t0, sufficiency=verdict.level, evidence_count=len(evidence),
            cloud_allowed=cloud_allowed, system_override=override,
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
        evidence_count: int = 0, cloud_allowed: bool = True,
    ) -> Iterator[dict]:
        text = ""
        try:
            for ev in self.router.generate(
                content, max_tokens=max_tokens, system_override=system_override,
                cloud_allowed=cloud_allowed,
            ):
                if ev["type"] == "delta":
                    text += ev["text"]
                    yield {"type": "answer_delta", "text": ev["text"]}
                elif ev["type"] == "error":
                    # 云端已吐出部分正文后失败：Router 不会拼接本地续写
                    # （见 router.py 模块docstring），这里原样透传让请求干净结束。
                    yield ev
                elif ev["type"] == "done":
                    cited = citation_coverage(text, evidence_count)
                    # 两条路径预算不同（architecture.md §6.5），按 served_by 选对应阈值。
                    budget = (
                        self.cfg.cloud_ttft_budget_s if ev["served_by"] == "claude"
                        else self.cfg.ttft_budget_s
                    )
                    yield {
                        "type": "done",
                        # architecture.md §7："done 必须包含 served_by"，
                        # 由 Router 决策产出，这里只负责透传，不重新判断。
                        "served_by": ev["served_by"],
                        "sufficiency": sufficiency.value,
                        "template_version": template_version(),
                        "ttft_s": ev["ttft_s"],
                        "ttft_over_budget": ev["ttft_s"] > budget,
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
