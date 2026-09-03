"""会话层（T-104）：在一次性问答编排（Orchestrator）之上加一层会话状态，
让同一次对话里能做独立问答、项目问答、追问，并在切换/清空上下文时不泄漏。

不新增一套推理逻辑——检索、充分性判定、证据选取、生成全部仍由
`services/orchestrator/answering.Orchestrator` 完成。这一层只做三件事：

1. **范围判定**：按 architecture.md §3 的优先级链，把这一轮问题归到
   "显式选择项目" / "显式新问题" / "追问" / "通用知识库独立问题" 之一。
2. **追问检索**：先用上一轮实体收紧这一轮检索、并把上一轮引用过的块
   （按 rowid 精确取回）纳入候选；仍不够时才放开限制（但项目边界永不放开）。
3. **状态派生**：一轮问答结束后，从这一轮已经产出的事件（不额外调用生成模型）
   派生"简要结论""实体""引用块 ID"存回会话，供下一轮追问使用。

会话状态只保存 architecture.md §3 列出的字段——语言、活动项目/版本、
上一轮的规范化问题/实体/简要结论/引用块 ID/未解决点。不做完整聊天历史的无限累加。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.prompts.answer import Language  # noqa: E402
from services.orchestrator.answering import (  # noqa: E402
    AnswerConfig, AnswerRequest, Orchestrator, assess, declined,
)
from services.retrieval.embed import Embedder  # noqa: E402
from services.retrieval.search import Hit, hits_by_rowid, hybrid_search  # noqa: E402
from services.retrieval.store import ChunkStore  # noqa: E402
from services.retrieval.tokenize import detect_technology, matched_terms  # noqa: E402


# ---------------------------------------------------------------------------
# 会话状态
# ---------------------------------------------------------------------------

@dataclass
class TurnContext:
    """上一轮遗留给下一轮的最小状态（architecture.md §3，不是完整历史）。"""

    question: str                         # 上一轮的规范化问题（用户原始文本，非追问拼接后的文本）
    entities: dict[str, object]           # technology/project_id/module/symbol/version/keywords
    brief_conclusion: str | None          # 上一轮答案的简要结论（派生自答案正文，非模型另外生成）
    cited_chunk_ids: list[int]            # 上一轮答案实际引用到的块 rowid
    open_issue: str | None = None         # 上一轮充分性不是 sufficient 时的简短标记，供追问参考


@dataclass
class SessionState:
    """服务端持有的会话状态。内存字典存储，参照 apps/gateway/main.py 的 `_pending`
    模式：容量上限、超限淘汰，不落盘。
    """

    session_id: str
    language: Language
    active_project_id: str | None = None
    active_version: str | None = None
    last_turn: TurnContext | None = None
    # CR-021：`POST .../turns` 建好一轮（携带当时的会话状态构造出 AnswerRequest）
    # 之后、`GET .../stream` 取走之前，会话可能被清空——那样一来这个已经构造好的
    # 旧请求（可能带着旧项目的 extra_hit_rowids/project_id）仍会被取流执行，
    # 执行完还会把旧状态重新写回会话，等于让"清空"形同虚设。每次 clear() 时
    # 递增这个计数器，取流前核对建轮时捕获的 epoch 是否仍然等于当前 epoch，
    # 不等就拒绝——比逐个从 `_pending_turns` 里删除更简单可靠，不用担心
    # 清空、建轮两个动作在字典操作上的先后细节。
    epoch: int = 0

    def clear(self) -> None:
        """清空会话上下文：保留 session_id 与 language，其余全部重置。

        scope.md：用户可随时清空会话上下文；清空后不得使用前一项目的材料——
        因此这里必须连 `last_turn`（含上一轮引用的块 rowid）一起清掉，
        否则下一轮追问检索仍会把旧项目的块 id 传回 Orchestrator。`epoch` 递增
        使清空之前建好、尚未取流的旧轮次在取流时被拒绝（CR-021）。
        """
        self.active_project_id = None
        self.active_version = None
        self.last_turn = None
        self.epoch += 1


# ---------------------------------------------------------------------------
# 追问信号判定
# ---------------------------------------------------------------------------

# 设计取舍（T-104 本轮唯一没有被文档定案的判断）：
#
# 优先依赖显式信号——请求体里的 `project_id`（切换/选择项目）与 `new_topic`
# （明确开始新问题）——这两个信号最可靠，判定在 resolve_turn() 里优先处理，
# 根本不会走到这里。这里的启发式只用于"两个显式信号都没给"的情形。
#
# 启发式选择**保守、少而准的标记短语**，不用单独的代词（"它"/"this"/"it"/
# "that"）：中英文里这类词出现在一句完全独立的新问题里的概率并不低
# （"How do I configure it in application.yml" 并不是追问），单独当信号
# 误判率太高。改用更具体、日常追问里才会出现的短语/开头词——
# 命中率会偏低（漏判一些隐晦的追问，被当成独立问题重新检索），
# 但比"逢代词就当追问、把不相关的旧证据强行搭进这一轮"更安全：
# 检索层面前者的代价是"检索范围稍宽、多花一次查询"，后者的代价是
# "答案可能被上一轮无关的项目材料污染"——两害相权取其轻。
#
# CR-026：最初的列表里有"这个"/"那个"，同属这一类过于宽泛的指代词——
# 独立复现"这个 Spring Boot 应用启动为什么失败？""这个 bug 怎么排查？"
# 这类完整独立问题（指代词后面直接跟一个全新的名词短语，不指向历史）都会
# 被误判为追问。这跟"它"/"this"/"it"被排除在外是同一个道理，已移除。
# 英文的"what about"/"how about"结构上是同一种模式（标记词后直接跟全新
# 名词短语也很常见，如"What about caching the response with Redis?"），
# 为一致性一并移除，即使本轮没有对应的具体误判复现。
# **残留风险（如实记录，未验证但推理上同样可能存在）**："这样"/"那样"
# 单独出现在独立新问题开头时同样有一定误判空间（如"这样配置有什么风险"）——
# 目前尚未复现出具体误判案例，暂时保留，后续如出现真实误判应参照同样的
# 思路收紧或移除。
_FOLLOWUP_MARKERS_ZH = (
    "这样", "那样", "上面", "上述", "刚才", "刚刚",
    "该怎么", "继续", "然后呢", "接着", "那要怎么", "那怎么", "还有呢",
    "为什么呢",
)
_FOLLOWUP_MARKERS_EN = (
    "that approach", "this approach", "the above", "previous answer",
    "previous step", "previously", "second one",
    "why is that", "why does that", "and then",
)


def looks_like_followup(question: str) -> bool:
    """轻量启发式：问题文本里是否有常见的指代/省略表达。

    只在 `resolve_turn()` 判定没有显式信号、且会话确有上一轮时才会被调用——
    见该函数的优先级链。
    """
    q = question.strip()
    low = q.lower()
    if any(m in q for m in _FOLLOWUP_MARKERS_ZH):
        return True
    return any(m in low for m in _FOLLOWUP_MARKERS_EN)


# ---------------------------------------------------------------------------
# 范围判定（architecture.md §3 优先级链）
# ---------------------------------------------------------------------------

class TurnKind(str, Enum):
    PROJECT_EXPLICIT = "project_explicit"     # 优先级1：本轮显式选择/切换了项目
    NEW_TOPIC = "new_topic"                   # 优先级2：显式开始新问题
    FOLLOWUP = "followup"                     # 优先级3：追问信号命中
    PROJECT_CONTINUED = "project_continued"   # 无信号，但会话已有活动项目——延续该项目范围
    GENERAL = "general"                       # 优先级4：无信号也无活动项目——通用知识库独立问题


@dataclass
class ResolvedTurn:
    kind: TurnKind
    project_id: str | None
    carry_forward: bool   # 是否复用上一轮证据/实体（只有 FOLLOWUP 为真）


def resolve_turn(
    question: str, body_project_id: str | None, new_topic: bool, session: SessionState,
) -> ResolvedTurn:
    """按 architecture.md §3 的优先级判定这一轮的范围：

    1. 用户显式选择的项目和版本；
    2. 用户明确开始新问题；
    3. 对上一轮实体、结论或方案的追问信号；
    4. 通用知识库中的独立问题。

    第 4 档在这里拆成两种：会话已有活动项目时，没有以上任何信号并不代表
    "离开项目回到通用知识库"——那需要用户显式清空或切换，见 SessionState.clear()——
    因此细分出 PROJECT_CONTINUED；真正的"通用知识库独立问题"只在会话从未
    选定过项目（或已被清空）时才成立。
    """
    if body_project_id is not None:
        return ResolvedTurn(TurnKind.PROJECT_EXPLICIT, body_project_id, carry_forward=False)
    if new_topic:
        return ResolvedTurn(TurnKind.NEW_TOPIC, session.active_project_id, carry_forward=False)
    if session.last_turn is not None and looks_like_followup(question):
        return ResolvedTurn(TurnKind.FOLLOWUP, session.active_project_id, carry_forward=True)
    if session.active_project_id is not None:
        return ResolvedTurn(TurnKind.PROJECT_CONTINUED, session.active_project_id, carry_forward=False)
    return ResolvedTurn(TurnKind.GENERAL, None, carry_forward=False)


# ---------------------------------------------------------------------------
# 追问检索与 prompt 构建
# ---------------------------------------------------------------------------

def _dedup_merge(prior: list[Hit], fresh: list[Hit]) -> list[Hit]:
    seen = {h.rowid for h in prior}
    return prior + [h for h in fresh if h.rowid not in seen]


def _enrich_question(question: str, brief_conclusion: str | None, language: Language) -> str:
    """把上一轮简要结论拼进这一轮问题文本，作为追问的最小上下文。

    不把上一轮的证据正文或完整对话历史塞进去——只有这一句结论。
    它同时服务两个目的：(1) 帮检索的 embedding 理解"这个/那样"具体指什么；
    (2) 作为生成阶段 user message 里唯一的"上一轮上下文"，不重复证据。
    """
    if not brief_conclusion:
        return question
    if language == "en":
        return f"(Continuing from the previous answer: {brief_conclusion}) {question}"
    return f"（承接上一轮结论：{brief_conclusion}）{question}"


def build_followup_request(
    store: ChunkStore, embedder: Embedder, session: SessionState,
    question: str, language: Language, cfg: AnswerConfig,
    body_version: str | None = None,
) -> tuple[AnswerRequest, bool]:
    """追问的检索范围与证据来源。

    先纳入/收紧到上一轮的证据与范围：把上一轮实体（technology/module/symbol）
    当作这一轮检索的过滤条件，并把上一轮引用过的块 rowid 一并带给 Orchestrator
    （见 AnswerRequest.extra_hit_rowids，Orchestrator.answer() 会把它们纳入候选）。

    版本按 architecture.md §3 的优先级：本轮显式传入的 `body_version` 优先于
    上一轮实体里记的版本，再优先于会话当前的活动版本（CR-022：显式版本此前
    在追问分支被完全忽略，即便用户明确要求切到另一个版本也不生效）。

    必要时才放开：这里的预检索只用来判断"收紧后的范围是否大概率给不出足够证据"——
    真正的充分性判定与证据选取仍在 Orchestrator.answer() 里做一次（这里的判定
    只影响传给它的过滤条件，不重复生成流程）。放开限制**只丢弃 technology/
    module/symbol/version**，绝不放开 project_id——切换/清空之外，追问不能突破
    项目边界。
    """
    last = session.last_turn
    assert last is not None
    vector = embedder.encode_one(question)

    effective_version = body_version or last.entities.get("version") or session.active_version

    tight_kwargs: dict[str, str | None] = {
        "technology": last.entities.get("technology"),
        "project_id": session.active_project_id,
        "module": last.entities.get("module"),
        "symbol": last.entities.get("symbol"),
        "version": effective_version,
    }
    prior_hits = hits_by_rowid(store, last.cited_chunk_ids, vector)
    tight_fresh = hybrid_search(
        store, question, vector, limit=cfg.max_evidence * 2, candidates=cfg.candidates, **tight_kwargs,
    )
    # 只用来决定要不要放开限制，不重复 Orchestrator 内部的完整充分性判定逻辑——
    # 直接复用同一个 assess()，标准必须一致，否则"要不要放开"这个决策
    # 会用一套和最终判定不同的尺子，自相矛盾。
    tight_verdict = assess(_dedup_merge(prior_hits, tight_fresh), cfg)
    if tight_verdict.level.value != "insufficient":
        chosen, widened = tight_kwargs, False
    else:
        chosen, widened = {
            "technology": None, "project_id": session.active_project_id,
            "module": None, "symbol": None, "version": None,
        }, True

    enriched = _enrich_question(question, last.brief_conclusion, language)
    req = AnswerRequest(
        question=enriched, language=language,
        extra_hit_rowids=list(last.cited_chunk_ids),
        **chosen,
    )
    return req, widened


def build_request_for_turn(
    store: ChunkStore, embedder: Embedder, session: SessionState, *,
    question: str, project_id: str | None, technology: str | None,
    module: str | None, symbol: str | None, version: str | None,
    new_topic: bool, max_tokens: int | None, cfg: AnswerConfig,
) -> tuple[AnswerRequest, ResolvedTurn, bool]:
    """把这一轮的 HTTP 请求体转成 Orchestrator 能消费的 AnswerRequest。

    范围判定的优先级链在 resolve_turn() 里；这里只根据判定结果分流到
    "追问"（复用上一轮证据/范围）或"非追问"（这一轮的显式过滤条件，
    与上一轮完全无关——避免切换项目/开始新问题时意外带出旧材料）两条路径。
    """
    resolved = resolve_turn(question, project_id, new_topic, session)
    if resolved.kind is TurnKind.FOLLOWUP:
        req, widened = build_followup_request(
            store, embedder, session, question, session.language, cfg, body_version=version,
        )
        if max_tokens is not None:
            req = replace(req, max_tokens=max_tokens)
        return req, resolved, widened

    # 版本优先级（architecture.md §3）：本轮显式传入 > 会话当前活动版本；
    # 显式切换项目（PROJECT_EXPLICIT）且未显式给版本时不回落到旧项目的活动
    # 版本——不同项目的版本编号互不相干，沿用旧版本号只会带出错误的过滤条件
    # （CR-022）。
    if version is not None:
        effective_version = version
    elif resolved.kind is TurnKind.PROJECT_EXPLICIT:
        effective_version = None
    else:
        effective_version = session.active_version

    req = AnswerRequest(
        question=question, language=session.language,
        technology=technology, project_id=resolved.project_id,
        module=module, symbol=symbol, version=effective_version, max_tokens=max_tokens,
    )
    return req, resolved, False


# ---------------------------------------------------------------------------
# 一轮结束后：从已产出的事件派生状态，不额外调用生成模型
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"[。！？.!?]")


def brief_conclusion(answer_text: str, limit: int = 80) -> str | None:
    """从答案正文派生"简要结论"：首句截断，不额外调用模型。

    拒答（NO_EVIDENCE）时没有结论可留：下一轮追问不该拿"未作答"当上下文。
    """
    text = answer_text.strip()
    if not text or declined(text):
        return None
    m = _SENTENCE_END.search(text)
    sentence = text[: m.end()] if m else text
    if len(sentence) > limit:
        sentence = sentence[:limit].rstrip() + "…"
    return sentence


def stream_and_record(
    orchestrator: Orchestrator, req: AnswerRequest, session: SessionState,
    resolved: ResolvedTurn, raw_question: str,
) -> Iterator[dict]:
    """转发 Orchestrator.answer() 的事件流给客户端；流结束后用这一轮已经
    产出的信息（技术域/项目/模块/符号/命中的关键词、答案正文、实际引用到的
    块 rowid）更新会话状态——不为此另起一次模型推理。

    出错时不更新会话状态：保留上一轮仍然有效的上下文，好过用一次失败的
    请求把它冲掉。
    """
    answer_parts: list[str] = []
    cited_indices: list[int] = []
    sources: list[dict] = []
    verdict_level: str | None = None
    had_error = False

    for ev in orchestrator.answer(req):
        et = ev.get("type")
        if et == "retrieval":
            verdict_level = ev.get("sufficiency")
        elif et == "answer_delta":
            answer_parts.append(ev["text"])
        elif et == "done":
            cited_indices = ev.get("cited_evidence") or []
        elif et == "sources":
            sources = ev.get("items") or []
        elif et == "error":
            had_error = True
        yield ev

    if had_error:
        return

    full_answer = "".join(answer_parts)
    cited_chunk_ids = [
        s["chunk_id"] for s in sources
        if s.get("index") in cited_indices and s.get("chunk_id") is not None
    ]
    entities: dict[str, object] = {
        # Orchestrator 内部在 req.technology 为空时会自行探测（见 answering.py
        # 的 `tech = req.technology or detect_technology(req.question)`）；
        # 这里同样兜底探测一次，让下一轮追问收紧检索时能用上这个技术域，
        # 而不是因为这一轮没显式传参就白白丢掉它。
        "technology": req.technology or detect_technology(raw_question),
        "project_id": resolved.project_id,
        "module": req.module, "symbol": req.symbol, "version": req.version,
        # 复用 T-025 已有的中文技术概念映射，不新起一套关键词抽取——
        # 见 services/retrieval/tokenize.matched_terms。只留前几个，
        # 会话状态不是完整关键词索引。
        "keywords": matched_terms(raw_question)[:5],
    }
    open_issue = verdict_level if verdict_level != "sufficient" else None

    session.active_project_id = resolved.project_id
    # req.version 已经在 build_request_for_turn()/build_followup_request() 里按
    # 正确优先级解析过（显式 > 会话活动版本，切项目且未显式给版本时为 None）——
    # 这里无条件写回，让"切换项目未带版本"能正确清空旧版本（CR-022），而不是
    # 像旧代码那样只在 truthy 时才更新、导致旧版本号残留。
    session.active_version = req.version
    session.last_turn = TurnContext(
        question=raw_question, entities=entities,
        brief_conclusion=brief_conclusion(full_answer),
        cited_chunk_ids=cited_chunk_ids, open_issue=open_issue,
    )
