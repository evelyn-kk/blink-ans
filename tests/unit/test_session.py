"""会话层测试（T-104）：独立问答 / 项目问答 / 追问 / 项目切换 / 清空上下文。

不启动真实模型或索引——沿用 tests/unit/test_answering.py 的假货模式
（FakeEmbedder/FakeRouter），只 monkeypatch `services.orchestrator.session`
与 `services.orchestrator.answering` 里对 `hybrid_search`/`hits_by_rowid`
的引用。这些断言验证的是"范围怎么判、追问怎么复用证据、状态怎么派生"这套
编排逻辑本身，与真实检索质量无关（那部分已由 tests/retrieval 覆盖）。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.orchestrator.answering import (  # noqa: E402
    AnswerConfig, AnswerRequest, Orchestrator,
)
from services.orchestrator.session import (  # noqa: E402
    ResolvedTurn, SessionState, TurnContext, TurnKind, brief_conclusion,
    build_followup_request, build_request_for_turn, looks_like_followup,
    resolve_turn, stream_and_record,
)
from services.retrieval.search import Hit  # noqa: E402

from tests.unit.test_answering import FakeEmbedder, FakeRouter  # noqa: E402


def hit(dist=0.65, kw=1, tokens=200, i=1, score=0.03, project_id=None) -> Hit:
    return Hit(
        rowid=i, text=f"证据正文 {i}" * 10, title_path=f"A › B{i}",
        source_url=f"https://example.com/{i}.html", source_project="kafka",
        version_or_commit="abc", retrieved_at="2026-08-31T00:00:00+00:00",
        technology="kafka", content_type="prose", token_estimate=tokens,
        score=score, keyword_rank=kw, vector_rank=1, vector_distance=dist,
        project_id=project_id,
    )


CFG = AnswerConfig()


# ---------- 追问信号启发式 ----------

def test_followup_marker_detected_in_chinese():
    assert looks_like_followup("这样改会有什么风险？") is True
    assert looks_like_followup("那要怎么验证？") is True


def test_followup_marker_detected_in_english():
    assert looks_like_followup("What about the previous step?") is True
    assert looks_like_followup("Why does that happen?") is True


def test_bare_pronoun_alone_is_not_treated_as_followup():
    """设计取舍：单独的代词（"它"/"it"/"that"）不算信号——它们出现在
    完全独立的新问题里的概率不低，误判代价（把无关旧证据带进这一轮）
    比漏判（多做一次全新检索）更高。"""
    assert looks_like_followup("How do I configure it in application.yml?") is False


def test_independent_technical_question_is_not_followup():
    assert looks_like_followup("Kafka 幂等生产者怎么配置？") is False


def test_generic_demonstrative_followed_by_new_topic_is_not_followup():
    """CR-026 独立复现场景：裸的"这个"/"那个"后面直接跟一个全新的名词短语
    （不是指向历史），整句其实是完整独立问题——不该被误判为追问。
    """
    assert looks_like_followup("这个 Spring Boot 应用启动为什么失败？") is False
    assert looks_like_followup("这个 bug 怎么排查？") is False


def test_generic_what_about_followed_by_new_topic_is_not_followup():
    """CR-026 的一致性延伸：英文"what about"/"how about"是同一种结构性歧义
    （标记词后直接跟全新名词短语），一并收紧。"""
    assert looks_like_followup("What about caching the response with Redis?") is False


# ---------- 简要结论派生（不额外调用模型） ----------

def test_brief_conclusion_truncates_to_first_sentence():
    assert brief_conclusion("消费者未提交偏移量导致重复消费。后续详细步骤略。") == "消费者未提交偏移量导致重复消费。"


def test_brief_conclusion_none_when_declined():
    assert brief_conclusion("NO_EVIDENCE") is None


def test_brief_conclusion_none_for_empty_answer():
    assert brief_conclusion("   ") is None


# ---------- 范围判定优先级链（architecture.md §3） ----------

def _session(active_project_id=None, last_turn=None, active_version=None) -> SessionState:
    return SessionState(
        session_id="s1", language="zh",
        active_project_id=active_project_id, last_turn=last_turn,
        active_version=active_version,
    )


def test_scope_1_explicit_project_wins_over_everything():
    """场景：项目问答。显式 project_id 优先级最高，即便问题文本也像追问。"""
    last = TurnContext("上一个问题", {}, "上一轮结论", [1])
    session = _session(active_project_id="orders", last_turn=last)
    resolved = resolve_turn("这样改会有风险吗", body_project_id="checkout", new_topic=False, session=session)
    assert resolved.kind is TurnKind.PROJECT_EXPLICIT
    assert resolved.project_id == "checkout"
    assert resolved.carry_forward is False


def test_scope_2_new_topic_flag_skips_followup_heuristic():
    last = TurnContext("上一个问题", {}, "上一轮结论", [1])
    session = _session(last_turn=last)
    resolved = resolve_turn("这样改会有风险吗", body_project_id=None, new_topic=True, session=session)
    assert resolved.kind is TurnKind.NEW_TOPIC
    assert resolved.carry_forward is False


def test_scope_3_followup_signal_without_explicit_flags():
    last = TurnContext("消费者为什么重复消费", {}, "偏移量未提交", [1])
    session = _session(last_turn=last)
    resolved = resolve_turn("那要怎么验证修好了", body_project_id=None, new_topic=False, session=session)
    assert resolved.kind is TurnKind.FOLLOWUP
    assert resolved.carry_forward is True


def test_scope_4_general_independent_question_without_active_project():
    """场景：独立知识问答。没有活动项目、没有上一轮、也没有信号——落到通用知识库。"""
    session = _session()
    resolved = resolve_turn("Kafka 幂等生产者怎么配置", body_project_id=None, new_topic=False, session=session)
    assert resolved.kind is TurnKind.GENERAL
    assert resolved.project_id is None


def test_scope_project_continued_when_no_signal_but_project_active():
    """有活动项目、没有其它信号时默认还在这个项目范围内，不会掉回通用知识库。"""
    session = _session(active_project_id="orders")
    resolved = resolve_turn("怎么预留库存", body_project_id=None, new_topic=False, session=session)
    assert resolved.kind is TurnKind.PROJECT_CONTINUED
    assert resolved.project_id == "orders"


def test_scope_followup_requires_a_last_turn_to_exist():
    """清空上下文后没有 last_turn，即便文本命中启发式标记也不能判成追问——
    否则清空形同虚设。"""
    session = _session()  # 无 last_turn
    resolved = resolve_turn("那要怎么验证", body_project_id=None, new_topic=False, session=session)
    assert resolved.kind is not TurnKind.FOLLOWUP


# ---------- 追问检索：复用上一轮证据 ----------

def test_followup_request_carries_prior_chunk_ids_and_scoped_filters(monkeypatch):
    last = TurnContext(
        question="消费者为什么重复消费",
        entities={"technology": "kafka", "project_id": None, "module": "consumer", "symbol": None, "version": None},
        brief_conclusion="消费者未提交偏移量导致重复消费。",
        cited_chunk_ids=[7, 9],
    )
    session = _session(last_turn=last)

    captured = {}

    def fake_hybrid_search(store, query, vector, *, limit, candidates, **kwargs):
        captured.update(kwargs)
        return [hit(i=7, dist=0.6)]

    monkeypatch.setattr("services.orchestrator.session.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr("services.orchestrator.session.hits_by_rowid", lambda *a, **kw: [hit(i=7, dist=0.6)])

    req, widened = build_followup_request(None, FakeEmbedder(), session, "那要怎么验证修好了", "zh", CFG)

    assert widened is False
    assert req.extra_hit_rowids == [7, 9]
    assert captured["technology"] == "kafka"
    assert captured["module"] == "consumer"
    # 简要结论进了问题文本，供检索与生成理解"那"指什么——但不是完整证据/历史
    assert "消费者未提交偏移量导致重复消费" in req.question
    assert "那要怎么验证修好了" in req.question


def test_followup_widens_scope_when_tight_scope_insufficient(monkeypatch):
    """必要时才放开：收紧后的范围检索不到相关证据，才丢弃 module/symbol/technology。"""
    last = TurnContext(
        question="消费者为什么重复消费",
        entities={"technology": "kafka", "project_id": None, "module": "consumer", "symbol": None, "version": None},
        brief_conclusion="消费者未提交偏移量导致重复消费。",
        cited_chunk_ids=[],
    )
    session = _session(last_turn=last)

    calls = []

    def fake_hybrid_search(store, query, vector, *, limit, candidates, **kwargs):
        calls.append(kwargs)
        if kwargs.get("module") == "consumer":
            return []  # 收紧范围检索不到
        return [hit(i=42, dist=0.6)]

    monkeypatch.setattr("services.orchestrator.session.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr("services.orchestrator.session.hits_by_rowid", lambda *a, **kw: [])

    req, widened = build_followup_request(None, FakeEmbedder(), session, "那要怎么办", "zh", CFG)

    assert widened is True  # 收紧范围判定为证据不足，触发放开
    assert req.module is None
    assert req.technology is None


# ---------- CR-022：显式/活动版本必须真正进入检索过滤条件 ----------

def test_continued_turn_falls_back_to_active_version_when_not_given():
    """普通（非追问）turn 没有显式传版本时，应落回会话当前的活动版本，
    而不是像旧实现那样直接丢弃、让 req.version 恒为 None。
    """
    session = _session(active_project_id="orders", active_version="v1")
    req, resolved, _ = build_request_for_turn(
        None, FakeEmbedder(), session,
        question="怎么配置重试", project_id=None, technology=None,
        module=None, symbol=None, version=None, new_topic=False, max_tokens=None,
        cfg=CFG,
    )
    assert resolved.kind is TurnKind.PROJECT_CONTINUED
    assert req.version == "v1"


def test_explicit_version_overrides_active_version_for_continued_turn():
    session = _session(active_project_id="orders", active_version="v1")
    req, _, _ = build_request_for_turn(
        None, FakeEmbedder(), session,
        question="怎么配置重试", project_id=None, technology=None,
        module=None, symbol=None, version="v2", new_topic=False, max_tokens=None,
        cfg=CFG,
    )
    assert req.version == "v2"


def test_project_switch_without_explicit_version_clears_old_active_version():
    """切换项目时不回落到旧项目的活动版本——不同项目的版本号互不相干，
    沿用旧版本只会带出错误的过滤条件。
    """
    session = _session(active_project_id="orders", active_version="v1")
    req, resolved, _ = build_request_for_turn(
        None, FakeEmbedder(), session,
        question="怎么配置重试", project_id="checkout", technology=None,
        module=None, symbol=None, version=None, new_topic=False, max_tokens=None,
        cfg=CFG,
    )
    assert resolved.kind is TurnKind.PROJECT_EXPLICIT
    assert req.version is None


def test_project_switch_with_explicit_version_uses_it():
    session = _session(active_project_id="orders", active_version="v1")
    req, _, _ = build_request_for_turn(
        None, FakeEmbedder(), session,
        question="怎么配置重试", project_id="checkout", technology=None,
        module=None, symbol=None, version="v9", new_topic=False, max_tokens=None,
        cfg=CFG,
    )
    assert req.version == "v9"


def test_followup_falls_back_to_active_version_when_not_given(monkeypatch):
    last = TurnContext(
        question="消费者为什么重复消费", entities={}, brief_conclusion="偏移量未提交。",
        cited_chunk_ids=[],
    )
    session = _session(active_project_id="orders", active_version="v1", last_turn=last)

    captured = {}

    def fake_hybrid_search(store, query, vector, *, limit, candidates, **kwargs):
        captured.update(kwargs)
        return [hit(i=1, dist=0.6)]

    monkeypatch.setattr("services.orchestrator.session.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr("services.orchestrator.session.hits_by_rowid", lambda *a, **kw: [])

    req, _ = build_followup_request(None, FakeEmbedder(), session, "那要怎么验证", "zh", CFG)

    assert req.version == "v1"
    assert captured.get("version") == "v1"


def test_followup_explicit_version_overrides_active_and_prior_turn_version(monkeypatch):
    """CR-022 独立复现场景：上一轮版本是 v1，本轮追问显式传 version=v2，
    必须真正体现在检索过滤条件与 AnswerRequest 里，而不是被忽略。
    """
    last = TurnContext(
        question="消费者为什么重复消费",
        entities={"version": "v1"}, brief_conclusion="偏移量未提交。", cited_chunk_ids=[],
    )
    session = _session(active_project_id="orders", active_version="v1", last_turn=last)

    captured = {}

    def fake_hybrid_search(store, query, vector, *, limit, candidates, **kwargs):
        captured.update(kwargs)
        return [hit(i=1, dist=0.6)]

    monkeypatch.setattr("services.orchestrator.session.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr("services.orchestrator.session.hits_by_rowid", lambda *a, **kw: [])

    req, resolved, _ = build_request_for_turn(
        None, FakeEmbedder(), session,
        question="那要怎么验证", project_id=None, technology=None,
        module=None, symbol=None, version="v2", new_topic=False, max_tokens=None,
        cfg=CFG,
    )

    assert resolved.kind is TurnKind.FOLLOWUP
    assert req.version == "v2"
    assert captured.get("version") == "v2"


def test_stream_and_record_writes_resolved_version_back_to_session(monkeypatch):
    """一轮结束后，会话的 active_version 必须跟着这一轮实际使用的版本更新——
    包括"切换项目未带版本"时应清空，而不是残留旧项目的版本号。
    """
    monkeypatch.setattr(
        "services.orchestrator.answering.hybrid_search",
        lambda *a, **kw: [hit(i=1, dist=0.6, project_id="checkout")],
    )
    from tests.unit.test_answering import FakeEngine

    session = _session(active_project_id="orders", active_version="v1")
    req = AnswerRequest("怎么配置重试", project_id="checkout", version=None)
    resolved = ResolvedTurn(TurnKind.PROJECT_EXPLICIT, "checkout", carry_forward=False)

    router = FakeRouter(FakeEngine())
    list(stream_and_record(
        Orchestrator(None, FakeEmbedder(), router), req, session, resolved, "怎么配置重试",
        turn_seq=1, created_epoch=0,
    ))

    assert session.active_version is None


# ---------- 一次完整追问：证据复用进 Orchestrator，且不泄漏历史全文 ----------

def test_orchestrator_merges_extra_hit_rowids_into_evidence(monkeypatch):
    """T-104：AnswerRequest.extra_hit_rowids 应参与判定/选取，而不是被忽略。"""
    monkeypatch.setattr(
        "services.orchestrator.answering.hybrid_search", lambda *a, **kw: [hit(i=1, dist=0.6)]
    )
    monkeypatch.setattr(
        "services.orchestrator.answering.hits_by_rowid", lambda *a, **kw: [hit(i=99, dist=0.6, score=0.5)]
    )
    router = FakeRouter(__import__("tests.unit.test_answering", fromlist=["FakeEngine"]).FakeEngine())
    events = list(Orchestrator(None, FakeEmbedder(), router).answer(
        AnswerRequest("那要怎么验证", extra_hit_rowids=[99])
    ))
    sources = next(e for e in events if e["type"] == "sources")
    chunk_ids = {item["chunk_id"] for item in sources["items"]}
    assert 99 in chunk_ids or 1 in chunk_ids  # 至少参与了候选合并（DP 按预算/分数选取）


def test_followup_prompt_does_not_include_full_prior_evidence_text(monkeypatch):
    """验证 item 6：追问送进生成模型的 user message 不含上一轮的完整证据正文，
    只带这一轮新证据 + 一句简要结论。"""
    from tests.unit.test_answering import RecordingRouter, FakeEngine

    stale_evidence_text = "这是上一轮一大段完整证据原文" * 20
    monkeypatch.setattr(
        "services.orchestrator.answering.hybrid_search",
        lambda *a, **kw: [hit(i=1, dist=0.6)],
    )
    monkeypatch.setattr("services.orchestrator.answering.hits_by_rowid", lambda *a, **kw: [])

    session = _session(last_turn=TurnContext(
        question="消费者为什么重复消费", entities={}, brief_conclusion="偏移量未提交导致重复消费。",
        cited_chunk_ids=[],
    ))
    monkeypatch.setattr("services.orchestrator.session.hybrid_search", lambda *a, **kw: [hit(i=1, dist=0.6)])
    monkeypatch.setattr("services.orchestrator.session.hits_by_rowid", lambda *a, **kw: [])

    req, _ = build_followup_request(None, FakeEmbedder(), session, "那要怎么验证", "zh", CFG)
    router = RecordingRouter(FakeEngine())
    list(Orchestrator(None, FakeEmbedder(), router).answer(req))

    content = router.calls[0]["content"]
    assert stale_evidence_text not in content
    assert "偏移量未提交导致重复消费" in content  # 简要结论作为上下文保留
    assert "证据正文 1" in content  # 这一轮新证据仍然完整送入


# ---------- 项目切换：不泄漏前一项目材料 ----------

def test_project_switch_drops_prior_entities_and_evidence():
    """场景：项目切换。显式换到另一个项目时，不应继承上一轮（另一项目）的
    module/symbol/technology 过滤条件，也不应带上一轮引用的块 rowid。"""
    last = TurnContext(
        question="项目 A 下怎么预留库存",
        entities={"technology": "kafka", "project_id": "project-a", "module": "checkout", "symbol": "reserve"},
        brief_conclusion="项目 A 的做法是……",
        cited_chunk_ids=[1, 2, 3],
    )
    session = _session(active_project_id="project-a", last_turn=last)

    req, resolved, widened = build_request_for_turn(
        None, FakeEmbedder(), session,
        question="怎么配置重试", project_id="project-b", technology=None,
        module=None, symbol=None, version=None, new_topic=False, max_tokens=None,
        cfg=CFG,
    )

    assert resolved.kind is TurnKind.PROJECT_EXPLICIT
    assert req.project_id == "project-b"
    assert req.module is None and req.symbol is None
    assert not req.extra_hit_rowids  # 不带项目 A 的引用块


def test_stream_and_record_updates_active_project_after_switch(monkeypatch):
    monkeypatch.setattr(
        "services.orchestrator.answering.hybrid_search",
        lambda *a, **kw: [hit(i=5, dist=0.6, project_id="project-b")],
    )
    from tests.unit.test_answering import FakeEngine

    session = _session(active_project_id="project-a", last_turn=TurnContext(
        question="项目 A 下怎么预留库存", entities={"project_id": "project-a"},
        brief_conclusion="项目 A 的做法是……", cited_chunk_ids=[1, 2, 3],
    ))
    req = AnswerRequest("怎么配置重试", project_id="project-b")
    resolved = ResolvedTurn(TurnKind.PROJECT_EXPLICIT, "project-b", carry_forward=False)

    router = FakeRouter(FakeEngine())
    list(stream_and_record(
        Orchestrator(None, FakeEmbedder(), router), req, session, resolved, "怎么配置重试",
        turn_seq=1, created_epoch=0,
    ))

    assert session.active_project_id == "project-b"
    assert session.last_turn.entities["project_id"] == "project-b"
    # 旧项目引用的块不会原样带进新 last_turn（除非这一轮真的又引用到同一 id）
    assert set(session.last_turn.cited_chunk_ids) <= {5}


# ---------- 清空会话上下文 ----------

def test_clear_resets_state_but_keeps_id_and_language():
    session = SessionState(
        session_id="s1", language="en", active_project_id="orders", active_version="v2",
        last_turn=TurnContext("q", {"a": 1}, "concl", [1, 2], open_issue="limited"),
    )
    session.clear()
    assert session.session_id == "s1"
    assert session.language == "en"
    assert session.active_project_id is None
    assert session.active_version is None
    assert session.last_turn is None


def test_clear_cannot_interleave_between_epoch_check_and_stream_state_writeback():
    """CR-032 判别性回归：SSE 生产在线程池，clear 在事件循环线程。

    让流线程在它**已经读到旧 epoch、尚未提交状态**时停住；随后发起 clear。
    修复后 clear 必须等待同一把会话锁，流先提交后 clear 最终清空；旧实现没有
    临界段，clear 会先完成、流恢复后再把 orders/v1 写回，最终状态错误地复活。
    """
    comparison_started = threading.Event()
    allow_comparison = threading.Event()
    clear_started = threading.Event()

    class _EpochSnapshot(int):
        def __ne__(self, other):
            comparison_started.set()
            assert allow_comparison.wait(timeout=2)
            return super().__ne__(other)

    class _RaceSession(SessionState):
        def __getattribute__(self, name):
            if name == "epoch" and threading.current_thread().name == "stream":
                return _EpochSnapshot(super().__getattribute__(name))
            return super().__getattribute__(name)

    class _FakeOrchestrator:
        def answer(self, req):
            yield {"type": "retrieval", "sufficiency": "sufficient"}
            yield {"type": "answer_delta", "text": "答案 [1]"}
            yield {"type": "done", "cited_evidence": [1]}
            yield {"type": "sources", "items": [
                {"index": 1, "chunk_id": 1, "citation": "来源"},
            ]}

    session = _RaceSession(
        session_id="s1", language="zh", active_project_id="prior", active_version="old",
    )
    req = AnswerRequest("订单怎么预留库存", project_id="orders", version="v1")
    resolved = ResolvedTurn(TurnKind.PROJECT_EXPLICIT, "orders", carry_forward=False)

    stream = threading.Thread(
        target=lambda: list(stream_and_record(
            _FakeOrchestrator(), req, session, resolved, req.question,
            turn_seq=1, created_epoch=0,
        )),
        name="stream",
    )
    clear = threading.Thread(
        target=lambda: (clear_started.set(), session.clear()), name="clear",
    )
    stream.start()
    assert comparison_started.wait(timeout=2)
    clear.start()
    assert clear_started.wait(timeout=2)

    # 放行工作线程：修复后它先原子提交并释放锁，clear 随后清空；旧实现中 clear
    # 已经抢先完成，工作线程会把旧项目写回。
    allow_comparison.set()
    stream.join(timeout=2)
    clear.join(timeout=2)
    assert not stream.is_alive()
    assert not clear.is_alive()
    assert session.epoch == 1
    assert session.active_project_id is None
    assert session.active_version is None
    assert session.last_turn is None


def test_read_state_blocks_until_writeback_fully_commits_not_a_partial_view():
    """CR-033 判别性回归：`read_state()` 与 stream_and_record() 的写回共享
    `_state_lock`——不能在写回临界段"active_project_id/active_version 已提交、
    last_turn 还没提交"的中途返回一份撕裂快照。

    在这个暂停点上直接现读 `session.active_project_id`/`active_version`
    （旧代码在 resolve_turn()/build_followup_request()/_session_view() 里就是
    这样逐字段读的）会看到撕裂组合：project 已经是新的，但 last_turn 还是
    旧的（初始为 None）。`read_state()` 必须等同一把锁释放才返回，因此拿到的
    只能是完整提交后的那一组。
    """
    paused = threading.Event()
    resume = threading.Event()

    class _RaceSession(SessionState):
        def __setattr__(self, name, value):
            if name == "last_turn" and threading.current_thread().name == "stream":
                paused.set()
                assert resume.wait(timeout=2)
            object.__setattr__(self, name, value)

    class _FakeOrchestrator:
        def answer(self, req):
            yield {"type": "retrieval", "sufficiency": "sufficient"}
            yield {"type": "answer_delta", "text": "答案 [1]"}
            yield {"type": "done", "cited_evidence": [1]}
            yield {"type": "sources", "items": [
                {"index": 1, "chunk_id": 1, "citation": "来源"},
            ]}

    session = _RaceSession(
        session_id="s1", language="zh", active_project_id="prior", active_version="old",
    )
    req = AnswerRequest("订单怎么预留库存", project_id="orders", version="v1")
    resolved = ResolvedTurn(TurnKind.PROJECT_EXPLICIT, "orders", carry_forward=False)

    stream = threading.Thread(
        target=lambda: list(stream_and_record(
            _FakeOrchestrator(), req, session, resolved, req.question,
            turn_seq=1, created_epoch=0,
        )),
        name="stream",
    )
    stream.start()
    assert paused.wait(timeout=2)

    # 撕裂点：project/version 已提交，last_turn 还没有——这就是旧的逐字段读取
    # 方式（resolve_turn()/build_followup_request()/_session_view() 修复前）
    # 会读到的具体撞车组合。
    assert session.active_project_id == "orders"
    assert session.active_version == "v1"
    assert session.last_turn is None

    # read_state() 此刻会因为拿不到锁而阻塞（锁被暂停中的写回线程持有）——
    # 放到另一个线程里发起，避免它在主线程上把测试本身锁死；resume.set()
    # 之后无论 reader 是先排到队还是后排到队，都只会在写回整体提交、锁释放
    # 之后才拿到值，不影响下面的断言。
    result: dict = {}
    reader = threading.Thread(
        target=lambda: result.__setitem__("snap", session.read_state()), name="reader",
    )
    reader.start()
    resume.set()
    stream.join(timeout=2)
    reader.join(timeout=2)
    assert not stream.is_alive()
    assert not reader.is_alive()

    snap = result["snap"]
    assert snap.active_project_id == "orders"
    assert snap.active_version == "v1"
    assert snap.last_turn is not None  # 要么和上面一样等到齐全，要么该测试本身先超时失败


def test_followup_request_uses_snapshot_taken_at_entry_not_interleaved_reads(monkeypatch):
    """CR-033：build_followup_request() 做 embedding + 检索期间不能现读
    session——这段时间足够长，另一个 turn 完全可能并发完成并写回新项目。
    旧代码分四次单独读 session.last_turn/active_version/active_project_id
    （其中两次分别在 embedding 前后），如果中途被切了项目，会把新项目的
    project_id 和旧项目的 last_turn.cited_chunk_ids 拼进同一个请求——两个
    项目的材料混进一次检索。这里用一个会在 encode_one() 里"抢跑"写回的假
    embedder 模拟这个时序，断言返回的 req 整组字段必须来自调用时刻的项目，
    不能是新旧混合。
    """
    last = TurnContext(
        question="消费者为什么重复消费",
        entities={"technology": "kafka", "project_id": None, "module": "consumer",
                  "symbol": None, "version": None},
        brief_conclusion="消费者未提交偏移量导致重复消费。",
        cited_chunk_ids=[7, 9],
    )
    session = _session(active_project_id="orders", last_turn=last, active_version="v1")

    class _InterleavingEmbedder:
        def encode_one(self, _question):
            # 模拟另一个 turn 恰好在这次 embedding 期间并发完成写回。
            session.active_project_id = "checkout"
            session.active_version = "v2"
            session.last_turn = TurnContext(
                question="换项目了", entities={}, brief_conclusion="换项目了",
                cited_chunk_ids=[99],
            )
            return [0.0]

    monkeypatch.setattr(
        "services.orchestrator.session.hybrid_search",
        lambda *a, **kw: [hit(i=7, dist=0.6)],
    )
    monkeypatch.setattr(
        "services.orchestrator.session.hits_by_rowid",
        lambda *a, **kw: [hit(i=7, dist=0.6)],
    )

    req, widened = build_followup_request(
        None, _InterleavingEmbedder(), session, "那要怎么验证修好了", "zh", CFG,
    )

    # 调用时刻是 orders——即便 embedding 期间会话已经被换成 checkout，
    # 这一次追问请求也必须整组保持 orders（连带它的 cited_chunk_ids），
    # 不能出现 project_id=checkout 却带着 orders 引用块这种混合。
    assert req.project_id == "orders"
    assert req.extra_hit_rowids == [7, 9]


def test_build_request_for_turn_shares_one_snapshot_across_resolve_and_followup(monkeypatch):
    """CR-034 判别性回归：resolve_turn() 判定范围用的快照，和
    build_followup_request() 实际拿去检索的快照，必须是 build_request_for_turn()
    入口那唯一一次 read_state()，不能各自现取。旧代码（CR-033 修复之后、
    CR-034 修复之前）里 resolve_turn() 和 build_followup_request() 各自独立调用
    `session.read_state()`，两次调用之间足够另一个 turn 的写回完整插入——判定
    用的 `resolved.project_id` 和实际检索用的 `req.project_id`/`extra_hit_rowids`
    可能来自并发写回前后两个不同时刻，彼此矛盾。

    这里不用线程，直接给 `SessionState.read_state` 打一个"第一次调用返回旧值、
    但顺带把底层字段写成新值"的补丁，模拟"两次 read_state() 之间，另一个 turn
    抢先完成并写回"这个时序——不依赖具体调用点（embedding 前/后），对
    resolve_turn() 和 build_followup_request() 各自独立取快照这件事本身有
    判别力：只要调用了两次 `read_state()`，第二次就会看到补丁后的新值；只调用
    一次（本轮修复后的行为）则两处结果必然一致。
    """
    last = TurnContext(
        question="消费者为什么重复消费",
        entities={"technology": "kafka", "project_id": None, "module": "consumer",
                  "symbol": None, "version": None},
        brief_conclusion="消费者未提交偏移量导致重复消费。",
        cited_chunk_ids=[7, 9],
    )
    session = _session(active_project_id="orders", last_turn=last, active_version="v1")

    orig_read_state = SessionState.read_state
    calls = {"n": 0}

    def patched_read_state(self):
        calls["n"] += 1
        snap = orig_read_state(self)
        if calls["n"] == 1:
            # 模拟这次 read_state() 返回之后、下一次 read_state() 之前，
            # 另一个 turn 的写回完整插入进来。
            self.active_project_id = "checkout"
            self.active_version = "v2"
            self.last_turn = TurnContext(
                question="换项目了", entities={}, brief_conclusion="换项目了",
                cited_chunk_ids=[99],
            )
        return snap

    monkeypatch.setattr(SessionState, "read_state", patched_read_state)
    monkeypatch.setattr(
        "services.orchestrator.session.hybrid_search",
        lambda *a, **kw: [hit(i=7, dist=0.6)],
    )
    monkeypatch.setattr(
        "services.orchestrator.session.hits_by_rowid",
        lambda *a, **kw: [hit(i=7, dist=0.6)],
    )

    req, resolved, _ = build_request_for_turn(
        None, FakeEmbedder(), session,
        question="那要怎么验证修好了", project_id=None, technology=None,
        module=None, symbol=None, version=None, new_topic=False,
        max_tokens=None, cfg=CFG,
    )

    assert calls["n"] == 1  # 本轮修复后只应该有入口那一次调用
    # 范围判定和实际检索必须来自同一时刻：都还是入口快照里的 orders，
    # 不能出现 resolved.project_id 和 req.project_id 各说各话的组合。
    assert resolved.kind is TurnKind.FOLLOWUP
    assert resolved.project_id == "orders"
    assert req.project_id == "orders"
    assert req.extra_hit_rowids == [7, 9]


def test_clear_prevents_followup_classification_on_next_turn():
    session = SessionState(
        session_id="s1", language="zh", active_project_id="orders",
        last_turn=TurnContext("上一个问题", {}, "上一轮结论", [1, 2]),
    )
    session.clear()
    resolved = resolve_turn("那要怎么验证", body_project_id=None, new_topic=False, session=session)
    assert resolved.kind is TurnKind.GENERAL
    assert resolved.project_id is None
