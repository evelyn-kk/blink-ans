"""路由级回归：会话 API 的建轮/清空/取流交互（CR-021、CR-030、CR-031）。

不启动真实模型或索引——直接调用 `apps.gateway.main` 里的路由处理函数
（它们只是普通的 async def，不经过 `TestClient` 也能直接调用），并用
monkeypatch 替换掉模块级的 `_sessions`/`_pending_turns`/`_orchestrator`/`_store`。
这样能验证真实的路由控制流（建轮时捕获 epoch/turn_seq、取流时核对二者、
清空后拒绝旧轮次、乱序完成不回滚会话状态、SSE 惰性消费期间清空不被绕过），
而不需要触发 `lifespan()` 里的真实 MLX 模型加载。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import apps.gateway.main as gw  # noqa: E402
from services.orchestrator.answering import AnswerConfig  # noqa: E402
from services.orchestrator.session import SessionState  # noqa: E402


class FakeOrchestrator:
    """`stream_and_record()` 只调用 `orchestrator.answer(req)`，
    这里给一个能跑完一轮完整成功流程的假货。
    """

    def __init__(self) -> None:
        self.cfg = AnswerConfig()

    def answer(self, req):
        yield {"type": "retrieval", "hits": 1, "sufficiency": "sufficient",
               "top_distance": 0.5, "keyword_hits": 1, "reason": "ok", "elapsed_ms": 1.0}
        yield {"type": "status", "message": "已选取 1 条证据", "evidence_tokens": 10}
        yield {"type": "answer_delta", "text": "答案 [1]"}
        yield {"type": "done", "served_by": "local", "sufficiency": "sufficient",
               "template_version": "x", "ttft_s": 0.1, "ttft_over_budget": False,
               "prompt_tokens": 10, "prefilled_tokens": 10, "prefix_reused": True,
               "decode_tps": 20.0, "cited_evidence": [1], "evidence_count": 1, "total_s": 0.2}
        yield {"type": "sources", "items": [
            {"index": 1, "citation": "来源", "url": "https://example.com", "chunk_id": 1},
        ]}


def _run(coro):
    return asyncio.run(coro)


async def _drain(response):
    """`StreamingResponse` 不会在构造时执行它包的异步生成器——`sse_stream()`
    的 body 要等真的被 ASGI 服务器迭代消费时才会跑。测试里直接调用
    `stream_turn()` 拿到的 response 对象本身不会触发 `stream_and_record()`
    的状态写回逻辑，必须显式把 `body_iterator` 迭代完，才算真正"跑完这次流"。
    """
    async for _ in response.body_iterator:
        pass
    return response


@pytest.fixture(autouse=True)
def _isolated_gateway_state(monkeypatch):
    """每个测试用独立的会话/待取轮次字典和假编排器，不污染真实模块状态。"""
    monkeypatch.setattr(gw, "_sessions", {})
    monkeypatch.setattr(gw, "_pending_turns", {})
    monkeypatch.setattr(gw, "_orchestrator", FakeOrchestrator())
    monkeypatch.setattr(gw, "_store", object())  # create_turn 只判断 is not None
    yield


def test_create_turn_then_clear_then_stream_rejects_stale_turn():
    """CR-021 独立复现场景：建轮 → 清空会话 → 取流，必须拒绝这个已失效的轮次，
    而不是执行它并把清空前的旧状态重新写回会话。
    """
    session_view = _run(gw.create_session(gw.SessionCreateBody(
        language="zh", project_id="orders", version=None,
    )))
    sid = session_view["session_id"]

    turn = _run(gw.create_turn(sid, gw.TurnBody(question="怎么预留库存")))
    tid = turn["turn_id"]
    assert tid in gw._pending_turns

    _run(gw.clear_session(sid))
    assert gw._sessions[sid].active_project_id is None

    with pytest.raises(HTTPException) as exc_info:
        _run(gw.stream_turn(sid, tid))
    assert exc_info.value.status_code == 409

    # 被拒绝的轮次已经从待取字典里移除，不会被重复处理或残留占位。
    assert tid not in gw._pending_turns
    # 清空后的会话状态没有被这次被拒绝的取流悄悄改回去。
    assert gw._sessions[sid].active_project_id is None
    assert gw._sessions[sid].last_turn is None


def test_create_turn_then_stream_without_clear_succeeds():
    """对照组：没有清空时，正常建轮→取流应该成功，且会话状态按这一轮结果更新——
    确认 CR-021 的修复没有连带破坏正常路径。
    """
    session_view = _run(gw.create_session(gw.SessionCreateBody(
        language="zh", project_id=None, version=None,
    )))
    sid = session_view["session_id"]

    turn = _run(gw.create_turn(sid, gw.TurnBody(question="怎么预留库存")))
    tid = turn["turn_id"]

    response = _run(gw.stream_turn(sid, tid))
    assert response.status_code == 200
    assert tid not in gw._pending_turns


def test_stream_turn_rejects_unknown_turn_id():
    session_view = _run(gw.create_session(gw.SessionCreateBody(
        language="zh", project_id=None, version=None,
    )))
    sid = session_view["session_id"]

    with pytest.raises(HTTPException) as exc_info:
        _run(gw.stream_turn(sid, "no-such-turn"))
    assert exc_info.value.status_code == 404


def test_older_turn_completing_after_newer_turn_does_not_roll_back_session():
    """CR-030 独立复现场景（用审查方给出的原始描述）：先为 orders/v1 建一个轮次
    （旧轮次），再为 checkout/v2 建一个显式新轮次（新轮次）。**先完整消费新流**，
    会话状态变成 checkout/v2；**再消费旧流**——旧流仍应正常流完给客户端
    （它对应的请求本身没有错，只是完成得晚），但不能把会话状态覆盖回
    orders/v1，那样等于凭空回滚了用户已经看到、已经确认生效的最新状态。

    这正是真实 SSE 流会发生的情形：两个轮次可以同时挂起，耗时不同，完成顺序
    不保证等于创建顺序。旧的 epoch 机制（CR-021）只在 clear() 时变化，同一个
    epoch 内的乱序完成完全不受它保护，需要单独的单调序号（turn_seq）。
    """
    session_view = _run(gw.create_session(gw.SessionCreateBody(
        language="zh", project_id=None, version=None,
    )))
    sid = session_view["session_id"]

    older = _run(gw.create_turn(sid, gw.TurnBody(
        question="订单怎么预留库存", project_id="orders", version="v1",
    )))
    newer = _run(gw.create_turn(sid, gw.TurnBody(
        question="结账怎么配置重试", project_id="checkout", version="v2",
    )))

    # 新轮次先完成：会话状态应该更新为 checkout/v2。
    response_newer = _run(gw.stream_turn(sid, newer["turn_id"]))
    assert response_newer.status_code == 200
    _run(_drain(response_newer))
    assert gw._sessions[sid].active_project_id == "checkout"
    assert gw._sessions[sid].active_version == "v2"

    # 旧轮次后完成：仍能正常拿到响应（不是错误），但不该把会话状态盖回去。
    response_older = _run(gw.stream_turn(sid, older["turn_id"]))
    assert response_older.status_code == 200
    _run(_drain(response_older))
    assert gw._sessions[sid].active_project_id == "checkout"
    assert gw._sessions[sid].active_version == "v2"


def test_clear_during_lazy_sse_consumption_is_not_undone_by_late_writeback():
    """CR-031 独立复现场景（用审查方给出的原始步骤）：`stream_turn()` 只在构造
    `StreamingResponse` **之前**核对过一次 epoch（CR-021）——但 SSE 是惰性消费的，
    响应对象构造完就立刻返回，真正执行生成器（进而真正写回状态）要等到响应体
    被 drain 的那一刻。这中间的窗口里，用户完全可以调用 `clear()`。

    步骤：为 orders/v1 建轮 → 调 `stream_turn()` 拿到 response（此时 epoch 检查
    已经通过）→ **在 drain 之前**调用 `clear_session()`（会话状态变 `None`，
    epoch 递增）→ 现在才 drain response。清空动作不能被这次"迟到"的写回撤销。
    """
    session_view = _run(gw.create_session(gw.SessionCreateBody(
        language="zh", project_id=None, version=None,
    )))
    sid = session_view["session_id"]

    turn = _run(gw.create_turn(sid, gw.TurnBody(
        question="订单怎么预留库存", project_id="orders", version="v1",
    )))
    response = _run(gw.stream_turn(sid, turn["turn_id"]))
    assert response.status_code == 200

    # 响应已经构造好、还没被 drain——这时候清空。
    _run(gw.clear_session(sid))
    assert gw._sessions[sid].active_project_id is None

    # 现在才真正消费这个（对清空来说已经过期的）流。
    _run(_drain(response))

    assert gw._sessions[sid].active_project_id is None
    assert gw._sessions[sid].active_version is None
    assert gw._sessions[sid].last_turn is None
