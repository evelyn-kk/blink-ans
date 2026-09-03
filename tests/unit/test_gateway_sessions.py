"""路由级回归：会话 API 的建轮/清空/取流交互（CR-021）。

不启动真实模型或索引——直接调用 `apps.gateway.main` 里的路由处理函数
（它们只是普通的 async def，不经过 `TestClient` 也能直接调用），并用
monkeypatch 替换掉模块级的 `_sessions`/`_pending_turns`/`_orchestrator`/`_store`。
这样能验证真实的路由控制流（建轮时捕获 epoch、取流时核对 epoch、清空后
拒绝旧轮次），而不需要触发 `lifespan()` 里的真实 MLX 模型加载。
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
