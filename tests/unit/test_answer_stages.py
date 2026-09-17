"""T-008 阶段打点：done.stages 的起点、顺序与"首个非空正文"语义。

时钟由测试控制，断言的是具体偏移值而不是"大于 0"——只会变绿的计时断言不算回归。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.orchestrator.answering import AnswerRequest, Orchestrator  # noqa: E402
from services.orchestrator.stages import StageClock  # noqa: E402
from tests.unit.test_answering import (  # noqa: E402
    BothBackendsFailRouter, FakeEmbedder, FakeEngine, FakeRouter, hit,
)


class ManualTime:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _use_clock(monkeypatch, now: float) -> ManualTime:
    clock = ManualTime(now)
    monkeypatch.setattr("services.orchestrator.stages.time", SimpleNamespace(perf_counter=clock))
    return clock


class TimedRouter(FakeRouter):
    """每个后端事件前推进时钟：空白增量在 t=106，首个非空正文在 t=107。"""

    def __init__(self, clock: ManualTime) -> None:
        super().__init__(FakeEngine())
        self.clock = clock

    def generate(self, content, *, max_tokens, system_override=None, cloud_allowed=True):
        self.clock.now = 106.0
        yield {"type": "delta", "text": " \n"}
        self.clock.now = 107.0
        yield {"type": "delta", "text": "答案 [1]"}
        self.clock.now = 109.0
        yield {"type": "delta", "text": "补充"}
        self.clock.now = 110.0
        yield {"type": "done", "ttft_s": 0.1, "prompt_tokens": 10, "prefilled_tokens": 10,
               "prefix_reused": True, "decode_tps": 20.0, "served_by": "local"}


def test_voice_marks_are_the_origin_and_first_delta_skips_blank_text(monkeypatch):
    clock = _use_clock(monkeypatch, 104.0)

    def search(*_a, **_kw):
        clock.now = 105.0
        return [hit()]

    monkeypatch.setattr("services.orchestrator.answering.hybrid_search", search)
    req = AnswerRequest("怎么预留库存", stage_marks={
        "final_chunk_received": 100.0, "asr_final": 101.5, "answer_requested": 103.0,
    })
    events = list(Orchestrator(None, FakeEmbedder(), TimedRouter(clock)).answer(req))
    done = next(e for e in events if e["type"] == "done")

    assert done["stages"] == {
        "origin": "final_chunk_received",
        "ms": {
            "final_chunk_received": 0.0,
            "asr_final": 1500.0,
            "answer_requested": 3000.0,
            "answer_started": 4000.0,
            "retrieval_done": 5000.0,
            "generation_started": 5000.0,
            "first_answer_delta": 7000.0,
            "answer_done": 10000.0,
        },
    }
    # 只加字段：既有 done 键与答案正文不变。
    assert done["served_by"] == "local"
    assert "".join(e["text"] for e in events if e["type"] == "answer_delta") == " \n答案 [1]补充"


def test_text_request_without_marks_starts_at_orchestrator(monkeypatch):
    _use_clock(monkeypatch, 50.0)
    monkeypatch.setattr("services.orchestrator.answering.hybrid_search", lambda *a, **kw: [hit()])
    events = list(Orchestrator(None, FakeEmbedder(), FakeRouter(FakeEngine()))
                  .answer(AnswerRequest("怎么预留库存")))
    done = next(e for e in events if e["type"] == "done")
    assert done["stages"]["origin"] == "answer_started"
    assert list(done["stages"]["ms"]) == [
        "answer_started", "retrieval_done", "generation_started", "first_answer_delta", "answer_done",
    ]


def test_fallback_done_also_carries_stages(monkeypatch):
    """双后端失败的兜底 done 同样带打点，不因分支不同而漏掉。"""
    _use_clock(monkeypatch, 1.0)
    monkeypatch.setattr("services.orchestrator.answering.hybrid_search", lambda *a, **kw: [hit(dist=0.6)])
    events = list(Orchestrator(None, FakeEmbedder(), BothBackendsFailRouter(FakeEngine()))
                  .answer(AnswerRequest("怎么预留库存", stage_marks={"answer_requested": 0.5})))
    done = next(e for e in events if e["type"] == "done")
    assert done["served_by"] == "none"
    assert done["stages"]["origin"] == "answer_requested"
    assert {"generation_started", "first_answer_delta", "answer_done"} <= set(done["stages"]["ms"])


def test_policy_refusal_done_carries_stages_without_generation(monkeypatch):
    _use_clock(monkeypatch, 1.0)
    monkeypatch.setattr("services.orchestrator.answering.hybrid_search", lambda *a, **kw: [hit(dist=0.74)])
    events = list(Orchestrator(None, FakeEmbedder(), FakeRouter(FakeEngine()))
                  .answer(AnswerRequest("怎么用 Rust 写词法分析器")))
    done = next(e for e in events if e["type"] == "done")
    assert done["served_by"] == "policy"
    assert "generation_started" not in done["stages"]["ms"]
    assert "answer_done" in done["stages"]["ms"]


def test_closing_the_answer_stream_still_reaches_the_backend_generator(monkeypatch):
    """新增的包装层不能让断开失效：后端生成器的 finally 必须立即执行（CR-011）。

    判别性有限，如实写明：CPython 下外层生成器关闭后内层引用计数归零也会被关闭，
    所以删掉 answer() 里显式的 events.close() 这条测试仍通过——它守的是"断开能
    传到后端"这一行为，显式 close 只是不依赖引用计数时机的防御。
    """
    closed = []

    class HoldingRouter(FakeRouter):
        def generate(self, content, *, max_tokens, system_override=None, cloud_allowed=True):
            try:
                yield {"type": "delta", "text": "开头"}
                yield {"type": "delta", "text": "不会被读到"}
            finally:
                closed.append(True)

    monkeypatch.setattr("services.orchestrator.answering.hybrid_search", lambda *a, **kw: [hit()])
    stream = Orchestrator(None, FakeEmbedder(), HoldingRouter(FakeEngine())).answer(AnswerRequest("怎么预留库存"))
    for ev in stream:
        if ev["type"] == "answer_delta":
            break
    stream.close()
    assert closed == [True]


def test_stage_clock_keeps_first_mark_and_orders_by_time():
    now = ManualTime(3.0)
    clock = StageClock({"b": 2.0, "a": 1.0}, clock=now)
    clock.mark("c")
    now.now = 4.0
    clock.mark("c")
    assert clock.snapshot() == {"origin": "a", "ms": {"a": 0.0, "b": 1000.0, "c": 2000.0}}
    assert StageClock().snapshot() == {"origin": None, "ms": {}}
