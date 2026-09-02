"""SSE 桥接层的回归（CR-001 / CR-011）。

这层把同步生成器搬到 `asyncio.to_thread` 上再喂给 SSE，两个已知陷阱：

- **CR-001**：`asyncio.Queue` 不是线程安全的，跨线程写必须经 `call_soon_threadsafe`。
- **CR-011**：`to_thread` 起的线程**不可取消**。客户端断开后生产者会继续把整段答案
  生成完，全程占着推理引擎的锁（生成在 `with self._lock` 内串行），
  后续请求只能干等；事件还会一直堆进无界队列。

修复靠三件事：停止标志、有界槽位、关闭源生成器。下面按这三件事分别断言。
不用 pytest-asyncio（依赖清单里没有），直接 `asyncio.run` 跑。
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from apps.gateway.sse import sse_stream  # noqa: E402


def _delta(i: int) -> dict:
    return {"type": "answer_delta", "text": str(i)}


def test_all_events_are_delivered_and_framed():
    async def run():
        src = iter([{"type": "retrieval", "hits": 3}, _delta(1), {"type": "done"}])
        return [chunk async for chunk in sse_stream(src)]

    out = asyncio.run(run())
    assert len(out) == 3
    assert out[0].startswith("event: retrieval\ndata: {")
    assert all(c.endswith("\n\n") for c in out), "SSE 记录必须以空行结束"


def test_non_ascii_is_not_escaped():
    """中文被转义成 \\uXXXX 不算错，但会让事件体积翻几倍。"""
    async def run():
        return [c async for c in sse_stream(iter([{"type": "status", "message": "已选取 5 条证据"}]))]

    assert "已选取 5 条证据" in asyncio.run(run())[0]


def test_producer_exception_becomes_an_error_event():
    def boom():
        yield _delta(1)
        raise ValueError("检索炸了")

    async def run():
        return [c async for c in sse_stream(boom(), stage="orchestrator")]

    out = asyncio.run(run())
    assert "event: error" in out[-1]
    assert "ValueError" in out[-1] and "检索炸了" in out[-1]


# ---------- CR-011：断开必须真的把生成停下来 ----------

def test_disconnect_closes_the_source_generator():
    """断开时必须 close() 源生成器。

    这一步是整条链路的关键：GeneratorExit 会一路送到 `engine.stream()` 的
    finally，让它还原前缀 KV 并退出 `with self._lock`。少了它，
    断开的长请求会一直占着引擎锁，把后续每个请求都堵死。
    """
    closed = threading.Event()

    def source():
        try:
            i = 0
            while True:
                i += 1
                yield _delta(i)
        finally:
            closed.set()

    async def run():
        agen = sse_stream(source())
        got = [await agen.__anext__() for _ in range(3)]
        await agen.aclose()          # 客户端断开
        return got

    assert len(asyncio.run(run())) == 3
    assert closed.wait(2.0), "断开后源生成器未被关闭，生成会继续占着引擎锁"


def test_disconnect_stops_the_producer_thread():
    """生产线程必须自行收尾，而不是把整段答案生成完。"""
    produced: list[int] = []
    finished = threading.Event()

    def source():
        for i in range(100_000):
            produced.append(i)
            yield _delta(i)
        finished.set()

    async def run():
        agen = sse_stream(source())
        await agen.__anext__()
        await agen.aclose()

    asyncio.run(run())
    settled = len(produced)
    threading.Event().wait(0.5)      # 给还没停的线程留出继续跑的机会
    assert not finished.is_set(), "断开后生产者仍把整段生成跑完了"
    assert len(produced) == settled, "断开后生产者还在继续产出事件"


def test_queue_is_bounded_when_the_client_stalls():
    """消费端不读时，积压必须有上界，否则一个卡住的客户端就能撑爆内存。"""
    produced: list[int] = []
    slots = 4

    def source():
        for i in range(100_000):
            produced.append(i)
            yield _delta(i)

    async def run():
        agen = sse_stream(source(), slots=slots)
        await agen.__anext__()
        await asyncio.sleep(0.5)     # 客户端读了一条就不读了
        await agen.aclose()

    asyncio.run(run())
    # 上界 = 槽位数 + 已被消费端取走归还的那一个
    assert len(produced) <= slots + 2, (
        f"消费端停住 0.5s 后仍产出 {len(produced)} 条，队列没有上界"
    )


def test_stalled_client_that_resumes_still_gets_every_event():
    """反压不能变成丢事件：慢客户端只该被拖慢，答案必须完整。"""
    expected = 50

    async def run():
        agen = sse_stream(iter([_delta(i) for i in range(expected)]), slots=4)
        out = []
        async for chunk in agen:
            if len(out) == 1:
                await asyncio.sleep(0.3)     # 中途卡一下，让生产者撞上满槽位
            out.append(chunk)
        return out

    assert len(asyncio.run(run())) == expected
