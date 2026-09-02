"""把同步生成器的事件流桥接成 SSE 输出。

这层单独存在的理由有两个：

1. **跨线程写队列的正确姿势只应有一份。** `asyncio.Queue` 不是线程安全的，
   工作线程直接 `put_nowait` 会在唤醒等待中的消费协程时走
   `Future.set_result -> loop.call_soon`，而 `call_soon` 不唤醒事件循环的
   selector，流可能就此卡住（CR-001）。I4 的音频分片上传是同样的形状，
   照抄一遍迟早会漏掉其中一步。

2. **客户端断开必须真的把生成停下来。** `asyncio.to_thread` 起的线程不可取消，
   断开后生产者会继续把整段答案生成完，全程占着推理引擎的锁，
   后续请求只能干等，事件还会堆进无界队列（CR-011）。
   这里用「停止标志 + 有界槽位 + 关闭源生成器」三件套做协作式取消。
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import AsyncIterator, Iterator

# 单个流允许积压的事件数。生成一路按 token 出事件，消费端变慢或断开时
# 无界队列会一直涨；256 够覆盖网络抖动，又远小于一次回答的 700 token 上限，
# 因此反压真会生效而不是形同虚设。
STREAM_SLOTS = 256
# 生产线程等槽位时的轮询间隔——决定断开后最坏多久才停下来
STOP_POLL_S = 0.2


def format_event(ev: dict) -> str:
    return f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"


async def sse_stream(
    events: Iterator[dict],
    *,
    stage: str = "orchestrator",
    slots: int = STREAM_SLOTS,
) -> AsyncIterator[str]:
    """在工作线程上消费 `events`，把每个事件转成一条 SSE 记录产出。

    `events` 必须是**生成器**：断开时靠 `close()` 把 GeneratorExit 送进去，
    让链路末端（`engine.stream()`）的 finally 还原前缀 KV 并释放引擎锁。
    """
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # 客户端断开的停止信号。to_thread 的线程无法被取消，只能协作式停止：
    # 生产者在每个事件的间隙查这个标志。
    stop = threading.Event()
    # 队列上界：消费一个还一个槽位，生产者取不到就阻塞，
    # 从而把客户端的读取速度反压回生成侧。
    free = threading.Semaphore(slots)

    def emit(item) -> bool:
        """把事件交给事件循环。返回 False 表示消费端已走，生产者应停止。"""
        while not free.acquire(timeout=STOP_POLL_S):
            if stop.is_set():
                return False
        if stop.is_set():
            free.release()
            return False
        loop.call_soon_threadsafe(q.put_nowait, item)
        return True

    def produce() -> None:
        try:
            for ev in events:
                if not emit(ev):
                    break
        except Exception as exc:
            emit({"type": "error", "stage": stage,
                  "message": f"{type(exc).__name__}: {exc}"})
        finally:
            # close() 把 GeneratorExit 一路送到 engine.stream() 的 finally：
            # 还原前缀 KV、退出 with self._lock。少了这一步，
            # 断开的请求会一直占着引擎锁，把后续请求全堵住。
            close = getattr(events, "close", None)
            if close is not None:
                close()
            # 终止哨兵不占槽位——断开时槽位可能全满，占用就会死等
            loop.call_soon_threadsafe(q.put_nowait, None)

    task = asyncio.create_task(asyncio.to_thread(produce))
    try:
        while (ev := await q.get()) is not None:
            free.release()
            yield format_event(ev)
    finally:
        # 先置位再等：客户端断开时本协程是被取消的，下面的 await 会立刻抛
        # CancelledError；届时标志已经立好，生产线程仍会在一个事件的间隙内自行收尾。
        stop.set()
        free.release()      # 唤醒可能正卡在满槽位上的生产线程
        await task
