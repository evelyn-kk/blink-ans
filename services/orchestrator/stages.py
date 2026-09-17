"""端到端阶段打点（architecture.md §6.1）。

所有打点都是**同一进程**的 `time.perf_counter()` 读数：网关在 HTTP 边界记下
语音侧的时刻，经 `AnswerRequest.stage_marks` 交给编排层，编排层在同一时钟上
继续打点，最后在 `done` 事件里换算成相对起点的毫秒偏移。

名字只承诺它实际量到的东西（AGENTS.md §5.3）：
- `final_chunk_received` 是服务端收到带 `final=true` 的 PCM 分片的时刻，**不是**
  用户停止说话的时刻——浏览器端 VAD 的静音窗口、排空在途 partial 与上传都在它之前，
  只能由客户端自己的时钟量；
- `first_answer_delta` 是编排层产出第一个非空白 `answer_delta` 的时刻，不判断这段
  正文是否"有意义"，也不含网络发送与客户端接收。
"""

from __future__ import annotations

import time
from typing import Callable


class StageClock:
    """按发生顺序记录命名时刻；起点是最早记录的那个打点。"""

    def __init__(
        self,
        marks: dict[str, float] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        # 调用时才取 time.perf_counter：写成默认参数会在定义时绑定，
        # 测试替换时钟就不生效（AGENTS.md §5.3 T-017 那一行的同一个坑）。
        self._clock = clock or time.perf_counter
        self._marks: dict[str, float] = dict(marks or {})

    def mark(self, name: str) -> None:
        """记录一次；同名打点只保留第一次，重复调用不会把"首个"改成"最后一个"。"""
        if name not in self._marks:
            self._marks[name] = self._clock()

    def snapshot(self) -> dict:
        if not self._marks:
            return {"origin": None, "ms": {}}
        origin, t0 = min(self._marks.items(), key=lambda item: item[1])
        return {
            "origin": origin,
            "ms": {
                name: round((t - t0) * 1000, 1)
                for name, t in sorted(self._marks.items(), key=lambda item: item[1])
            },
        }
