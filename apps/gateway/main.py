"""API 网关 —— I0 最小运行示例。

目的不是实现问答，而是验证架构主线成立：
FastAPI 单进程 + 常驻 MLX 模型 + KV cache 跨请求驻留 + SSE 流式输出。

当前 /healthz 中转写、索引、外部检索三项为占位，将在 I1/I4/I6 逐步接入。
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.inference.engine import DEFAULT_MODEL, InferenceEngine  # noqa: E402

# 固定不变的系统前缀。KV cache 只能复用前缀，因此这里的内容
# 绝不能包含随请求变化的信息（时间戳、用户名、检索结果）。
SYSTEM_PREFIX = """你是一名 Java / Spring 云原生方向的资深后端工程师，负责回答生产环境问题。
回答必须依据给定证据，不得编造配置项名称、方法签名或指标名。
证据不足时明确说明"现有证据不足以确定"，并指出还需核实什么。"""

engine = InferenceEngine(DEFAULT_MODEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 冷加载 62s + 预热，必须在启动阶段完成，不能让第一个用户承担。
    await asyncio.to_thread(engine.load, SYSTEM_PREFIX)
    yield


app = FastAPI(title="blink-ans gateway", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    s = engine.status
    return {
        "status": "ok" if s.loaded else ("error" if s.error else "loading"),
        "components": {
            "inference": {
                "ready": s.loaded,
                "model": s.model_id,
                "load_seconds": s.load_seconds,
                "warmup_seconds": s.warmup_seconds,
                "resident_prefix_tokens": s.prefix_tokens,
                "error": s.error,
            },
            # 以下为占位，分别在 I4 / I1 / I6 接入
            "transcription": {"ready": False, "note": "I4 接入 mlx-whisper"},
            "index": {"ready": False, "note": "I1 接入 sqlite-vec + FTS5"},
            "external_search": {"ready": False, "note": "I6 接入官方来源白名单"},
        },
    }


class DebugAsk(BaseModel):
    question: str
    max_tokens: int = 256
    use_prefix: bool = True


@app.post("/v1/debug/stream")
async def debug_stream(body: DebugAsk):
    """I0 验证用端点：证明 SSE 流式与前缀 KV 复用在服务端可用。

    这不是最终的问答接口——真正的 /v1/answers 在 I2 实现，
    届时会加入检索、证据构建与来源引用。
    """

    async def gen():
        q: asyncio.Queue = asyncio.Queue()

        def produce():
            try:
                for ev in engine.stream(body.question, body.max_tokens, body.use_prefix):
                    q.put_nowait(ev)
            except Exception as exc:
                q.put_nowait({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                q.put_nowait(None)

        task = asyncio.create_task(asyncio.to_thread(produce))
        while (ev := await q.get()) is not None:
            yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
        await task

    return StreamingResponse(gen(), media_type="text/event-stream")
