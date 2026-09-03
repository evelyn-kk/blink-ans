"""API 网关。

I2 交付文本问答闭环：`POST /v1/answers` 建立会话，
`GET /v1/answers/{id}/stream` 以 SSE 推送检索状态、答案增量与来源。

事件类型在此定死，后续迭代（语音、实时检索）只增加事件，不改已有语义：
    retrieval / status / answer_delta / sources / done / error
I4 会补上 transcript 事件。
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from packages.prompts.answer import SYSTEM_PROMPT, template_version  # noqa: E402
from services.inference.engine import DEFAULT_MODEL, InferenceEngine  # noqa: E402
from services.orchestrator.answering import (  # noqa: E402
    AnswerRequest, Orchestrator,
)
from services.retrieval.embed import Embedder  # noqa: E402
from services.retrieval.store import ChunkStore  # noqa: E402

from apps.gateway.sse import sse_stream  # noqa: E402

engine = InferenceEngine(DEFAULT_MODEL)
embedder = Embedder()
_store: ChunkStore | None = None
_store_error: str | None = None
_orchestrator: Orchestrator | None = None

# 待取的问答会话。单用户本地服务，用内存字典即可；
# 未被消费的会话在下次创建时按容量上限淘汰，避免长时间运行后无限增长。
_pending: dict[str, AnswerRequest] = {}
_MAX_PENDING = 64


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _store_error, _orchestrator

    def boot():
        global _store, _store_error, _orchestrator
        # 模型冷加载与索引打开都放在启动阶段：I0 实测冷启动 TTFT 是热态的两倍，
        # 不能让第一个真实用户承担这个代价。
        engine.load(SYSTEM_PROMPT)
        embedder.load()
        try:
            _store = ChunkStore()
        except Exception as exc:
            _store_error = f"{type(exc).__name__}: {exc}"
            return
        if engine.status.loaded:
            _orchestrator = Orchestrator(_store, embedder, engine)

    await asyncio.to_thread(boot)
    yield
    if _store is not None:
        _store.close()


app = FastAPI(title="blink-ans gateway", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    s = engine.status
    index_ready = _store is not None
    ready = s.loaded and index_ready
    return {
        "status": "ok" if ready else ("error" if (s.error or _store_error) else "loading"),
        "components": {
            "inference": {
                "ready": s.loaded, "model": s.model_id,
                "load_seconds": s.load_seconds, "warmup_seconds": s.warmup_seconds,
                "resident_prefix_tokens": s.prefix_tokens,
                "template_version": template_version(),
                "error": s.error,
            },
            "index": {
                "ready": index_ready,
                "chunks": _store.count() if _store else None,
                "embedding_model": _store.meta.get("embedding_model") if _store else None,
                "dictionary_version": _store.meta.get("dictionary_version") if _store else None,
                "error": _store_error,
            },
            "transcription": {"ready": False, "note": "I4 接入 mlx-whisper"},
            "external_search": {"ready": False, "note": "I6 接入官方来源白名单"},
        },
    }


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    technology: str | None = None
    project: str | None = Field(default=None, description="外部来源项目 ID，例如 kafka")
    project_id: str | None = Field(default=None, description="用户项目 ID；严格限制项目材料")
    module: str | None = Field(default=None, description="用户项目模块精确过滤")
    symbol: str | None = Field(default=None, description="用户项目符号精确过滤")
    max_tokens: int | None = Field(default=None, ge=32, le=2048)


@app.post("/v1/answers", status_code=201)
async def create_answer(body: AskBody):
    if _orchestrator is None:
        raise HTTPException(503, "服务尚未就绪，请查看 /healthz")

    if len(_pending) >= _MAX_PENDING:
        for k in list(_pending)[: len(_pending) - _MAX_PENDING + 1]:
            _pending.pop(k, None)

    aid = uuid.uuid4().hex[:16]
    _pending[aid] = AnswerRequest(
        question=body.question, technology=body.technology,
        project=body.project, project_id=body.project_id,
        module=body.module, symbol=body.symbol, max_tokens=body.max_tokens,
    )
    return {"answer_id": aid, "stream_url": f"/v1/answers/{aid}/stream"}


@app.get("/v1/answers/{answer_id}/stream")
async def stream_answer(answer_id: str):
    req = _pending.pop(answer_id, None)
    if req is None:
        raise HTTPException(404, "会话不存在或已被消费")
    if _orchestrator is None:
        raise HTTPException(503, "服务尚未就绪")

    return StreamingResponse(
        sse_stream(_orchestrator.answer(req)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).resolve().parents[1] / "pwa" / "index.html").read_text(encoding="utf-8")
