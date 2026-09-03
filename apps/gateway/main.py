"""API 网关。

I2 交付文本问答闭环：`POST /v1/answers` 建立会话，
`GET /v1/answers/{id}/stream` 以 SSE 推送检索状态、答案增量与来源。

事件类型在此定死，后续迭代（语音、实时检索）只增加事件，不改已有语义：
    retrieval / status / answer_delta / sources / done / error
I4 会补上 transcript 事件。
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from packages.config.env import load_dotenv  # noqa: E402
from packages.prompts.answer import (  # noqa: E402
    SUPPORTED_LANGUAGES, Language, system_prompt, template_version,
)
from services.inference.backend import LocalBackend  # noqa: E402
from services.inference.claude_backend import ClaudeBackend  # noqa: E402
from services.inference.engine import DEFAULT_MODEL, InferenceEngine  # noqa: E402
from services.inference.router import Router  # noqa: E402
from services.orchestrator.answering import (  # noqa: E402
    AnswerConfig, AnswerRequest, Orchestrator,
)
from services.retrieval.embed import Embedder  # noqa: E402
from services.retrieval.store import ChunkStore  # noqa: E402

from apps.gateway.sse import sse_stream  # noqa: E402

# 凭据来自仓库根 .env（T-107）；ANTHROPIC_API_KEY 就是从这里进环境变量的，
# 云端后端本身不重新实现加载逻辑（见 services/inference/claude_backend.py）。
load_dotenv(ROOT / ".env")

engine = InferenceEngine(DEFAULT_MODEL)
embedder = Embedder()
_store: ChunkStore | None = None
_store_error: str | None = None
_orchestrator: Orchestrator | None = None
_router: Router | None = None

# 显式离线模式的唯一入口（T-028，architecture.md §6.4 三个降级触发条件之一）：
# 环境变量 BLINK_OFFLINE=1（或 true/yes，大小写不敏感）。没有做成 HTTP 参数——
# 离线是"这台机器现在没有网络"这一档的判断，不是单个问题的属性。
# `Router.offline` 是可变属性，未来若要加运行时切换的管理接口，直接改它即可。
_OFFLINE_TRUE = {"1", "true", "yes"}
_offline_mode = os.environ.get("BLINK_OFFLINE", "").strip().lower() in _OFFLINE_TRUE

# T-022：本地引擎只有单槽常驻前缀（T-027 已裁决不重新做多槽——本地已降级为
# 断网/失败兜底，多语言常驻前缀的收益不值当），启动时必须选定一个语言把它
# 预热进 KV cache。这个环境变量就是那个选择，同时也是 ClaudeBackend 默认携带
# 的 system prompt 语言、以及 AnswerConfig.default_language（Orchestrator 靠
# 它判断请求语言是否命中常驻前缀，见 services/orchestrator/answering.py）。
# scope.md：语言在会话开始前由用户选定，但会话/语音客户端要到 I3/I4 才接入，
# 本轮 API 层还没有会话概念——因此这里只提供进程级默认值，单次请求可以在
# POST /v1/answers 里用 `language` 字段覆盖它（覆盖值不影响本地常驻前缀，
# 只影响这一次传给生成后端的 system_override，见 Orchestrator.answer()）。
_DEFAULT_LANGUAGE = os.environ.get("BLINK_DEFAULT_LANGUAGE", "zh").strip().lower()
if _DEFAULT_LANGUAGE not in SUPPORTED_LANGUAGES:
    raise RuntimeError(
        f"BLINK_DEFAULT_LANGUAGE={_DEFAULT_LANGUAGE!r} 不受支持，"
        f"仅支持 {SUPPORTED_LANGUAGES}"
    )

# 待取的问答会话。单用户本地服务，用内存字典即可；
# 未被消费的会话在下次创建时按容量上限淘汰，避免长时间运行后无限增长。
_pending: dict[str, AnswerRequest] = {}
_MAX_PENDING = 64


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _store_error, _orchestrator, _router

    def boot():
        global _store, _store_error, _orchestrator, _router
        # 模型冷加载与索引打开都放在启动阶段：I0 实测冷启动 TTFT 是热态的两倍，
        # 不能让第一个真实用户承担这个代价。
        # 常驻前缀只有一槽（T-027），预热哪个语言由 _DEFAULT_LANGUAGE 决定。
        default_prompt = system_prompt(_DEFAULT_LANGUAGE)
        engine.load(default_prompt)
        embedder.load()
        try:
            _store = ChunkStore()
        except Exception as exc:
            _store_error = f"{type(exc).__name__}: {exc}"
            return
        if engine.status.loaded:
            local = LocalBackend(engine)
            # ClaudeBackend 总是构造出来——它的 available() 只查环境变量，
            # 缺 ANTHROPIC_API_KEY 时路由自然只会用本地，不需要在这里判空后
            # 传 None（传 None 也可以，但会在两处维护"有没有配凭据"的判断）。
            # 构造时携带与本地常驻前缀相同语言的默认 system prompt——
            # 请求语言与其不同时，Orchestrator 会逐请求传 system_override 覆盖它
            # （见 services/orchestrator/answering.py，ClaudeBackend.stream() 本就
            # 支持 per-call 覆盖，不需要为云端另建多份常驻状态）。
            cloud = ClaudeBackend(default_prompt)
            _router = Router(local, cloud, offline=_offline_mode)
            _orchestrator = Orchestrator(
                _store, embedder, _router,
                config=AnswerConfig(default_language=_DEFAULT_LANGUAGE),
            )

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
                "resident_prefix_language": _DEFAULT_LANGUAGE,
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
            # architecture.md §7：healthz 要分别报告本地生成与云端生成状态。
            # "ready" 只反映凭据是否配置（ClaudeBackend.available()），不代表
            # 探测过真实连通性——探测本身就是一次会花钱、可能超时的调用。
            "cloud_generation": {
                "ready": bool(_router and _router.cloud and _router.cloud.available()),
                "backend": _router.cloud.name if _router and _router.cloud else None,
                "offline_mode": _router.offline if _router else _offline_mode,
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
    # T-022：会话/语音客户端到 I3/I4 才接入（scope.md：语言在会话开始前选定），
    # 本轮 API 层还没有会话概念，因此逐请求可选——不传时落回
    # BLINK_DEFAULT_LANGUAGE（即本地常驻前缀预热的语言）。类型用 pydantic 对
    # Literal 的原生校验，非法值直接 422，不会流到 Orchestrator 里才发现。
    language: Language | None = Field(
        default=None, description="回答语言，不传则用服务端默认语言"
    )


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
        language=body.language or _DEFAULT_LANGUAGE,
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
