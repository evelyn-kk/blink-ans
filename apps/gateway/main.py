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
from services.inference.claude_backend import ClaudeBackend, probe_network_floor  # noqa: E402
from services.inference.engine import DEFAULT_MODEL, InferenceEngine  # noqa: E402
from services.inference.router import Router  # noqa: E402
from services.orchestrator.answering import (  # noqa: E402
    AnswerConfig, AnswerRequest, Orchestrator,
)
from services.orchestrator.session import (  # noqa: E402
    ResolvedTurn, SessionState, build_request_for_turn, stream_and_record,
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

# 待取的一次性问答请求（POST /v1/answers 用）。单用户本地服务，用内存字典即可；
# 未被消费的请求在下次创建时按容量上限淘汰，避免长时间运行后无限增长。
_pending: dict[str, AnswerRequest] = {}
_MAX_PENDING = 64

# T-104：会话状态与"待取的会话内一轮问答"，与上面的 `_pending` 是两套独立语义——
# `_pending` 是"一次性问答的暂存"，这里的 `_sessions` 是跨多个 turn 存活的会话
# 状态（architecture.md §3：语言/活动项目/版本/上一轮实体与结论），
# `_pending_turns` 只是"这一个 turn 建好了、还没被 GET .../stream 取走"的暂存，
# 生命周期与一次性问答的 `_pending` 完全类似，只是多带了 session_id 与范围判定
# 结果，供取流时更新对应会话。同样内存字典即可，不落盘，容量上限、超限淘汰。
_sessions: dict[str, SessionState] = {}
_MAX_SESSIONS = 64
# 元组最后两项分别是建轮时的 session.epoch 快照（CR-021：取流前核对是否仍等于
# 会话当前 epoch，会话被 clear() 过就会不等，届时拒绝这个轮次而不是执行它）与
# session.turn_seq 快照（CR-030：乱序完成时不让旧轮次的写回盖掉新轮次的状态）。
_pending_turns: dict[str, tuple[str, AnswerRequest, ResolvedTurn, str, int, int]] = {}
_MAX_PENDING_TURNS = 64


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
        # CR-035：engine.load()/embedder.load() 现在都只把失败写进各自的
        # error 状态、不再抛——没有 Metal 设备的机器（例如这个审查沙箱）以前
        # 会在这两行直接崩掉整个 boot()，_store/_orchestrator/_router 全部
        # 建不起来，/healthz 也没机会呈现"本地推理/嵌入不可用"，只能眼看
        # 进程启动失败。现在两者失败都会继续往下走，让 /healthz 能报出具体
        # 哪个组件坏了。
        embedder.load()
        try:
            _store = ChunkStore()
        except Exception as exc:
            _store_error = f"{type(exc).__name__}: {exc}"
            return
        # 检索/嵌入永远走本地（architecture.md §7）：embedder 没加载成功，
        # Orchestrator 拿到手也是个每次检索都抛异常的半成品，不如干脆不建——
        # /healthz 已经能从 embedder.error 单独报出这个组件坏了。
        if engine.status.loaded and embedder.error is None:
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


def _healthz_status(
    core_ready: bool, cloud_ready: bool, s, store_error: str | None,
    embedder_error: str | None = None,
) -> str:
    """T-029：三档状态——`/healthz` 要能区分"完全正常"和"云端挂了但本地能兜底"，
    不能把这两种都报成同一个 `"error"`（检索/嵌入永远走本地，architecture.md §7）。

    - **硬错误（"error"）**：本地引擎、索引或嵌入模型没就绪——这三个不 ready
      服务就完全不能用，与云端状态无关（CR-035：嵌入模型加载失败此前不会走到
      这里判断——它会让 boot() 直接崩溃，进程都起不来，`/healthz` 无从谈起；
      现在 embedder.load() 失败只置 error 不再抛，这里才第一次真正需要把它
      纳入硬错误判断）。
    - **降级可用（"degraded"）**：本地 + 索引 + 嵌入都 ready，但云端不可用/
      未配置——服务仍然可用（本地兜底），不该跟硬错误混在一起报。
    - **正常（"ok"）**：本地 + 索引 + 嵌入 + 云端都 ready。
    - **"loading"**：核心组件尚未就绪，但也没有记录错误——还在启动过程中，
      沿用原有语义。
    """
    core_error = bool(s.error or store_error or embedder_error)
    if core_ready and cloud_ready:
        return "ok"
    if core_ready:
        return "degraded"
    return "error" if core_error else "loading"


@app.get("/healthz")
async def healthz():
    s = engine.status
    index_ready = _store is not None
    # CR-035：检索/嵌入永远走本地——embedder 没加载成功，服务和本地引擎没
    # 加载成功一样，都是不能用，必须一起进 core_ready，不能只看 engine/index。
    embedding_ready = embedder.error is None
    core_ready = s.loaded and index_ready and embedding_ready
    # "ready" 只反映凭据是否配置（ClaudeBackend.available()），语义不变——
    # 探测过真实连通性的结果单独放进 network_floor，不混进这个字段。
    cloud_ready = bool(_router and _router.cloud and _router.cloud.available())

    # 网络地板：只在云端凭据已配置时才探测（没配凭据没必要连一次官方 endpoint）。
    # CR-027：probe_network_floor() 内部是阻塞 I/O（socket.create_connection +
    # TLS 握手，最长 timeout_s 秒），`healthz()` 是 async 路由——直接同步调用会
    # 占住事件循环，暂停 SSE 流和其他所有并发请求，这正是健康检查最不该有的副作用
    # （审查方用 monkeypatch 成 0.2s 的假探测复现：并发的 `asyncio.sleep(0.01)`
    # 任务被拖到 0.217s 才完成）。改用 `asyncio.to_thread()` 丢到线程池执行，
    # 不阻塞事件循环。probe_network_floor() 内部已经把连接异常兜底成 error 字段
    # 而不抛出，这里再包一层 try/except 纯属防御性——保证这一项探测无论如何都
    # 不能把整个 /healthz 拖到 500（要观测的是"这一项测不出来"，不是让请求本身失败）。
    network_floor = None
    if cloud_ready:
        try:
            network_floor = await asyncio.to_thread(probe_network_floor)
        except Exception as exc:
            network_floor = {"host": None, "tcp_connect_s": None, "tcp_tls_s": None,
                              "error": f"{type(exc).__name__}: {exc}"}

    return {
        "status": _healthz_status(core_ready, cloud_ready, s, _store_error, embedder.error),
        "components": {
            "inference": {
                "ready": s.loaded, "model": s.model_id,
                "load_seconds": s.load_seconds, "warmup_seconds": s.warmup_seconds,
                "resident_prefix_tokens": s.prefix_tokens,
                "resident_prefix_language": _DEFAULT_LANGUAGE,
                "template_version": template_version(),
                "error": s.error,
            },
            # CR-035：检索/嵌入永远走本地，加载失败此前是个没人能观测到的裸
            # 异常（boot() 直接崩溃）——现在 embedder.load() 只置 error，这里
            # 单独报出来，不再混进 index 组件（index 报的是索引文件本身的
            # 状态，和"负责查询期间把问题转成向量的这个模型是否加载成功"是
            # 两回事）。
            "embedding": {
                "ready": embedding_ready, "model": embedder.model_id,
                "error": embedder.error,
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
            # 探测过真实连通性——真正探测过连通性的结果放在 network_floor 里，
            # 且那次探测本身不发起任何模型请求、不花钱（只是 TCP+TLS 握手，见
            # probe_network_floor()），与"ready 不代表探测过连通性"这句注释
            # 描述的是"没有靠一次会花钱的模型调用去确认"，不是完全不探测网络。
            "cloud_generation": {
                "ready": cloud_ready,
                "backend": _router.cloud.name if _router and _router.cloud else None,
                "offline_mode": _router.offline if _router else _offline_mode,
                # T-029：网络地板（TCP 连接 + TLS 握手耗时）。只在凭据已配置时探测；
                # 未配置凭据时为 None（没必要连一次官方 endpoint）。探测失败时
                # tcp_connect_s/tcp_tls_s 为 None、error 说明原因，不代表 /healthz 本身失败。
                "network_floor": network_floor,
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


# ---------------------------------------------------------------------------
# T-104：会话——独立问答/项目问答/追问，同一次对话跨多个 turn 保留最小状态。
# `POST /v1/answers` 保持不变（一次性问答，不建会话），会话是并存的新入口。
# ---------------------------------------------------------------------------

class SessionCreateBody(BaseModel):
    language: Language | None = Field(default=None, description="不传则用服务端默认语言")
    project_id: str | None = Field(default=None, description="创建时即选定活动项目，可选")
    version: str | None = Field(default=None, description="创建时即选定活动版本，可选")


def _session_view(s: SessionState) -> dict:
    # CR-033：三个字段必须来自同一次 read_state()，不能分开单独读——否则可能
    # 在另一线程的写回中途读到"项目已经切换，last_turn 却还是旧项目"的撕裂组合。
    snap = s.read_state()
    return {
        "session_id": s.session_id,
        "language": s.language,
        "active_project_id": snap.active_project_id,
        "active_version": snap.active_version,
        "last_turn": None if snap.last_turn is None else {
            "question": snap.last_turn.question,
            "entities": snap.last_turn.entities,
            "brief_conclusion": snap.last_turn.brief_conclusion,
            "cited_chunk_ids": snap.last_turn.cited_chunk_ids,
            "open_issue": snap.last_turn.open_issue,
        },
    }


@app.post("/v1/sessions", status_code=201)
async def create_session(body: SessionCreateBody):
    if len(_sessions) >= _MAX_SESSIONS:
        for k in list(_sessions)[: len(_sessions) - _MAX_SESSIONS + 1]:
            _sessions.pop(k, None)

    sid = uuid.uuid4().hex[:16]
    _sessions[sid] = SessionState(
        session_id=sid,
        language=body.language or _DEFAULT_LANGUAGE,
        active_project_id=body.project_id,
        active_version=body.version,
    )
    return _session_view(_sessions[sid])


@app.get("/v1/sessions/{session_id}")
async def get_session(session_id: str):
    s = _sessions.get(session_id)
    if s is None:
        raise HTTPException(404, "会话不存在或已过期")
    return _session_view(s)


@app.post("/v1/sessions/{session_id}/clear")
async def clear_session(session_id: str):
    s = _sessions.get(session_id)
    if s is None:
        raise HTTPException(404, "会话不存在或已过期")
    s.clear()
    return _session_view(s)


class TurnBody(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # 显式选择/切换项目——architecture.md §3 优先级第 1 档。传了就以它为准，
    # 不管这次问题文本本身像不像追问。
    project_id: str | None = Field(default=None, description="显式选择/切换项目")
    technology: str | None = None
    module: str | None = Field(default=None, description="用户项目模块精确过滤")
    symbol: str | None = Field(default=None, description="用户项目符号精确过滤")
    version: str | None = Field(default=None, description="显式选择/切换版本")
    # 显式开始新问题——优先级第 2 档，跳过追问启发式，不复用上一轮证据。
    new_topic: bool = Field(default=False, description="不追问上一轮，按新问题检索")
    max_tokens: int | None = Field(default=None, ge=32, le=2048)


@app.post("/v1/sessions/{session_id}/turns", status_code=201)
async def create_turn(session_id: str, body: TurnBody):
    if _orchestrator is None or _store is None:
        raise HTTPException(503, "服务尚未就绪，请查看 /healthz")
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "会话不存在或已过期")

    if len(_pending_turns) >= _MAX_PENDING_TURNS:
        for k in list(_pending_turns)[: len(_pending_turns) - _MAX_PENDING_TURNS + 1]:
            _pending_turns.pop(k, None)

    req, resolved, widened = build_request_for_turn(
        _store, embedder, session,
        question=body.question, project_id=body.project_id,
        technology=body.technology, module=body.module, symbol=body.symbol,
        version=body.version, new_topic=body.new_topic, max_tokens=body.max_tokens,
        cfg=_orchestrator.cfg,
    )
    tid = uuid.uuid4().hex[:16]
    # CR-030：每建一轮就递增一次会话的单调序号，捕获成这一轮的版本号——
    # 取流时凭它判断"有没有更新的轮次已经先我一步完成并写回过"，见
    # services/orchestrator/session.stream_and_record() 的说明。
    session.turn_seq += 1
    _pending_turns[tid] = (session_id, req, resolved, body.question, session.epoch, session.turn_seq)
    return {
        "turn_id": tid,
        "stream_url": f"/v1/sessions/{session_id}/turns/{tid}/stream",
        "scope": resolved.kind.value,
        "project_id": resolved.project_id,
        "widened_retrieval": widened,
    }


@app.get("/v1/sessions/{session_id}/turns/{turn_id}/stream")
async def stream_turn(session_id: str, turn_id: str):
    pending = _pending_turns.pop(turn_id, None)
    if pending is None:
        raise HTTPException(404, "会话轮次不存在或已被消费")
    sid, req, resolved, raw_question, created_epoch, turn_seq = pending
    if sid != session_id:
        raise HTTPException(404, "会话轮次不存在或已被消费")
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "会话不存在或已过期")
    if session.epoch != created_epoch:
        # CR-021：这个轮次是在上一次 clear() 之前建的，携带的是清空前的会话
        # 状态（比如旧项目的 extra_hit_rowids）。清空之后必须让它失效，
        # 否则执行完还会把这份旧状态重新写回会话，等于撤销了这次清空。
        raise HTTPException(409, "会话已被清空，此轮次已失效，请重新发起")
    if _orchestrator is None:
        raise HTTPException(503, "服务尚未就绪")

    return StreamingResponse(
        sse_stream(stream_and_record(
            _orchestrator, req, session, resolved, raw_question, turn_seq, created_epoch,
        )),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).resolve().parents[1] / "pwa" / "index.html").read_text(encoding="utf-8")
