"""`/healthz` 的三档状态回归（T-029）。

不启动真实模型或索引——直接调用 `apps.gateway.main.healthz()`（普通 async def，
不经过 `TestClient` 也能直接调用），并用 monkeypatch 替换掉模块级的
`engine.status`/`_store`/`_store_error`/`_router`。模式抄
`tests/unit/test_gateway_sessions.py`（CR-021 那轮定下的），避免触发 lifespan()
里的真实 MLX 模型加载。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import apps.gateway.main as gw  # noqa: E402
from services.inference.engine import EngineStatus  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


class FakeStore:
    def __init__(self, chunks: int = 10) -> None:
        self._chunks = chunks
        self.meta = {"embedding_model": "fake-embed", "dictionary_version": "v1"}

    def count(self) -> int:
        return self._chunks


class FakeCloudBackend:
    name = "claude"

    def __init__(self, available: bool = True) -> None:
        self._available = available

    def available(self) -> bool:
        return self._available


class FakeRouter:
    def __init__(self, *, cloud=None, offline: bool = False) -> None:
        self.cloud = cloud
        self.offline = offline


def _loaded_status(error: str | None = None) -> EngineStatus:
    return EngineStatus(
        model_id="fake-model", loaded=error is None, load_seconds=1.0,
        warmup_seconds=0.5, error=error, prefix_tokens=100,
    )


def test_healthz_ok_when_local_index_and_cloud_all_ready(monkeypatch):
    monkeypatch.setattr(gw.engine, "status", _loaded_status())
    monkeypatch.setattr(gw, "_store", FakeStore())
    monkeypatch.setattr(gw, "_store_error", None)
    monkeypatch.setattr(gw, "_router", FakeRouter(cloud=FakeCloudBackend(available=True)))
    monkeypatch.setattr(gw, "probe_network_floor", lambda: {
        "host": "api.anthropic.com", "tcp_connect_s": 0.02, "tcp_tls_s": 0.05, "error": None,
    })

    body = _run(gw.healthz())

    assert body["status"] == "ok"
    assert body["components"]["cloud_generation"]["ready"] is True
    assert body["components"]["cloud_generation"]["network_floor"]["error"] is None


def test_healthz_degraded_when_local_and_index_ready_but_cloud_unavailable(monkeypatch):
    """T-029 核心场景：云端挂了/未配置，本地能兜底——不该跟硬错误混报同一个 error。"""
    monkeypatch.setattr(gw.engine, "status", _loaded_status())
    monkeypatch.setattr(gw, "_store", FakeStore())
    monkeypatch.setattr(gw, "_store_error", None)
    monkeypatch.setattr(gw, "_router", FakeRouter(cloud=FakeCloudBackend(available=False)))
    probe_called = {"value": False}

    def _should_not_be_called():
        probe_called["value"] = True
        raise AssertionError("未配置/不可用凭据时不该探测网络")

    monkeypatch.setattr(gw, "probe_network_floor", _should_not_be_called)

    body = _run(gw.healthz())

    assert body["status"] == "degraded"
    assert body["components"]["inference"]["ready"] is True
    assert body["components"]["index"]["ready"] is True
    assert body["components"]["cloud_generation"]["ready"] is False
    assert body["components"]["cloud_generation"]["network_floor"] is None
    assert probe_called["value"] is False


def test_healthz_degraded_when_cloud_router_missing_entirely(monkeypatch):
    """启动阶段索引打开失败等场景下 `_router` 可能是 None——同样应归为降级
    （只要本地+索引 ready），不是硬错误。
    """
    monkeypatch.setattr(gw.engine, "status", _loaded_status())
    monkeypatch.setattr(gw, "_store", FakeStore())
    monkeypatch.setattr(gw, "_store_error", None)
    monkeypatch.setattr(gw, "_router", None)
    monkeypatch.setattr(gw, "probe_network_floor", lambda: (_ for _ in ()).throw(
        AssertionError("不该被调用")
    ))

    body = _run(gw.healthz())

    assert body["status"] == "degraded"
    assert body["components"]["cloud_generation"]["ready"] is False
    assert body["components"]["cloud_generation"]["network_floor"] is None


def test_healthz_error_when_local_engine_not_loaded(monkeypatch):
    """本地引擎没就绪——即使云端一切正常，也必须是硬错误，不是降级
    （检索/嵌入永远走本地，架构上无法绕开）。
    """
    monkeypatch.setattr(gw.engine, "status", _loaded_status(error="RuntimeError: 模型加载失败"))
    monkeypatch.setattr(gw, "_store", FakeStore())
    monkeypatch.setattr(gw, "_store_error", None)
    monkeypatch.setattr(gw, "_router", FakeRouter(cloud=FakeCloudBackend(available=True)))
    monkeypatch.setattr(gw, "probe_network_floor", lambda: {
        "host": "api.anthropic.com", "tcp_connect_s": 0.02, "tcp_tls_s": 0.05, "error": None,
    })

    body = _run(gw.healthz())

    assert body["status"] == "error"
    assert body["components"]["inference"]["ready"] is False


def test_healthz_error_when_index_not_ready_with_store_error(monkeypatch):
    monkeypatch.setattr(gw.engine, "status", _loaded_status())
    monkeypatch.setattr(gw, "_store", None)
    monkeypatch.setattr(gw, "_store_error", "RuntimeError: 索引打开失败")
    monkeypatch.setattr(gw, "_router", None)

    body = _run(gw.healthz())

    assert body["status"] == "error"
    assert body["components"]["index"]["ready"] is False


def test_healthz_loading_when_core_not_ready_and_no_error_yet():
    """启动尚未完成（还没报错，也还没就绪）——沿用原有的 loading 语义。"""
    status = EngineStatus(model_id="fake-model")  # loaded=False, error=None（默认值）
    import apps.gateway.main as gw2

    orig_status = gw2.engine.status
    orig_store = gw2._store
    orig_store_error = gw2._store_error
    orig_router = gw2._router
    try:
        gw2.engine.status = status
        gw2._store = None
        gw2._store_error = None
        gw2._router = None
        body = _run(gw2.healthz())
        assert body["status"] == "loading"
    finally:
        gw2.engine.status = orig_status
        gw2._store = orig_store
        gw2._store_error = orig_store_error
        gw2._router = orig_router


def test_healthz_network_probe_failure_is_captured_not_raised(monkeypatch):
    """探测本身抛异常时（即便 probe_network_floor 内部理论上已经兜底），/healthz
    外层的防御性 try/except 也要能兜住，返回而不是让整个请求 500。
    """
    monkeypatch.setattr(gw.engine, "status", _loaded_status())
    monkeypatch.setattr(gw, "_store", FakeStore())
    monkeypatch.setattr(gw, "_store_error", None)
    monkeypatch.setattr(gw, "_router", FakeRouter(cloud=FakeCloudBackend(available=True)))

    def _raise():
        raise OSError("simulated unexpected failure")

    monkeypatch.setattr(gw, "probe_network_floor", _raise)

    body = _run(gw.healthz())

    assert body["status"] == "ok"  # 本地/索引/云端凭据都 ready，探测失败不改变这一点
    nf = body["components"]["cloud_generation"]["network_floor"]
    assert nf["error"] is not None
    assert nf["tcp_connect_s"] is None


def test_healthz_slow_probe_does_not_block_event_loop(monkeypatch):
    """CR-027 判别性回归：`probe_network_floor()` 是阻塞 I/O（最长 `timeout_s` 秒），
    `healthz()` 是 async 路由——如果同步直接调用会占住事件循环，暂停 SSE 流和
    其他所有并发请求。这里用 `time.sleep(0.2)` 模拟慢探测，验证并发调度的另一个
    协程（`asyncio.sleep(0.01)`）不会被拖慢——它理应仍在 ~0.01s 左右完成，而不是
    被拖到 ~0.2s（旧实现同步调用会复现审查方给出的那个数字：0.217s）。
    """
    import time as time_module

    monkeypatch.setattr(gw.engine, "status", _loaded_status())
    monkeypatch.setattr(gw, "_store", FakeStore())
    monkeypatch.setattr(gw, "_store_error", None)
    monkeypatch.setattr(gw, "_router", FakeRouter(cloud=FakeCloudBackend(available=True)))
    monkeypatch.setattr(gw, "probe_network_floor", lambda: (time_module.sleep(0.2), {
        "host": "api.anthropic.com", "tcp_connect_s": 0.2, "tcp_tls_s": 0.2, "error": None,
    })[1])

    async def _quick_task(start):
        # 从共享起点 `start` 算起，而不是从这个协程自己开始执行的那一刻算起——
        # 否则如果事件循环被 healthz() 的同步阻塞占住，这个协程根本没机会被
        # 调度到，等它终于跑起来时再测"自己的" 0.01s 睡眠时长永远是对的，
        # 测不出"它等了多久才轮到自己"这个真正要验证的东西。
        await asyncio.sleep(0.01)
        return time_module.perf_counter() - start

    async def _both():
        start = time_module.perf_counter()
        return await asyncio.gather(gw.healthz(), _quick_task(start))

    _, quick_elapsed = _run(_both())

    assert quick_elapsed < 0.1, (
        f"并发的 asyncio.sleep(0.01) 被拖到 {quick_elapsed:.3f}s 才完成——"
        "说明 healthz() 里的慢探测阻塞了事件循环"
    )
