"""CR-036 判别性回归：真实 `lifespan()`/`boot()` 路径不能触发二次 mlx 原生导入。

其余 `/healthz` 相关测试（`test_gateway_healthz.py`）都刻意绕开 `TestClient`，
直接调用 `healthz()` 这个 async def，避免触发真实 MLX 模型加载。这里反过来——
CR-036 的问题恰恰出在 `boot()` 内部（`engine.load()` 之后无条件跟着
`embedder.load()`），只调用 `healthz()` 测不到它，必须用 `TestClient(app)`
真的走一遍 `lifespan()`，才能验证 `boot()` 实际执行时确实没有让
`Embedder.load()` 碰到第二次原生扩展导入（codex 复审 R32 就是用
`TestClient(app)` 在真实无 Metal 沙箱里复现出 nanobind 致命错误的）。

这个模块级的 `engine`/`embedder` 单例会被真实 `boot()` 调用改动状态，测试
结束后必须显式还原，否则会污染同一个 pytest 进程里其他依赖它们默认状态的
测试。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import apps.gateway.main as gw  # noqa: E402
from services import mlx_runtime  # noqa: E402
from services.inference.engine import EngineStatus  # noqa: E402


class _FakeChunkStore:
    """不碰真实索引文件——这条回归只关心 mlx 加载路径，索引本身不是重点。"""

    def __init__(self) -> None:
        self.meta = {"embedding_model": "fake", "dictionary_version": "v1"}

    def count(self) -> int:
        return 0

    def close(self) -> None:
        pass


def test_boot_never_attempts_second_mlx_import_after_engine_import_fails(monkeypatch):
    """让真实 `lifespan()` 里第一次 `import mlx_lm` 失败（`sys.modules` 标准
    手法，任何机器上都能确定性复现），装一个"被调用就报错"的 `mlx_embeddings`
    哨兵代表 nanobind 会在这一步 abort 整个进程。断言哨兵从未被调用——验证的
    是 `boot()` 真的没有走到那一步，而不是走了但侥幸没崩（那一步在真实无
    Metal 环境下不是能被 `pytest.raises` 接住的 Python 异常，是直接终止进程，
    没法在这条测试里"故意让它崩一次"来复现，只能验证没走到）。

    CR-035 之后、CR-036 之前的旧实现里 `boot()` 无条件调用
    `embedder.load()`，这条回归会在那份代码上失败——哨兵会被真的调用到。
    """
    monkeypatch.setitem(sys.modules, "mlx_lm", None)

    trap_calls = {"n": 0}
    trap = types.ModuleType("mlx_embeddings")

    def _trap_load(*_a, **_kw):
        trap_calls["n"] += 1
        raise AssertionError("boot() 不该在 engine 导入失败后还让 embedder 碰原生导入")

    trap.load = _trap_load
    monkeypatch.setitem(sys.modules, "mlx_embeddings", trap)

    monkeypatch.setattr(mlx_runtime, "broken", False)
    monkeypatch.setattr(mlx_runtime, "broken_reason", None)
    monkeypatch.setattr(gw, "ChunkStore", _FakeChunkStore)

    orig_store = gw._store
    orig_store_error = gw._store_error
    orig_orchestrator = gw._orchestrator
    orig_router = gw._router
    try:
        with TestClient(gw.app) as client:
            resp = client.get("/healthz")
    finally:
        # 真实模块级单例：boot() 会原地改 engine.status/embedder._model/
        # embedder.error，不是重新赋值一个新对象，必须显式复位，不能靠
        # monkeypatch 自动回滚（monkeypatch 只回滚它自己设置过的属性）。
        gw.engine.status = EngineStatus(model_id=gw.DEFAULT_MODEL)
        gw.embedder._model = None
        gw.embedder._tok = None
        gw.embedder.error = None
        gw._store = orig_store
        gw._store_error = orig_store_error
        gw._orchestrator = orig_orchestrator
        gw._router = orig_router

    assert trap_calls["n"] == 0, "embedder.load() 不该真的执行到 mlx_embeddings.load"
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert body["components"]["inference"]["ready"] is False
    assert body["components"]["embedding"]["ready"] is False
