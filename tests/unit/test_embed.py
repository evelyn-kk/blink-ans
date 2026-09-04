"""CR-035：嵌入模型加载失败必须落进 error 状态，不能裸抛。

检索/嵌入永远走本地（architecture.md §7）——加载不了，整个服务就用不了。
用 `sys.modules["mlx_embeddings"] = None` 强制下一次导入抛 `ImportError`
（Python 官方文档：模块名在 `sys.modules` 里映射到 `None` 时触发），不依赖
真实缺 Metal，任何机器上都能确定性复现。

CR-036/CR-037：这两条是 codex 复审 CR-035 时发现的两个新问题——分别是
"mlx_lm 先导入失败后再加载嵌入模型会触发 nanobind 致命重复注册（整个进程
被 abort，不是能在测试里安全触发的 Python 异常，见
`services/mlx_runtime.py`）"和"加载失败后即使后续重试成功，旧 error 也不会
清除，健康状态永久错误"。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services import mlx_runtime  # noqa: E402
from services.retrieval.embed import Embedder  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_mlx_runtime_broken_flag(monkeypatch):
    """CR-036：`mlx_runtime.broken` 是进程级全局哨兵，一旦被某条测试触发
    `mark_broken()` 就不会自己复原（这是它故意的设计），会一直污染同一个
    pytest 进程里后面所有测试。`monkeypatch.setattr` 记录的是调用时刻的值，
    不管测试期间这两个全局变量被 `mark_broken()` 怎么改，teardown 时都会
    强制恢复成这里设的值。
    """
    monkeypatch.setattr(mlx_runtime, "broken", False)
    monkeypatch.setattr(mlx_runtime, "broken_reason", None)


def test_load_captures_mlx_embeddings_import_failure_instead_of_raising(monkeypatch):
    """旧实现里 `from mlx_embeddings import load` 在 `try` 之外——这个异常会
    直接冲出 `load()`，让 `apps/gateway` 的 `boot()` 崩溃、整个网关起不来。
    修复后 `load()` 必须正常返回，把失败写进 `self.error`。
    """
    monkeypatch.setitem(sys.modules, "mlx_embeddings", None)

    emb = Embedder("fake-model")
    emb.load()  # 不能抛

    assert emb._model is None
    assert emb.error is not None


def test_encode_raises_clear_runtime_error_when_model_never_loaded(monkeypatch):
    """`encode()` 不能在模型没加载成功时直接走到 `mlx_embeddings.generate()`
    ——那样报的错和"嵌入模型加载失败"这件事没有直接关系，排障要多绕一层。
    这里必须是一个带着 `self.error` 原因的、稳定可测试的 `RuntimeError`。
    """
    monkeypatch.setitem(sys.modules, "mlx_embeddings", None)

    emb = Embedder("fake-model")
    with pytest.raises(RuntimeError, match="嵌入模型未就绪"):
        emb.encode(["测试文本"])


# ---------- CR-036：mlx_runtime.broken 时绝不能再尝试导入原生扩展 ----------

def test_load_skips_import_when_mlx_runtime_already_broken(monkeypatch):
    """`mlx_runtime.broken` 已经被别的组件（比如 `InferenceEngine`）置位时，
    `Embedder.load()` 绝不能再尝试 `import mlx_embeddings`——哪怕这次导入
    "本来会成功"，nanobind 对同一个 C++ 类型二次注册是 fatal error，真正
    执行到那一步就已经晚了（会直接 abort 整个进程，不是能在测试里安全
    复现的 Python 异常）。这里放一个"被调用就报错"的哨兵模块，验证
    `load()` 真的没有走到 import 这一步，而不是走了但侥幸没坏。
    """
    trap = types.ModuleType("mlx_embeddings")

    def _trap_load(*_a, **_kw):
        raise AssertionError("mlx_runtime.broken 时不该再尝试导入/加载")

    trap.load = _trap_load
    monkeypatch.setitem(sys.modules, "mlx_embeddings", trap)
    monkeypatch.setattr(mlx_runtime, "broken", True)
    monkeypatch.setattr(mlx_runtime, "broken_reason", "ImportError: 模拟 InferenceEngine 先失败")

    emb = Embedder("fake-model")
    emb.load()  # 不能抛，也不能碰 trap.load

    assert emb._model is None
    assert emb.error is not None
    assert "跳过加载" in emb.error


# ---------- CR-037：重试成功后不能留着上一次的旧 error ----------

def test_error_clears_after_a_later_successful_load(monkeypatch):
    """第一次 `load()` 失败之后，如果是非导入类失败（比如权重文件损坏——
    这种失败发生在原生扩展已经导入成功之后，不触发 `mlx_runtime.mark_broken()`，
    重试在原理上是安全的），后续重试成功时旧的 `error` 不能永远滞留，否则
    `/healthz` 会永久报一个已经不存在的故障。
    """
    calls = {"n": 0}
    fake_module = types.ModuleType("mlx_embeddings")

    def _fake_load(_model_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("模拟权重文件损坏")
        return "fake-model-obj", "fake-tok-obj"

    fake_module.load = _fake_load
    monkeypatch.setitem(sys.modules, "mlx_embeddings", fake_module)

    emb = Embedder("fake-model")
    emb.load()
    assert emb._model is None
    assert emb.error is not None
    assert mlx_runtime.broken is False  # 非导入类失败，不该触发进程级哨兵

    emb.load()  # 重试
    assert emb._model == "fake-model-obj"
    assert emb.error is None
