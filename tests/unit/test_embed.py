"""CR-035：嵌入模型加载失败必须落进 error 状态，不能裸抛。

检索/嵌入永远走本地（architecture.md §7）——加载不了，整个服务就用不了。
用 `sys.modules["mlx_embeddings"] = None` 强制下一次导入抛 `ImportError`
（Python 官方文档：模块名在 `sys.modules` 里映射到 `None` 时触发），不依赖
真实缺 Metal，任何机器上都能确定性复现。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.retrieval.embed import Embedder  # noqa: E402


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
