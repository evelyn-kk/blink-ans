"""嵌入模型封装。

I0 实测（bge-m3-mlx-8bit，1024 维）：
- 查询侧单条 17 ms，远低于 0.8 秒检索预算，可直接放在请求路径上。
- 建索引 12.9 chunk/s，且批量化无收益（已跑满算力），属一次性离线成本。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services import mlx_runtime  # noqa: E402

DEFAULT_MODEL = "mlx-community/bge-m3-mlx-8bit"
DIM = 1024


class Embedder:
    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        self.model_id = model_id
        self._model = None
        self._tok = None
        self._lock = threading.Lock()
        # CR-035：检索/嵌入永远走本地（architecture.md §7），云端只顶生成——
        # 嵌入模型加载不了，服务就完全不能用，不该是一个没人能观测到的裸异常。
        self.error: str | None = None

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            # CR-037：每次尝试都先清掉上一次的 error——不然重试成功之后旧的
            # 错误消息会永远滞留，/healthz 报出一个已经不存在的故障。
            self.error = None
            # CR-036：`mlx_lm`/`mlx_embeddings` 底层共用同一份 mlx.core 原生
            # 扩展。如果 InferenceEngine 那边已经导入失败过（可能是 nanobind
            # 的 C++ 类型注册表停在半路），这里再导入一次会被判定为类型重复
            # 注册，直接 fatal error 中止整个进程——不是 Python 异常，
            # try/except 拦不住。见 services/mlx_runtime.py，检查这个进程级
            # 哨兵，已经破损就不再尝试。
            if mlx_runtime.broken:
                self.error = f"跳过加载：{mlx_runtime.broken_reason}"
                return
            try:
                from mlx_embeddings import load
                self._model, self._tok = load(self.model_id)
            except Exception as exc:
                # 没有 Metal 设备的机器上 `import mlx_embeddings` 直接抛
                # ImportError；不捕获的话会一路冲出 load()，让 boot() 崩溃、
                # 整个网关起不来——即便云端生成本可用，检索这一步也是本地
                # 强制的，所以这里必须能被上层观测到，而不是让进程直接退出。
                self.error = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, ImportError):
                    mlx_runtime.mark_broken(self.error)

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.load()
        if self._model is None:
            # 模型没能加载成功时才会到这——给一个稳定、可测试的诊断，而不是让
            # 下面 `import mlx.core`/`mlx_embeddings.generate` 在没有 Metal 的
            # 机器上再抛一次没头没脑的 ImportError。放到 load() 成功之后才导入：
            # 能走到这里说明 mlx_embeddings 刚刚已经导入成功过，同一个运行时
            # 里的 mlx.core 没有理由导入失败。
            raise RuntimeError(f"嵌入模型未就绪：{self.error}")
        import mlx.core as mx
        from mlx_embeddings import generate

        with self._lock:
            out = generate(self._model, self._tok, texts=texts)
            for attr in ("text_embeds", "embeddings", "pooler_output"):
                v = getattr(out, attr, None)
                if v is not None:
                    mx.eval(v)
                    return v.tolist()
        raise RuntimeError(f"无法从 {type(out)} 中取出嵌入向量")

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]
