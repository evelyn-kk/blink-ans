"""嵌入模型封装。

I0 实测（bge-m3-mlx-8bit，1024 维）：
- 查询侧单条 17 ms，远低于 0.8 秒检索预算，可直接放在请求路径上。
- 建索引 12.9 chunk/s，且批量化无收益（已跑满算力），属一次性离线成本。
"""

from __future__ import annotations

import threading

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
            try:
                from mlx_embeddings import load
                self._model, self._tok = load(self.model_id)
            except Exception as exc:
                # 没有 Metal 设备的机器上 `import mlx_embeddings` 直接抛
                # ImportError；不捕获的话会一路冲出 load()，让 boot() 崩溃、
                # 整个网关起不来——即便云端生成本可用，检索这一步也是本地
                # 强制的，所以这里必须能被上层观测到，而不是让进程直接退出。
                self.error = f"{type(exc).__name__}: {exc}"

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
