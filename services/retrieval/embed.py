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

    def load(self) -> None:
        from mlx_embeddings import load

        with self._lock:
            if self._model is None:
                self._model, self._tok = load(self.model_id)

    def encode(self, texts: list[str]) -> list[list[float]]:
        import mlx.core as mx
        from mlx_embeddings import generate

        self.load()
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
