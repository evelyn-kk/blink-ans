"""常驻推理引擎。

设计要点来自 I0 基准（见 progress.md「I0 基准数据」）：

1. prefill 约 352 tok/s 且随上下文线性增长，是本机最硬的约束。
2. KV cache 预热后首 token 仅需 0.095s，比冷 prefill 快约 30 倍。

因此模型必须常驻单进程，且 KV cache 对象要能跨请求存活——这是投机式提前 prefill
（在用户说话期间就把证据 prefill 好）能够成立的前提，也是服务端选 Python 单进程的原因。
"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator

DEFAULT_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


@dataclass
class EngineStatus:
    model_id: str
    loaded: bool = False
    load_seconds: float | None = None
    warmup_seconds: float | None = None
    error: str | None = None
    prefix_tokens: int = 0


class InferenceEngine:
    """持有常驻模型与可复用的前缀 KV cache。

    线程安全说明：MLX 的求值不是线程安全的，因此所有生成都在 _lock 下串行。
    单用户本地服务场景下这是可接受的；并发需求出现时再改为请求队列。
    """

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        self.status = EngineStatus(model_id=model_id)
        self._model = None
        self._tokenizer = None
        self._prefix_cache = None
        self._prefix_tokens: list[int] = []
        self._system_prompt = ""
        self._lock = threading.Lock()

    # ---------- 生命周期 ----------

    def load(self, system_prompt: str | None = None) -> None:
        from mlx_lm import load

        t0 = time.perf_counter()
        try:
            self._model, self._tokenizer = load(self.status.model_id)
        except Exception as exc:
            self.status.error = f"{type(exc).__name__}: {exc}"
            return
        self.status.load_seconds = round(time.perf_counter() - t0, 2)
        self.status.loaded = True

        # 冷启动的首次生成明显慢于热态（4B 实测 3.18s vs 1.57s），
        # 因此启动时必须预热，避免第一个真实用户吃到冷启动代价。
        if system_prompt:
            self.warm_prefix(system_prompt)

        t1 = time.perf_counter()
        for _ in self.stream("你好", max_tokens=1):
            break
        self.status.warmup_seconds = round(time.perf_counter() - t1, 2)

    def warm_prefix(self, system_prompt: str) -> int:
        """把固定不变的提示词前缀预先算进 KV cache 并常驻。

        必须在 **chat template 渲染后的 token 空间**里切分前缀，不能直接用原始文本拼接：
        否则模型看不到 <|im_start|>assistant 这类角色标记，会变成续写用户输入而非作答。
        （I0 最小运行示例中曾因此产生错误输出，见 progress.md 变更记录。）

        做法：分别渲染"只有系统提示"和"系统提示 + 探针内容"两个完整 prompt，
        取二者的最长公共 token 前缀作为可复用部分。

        只对**前缀**有效：一旦模板中间插入随请求变化的内容，整段缓存失效。
        实测收益正比于前缀占比（235/1056 token 的前缀省下 19% TTFT），
        属于投机式 prefill 失败时的保底手段。
        """
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        self._system_prompt = system_prompt
        # 用两个不同的变化部分求公共前缀，避免把探针内容本身算进前缀
        a = self._tokenizer.encode(self._render(system_prompt, "AAAA"))
        b = self._tokenizer.encode(self._render(system_prompt, "BBBB"))
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        tokens = a[:n]

        with self._lock:
            cache = make_prompt_cache(self._model)
            self._model(mx.array(tokens)[None], cache=cache)
            mx.eval([c.state for c in cache])
            self._prefix_cache = cache
            self._prefix_tokens = tokens
        self.status.prefix_tokens = len(tokens)
        return len(tokens)

    def _render(self, system_prompt: str, user_content: str) -> str:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_content})
        try:
            return self._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

    # ---------- 生成 ----------

    def stream(
        self, user_content: str, max_tokens: int = 512, use_prefix: bool = True
    ) -> Iterator[dict]:
        """流式生成。产出 {"type": "delta"|"done", ...}。

        传入的是**用户内容**（证据 + 问题），chat template 由引擎内部套用。
        use_prefix=True 时复用常驻的系统前缀 KV，只 prefill 变化部分。
        """
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache

        if not self.status.loaded:
            raise RuntimeError("模型尚未加载")

        full = self._tokenizer.encode(self._render(self._system_prompt, user_content))
        reuse = (
            use_prefix
            and self._prefix_cache is not None
            and full[: len(self._prefix_tokens)] == self._prefix_tokens
        )
        prompt = full[len(self._prefix_tokens):] if reuse else full
        prefilled = len(prompt)

        with self._lock:
            cache = copy.deepcopy(self._prefix_cache) if reuse else make_prompt_cache(self._model)

            t0 = time.perf_counter()
            ttft = None
            n = 0
            for resp in stream_generate(
                self._model, self._tokenizer, prompt,
                max_tokens=max_tokens, prompt_cache=cache,
            ):
                if ttft is None:
                    ttft = round(time.perf_counter() - t0, 4)
                n += 1
                yield {"type": "delta", "text": resp.text}

            total = time.perf_counter() - t0
            yield {
                "type": "done",
                "ttft_s": ttft,
                "total_s": round(total, 4),
                "tokens": n,
                "decode_tps": round((n - 1) / max(total - (ttft or 0), 1e-6), 1) if n > 1 else 0.0,
                "prompt_tokens": len(full),
                "prefilled_tokens": prefilled,
                "prefix_reused": reuse,
            }
