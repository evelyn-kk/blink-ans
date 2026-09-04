"""常驻推理引擎。

设计要点来自 I0 基准（见 progress.md「I0 基准数据」）：

1. prefill 约 352 tok/s 且随上下文线性增长，是本机最硬的约束。
2. KV cache 预热后首 token 仅需 0.095s，比冷 prefill 快约 30 倍。

因此模型必须常驻单进程，且 KV cache 对象要能跨请求存活——这是投机式提前 prefill
（在用户说话期间就把证据 prefill 好）能够成立的前提，也是服务端选 Python 单进程的原因。
"""

from __future__ import annotations

import copy
import sys
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
        # CR-035：导入本身要和真正的模型加载用同一段 try/except——没有 Metal 设备
        # 的机器上 `import mlx_lm` 就会直接抛 ImportError，如果导入留在 try 之外，
        # 这个异常会一路冲出 load()，调用方（apps/gateway boot()、
        # tests/integration 的 module fixture）设计好的"读 status.error/
        # pytest.skip"退化路径根本没机会跑到，会变成启动崩溃或 pytest setup error
        # 而不是一个可观测、可测试的降级状态。
        t0 = time.perf_counter()
        try:
            from mlx_lm import load
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
        with self._lock:
            return self._warm_prefix_locked(system_prompt)

    def _warm_prefix_locked(self, system_prompt: str) -> int:
        """warm_prefix 的实现体。**调用方必须已持有 `_lock`。**

        拆出来是因为 `_restore_prefix()` 在 `stream()` 的锁内被调用，
        而它的回退路径需要重建前缀。`threading.Lock` 不可重入，
        直接调用公开的 `warm_prefix()` 会当场自锁死（CR-010）。
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

        cache = make_prompt_cache(self._model)
        self._model(mx.array(tokens)[None], cache=cache)
        mx.eval([c.state for c in cache])
        self._prefix_cache = cache
        self._prefix_tokens = tokens
        self.status.prefix_tokens = len(tokens)
        return len(tokens)

    def count_tokens(self, text: str) -> int:
        """用真实分词器计数。

        切块的 token_estimate 是不加载分词器的粗略估计，对英文 P90 低估约 16%，
        用它控制上下文预算会让首 token 时延的尾部失控。
        分词几百 token 只需毫秒级，没有理由继续估算。
        """
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def _restore_prefix(self, cache) -> None:
        """把缓存裁剪回常驻前缀长度。**调用方已持有 `_lock`**（见 stream 的 finally）。

        裁剪失败时（模型的 cache 类型不支持）退回重建前缀——
        宁可慢一次，也不能让上一次请求的内容泄漏到下一次。
        """
        from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

        target = len(self._prefix_tokens)
        try:
            extra = cache[0].offset - target
            if extra > 0 and can_trim_prompt_cache(cache):
                trim_prompt_cache(cache, extra)
            if cache[0].offset == target:
                return
        except Exception:
            pass

        # 走到这里说明裁剪没能还原缓存，只能丢弃重建。
        # 丢弃这一步必须先做完：即使重建失败，也不能留着带上一次请求内容的缓存，
        # 否则下一次请求会串话——串话比慢一次严重得多。
        self._prefix_cache = None
        self._prefix_tokens = []
        self.status.prefix_tokens = 0
        if not self._system_prompt:
            return
        try:
            # 已在锁内，必须走不重复加锁的实现体
            self._warm_prefix_locked(self._system_prompt)
        except Exception as exc:
            # 这里是 stream() 的 finally，客户端断流时还会带着 GeneratorExit。
            # 让重建异常盖掉原本的退出原因只会更难排查，因此就地降级：
            # 前缀归零，下一次请求走完整 prefill，仅仅是慢，不会错。
            # status.prefix_tokens 归零使降级在 /healthz 上可见，不是静默失败。
            print(f"[engine] 前缀缓存重建失败，已降级为无前缀: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)

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
        self,
        user_content: str,
        max_tokens: int = 512,
        use_prefix: bool = True,
        system_override: str | None = None,
    ) -> Iterator[dict]:
        """流式生成。产出 {"type": "delta"|"done", ...}。

        传入的是**用户内容**（证据 + 问题），chat template 由引擎内部套用。
        use_prefix=True 时复用常驻的系统前缀 KV，只 prefill 变化部分。
        """
        # 先判状态再导入：未加载时的错误路径不应依赖 Metal/MLX，
        # 否则在没有 GPU 的环境里这条路径会抛 ImportError 而非 RuntimeError，
        # 纯逻辑测试也就无法在 CI 上作为门禁运行。
        if not self.status.loaded:
            raise RuntimeError("模型尚未加载")

        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache

        # 覆盖系统提示词时前缀必然不匹配，直接放弃复用。
        # 证据不足分支用的是另一套短提示词，占比很小，不值得为它再驻留一份 KV。
        system = system_override if system_override is not None else self._system_prompt
        full = self._tokenizer.encode(self._render(system, user_content))
        reuse = (
            use_prefix
            and system_override is None
            and self._prefix_cache is not None
            and full[: len(self._prefix_tokens)] == self._prefix_tokens
        )
        prompt = full[len(self._prefix_tokens):] if reuse else full
        prefilled = len(prompt)

        with self._lock:
            # 不拷贝常驻前缀，用完裁剪回去。
            # deepcopy 一份 322 token 的 KV 约 47MB，实测吃掉约 0.2 秒——
            # 占 2.5 秒首 token 预算的 8%，而生成本就在锁内串行，无需副本。
            trimmed = False
            if reuse:
                cache = self._prefix_cache
                trimmed = True
            else:
                cache = make_prompt_cache(self._model)

            t0 = time.perf_counter()
            ttft = None
            n = 0
            try:
                for resp in stream_generate(
                    self._model, self._tokenizer, prompt,
                    max_tokens=max_tokens, prompt_cache=cache,
                ):
                    if ttft is None:
                        ttft = round(time.perf_counter() - t0, 4)
                    n += 1
                    yield {"type": "delta", "text": resp.text}
            finally:
                # 无论正常结束还是中途异常/断流，都必须把缓存还原为纯前缀，
                # 否则下一次请求会带着上一次的证据与答案，产生串话。
                if trimmed:
                    self._restore_prefix(cache)

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
