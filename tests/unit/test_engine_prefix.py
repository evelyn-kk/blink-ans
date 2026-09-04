"""前缀 KV 缓存还原的纯逻辑回归（CR-010）。

`stream()` 在 `with self._lock` 内的 finally 里调用 `_restore_prefix()`；
裁剪失败的回退路径要重建前缀，而 `threading.Lock` **不可重入**——
旧实现在这里调用会再次加锁的公开 `warm_prefix()`，必然当场自锁：
生成线程永远不返回，引擎锁永远不释放，之后的每个请求都被堵死。

本文件用假的 mlx / mlx_lm 模块跑真实控制流，因此**不需要 Metal**，
可以进快速门禁（CR-002 要求纯逻辑测试能作 CI 门禁）。
每条测试都带超时断言：自锁的表现是挂住，不是报错。
"""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.inference.engine import InferenceEngine  # noqa: E402

SYSTEM = "你是一名 Java 后端工程师，依据证据作答。"
QUESTION = "Kafka 消费者重复消费的常见原因是什么？"
TIMEOUT_S = 5.0


class _Arr:
    """mx.array 的替身：只需支持 `[None]` 与取长度。"""

    def __init__(self, data): self.data = list(data)
    def __getitem__(self, _): return self


class _Cache:
    """mlx_lm 的 prompt cache 替身，只保留本测试关心的 offset。"""

    def __init__(self): self.offset = 0
    @property
    def state(self): return None


class _Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]

    def apply_chat_template(self, msgs, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + "<assistant>"


def _install_fake_mlx(monkeypatch, *, can_trim: bool,
                      trim_raises: bool = False, warm_raises: bool = False) -> None:
    mx = types.ModuleType("mlx.core")
    mx.array = _Arr
    mx.eval = lambda _: None
    mlx = types.ModuleType("mlx")
    mlx.core = mx

    cache_mod = types.ModuleType("mlx_lm.models.cache")

    def make_prompt_cache(model):
        if warm_raises:
            raise RuntimeError("Metal 设备不可用")
        return [_Cache()]

    def trim_prompt_cache(cache, n):
        if trim_raises:
            raise RuntimeError("这个 cache 类型不支持裁剪")
        cache[0].offset -= n

    cache_mod.make_prompt_cache = make_prompt_cache
    cache_mod.can_trim_prompt_cache = lambda _: can_trim
    cache_mod.trim_prompt_cache = trim_prompt_cache

    models_mod = types.ModuleType("mlx_lm.models")
    models_mod.cache = cache_mod

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.models = models_mod

    def stream_generate(model, tok, prompt, max_tokens=0, prompt_cache=None):
        prompt_cache[0].offset += len(prompt)        # prefill
        for i in range(3):
            prompt_cache[0].offset += 1              # 每 decode 一个 token 前进一位
            yield types.SimpleNamespace(text=f"t{i}")

    mlx_lm.stream_generate = stream_generate

    for name, mod in (("mlx", mlx), ("mlx.core", mx), ("mlx_lm", mlx_lm),
                      ("mlx_lm.models", models_mod), ("mlx_lm.models.cache", cache_mod)):
        monkeypatch.setitem(sys.modules, name, mod)


def _engine(monkeypatch, **fake) -> InferenceEngine:
    _install_fake_mlx(monkeypatch, **fake)
    eng = InferenceEngine("fake-model")
    # 假模型：被调用时按输入长度推进 cache，与真实 prefill 的行为一致
    def model(arr, cache=None):
        if cache is not None:
            cache[0].offset += len(arr.data)
    eng._model = model
    eng._tokenizer = _Tokenizer()
    eng.status.loaded = True
    eng.warm_prefix(SYSTEM)
    return eng


def _stream_with_timeout(eng: InferenceEngine, question: str = QUESTION) -> list[dict]:
    """在子线程里跑完一次生成。超时即判定自锁——自锁的表现是挂住而非报错。"""
    out: list[dict] = []
    err: list[BaseException] = []

    def run():
        try:
            out.extend(eng.stream(question, max_tokens=8))
        except BaseException as exc:      # noqa: BLE001 — 要把异常带回主线程
            err.append(exc)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(TIMEOUT_S)
    assert not t.is_alive(), f"stream() {TIMEOUT_S}s 未返回，疑似在锁内重复加锁自锁"
    if err:
        raise err[0]
    return out


def test_warm_prefix_takes_the_lock_and_gives_it_back(monkeypatch):
    eng = _engine(monkeypatch, can_trim=True)
    assert eng.status.prefix_tokens == len(eng._prefix_tokens) > 0
    assert eng._lock.acquire(timeout=1), "warm_prefix 未释放锁"
    eng._lock.release()


def test_trim_path_reuses_the_resident_cache(monkeypatch):
    """能裁剪时走快路径：缓存对象原地复用并还原为纯前缀，不该重建。"""
    eng = _engine(monkeypatch, can_trim=True)
    before = eng._prefix_cache
    _stream_with_timeout(eng)
    assert eng._prefix_cache is before, "可裁剪时不应重建前缀"
    assert eng._prefix_cache[0].offset == len(eng._prefix_tokens), "缓存未还原为纯前缀"


def test_rebuild_fallback_does_not_deadlock_when_trim_unsupported(monkeypatch):
    """CR-010 主场景：cache 类型不支持裁剪 → 锁内重建前缀。"""
    eng = _engine(monkeypatch, can_trim=False)
    events = _stream_with_timeout(eng)
    assert [e["type"] for e in events][-1] == "done"
    assert eng._prefix_cache is not None, "回退路径未重建前缀"
    assert eng._prefix_cache[0].offset == len(eng._prefix_tokens)


def test_rebuild_fallback_does_not_deadlock_when_trim_raises(monkeypatch):
    """CR-010 点名的另一半：声称可裁剪、真裁剪时抛错。"""
    eng = _engine(monkeypatch, can_trim=True, trim_raises=True)
    _stream_with_timeout(eng)
    assert eng._prefix_cache is not None
    assert eng._prefix_cache[0].offset == len(eng._prefix_tokens)


def test_engine_lock_is_released_after_rebuild(monkeypatch):
    """自锁的实际危害是后续请求全被堵死，因此显式断言锁已归还。"""
    eng = _engine(monkeypatch, can_trim=False)
    _stream_with_timeout(eng)
    assert eng._lock.acquire(timeout=1), "生成结束后引擎锁未释放，后续请求会被堵死"
    eng._lock.release()
    # 再跑一次必须照样通得过：真自锁时第二次会直接挂住
    _stream_with_timeout(eng, "另一个完全不同的问题")


def test_prefix_is_dropped_before_rebuild_is_attempted(monkeypatch):
    """重建失败时宁可没有前缀，也不能留着带上一次请求内容的缓存。

    串话（下一次请求读到上一次的证据与答案）比慢一次严重得多；
    prefix_tokens 归零让这次降级在 /healthz 上看得见，不是静默失败。
    """
    eng = _engine(monkeypatch, can_trim=False)
    _install_fake_mlx(monkeypatch, can_trim=False, warm_raises=True)
    events = _stream_with_timeout(eng)
    assert [e["type"] for e in events][-1] == "done", "重建失败不该打断本次回答"
    assert eng._prefix_cache is None
    assert eng.status.prefix_tokens == 0


def test_stream_before_load_raises_runtime_error_without_mlx():
    """CR-002 的门禁性质：未加载时的错误路径不得依赖 Metal/MLX。"""
    with pytest.raises(RuntimeError, match="尚未加载"):
        next(InferenceEngine("fake-model").stream("测试"))


# ---------- CR-035：load() 里 mlx_lm 导入失败必须落进 status.error，不能裸抛 ----------

def test_load_captures_mlx_lm_import_failure_instead_of_raising(monkeypatch):
    """在没有 Metal 设备的机器上，`import mlx_lm` 本身就会抛 `ImportError`——
    这条判别性回归不依赖真实缺 Metal，而是用 `sys.modules["mlx_lm"] = None`
    这个标准手法强制下一次 `import mlx_lm`/`from mlx_lm import ...` 抛
    `ImportError`（见 Python 官方文档：模块名在 `sys.modules` 里映射到 `None`
    时触发这个行为），在任何机器上都能确定性复现。

    旧实现（`from mlx_lm import load` 在 `try` 之外）会让这个异常直接冲出
    `load()`；调用方（`apps/gateway` 的 `boot()`、
    `tests/integration/test_prefix_reuse.py` 的 module fixture）设计好的
    "读 status.error/pytest.skip" 退化路径根本没机会跑到。修复后 `load()`
    必须正常返回，并把这次导入失败写进 `status.error`。
    """
    monkeypatch.setitem(sys.modules, "mlx_lm", None)

    eng = InferenceEngine("fake-model")
    eng.load()  # 不能抛

    assert eng.status.loaded is False
    assert eng.status.error is not None
    assert "mlx_lm" in eng.status.error or "ImportError" in eng.status.error
