"""路由决策的回归（T-028，策略见 architecture.md §6.4）。

全部用假后端驱动——**不发起任何真实网络请求**，这是任务对本轮测试的硬约束
（会花钱、需要密钥、CI 环境未必有网）。假后端只需要满足
`services.inference.backend.GenerationBackend` 的最小接口：`name` / `available()`
/ `stream()`，与真实的 `ClaudeBackend`/`LocalBackend` 结构一致但不触碰网络或 MLX。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.inference.router import Router  # noqa: E402


def _done(**overrides) -> dict:
    base = {
        "type": "done", "ttft_s": 0.5, "total_s": 0.8, "tokens": 3,
        "decode_tps": 10.0, "prompt_tokens": 20, "prefilled_tokens": 20,
        "prefix_reused": False,
    }
    base.update(overrides)
    return base


class ScriptedBackend:
    """按脚本吐事件的假后端；`events` 是一串 dict 或异常类。"""

    def __init__(self, name: str, events: list, available: bool = True) -> None:
        self.name = name
        self._events = events
        self._available = available
        self.call_count = 0

    def available(self) -> bool:
        return self._available

    def stream(self, user_content, *, max_tokens, system_override=None):
        self.call_count += 1
        for ev in self._events:
            if isinstance(ev, type) and issubclass(ev, BaseException):
                raise ev("模拟故障")
            yield ev


# ---------- 正常路径 ----------

def test_cloud_success_reports_served_by_claude_and_never_touches_local():
    cloud = ScriptedBackend("claude", [{"type": "delta", "text": "答"}, _done()])
    local = ScriptedBackend("local", [{"type": "delta", "text": "本地答案"}, _done()])
    router = Router(local, cloud)

    events = list(router.generate("问题", max_tokens=100))

    assert local.call_count == 0
    done = next(e for e in events if e["type"] == "done")
    assert done["served_by"] == "claude"


def test_no_cloud_backend_configured_goes_straight_to_local():
    local = ScriptedBackend("local", [{"type": "delta", "text": "本地答案"}, _done()])
    router = Router(local, cloud=None)

    events = list(router.generate("问题", max_tokens=100))

    done = next(e for e in events if e["type"] == "done")
    assert done["served_by"] == "local"


def test_cloud_unavailable_skips_straight_to_local_without_calling_cloud_stream():
    """缺凭据时 available() 返回 False——路由不应该调用 stream() 后再处理异常。"""
    cloud = ScriptedBackend("claude", [RuntimeError], available=False)
    local = ScriptedBackend("local", [{"type": "delta", "text": "本地答案"}, _done()])
    router = Router(local, cloud)

    events = list(router.generate("问题", max_tokens=100))

    assert cloud.call_count == 0
    done = next(e for e in events if e["type"] == "done")
    assert done["served_by"] == "local"


# ---------- 断网 / 失败降级（T-028 要求的核心场景）----------

def test_cloud_connection_error_before_any_delta_falls_back_to_local():
    """典型断网场景：连接失败发生在拿到第一个字节之前，重新走一遍生成是安全的。"""
    cloud = ScriptedBackend("claude", [ConnectionError])
    local = ScriptedBackend("local", [{"type": "delta", "text": "本地答案"}, _done()])
    router = Router(local, cloud)

    events = list(router.generate("问题", max_tokens=100))

    assert local.call_count == 1
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert deltas == ["本地答案"]
    done = next(e for e in events if e["type"] == "done")
    assert done["served_by"] == "local"


def test_cloud_timeout_before_any_delta_falls_back_to_local():
    """超时是另一条常见失败路径，同样必须能被路由捕获并降级，不能让请求整体失败。"""
    class FakeTimeout(TimeoutError):
        pass

    cloud = ScriptedBackend("claude", [FakeTimeout])
    local = ScriptedBackend("local", [{"type": "delta", "text": "本地答案"}, _done()])
    router = Router(local, cloud)

    events = list(router.generate("问题", max_tokens=100))

    done = next(e for e in events if e["type"] == "done")
    assert done["served_by"] == "local"
    assert not any(e["type"] == "error" for e in events)


def test_cloud_midstream_failure_does_not_splice_local_continuation():
    """云端已经吐出正文后失败：不得悄悄接本地续写（半句云端半句本地），

    而是终止并交出一个 error 事件——本地后端完全不应该被调用。
    """
    cloud = ScriptedBackend("claude", [{"type": "delta", "text": "云端开了个头"}, RuntimeError])
    local = ScriptedBackend("local", [{"type": "delta", "text": "本地答案"}, _done()])
    router = Router(local, cloud)

    events = list(router.generate("问题", max_tokens=100))

    assert local.call_count == 0
    assert [e["text"] for e in events if e["type"] == "delta"] == ["云端开了个头"]
    assert events[-1]["type"] == "error"
    assert events[-1]["stage"] == "generation_cloud_midstream"
    assert not any(e["type"] == "done" for e in events)


# ---------- 强制本地的两个触发条件 ----------

def test_cloud_generation_allowed_false_forces_local_even_if_cloud_available():
    cloud = ScriptedBackend("claude", [{"type": "delta", "text": "云端答案"}, _done()])
    local = ScriptedBackend("local", [{"type": "delta", "text": "本地答案"}, _done()])
    router = Router(local, cloud)

    events = list(router.generate("问题", max_tokens=100, cloud_allowed=False))

    assert cloud.call_count == 0
    done = next(e for e in events if e["type"] == "done")
    assert done["served_by"] == "local"


def test_offline_mode_forces_local_even_if_cloud_available_and_allowed():
    cloud = ScriptedBackend("claude", [{"type": "delta", "text": "云端答案"}, _done()])
    local = ScriptedBackend("local", [{"type": "delta", "text": "本地答案"}, _done()])
    router = Router(local, cloud, offline=True)

    events = list(router.generate("问题", max_tokens=100, cloud_allowed=True))

    assert cloud.call_count == 0
    done = next(e for e in events if e["type"] == "done")
    assert done["served_by"] == "local"


# ---------- count_tokens 委托 ----------

def test_count_tokens_delegates_to_local_backend():
    class CountingLocal(ScriptedBackend):
        def count_tokens(self, text):
            return len(text)

    local = CountingLocal("local", [])
    router = Router(local, cloud=None)
    assert router.count_tokens("abcd") == 4
