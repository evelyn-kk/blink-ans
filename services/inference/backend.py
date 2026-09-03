"""生成后端契约。

T-028：路由要在多个生成实现之间切换（云端 Claude / 本地 Qwen3-4B），
编排层（`services/orchestrator/answering.py`）不应该知道自己在跟哪一个打交道。
这里把 `InferenceEngine.stream()` 已经在用的事件形状固化成一份显式契约。

选 `Protocol` 而不是 ABC：本仓库的测试已经在用纯 duck typing 构造假引擎
（见 `tests/unit/test_answering.py` 的 `FakeEngine`），要求测试双测显式继承一个
基类只会增加样板、不增加安全性——Python 生成器/流式接口本来就是结构化类型
更自然的场景。`runtime_checkable` 使 `isinstance` 检查在需要时仍然可用。

事件形状（与 `InferenceEngine.stream()` 保持兼容，`Orchestrator._generate()`
按这个形状透传）：
    {"type": "delta", "text": str}
    {"type": "done", "ttft_s": float, "total_s": float, "tokens": int,
     "decode_tps": float, "prompt_tokens": int, "prefilled_tokens": int,
     "prefix_reused": bool, ...可扩展字段...}
`served_by` **不**由后端自己产出——它是路由层（`services/inference/router.py`）
的决策结果，由路由在转发 `done` 事件时附加，后端不应该知道自己被起了什么名字
之外的事（backend 只声明 `name`，语义解释权在路由）。
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from services.inference.engine import InferenceEngine


@runtime_checkable
class GenerationBackend(Protocol):
    """任何生成后端都要满足的最小接口。"""

    name: str

    def available(self) -> bool:
        """当前是否可用。必须是纯判断，不发起真实请求——

        路由用它决定要不要尝试这个后端；如果“判断可用”本身要发请求
        （比如探测云端连通性），失败模式就绕回了“异常冒泡”这条要避免的路，
        且会在每个请求上多付一次往返。云端后端应该只检查凭据是否配置。
        """
        ...

    def stream(
        self, user_content: str, *, max_tokens: int, system_override: str | None = None
    ) -> Iterator[dict]:
        """流式生成。产出 delta/done（形状见模块docstring）。"""
        ...


class LocalBackend:
    """把现有 `InferenceEngine` 适配成 `GenerationBackend`。

    改动面最小：不改 `InferenceEngine` 本身（它的 `stream()` 签名已经满足协议，
    只是参数顺序不同），只加一层薄适配把 `max_tokens`/`system_override` 转成
    关键字调用，并暴露 `available()`/`count_tokens()` 给路由和编排层用。
    """

    name = "local"

    def __init__(self, engine: InferenceEngine) -> None:
        self._engine = engine

    def available(self) -> bool:
        return self._engine.status.loaded

    def count_tokens(self, text: str) -> int:
        """证据预算用真实分词器计数（见 orchestrator/answering.py）。

        这个方法留在本地后端上而不是路由/编排层里另起一份，是因为 token 预算
        本质上是本地 prefill 时延约束的产物（architecture.md §6.6 第2条）：
        云端路径不受本机算力线性约束，但证据选取逻辑目前仍统一按本地分词器
        计数——即便未来云端成为主路径，评估证据体量仍需要一把稳定的尺子。
        """
        return self._engine.count_tokens(text)

    def stream(
        self, user_content: str, *, max_tokens: int, system_override: str | None = None
    ) -> Iterator[dict]:
        yield from self._engine.stream(
            user_content, max_tokens=max_tokens, system_override=system_override
        )
