"""生成路由：云端优先，断网/失败/项目禁云端/离线模式时切本地兜底。

策略已在 architecture.md §6.4 定案，这里只是把它写成代码：

- **主路径：云端 Claude。** 默认生成后端。
- **本地是兜底，不是次优主力。** 触发条件三选一即切本地：
  1. 本轮命中证据里有项目材料显式 `cloud_generation_allowed=False`；
  2. 云端不可用（缺凭据）或请求失败/超时；
  3. 显式离线模式（见 `Router.offline`）。
- **失败切换，不做 deadline 对冲**（§6.4 第4条：T-026 未发现 P95 单独超标，
  对冲是纯成本无收益）。
- **不拼接半句云端半句本地。** 云端流已经吐出过正文后再失败，不能悄悄切本地
  重新生成——那样用户会看到"半句云端腔调 + 半句本地续写"缝在一起，语气和
  引用风格都可能对不上。这种情况下改为吐一个 `error` 事件、终止这次生成，
  而不是在没有失败迹象之前吐出的文字之后硬接别的模型。只有在云端**还没吐出
  任何正文**时失败，才符合"直接换后端重新走一遍生成"的条件——此时相当于
  请求还没真正开始过，重新来一遍不会有拼接问题。这也是最常见的真实断网场景
  （连接失败发生在拿到第一个字节之前）。

`served_by` 由这一层产出并写入每个请求的 `done` 事件与 stderr 日志——
architecture.md §7："`done` 必须包含 `served_by`...后端选择是服务端策略，
不暴露为客户端模型选择"。
"""

from __future__ import annotations

import sys
from typing import Iterator, Optional

from services.inference.backend import GenerationBackend, LocalBackend


class Router:
    def __init__(
        self,
        local: LocalBackend,
        cloud: Optional[GenerationBackend] = None,
        *,
        offline: bool = False,
    ) -> None:
        self.local = local
        self.cloud = cloud
        # 显式离线模式开关。当前唯一入口：调用方在构造 Router 时传入，
        # 或直接翻转这个属性（可变，非 frozen）——例如 `apps/gateway/main.py`
        # 读环境变量 `BLINK_OFFLINE=1` 决定初始值。没有做成 AnswerConfig 里的
        # 逐请求字段：离线是"这台机器/这次会话没有网络"这一档的判断，不是
        # 单个问题的属性，放在路由的进程级状态上比每次请求都传更贴近语义；
        # 未来若要做成运行时可切换的管理开关（比如加一个 /v1/offline 接口），
        # 直接改这个属性即可，不需要改调用方。
        self.offline = offline

    def count_tokens(self, text: str) -> int:
        """委托给本地后端的真实分词器。见 `LocalBackend.count_tokens` 的说明。"""
        return self.local.count_tokens(text)

    def generate(
        self,
        user_content: str,
        *,
        max_tokens: int,
        system_override: str | None = None,
        cloud_allowed: bool = True,
    ) -> Iterator[dict]:
        """产出事件流，`done` 事件带 `served_by`（"claude" 或 "local"）。

        cloud_allowed=False 强制走本地——调用方（Orchestrator）在本轮命中证据
        含 `cloud_generation_allowed=False` 时应该传 False。
        """
        use_cloud = (
            not self.offline
            and cloud_allowed
            and self.cloud is not None
            and self.cloud.available()
        )
        if use_cloud:
            yield from self._try_cloud(
                user_content, max_tokens=max_tokens, system_override=system_override
            )
        else:
            yield from self._local(
                user_content, max_tokens=max_tokens, system_override=system_override
            )

    def _try_cloud(
        self, user_content: str, *, max_tokens: int, system_override: str | None
    ) -> Iterator[dict]:
        started_delta = False
        try:
            for ev in self.cloud.stream(
                user_content, max_tokens=max_tokens, system_override=system_override
            ):
                if ev["type"] == "delta":
                    started_delta = True
                if ev["type"] == "done":
                    ev = {**ev, "served_by": self.cloud.name}
                yield ev
                if ev["type"] == "done":
                    print(f"[router] served_by={self.cloud.name}", file=sys.stderr)
                    return
            return
        except Exception as exc:
            print(
                f"[router] 云端生成异常（已吐出正文={started_delta}）: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if started_delta:
                # 已经流出去的文字不能悄悄接别的模型续写，见模块 docstring。
                yield {
                    "type": "error",
                    "stage": "generation_cloud_midstream",
                    "message": f"{type(exc).__name__}: {exc}",
                }
                return
            # 云端还没吐出任何正文就失败——等价于"请求还没真正开始过"，
            # 直接换本地重新走一遍生成是安全的。
            print("[router] 云端未吐出任何正文即失败，降级本地重新生成", file=sys.stderr)

        yield from self._local(
            user_content, max_tokens=max_tokens, system_override=system_override
        )

    def _local(
        self, user_content: str, *, max_tokens: int, system_override: str | None
    ) -> Iterator[dict]:
        for ev in self.local.stream(
            user_content, max_tokens=max_tokens, system_override=system_override
        ):
            if ev["type"] == "done":
                ev = {**ev, "served_by": self.local.name}
            yield ev
        print(f"[router] served_by={self.local.name}", file=sys.stderr)
