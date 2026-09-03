"""云端 Claude 生成后端（T-028，路由策略见 architecture.md §6.4）。

主路径：默认生成后端，覆盖 `cloud_generation_allowed=true` 的全部场景
（T-026 四档 P95 首字 2.79–3.54s，落在 §6.5 的 3.6 秒生成子预算内）。

调用方式抄 `bench/bench_llm_remote.py` 的 `_claude_client()`/`measure_claude()`
（T-026 的实测参考实现），但这里是生产代码，多了三件那份基准脚本不需要管的事：

1. **凭据缺失必须是"不可用"，不是异常。** `available()` 只查环境变量，
   不发请求；路由据此决定要不要连本地都不试就直接跳过云端。
2. **请求要有硬超时，且不能被 SDK 默认重试拖垮预算。** 生成阶段云端子预算是
   3.6 秒（architecture.md §6.5）；SDK 默认 `max_retries=2` 且指数退避，
   一次超时可能被悄悄重试成三次，总耗时远超预算却毫无察觉——因此这里
   显式把 `max_retries` 归零，超时/失败必须立刻交回给路由做决定。
3. **任何异常都不在这里兜底，原样向上抛。** 由 `Router` 统一捕获并降级到
   本地（`services/inference/router.py`），这里兜底会让路由看不到失败发生过。
"""

from __future__ import annotations

import os
import time
from typing import Iterator

DEFAULT_MODEL = "claude-opus-5"  # T-026 实测所用型号，见 bench/reports/llm-remote-*.json
CLAUDE_API_BASE_URL = "https://api.anthropic.com"
DEFAULT_TIMEOUT_S = 3.6  # architecture.md §6.5：generation_started→first_answer_text 云端子预算


class ClaudeBackend:
    name = "claude"

    def __init__(
        self,
        system_prompt: str,
        model: str = DEFAULT_MODEL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        effort: str = "low",
    ) -> None:
        self._system_prompt = system_prompt
        self._model = model
        self._timeout_s = timeout_s
        # effort="low" 是有意的（T-026 的选择，非本轮新决策）：3.6 秒预算下
        # 打开思考会把首字时延预算吃在用户看不到的地方。
        self._effort = effort

    def available(self) -> bool:
        """缺 `ANTHROPIC_API_KEY` 时判为不可用。

        不在这里报错——路由把"不可用"当正常输入处理（跳过云端走本地），
        而不是当异常捕获，语义更清楚也少一层 try/except。
        """
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _client(self):
        import anthropic  # 惰性导入：没装 anthropic 或没配凭据的环境不该在这里报错

        # 显式 base_url + 显式 api_key：不继承同机可能配置的 Kimi 兼容网关环境变量
        # （ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN）——T-109/CR 修过的真实坑，
        # 静默继承会把"Claude 请求"实际打到另一家端点。
        return anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url=CLAUDE_API_BASE_URL,
            timeout=self._timeout_s,
            max_retries=0,
        )

    def stream(
        self, user_content: str, *, max_tokens: int, system_override: str | None = None
    ) -> Iterator[dict]:
        """流式生成。异常（超时/连接失败/API 错误）原样向上抛，由 Router 捕获降级。"""
        system = system_override if system_override is not None else self._system_prompt
        client = self._client()

        t0 = time.perf_counter()
        ttft = None
        n = 0

        # 缓存断点打在 system 与整段 user 内容末尾（system+证据+问题合一条消息）。
        # 证据正文逐轮不同，缓存是否稳定命中不重要（T-026 的 cold 值本就达标，
        # 见 architecture.md §6.4）——断点打上不吃亏，命中就是额外收益。
        with client.messages.stream(
            model=self._model,
            max_tokens=max_tokens,
            output_config={"effort": self._effort},
            system=[{
                "type": "text", "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": [
                {"type": "text", "text": user_content,
                 "cache_control": {"type": "ephemeral"}},
            ]}],
        ) as stream:
            for ev in stream:
                if ev.type != "content_block_delta":
                    continue
                delta = ev.delta
                # thinking_delta 在 effort=low 下被压到最小，不单独统计——
                # 产品指标是用户看到第一个正文字符的时间，不是首个事件（T-026）。
                if delta.type == "text_delta":
                    if ttft is None:
                        ttft = round(time.perf_counter() - t0, 4)
                    n += 1
                    yield {"type": "delta", "text": delta.text}
            final = stream.get_final_message()

        total = time.perf_counter() - t0
        usage = final.usage
        cache_read = getattr(usage, "cache_read_input_tokens", None)
        yield {
            "type": "done",
            # 整轮没吐出正文（全被截断/被思考吃掉）时，ttft 记 total——
            # 产品视角下"一个字都没等到"等价于等满全程（同 bench_llm_remote._pack）。
            "ttft_s": ttft if ttft is not None else round(total, 4),
            "total_s": round(total, 4),
            "tokens": n,
            "decode_tps": round((n - 1) / max(total - (ttft or 0), 1e-6), 1) if n > 1 else 0.0,
            "prompt_tokens": usage.input_tokens,
            # 云端没有本地那种"前缀 KV 复用"概念；借用同一形状表达类比语义：
            # prefilled_tokens 记全部输入 token（云端不做分段 prefill），
            # prefix_reused 记 prompt cache 是否命中——两者都是"这次省了多少活儿"
            # 的近似，不是同一件事的精确对应，Orchestrator._generate() 只透传数值。
            "prefilled_tokens": usage.input_tokens,
            "prefix_reused": bool(cache_read),
            # 拿不到的数据不硬造：Claude API 没有这两个字段的等价物就不填。
            "cache_read_tokens": cache_read,
            "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", None),
        }
