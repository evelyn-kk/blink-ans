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

# T-029：Claude Opus 5（claude-opus-5）官方定价，2026-06-24 Anthropic 官方费率表
# （Current Models 表，$/MTok）。价格变动时先来这里核对更新，再看下面两个衍生倍率
# 是否仍然成立——费率表变了这两个常量不会自动跟着变。
PRICE_PER_MTOK_INPUT_USD = 5.00
PRICE_PER_MTOK_OUTPUT_USD = 25.00
# 缓存写入价 ≈ 输入价的 1.25 倍、缓存读取价 ≈ 输入价的 0.1 倍——这组倍率关系
# development-notes.md（T-026 附近）已记过一次，与官方费率表口径一致，直接复用。
PRICE_PER_MTOK_CACHE_WRITE_USD = PRICE_PER_MTOK_INPUT_USD * 1.25  # 6.25
PRICE_PER_MTOK_CACHE_READ_USD = PRICE_PER_MTOK_INPUT_USD * 0.1    # 0.50


def compute_cost_usd(
    *,
    prompt_tokens: int,
    output_tokens: int,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
) -> float:
    """按官方费率把这次云端请求的 token 用量换算成美元。

    口径：Anthropic 用量语义里 `usage.input_tokens`（这里的 prompt_tokens）本身就
    已经排除了缓存命中的部分——`cache_read_input_tokens`/`cache_creation_input_tokens`
    是分开计的两个桶，三者互不重叠，相加才是这次请求真正处理的总 token 数。
    因此这里直接按各自单价分别计费再相加，不需要做任何去重或扣减。

    本地后端没有这个函数对应的调用点——本地固定 cost_usd=0.0（自有硬件，摊销
    电费/硬件成本不计入这个指标，见 architecture.md/development-notes.md 里
    这项一贯的口径），不经过这里。

    保留 6 位小数：单次请求成本常在 $0.001 量级，4 位小数会把大多数请求截断成 0，
    抹掉埋点区分度；50 题回归汇总总成本时，6 位小数足够看出个位数美分级别的差异。
    """
    cost = (
        prompt_tokens / 1_000_000 * PRICE_PER_MTOK_INPUT_USD
        + output_tokens / 1_000_000 * PRICE_PER_MTOK_OUTPUT_USD
        + (cache_read_tokens or 0) / 1_000_000 * PRICE_PER_MTOK_CACHE_READ_USD
        + (cache_write_tokens or 0) / 1_000_000 * PRICE_PER_MTOK_CACHE_WRITE_USD
    )
    return round(cost, 6)


def probe_network_floor(timeout_s: float = 2.0) -> dict:
    """生产用的轻量版网络地板探测（/healthz 用，思路抄 bench/bench_llm_remote.py
    的 `probe_network_rtt()`，但只探一次，不做那份基准脚本的多次采样统计——
    `/healthz` 不需要那种精度，且探测本身已经是额外的一次真实网络往返，
    多探几次只会让 `/healthz` 更慢）。

    对 `api.anthropic.com` 做一次 TCP 连接 + TLS 握手，握手完立刻关闭连接。
    **不发送任何 HTTP 请求，不调用 `/v1/messages`，不花钱**——这一点是硬约束，
    不是这个函数手误的事：`/healthz` 可能被监控系统高频轮询，一旦这里改成
    发起真实模型请求，每次轮询都会计费。

    - `tcp_connect_s`：一次网络往返，稳态下每个请求要付的网络地板。
    - `tcp_tls_s`：TCP+TLS 完整握手，冷连接才付的一次性成本。

    网络不通/超时/证书问题等一律捕获，返回 `error` 字段而不是向上抛异常——
    调用方（`/healthz`）不能因为这一项测不出来就让整个响应跟着 500。

    CR-028：计时必须在 `ssl.create_default_context()`**之后**才开始——它要读本机
    证书库，耗时随平台/证书数量变化，混进计时会让"网络"数字实际测的是本机负载
    （复现：把 `create_default_context()` 延迟 50ms、socket/TLS 都设为立即返回，
    修复前 `tcp_connect_s`/`tcp_tls_s` 仍报 ≈0.05s）。`bench/bench_llm_remote.py`
    的 `probe_network_rtt()` 原本就是先建 context、循环内才计时，这里移植成单次
    探测时把 `t0` 错放到了 context 创建之前，属于移植引入的新回归，不是照抄旧代码
    的既有缺陷。
    """
    import socket
    import ssl

    host = CLAUDE_API_BASE_URL.split("://", 1)[-1].rstrip("/")
    try:
        ctx = ssl.create_default_context()
        t0 = time.perf_counter()
        with socket.create_connection((host, 443), timeout=timeout_s) as sock:
            tcp_connect_s = round(time.perf_counter() - t0, 4)
            with ctx.wrap_socket(sock, server_hostname=host):
                pass
        return {
            "host": host,
            "tcp_connect_s": tcp_connect_s,
            "tcp_tls_s": round(time.perf_counter() - t0, 4),
            "error": None,
        }
    except Exception as exc:
        return {
            "host": host,
            "tcp_connect_s": None,
            "tcp_tls_s": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


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
        cache_write = getattr(usage, "cache_creation_input_tokens", None)
        # 输出 token 数优先用 SDK 汇总的 usage.output_tokens（若有），比累加流式
        # delta 条数（n）更可靠——n 数的是「文本 delta 事件」条数，不是 token 数，
        # 二者在多字节/多 token 一个 delta 的情况下本就不该假定相等。usage 对象
        # 没有这个字段时（旧版 SDK）才退回 n 作为近似。
        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = n
        cost_usd = compute_cost_usd(
            prompt_tokens=usage.input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )
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
            "cache_write_tokens": cache_write,
            # T-029：云端场景才有真实美元成本，本地固定 0.0（见 compute_cost_usd 的说明）。
            "cost_usd": cost_usd,
        }
