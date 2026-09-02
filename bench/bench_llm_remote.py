"""远端生成模型基准：与 bench_llm.py 用同一把尺子，量「用户多久看到第一个字」。

本脚本存在的唯一理由，是 I0 那条结论只对 M4 成立：
「prefill 352 tok/s 线性增长，2000 token 上下文即需 6s，物理上不可能满足 5s 目标」。
换成云端推理，prefill 不再是瓶颈，但**网络成了新的瓶颈，而且方差大得多**。
究竟哪边快，是个实证问题，不该在文档里靠估算争论——I0 当初就是这样把 8B 淘汰掉的。

三条与本地基准不同的设计，都是被云端的特性逼出来的：

1. **测 P95，不只测中位数。** 本地时延由算力决定，重复三次就够稳；
   云端时延由网络抖动和对方排队决定，中位数好看而 P95 崩掉是常态。
   architecture.md 第 6 节的预算是按 P95 写的，所以这里必须出 P95。

2. **区分「首个事件」和「首个正文字符」。** 思考型模型会先流式吐 thinking，
   用户在屏幕上看到字的时间才是产品指标。只测首包会得到一个漂亮但无意义的数字。

3. **把「网络」和「最小请求」分开测**（CR-017）。早先这里只有一个叫
   「RTT 地板」的指标，实际发的却是一次完整的流式生成——里面含服务端排队、
   最小 prompt 的处理和首字生成，根本不是网络往返。现在拆成两个：
   - `network_rtt`：只做 TCP 连接 + TLS 握手，是**真正的网络层地板**；
   - `min_request`：最小请求的端到端 TTFT，含排队与最小生成成本。
   两者之差才是「服务端固定开销」。混成一个数，看到 P95 超标也分不清
   该换网络还是该换供应商。

4. **显式验证 prompt cache 是否命中**（CR-014）。缓存不命中**不报错**，
   只是变慢变贵；不验证就用报告决定路由，等于拿一个没测过的假设当结论。
   做法是两阶段断言：首轮必须有写入，后续轮必须有读取，否则该档位的
   缓存结论标记为不可用。

**上下文分档的口径**：各家分词器不同，固定 token 数没有可比性；
固定**同一段文本**才有。因此证据正文由 bench_llm.py 的同一份中文技术语料、
按同一个 Qwen 分词器切到目标档位生成，逐字相同地发给每一家。

但**不能说「逐字相同的提示词」**（CR-017）：本地基准会再套一层 Qwen chat
template，各家 API 也各自拼自己的模板，模板 token 不在我们手里。
可比的只有**证据正文**这一段；各家自报的 prompt_tokens 一并记进报告，
差值即为模板与分词器带来的膨胀，做结论时必须先把它剥掉。

用法:
    # 只测网络与最小请求，最省钱，先跑这个确认链路可用
    python bench/bench_llm_remote.py --providers claude --rtt-only

    # 完整分档
    python bench/bench_llm_remote.py --providers claude kimi
    python bench/bench_llm_remote.py --providers claude --models claude-haiku-4-5 claude-sonnet-5

凭据从环境变量读，缺哪家就跳过哪家，不影响其余：
    ANTHROPIC_API_KEY   Claude
    MOONSHOT_API_KEY    Kimi（月之暗面）
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import Any, Callable

from bench_llm import FILLER  # 语料必须与本地基准同源，否则两组数字不可比
from common import write_report

TOKENIZER_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

# 稳定前缀。生产里对应 packages/prompts/answer.py 的系统提示词（实测 341 token）。
# 这里刻意复制一段等长的稳定文本而不是 import：bench/ 不依赖 packages/，
# 且提示词马上要因 T-022 重写，基准不该跟着它变。
#
# **一条必须记住的约束**：各家的最小可缓存前缀是 512–4096 token 且随模型而变，
# 短于它会**静默不缓存**。341 token 的系统提示词很可能整个落在门槛以下——
# 也就是说，生产里那条"稳定前缀"可能根本不产生缓存收益。
# 本脚本把断点同时打在 system 和证据末尾，就是为了把这两种情况分开看：
# 若只有证据那个断点有读取量，说明系统提示词太短没进缓存。
SYSTEM_PREFIX = (
    "你是 Java / Spring 云原生方向的资深后端工程师，回答生产环境问题。"
    "只依据给出的证据作答；证据未提及的内容一律说明未涵盖，不得凭经验补充。"
    "不得编造配置项名称、方法签名、指标名或版本号。"
    "每一条技术论断后必须紧跟证据编号。涉及生产变更时必须说明回滚方式。"
    "不同大版本的配置不得混用。默认先给结论，再按需补充必要步骤与关键风险。"
) * 2

# 与 bench_llm.build_prompt 保持同一道题。本地那边会再套 chat template，
# 这里不套——各家 API 自己拼模板，重复套会变成把模板标记当正文发过去。
QUESTION = (
    "请根据上述证据回答：生产环境出现消息重复消费，应如何定位和处理？"
    "按结论、适用前提、实施步骤、失败模式、监控与验证的结构作答。"
)

DEFAULT_MODELS = {
    # 默认只测 Opus 5。想要延迟档位的对照，显式传
    # --models claude-haiku-4-5 claude-sonnet-5
    "claude": ["claude-opus-5"],
    # Kimi 的模型 ID 迭代较快，这里给的默认值不保证长期有效；
    # 报错时直接用 --models 传当前可用的 ID，脚本会把服务端原文打出来。
    "kimi": ["kimi-k2-0905-preview"],
}


def build_evidence(target_tokens: int) -> tuple[str, int]:
    """按 Qwen 分词器把语料切到目标档位，返回**证据正文**和它的 Qwen token 数。

    只返回证据，不拼问题也不拼系统提示词：调用方要把它们分别放进
    各自的消息槽位，缓存断点才能打在证据末尾（CR-014）。

    刻意不套 chat template：模板由各家服务端自己拼。
    也正因如此，报告里各家的 prompt_tokens 一定大于这里返回的 Qwen 计数，
    差值是模板与分词器的膨胀，不能直接拿来横比（CR-017）。
    """
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    unit = tk.encode(FILLER, add_special_tokens=False)
    if not unit:
        raise RuntimeError("分词器对填充语料返回空结果")
    reps = max(1, target_tokens // len(unit) + 1)
    body = tk.decode((unit * reps)[:target_tokens])
    evidence = f"以下是检索到的证据：\n\n{body}"
    return evidence, len(tk.encode(evidence, add_special_tokens=False))


# ---------- 各家适配器 ----------
#
# 每个适配器返回同一组字段，调用方不关心是哪家。
# 刻意不抽象出公共基类：两家的流式事件模型差异大到抽象只会掩盖差异，
# 而这个脚本的价值恰恰在于把差异量出来。


def measure_claude(model: str, evidence: str, max_tokens: int, effort: str) -> dict[str, float]:
    """Claude：用官方 anthropic SDK 的流式接口。

    effort 默认 low 是有意的：Opus 5 思考默认开启，而思考的 token 全部落在
    用户看到第一个字之前。在 5 秒硬预算下，这是产品指标而非省钱选项。
    """
    import anthropic

    client = anthropic.Anthropic()
    t0 = time.perf_counter()
    first_event = first_text = None
    text_chars = thinking_chars = 0

    # 缓存断点打在证据正文末尾：system + 证据构成稳定前缀，问题在其后（CR-014）。
    # 缓存是**前缀匹配**，断点之后的任何变化都不影响前面已缓存的部分。
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        output_config={"effort": effort},
        system=[{
            "type": "text", "text": SYSTEM_PREFIX,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": [
            {"type": "text", "text": evidence,
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": QUESTION},
        ]}],
    ) as stream:
        for ev in stream:
            if first_event is None:
                first_event = time.perf_counter() - t0
            if ev.type != "content_block_delta":
                continue
            delta = ev.delta
            if delta.type == "thinking_delta":
                thinking_chars += len(delta.thinking)
            elif delta.type == "text_delta":
                if first_text is None:
                    first_text = time.perf_counter() - t0
                text_chars += len(delta.text)
        final = stream.get_final_message()

    total = time.perf_counter() - t0
    usage = final.usage
    return _pack(
        first_event, first_text, total, text_chars, thinking_chars,
        prompt_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        # 写入约 1.25 倍价、读取约 0.1 倍价；两个数都要留痕，
        # 否则「省了多少钱」无从计算。
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", None),
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", None),
    )


def measure_kimi(model: str, evidence: str, max_tokens: int, effort: str) -> dict[str, float]:
    """Kimi：OpenAI 兼容接口，用 openai SDK 指向月之暗面的 base_url。

    这不是「用 OpenAI 的 shim 调 Claude」——是 Kimi 自己就以 OpenAI 兼容协议对外，
    官方 SDK 即为此。Claude 那一路走的是 anthropic 官方 SDK，两条路各用各的。
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["MOONSHOT_API_KEY"],
        base_url=os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"),
    )
    t0 = time.perf_counter()
    first_event = first_text = None
    text_chars = thinking_chars = 0
    prompt_tokens = output_tokens = None
    cache_read = None

    stream = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        # 与 Claude 同构：稳定的 system + 证据在前，问题在后。
        # Moonshot 的上下文缓存是自动的，靠前缀匹配，因此顺序同样重要。
        messages=[
            {"role": "system", "content": SYSTEM_PREFIX},
            {"role": "user", "content": f"{evidence}\n\n{QUESTION}"},
        ],
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if first_event is None:
            first_event = time.perf_counter() - t0
        if getattr(chunk, "usage", None):
            prompt_tokens = chunk.usage.prompt_tokens
            output_tokens = chunk.usage.completion_tokens
            # OpenAI 兼容协议把命中数放在 prompt_tokens_details.cached_tokens。
            # **字段不存在 ≠ 没命中**——只是这家没报。两者必须区分开，
            # 因此缺字段时留 None，由 cache_verdict 判成"无法验证"而不是"未命中"。
            details = getattr(chunk.usage, "prompt_tokens_details", None)
            if details is not None:
                cache_read = getattr(details, "cached_tokens", None)
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # 思考型模型把推理放在 reasoning_content；非思考模型没有这个字段。
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            thinking_chars += len(reasoning)
        if delta.content:
            if first_text is None:
                first_text = time.perf_counter() - t0
            text_chars += len(delta.content)

    total = time.perf_counter() - t0
    return _pack(
        first_event, first_text, total, text_chars, thinking_chars,
        prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        # 兼容协议不报写入量，只报读取量；写入留 None，别拿 0 冒充"没写入"。
        cache_write_tokens=None, cache_read_tokens=cache_read,
    )


def _pack(first_event, first_text, total, text_chars, thinking_chars,
          prompt_tokens, output_tokens,
          cache_write_tokens=None, cache_read_tokens=None) -> dict[str, float]:
    # first_text 为 None 表示整轮没吐出正文（全被思考吃掉，或被截断）。
    # 记 total 而不是记 0：产品视角下「一个字都没等到」等价于等满全程。
    ttft = first_text if first_text is not None else total
    return {
        "ttft_text_s": round(ttft, 4),
        "first_event_s": round(first_event or 0.0, 4),
        "total_s": round(total, 4),
        "text_chars": text_chars,
        "thinking_chars": thinking_chars,
        "prompt_tokens": prompt_tokens or 0,
        "output_tokens": output_tokens or 0,
        # None 表示"这家没报这个字段"，0 表示"报了且为零"。
        # 合并成 0 会让"无法验证"看起来像"确认未命中"，是本轮要避免的正是这种事。
        "cache_write_tokens": cache_write_tokens,
        "cache_read_tokens": cache_read_tokens,
    }


MEASURERS: dict[str, Callable[..., dict[str, float]]] = {
    "claude": measure_claude,
    "kimi": measure_kimi,
}
CREDENTIALS = {"claude": "ANTHROPIC_API_KEY", "kimi": "MOONSHOT_API_KEY"}


def summarize(samples: list[dict[str, float]]) -> dict[str, Any]:
    """中位数 + P95 + 全部原始值。

    common.repeat() 只出中位数，对本地够用；云端必须看尾部，故在此单独实现，
    不去改动其它基准已在依赖的 common.repeat。
    """
    keys = samples[0].keys()

    def p95(vals: list[float]) -> float:
        s = sorted(vals)
        # 样本量小，取「不小于 95% 分位」的那个实测值，不做插值——
        # 插值会造出一个从未真实发生过的数字。
        idx = min(len(s) - 1, max(0, int(round(0.95 * len(s) + 0.5)) - 1))
        return s[idx]

    def series(k: str) -> list[float] | None:
        """某一列的数值序列；只要有一个样本是 None 就整列作废。

        缓存字段可能是 None（这家没报）。把 None 当 0 求中位数，
        会把"无法验证"算成"确认为零"——正是 CR-014 要防的那种误读。
        """
        vals = [s[k] for s in samples]
        return None if any(v is None for v in vals) else vals

    med, p95s = {}, {}
    for k in keys:
        vals = series(k)
        if vals is None:
            med[k] = p95s[k] = None
        else:
            med[k] = round(statistics.median(vals), 4)
            p95s[k] = round(p95(vals), 4)

    return {
        "runs": len(samples),
        "cold": samples[0],
        "median": med,
        "p95": p95s,
        "cache": cache_verdict(samples),
        "samples": samples,
    }


def cache_verdict(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """两阶段判定 prompt cache 到底有没有命中（CR-014）。

    为什么必须显式判：**缓存不命中不报错**，请求照常成功、报告照常生成，
    只是更慢更贵。不验证就拿报告去决定云端路由，等于把一个从没测过的
    假设写进架构决策。

    判据是"首轮写入、后续读取"：

    - `hit`：首轮有写入量，且至少一个后续轮次读取量 > 0 —— 确认可用。
    - `miss`：字段报了，但后续轮次读取量全为 0 —— 确认不命中
      （常见原因是前缀短于该模型的最小可缓存长度，512–4096 token 且随模型而变）。
    - `unverified`：这家压根没报缓存字段 —— **不是未命中，是测不了**。
      两者必须分开：前者可以下结论，后者只能说"这条结论不成立"。

    `runs < 2` 时一律 `unverified`：一轮跑不出"后续读取"这件事。
    """
    if len(samples) < 2:
        return {"status": "unverified", "reason": "样本不足 2 轮，无法观察后续读取"}

    reads = [s.get("cache_read_tokens") for s in samples]
    writes = [s.get("cache_write_tokens") for s in samples]
    if all(r is None for r in reads):
        return {"status": "unverified",
                "reason": "供应商未在 usage 中报告缓存读取量，无法验证"}

    later_reads = [r for r in reads[1:] if r]
    if later_reads:
        return {
            "status": "hit",
            "first_write_tokens": writes[0],
            "later_read_tokens_max": max(later_reads),
            "later_runs_hit": f"{len(later_reads)}/{len(reads) - 1}",
        }
    return {
        "status": "miss",
        "reason": "后续轮次缓存读取量均为 0；常见原因是稳定前缀短于该模型的"
                  "最小可缓存长度（512–4096 token，随模型而变）",
        "first_write_tokens": writes[0],
    }


API_HOSTS = {
    "claude": "api.anthropic.com",
    "kimi": "api.moonshot.cn",
}


def probe_network_rtt(provider: str, runs: int) -> dict[str, Any]:
    """**真正的网络层地板**：TCP 连接 + TLS 握手的往返耗时（CR-017）。

    早先这个名字挂在一次完整的流式生成上，那里面含服务端排队、
    最小 prompt 的处理和首字生成，与网络往返不是一回事。
    混成一个数，看到 P95 超标就分不清该换网络还是该换供应商。

    这里不发任何模型请求，因此**不花钱**，也不受对方排队影响。
    握手完立刻关闭连接。

    **两个数的口径不同，别混用**（否则就是重犯 CR-017 那个错）：

    - `tcp_connect_s` ≈ **一次网络往返**。SDK 复用连接，所以这才是
      稳态下每个请求要付的网络地板。
    - `tcp_tls_s` = TCP + TLS 完整握手，是**冷连接**才付的一次性成本。
      keep-alive 生效后不再重复付，不能按请求数乘。

    实测（2026-09-02，同一台机器同一时刻）：
    Claude `api.anthropic.com` TCP 中位 0.018s；
    Kimi `api.moonshot.cn` TCP 中位 0.240s、TLS 完整握手中位 0.794s。
    单次往返差 13 倍——这个差距在 3.0 秒生成预算里不是可以忽略的量。
    """
    import socket
    import ssl

    host = API_HOSTS[provider]
    ctx = ssl.create_default_context()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        with socket.create_connection((host, 443), timeout=10) as sock:
            connected = time.perf_counter() - t0
            with ctx.wrap_socket(sock, server_hostname=host):
                pass
        samples.append({
            "tcp_connect_s": round(connected, 4),
            "tcp_tls_s": round(time.perf_counter() - t0, 4),
        })
    return {"host": host, **summarize(samples)}


def probe_min_request(provider: str, model: str, runs: int) -> dict[str, Any]:
    """最小请求的端到端 TTFT：网络 + 服务端排队 + 最小 prefill + 首字生成。

    **不是**网络地板（那是 probe_network_rtt）。它减去网络地板，
    才是「服务端固定开销」；完整请求再减去它，才是上下文长度带来的增量。

    刻意不打缓存断点：最小请求远低于任何模型的最小可缓存长度，
    打了也不会命中，反而让 cache_verdict 报出误导性的 miss。
    """
    fn = MEASURERS[provider]
    samples = [fn(model, "回答一个字：好", 4, "low") for _ in range(runs)]
    return summarize(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+", default=["claude", "kimi"],
                    choices=sorted(MEASURERS))
    ap.add_argument("--models", nargs="+", default=None,
                    help="覆盖默认模型列表；只在单个 provider 时有意义")
    ap.add_argument("--contexts", type=int, nargs="+", default=[512, 1024, 2048, 4096],
                    help="上下文档位，按 Qwen 分词器计，与 bench_llm.py 对齐")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--runs", type=int, default=5,
                    help="每档重复次数。云端方差大，默认高于本地基准的 3 次")
    ap.add_argument("--effort", default="low",
                    help="Claude 的 output_config.effort。5 秒预算下默认 low")
    ap.add_argument("--rtt-only", action="store_true",
                    help="只测网络往返与最小请求，不跑分档。用于先确认链路与凭据")
    args = ap.parse_args()

    active = []
    for p in args.providers:
        if os.environ.get(CREDENTIALS[p]):
            active.append(p)
        else:
            print(f"跳过 {p}：环境变量 {CREDENTIALS[p]} 未设置")
    if not active:
        print("没有可用的凭据，未测量任何供应商。")
        return

    evidences = {}
    if not args.rtt_only:
        for target in args.contexts:
            evidences[target] = build_evidence(target)

    results: list[dict[str, Any]] = []
    for provider in active:
        models = args.models or DEFAULT_MODELS[provider]
        for model in models:
            print(f"\n== {provider} / {model} ==", flush=True)
            try:
                net = probe_network_rtt(provider, args.runs)
            except Exception as exc:
                print(f"  网络探测失败: {type(exc).__name__}: {exc}")
                net = None
            else:
                print(f"  网络地板(TCP+TLS) 中位 {net['median']['tcp_tls_s']}s / "
                      f"P95 {net['p95']['tcp_tls_s']}s")

            try:
                minreq = probe_min_request(provider, model, args.runs)
            except Exception as exc:
                # 打服务端原文：模型 ID 过期、区域不可达、配额问题都在这里现形。
                print(f"  最小请求失败，跳过该模型: {type(exc).__name__}: {exc}")
                continue
            print(f"  最小请求 中位 {minreq['median']['ttft_text_s']}s / "
                  f"P95 {minreq['p95']['ttft_text_s']}s")
            if net:
                overhead = round(
                    minreq["median"]["ttft_text_s"] - net["median"]["tcp_tls_s"], 4)
                print(f"  → 服务端固定开销(最小请求 − 网络) 中位 {overhead}s")

            tiers = []
            for target, (evidence, qwen_tokens) in evidences.items():
                print(f"  证据 {target} (Qwen 计 {qwen_tokens} token) ...", flush=True)
                try:
                    stats = summarize([
                        MEASURERS[provider](model, evidence, args.max_tokens, args.effort)
                        for _ in range(args.runs)
                    ])
                except Exception as exc:
                    print(f"    失败: {type(exc).__name__}: {exc}")
                    continue
                tiers.append({
                    "target_evidence_tokens": target,
                    "qwen_evidence_tokens": qwen_tokens,
                    "provider_prompt_tokens": stats["median"]["prompt_tokens"],
                    **stats,
                })
                cache = stats["cache"]
                print(f"    首字 中位 {stats['median']['ttft_text_s']}s / "
                      f"P95 {stats['p95']['ttft_text_s']}s"
                      f"    缓存 {cache['status']}")

            results.append({
                "provider": provider,
                "model": model,
                "effort": args.effort if provider == "claude" else None,
                "network_rtt": net,
                "min_request": minreq,
                "results": tiers,
            })

    if not results:
        print("\n没有任何模型测量成功，不写报告。")
        return

    path = write_report("llm-remote", {
        "max_tokens_per_run": args.max_tokens,
        "tokenizer_for_context_tiers": TOKENIZER_ID,
        "providers": results,
    })
    print(f"\n报告已写入 {path}")

    # 直接对上 architecture.md 第 6 节：生成阶段新预算 3.0 秒，按 P95 判。
    budget = 3.0
    print(f"\n结论（按 P95 首字，生成阶段预算 {budget}s）:")
    for entry in results:
        ok = [t for t in entry["results"] if t["p95"]["ttft_text_s"] <= budget]
        tag = f"{entry['provider']}/{entry['model']}"
        if ok:
            best = max(ok, key=lambda t: t["qwen_evidence_tokens"])
            print(f"  {tag}: 证据上限约 {best['qwen_evidence_tokens']} token")
        else:
            print(f"  {tag}: 所有档位 P95 均超预算")

    # 缓存结论单列，且**不允许含糊**（CR-014）：
    # T-026 的完成条件之一就是"prompt cache 是否真命中"，
    # 没验证到就必须说没验证到，不能让报告看起来像验证过了。
    print("\nprompt cache 验证:")
    unverified = []
    for entry in results:
        tag = f"{entry['provider']}/{entry['model']}"
        for t in entry["results"]:
            c = t["cache"]
            line = f"  {tag} @{t['target_evidence_tokens']}: {c['status']}"
            if c["status"] == "hit":
                line += (f"  首轮写入 {c['first_write_tokens']} tok，"
                         f"后续读取 {c['later_runs_hit']} 轮命中")
            else:
                line += f"  —— {c.get('reason', '')}"
                unverified.append(f"{tag}@{t['target_evidence_tokens']}")
            print(line)
    if unverified:
        print("\n**缓存收益不可作为路由依据**：以下档位未确认命中 —— "
              + ", ".join(unverified))
        print("  T-027 若要把缓存算进成本或时延模型，必须先让这些档位变成 hit。")


if __name__ == "__main__":
    main()
