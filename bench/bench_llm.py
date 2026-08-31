"""本地生成模型基准：按上下文长度分档测首 token 时延。

关键结论不是"每秒多少 token"，而是"5 秒预算下能塞多少证据"。
prefill 时延随上下文长度增长，直接决定检索阶段的 top-k 上限，
因此本脚本按 512/1024/2048/4096 token 分档测量 TTFT。

用法:
    python bench/bench_llm.py --model mlx-community/Qwen3-4B-Instruct-2507-4bit
"""

from __future__ import annotations

import argparse
import gc
import time
from typing import Any

from common import Timer, peak_memory_gb, repeat, reset_peak_memory, write_report

# 用真实的中文技术语料做填充，保证分词结果贴近实际工作负载。
# 英文和中文的 token 密度差异很大，用英文 lorem ipsum 测出来的数字没有参考价值。
FILLER = """
Spring Boot 3.2 中，@Transactional 默认只对 RuntimeException 和 Error 回滚。
若业务代码捕获了受检异常并转换为返回值，事务不会回滚，导致 Outbox 表与业务表状态不一致。
Kafka 消费者在 max.poll.interval.ms 超时后会被踢出消费组并触发 rebalance，
此时未提交的 offset 会被其他消费者重新拉取，产生重复消费。
Redis 预扣库存使用 Lua 脚本保证原子性，但在主从切换时可能丢失未同步的写入，
需要配合数据库的最终一致性对账任务补偿。PostgreSQL 的 P95 慢查询通常来自
缺失的复合索引或统计信息过期，应先用 EXPLAIN ANALYZE 确认执行计划再考虑加索引。
Kubernetes 的 liveness probe 配置过于激进会在 GC 停顿期间误杀 Pod，
initialDelaySeconds 必须大于应用的冷启动时间加上 JIT 预热时间。
"""


def build_prompt(tokenizer, target_tokens: int) -> tuple[str, int]:
    """构造接近目标 token 数的提示词，返回最终字符串和实际 token 数。"""
    unit = tokenizer.encode(FILLER, add_special_tokens=False)
    if not unit:
        raise RuntimeError("分词器对填充语料返回空结果")
    reps = max(1, target_tokens // len(unit) + 1)
    body = tokenizer.decode(( unit * reps )[:target_tokens])

    question = (
        f"以下是检索到的证据：\n\n{body}\n\n"
        "请根据上述证据回答：生产环境出现消息重复消费，应如何定位和处理？"
        "按结论、适用前提、实施步骤、失败模式、监控与验证的结构作答。"
    )
    messages = [{"role": "user", "content": question}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        # Qwen3 等混合推理模型默认先输出 <think> 块，会把首段预算全部吃掉。
        # 模板支持时显式关闭；不支持的模型（如 Instruct-2507）忽略该参数。
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    else:
        prompt = question
    return prompt, len(tokenizer.encode(prompt))


def measure(model, tokenizer, prompt: str, max_tokens: int) -> dict[str, float]:
    """测一次生成：首 token 时延 + 解码吞吐。"""
    from mlx_lm import stream_generate

    t0 = time.perf_counter()
    ttft = None
    n = 0
    for item in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        if ttft is None:
            ttft = time.perf_counter() - t0
        n += 1
    total = time.perf_counter() - t0

    decode_time = max(total - (ttft or 0), 1e-6)
    return {
        "ttft_s": round(ttft or 0.0, 4),
        "total_s": round(total, 4),
        "generated_tokens": n,
        "decode_tps": round((n - 1) / decode_time, 2) if n > 1 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="MLX 模型仓库或本地路径")
    ap.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        help="要测的上下文 token 档位",
    )
    ap.add_argument("--max-tokens", type=int, default=160, help="每次生成的 token 数")
    ap.add_argument("--runs", type=int, default=3, help="每档重复次数，取中位数")
    args = ap.parse_args()

    from mlx_lm import load

    reset_peak_memory()
    with Timer() as t:
        model, tokenizer = load(args.model)
    load_s = round(t.elapsed, 2)
    weights_gb = peak_memory_gb()
    print(f"模型加载完成: {load_s}s, 权重驻留约 {weights_gb} GB")

    results: list[dict[str, Any]] = []
    for target in args.contexts:
        prompt, actual = build_prompt(tokenizer, target)
        reset_peak_memory()
        print(f"  测量上下文 {target} (实际 {actual} token) ...", flush=True)
        stats = repeat(lambda: measure(model, tokenizer, prompt, args.max_tokens), args.runs)
        entry = {
            "target_context_tokens": target,
            "actual_prompt_tokens": actual,
            "peak_memory_gb": peak_memory_gb(),
            **stats,
        }
        results.append(entry)
        m = stats["median"]
        print(
            f"    TTFT {m['ttft_s']}s | 解码 {m['decode_tps']} tok/s "
            f"| 峰值 {entry['peak_memory_gb']} GB"
        )
        gc.collect()

    path = write_report(
        "llm",
        {
            "model": args.model,
            "runtime": "mlx-lm",
            "load_seconds": load_s,
            "weights_resident_gb": weights_gb,
            "max_tokens_per_run": args.max_tokens,
            "results": results,
        },
    )
    print(f"\n报告已写入 {path}")

    # 直接给出 architecture.md 第 6 节关心的结论：2.5 秒首段预算下的上下文上限。
    budget = 2.5
    ok = [r for r in results if r["median"]["ttft_s"] <= budget]
    if ok:
        best = max(ok, key=lambda r: r["actual_prompt_tokens"])
        print(f"结论: {budget}s 首 token 预算下，上下文上限约 {best['actual_prompt_tokens']} token")
    else:
        print(f"结论: 所有档位 TTFT 均超过 {budget}s 预算，需要降级模型或压缩证据")


if __name__ == "__main__":
    main()
