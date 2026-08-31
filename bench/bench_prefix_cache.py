"""前缀 KV cache 的收益量化。

基准显示 prefill 约 350 tok/s，是本机的硬约束。唯一的出路是少 prefill 一些 token。
本脚本量化三种场景，用来决定 I2 的提示词该怎么排布：

  A 基线      每次完整 prefill（模板 + 证据 + 问题）
  B 前缀复用  模板部分的 KV 常驻，每次只 prefill 证据和问题
  C 完全命中  整个 prompt 已缓存（热点问题），只剩解码

B 成立的前提是"不变的排在前面、变化的排在后面"——KV cache 只能复用前缀。
这个结论会直接约束提示词模板的写法。

用法:
    python bench/bench_prefix_cache.py --model mlx-community/Qwen3-4B-Instruct-2507-4bit
"""

from __future__ import annotations

import argparse
import copy
import time

import mlx.core as mx
from common import Timer, peak_memory_gb, repeat, reset_peak_memory, write_report

# 固定不变的部分：系统指令 + 答案结构模板。真实系统里这段每次请求完全相同。
TEMPLATE = """你是一名 Java / Spring 云原生方向的资深后端工程师，负责回答生产环境问题。

回答必须严格遵循以下结构，每一节都不能省略：
1. 结论：一句话给出判断，不要铺垫。
2. 适用前提：这个结论在什么版本、什么配置、什么流量特征下成立。
3. 实施步骤：可直接执行的操作序列，涉及配置项时写出完整的键名和取值。
4. 失败模式：这个方案在什么情况下会失效，以及失效时的现象。
5. 监控与验证：改动后应该观察哪些指标，以及判断生效的具体阈值。
6. 来源：逐条列出依据的文档链接与版本号。

硬性要求：
- 只依据下方提供的证据作答，证据不足时明确说"现有证据不足以确定"，并说明还需要核实什么。
- 不得编造配置项名称、方法签名或指标名。
- 涉及版本差异时必须写清楚版本号，不要把不同大版本的配置混用。
- 生产变更必须提示回滚方式。
"""

EVIDENCE_UNIT = """
[证据] Spring Framework 6.1 参考文档 / 事务管理 / 回滚规则
默认情况下，事务只在抛出 RuntimeException 或 Error 时回滚，受检异常不触发回滚。
可通过 @Transactional(rollbackFor = Exception.class) 修改该行为。
在声明式事务中，自调用（同类内部方法调用）不会经过代理，事务注解不生效。
"""

QUESTION = "生产环境 Outbox 表和业务表状态不一致，怎么定位和修复？"


def build(tokenizer, evidence_tokens: int) -> tuple[list[int], int]:
    """返回完整 prompt 的 token 序列，以及固定模板前缀的长度。"""
    unit = tokenizer.encode(EVIDENCE_UNIT, add_special_tokens=False)
    reps = max(1, evidence_tokens // len(unit) + 1)
    evidence = tokenizer.decode((unit * reps)[:evidence_tokens])

    def render(body: str) -> list[int]:
        msgs = [{"role": "user", "content": body}]
        try:
            s = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            s = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        return tokenizer.encode(s)

    full = render(f"{TEMPLATE}\n【证据】\n{evidence}\n\n【问题】{QUESTION}")

    # 前缀长度 = 只含模板时的公共前缀。用逐位比较求真实公共前缀，
    # 避免因 chat template 收尾标记导致的偏差。
    template_only = render(TEMPLATE)
    n = 0
    for a, b in zip(full, template_only):
        if a != b:
            break
        n += 1
    return full, n


def warm_cache(model, tokens: list[int]):
    """把给定 token 序列的 KV 算进一个新 cache 并返回。"""
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    model(mx.array(tokens)[None], cache=cache)
    mx.eval([c.state for c in cache])
    return cache


def time_generate(model, tokenizer, prompt, cache, max_tokens: int) -> float:
    """返回首 token 时延。cache 会被就地修改，调用方负责传副本。"""
    from mlx_lm import stream_generate

    t0 = time.perf_counter()
    for _ in stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, prompt_cache=cache
    ):
        return time.perf_counter() - t0
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-4B-Instruct-2507-4bit")
    ap.add_argument("--evidence", type=int, default=800, help="证据部分的 token 数")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = load(args.model)
    full, prefix_len = build(tokenizer, args.evidence)
    suffix_len = len(full) - prefix_len
    print(f"完整 prompt {len(full)} token = 固定模板 {prefix_len} + 变化部分 {suffix_len}")

    reset_peak_memory()
    results = []

    # A 基线：每次完整 prefill
    stats = repeat(
        lambda: {"ttft_s": time_generate(model, tokenizer, full, make_prompt_cache(model), 8)},
        args.runs,
    )
    a = stats["median"]["ttft_s"]
    results.append({"scenario": "A_full_prefill", "prefilled_tokens": len(full), **stats})
    print(f"  A 基线（完整 prefill {len(full)} token）: {a:.3f}s")

    # B 前缀复用：模板 KV 常驻，只 prefill 变化部分
    base = warm_cache(model, full[:prefix_len])
    template_kv_gb = peak_memory_gb()
    stats = repeat(
        lambda: {
            "ttft_s": time_generate(
                model, tokenizer, full[prefix_len:], copy.deepcopy(base), 8
            )
        },
        args.runs,
    )
    b = stats["median"]["ttft_s"]
    results.append({"scenario": "B_prefix_reuse", "prefilled_tokens": suffix_len, **stats})
    print(f"  B 前缀复用（只 prefill {suffix_len} token）: {b:.3f}s  省下 {a - b:.3f}s ({(1 - b / a) * 100:.0f}%)")

    # C 完全命中：整个 prompt 已缓存，只剩最后一个 token
    hot = warm_cache(model, full[:-1])
    stats = repeat(
        lambda: {"ttft_s": time_generate(model, tokenizer, full[-1:], copy.deepcopy(hot), 8)},
        args.runs,
    )
    c = stats["median"]["ttft_s"]
    results.append({"scenario": "C_full_hit", "prefilled_tokens": 1, **stats})
    print(f"  C 完全命中（热点问题）: {c:.3f}s  省下 {a - c:.3f}s ({(1 - c / a) * 100:.0f}%)")

    path = write_report(
        "prefix-cache",
        {
            "model": args.model,
            "prompt_tokens": len(full),
            "template_prefix_tokens": prefix_len,
            "variable_suffix_tokens": suffix_len,
            "peak_memory_gb": peak_memory_gb(),
            "results": results,
        },
    )
    print(f"\n报告已写入 {path}")
    print(f"结论: 模板前缀常驻可把 TTFT 从 {a:.2f}s 降到 {b:.2f}s；热点完全命中可降到 {c:.2f}s")


if __name__ == "__main__":
    main()
