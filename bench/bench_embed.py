"""嵌入模型基准：单条查询时延与批量建索引吞吐。

两个数字用途不同：
- 单条时延落在请求路径上，直接吃 architecture.md 第 6 节的 0.8 秒检索预算。
- 批量吞吐决定 I1 的全量建索引要跑多久，属于离线成本。

用法:
    python bench/bench_embed.py --model mlx-community/bge-m3-mlx-8bit
"""

from __future__ import annotations

import argparse
import time

from common import Timer, peak_memory_gb, repeat, reset_peak_memory, write_report

# 查询侧：用户口语化的技术问题
QUERIES = [
    "Kafka 消费者重复消费怎么排查",
    "Spring Boot 事务不回滚是什么原因",
    "Redis 预扣库存超卖如何解决",
    "PostgreSQL 慢查询突然变慢 执行计划变成 Seq Scan",
    "Kubernetes liveness probe 把 Pod 杀掉了",
]

# 文档侧：模拟切块后的文档片段，长度接近真实语料
DOC_CHUNK = (
    "在 Spring Boot 3.2 中，@Transactional 注解默认的回滚规则只覆盖 RuntimeException "
    "和 Error。如果业务方法内部捕获了受检异常并转换成返回码，Spring 不会感知到异常，"
    "事务将正常提交。这在 Outbox 模式下尤其危险：业务表写入成功而 Outbox 记录未写入时，"
    "下游服务永远不会收到事件。推荐显式声明 rollbackFor = Exception.class，"
    "并通过集成测试验证异常路径下的回滚行为。"
) * 2


def embed_fn(model, tokenizer):
    """封装不同 mlx-embeddings 版本的调用差异。"""
    from mlx_embeddings import generate

    def run(texts: list[str]):
        out = generate(model, tokenizer, texts=texts)
        # 不同版本返回字段名不同，按优先级取归一化后的句向量
        for attr in ("text_embeds", "embeddings", "pooler_output", "last_hidden_state"):
            v = getattr(out, attr, None)
            if v is not None:
                return v
        raise RuntimeError(f"无法从 {type(out)} 中取出嵌入向量")

    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/bge-m3-mlx-8bit")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    from mlx_embeddings import load

    reset_peak_memory()
    with Timer() as t:
        model, tokenizer = load(args.model)
    load_s = round(t.elapsed, 2)
    run = embed_fn(model, tokenizer)

    probe = run(QUERIES[:1])
    dim = probe.shape[-1]
    print(f"模型加载 {load_s}s | 向量维度 {dim} | 权重约 {peak_memory_gb()} GB")

    results = []

    # 查询侧单条时延（请求路径上的成本）
    reset_peak_memory()
    stats = repeat(
        lambda: {"latency_s": _timed(run, [QUERIES[0]])}, args.runs
    )
    results.append({"mode": "query_single", "batch": 1, **stats})
    print(f"  单条查询嵌入: {stats['median']['latency_s'] * 1000:.1f} ms (冷 {stats['cold']['latency_s'] * 1000:.1f} ms)")

    # 文档侧批量吞吐（离线建索引成本）
    for b in args.batches:
        texts = [DOC_CHUNK] * b
        reset_peak_memory()
        stats = repeat(lambda: {"latency_s": _timed(run, texts)}, args.runs)
        tps = round(b / stats["median"]["latency_s"], 1)
        results.append(
            {
                "mode": "doc_batch",
                "batch": b,
                "chunks_per_second": tps,
                "peak_memory_gb": peak_memory_gb(),
                **stats,
            }
        )
        print(f"  批量 {b:>3}: {stats['median']['latency_s']:.3f}s -> {tps} chunk/s")

    path = write_report(
        "embed",
        {
            "model": args.model,
            "runtime": "mlx-embeddings",
            "dimension": dim,
            "load_seconds": load_s,
            "results": results,
        },
    )
    print(f"\n报告已写入 {path}")

    best = max(
        (r for r in results if r["mode"] == "doc_batch"),
        key=lambda r: r["chunks_per_second"],
    )
    q = next(r for r in results if r["mode"] == "query_single")
    print(f"结论: 查询侧 {q['median']['latency_s'] * 1000:.0f} ms/条；"
          f"建索引峰值 {best['chunks_per_second']} chunk/s (batch={best['batch']})")


def _timed(run, texts) -> float:
    import mlx.core as mx

    t0 = time.perf_counter()
    out = run(texts)
    mx.eval(out)  # MLX 惰性求值，必须显式 eval 才能测到真实耗时
    return round(time.perf_counter() - t0, 5)


if __name__ == "__main__":
    main()
