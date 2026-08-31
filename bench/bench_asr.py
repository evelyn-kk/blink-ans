"""本地语音转写基准：实时率与转写耗时。

architecture.md 第 6 节给转写留了 0.7 秒预算，但那是"语音结束到首段转写"，
对应的是流式分片场景。本脚本先测整段实时率（RTF = 音频时长 / 转写耗时），
RTF 必须显著大于 1 才有可能在 I4 做到流式低延迟。

测试音频用 macOS 的 say 合成，保证任何机器上都能复现。
注意：合成语音比真实口语干净得多，本脚本的数字只用于时延选型，
准确率必须在 I4 用真实录音重新评估。

用法:
    python bench/bench_asr.py --model mlx-community/whisper-large-v3-turbo
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from common import Timer, peak_memory_gb, repeat, reset_peak_memory, write_report

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

# 中英混杂的技术口语，贴近真实提问方式
# 技术术语词表偏置：whisper 的 initial_prompt 会影响解码时的先验。
# 中英混杂的技术口语里，Kafka / PostgreSQL 这类词一旦识别错，检索必然失败，
# 因此词表偏置的收益直接体现在检索命中率上，而不只是转写准确率。
GLOSSARY = (
    "以下是一段关于 Java 后端技术的讨论，涉及这些术语："
    "Kafka、RabbitMQ、Redis、PostgreSQL、MySQL、Oracle、Elasticsearch、"
    "Spring Boot、Spring Cloud、Hibernate、MyBatis、JPA、"
    "Kubernetes、Docker、Helm、Istio、OpenTelemetry、Prometheus、Grafana、"
    "Outbox、DLQ、offset、rebalance、幂等、预扣、超卖、扣减、对账、"
    "慢查询、执行计划、索引、事务、回滚、连接池、"
    "P95、P99、QPS、TPS、GC、JVM、OOM、CPU、liveness probe、readiness probe。"
)

UTTERANCES = {
    "short": "我们线上的 Kafka 消费者一直重复消费，offset 提交好像有问题，怎么排查？",
    "medium": (
        "我们的订单服务用 Spring Boot 三点二，最近发现库存扣减出现超卖。"
        "Redis 里用 Lua 脚本做的预扣，数据库那边是乐观锁。"
        "压测的时候 QPS 上到两千就开始出问题，帮我分析一下可能的原因和排查步骤。"
    ),
    "long": (
        "生产环境的 PostgreSQL 最近 P95 慢查询涨到了八百毫秒，之前一直是六十毫秒左右。"
        "我看了 EXPLAIN ANALYZE，执行计划从 Index Scan 变成了 Seq Scan。"
        "这个表大概两千万行，最近做过一次批量的数据归档，删掉了差不多三成的数据。"
        "应用层是 Spring Data JPA，没有改过查询代码。"
        "另外 Kubernetes 上的这个 Pod 最近也偶尔会被 liveness probe 杀掉，"
        "不确定这两件事有没有关联，帮我理一下排查思路和需要确认的监控指标。"
    ),
}


def synth(name: str, text: str) -> Path:
    """用 macOS say 合成中文语音并转成 16kHz 单声道 wav。"""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    wav = FIXTURE_DIR / f"{name}.wav"
    if wav.exists():
        return wav
    aiff = FIXTURE_DIR / f"{name}.aiff"
    # Tingting 是 macOS 的中文语音；不存在时回退到系统默认语音
    voices = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    voice = "Tingting" if "Tingting" in voices else None
    cmd = ["say"] + (["-v", voice] if voice else []) + ["-o", str(aiff), text]
    subprocess.run(cmd, check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(wav)],
        check=True,
        capture_output=True,
    )
    aiff.unlink(missing_ok=True)
    return wav


def duration_s(wav: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(wav)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument(
        "--compare-glossary",
        action="store_true",
        help="对比开启/关闭技术术语词表偏置的转写差异",
    )
    args = ap.parse_args()

    import mlx_whisper

    clips = []
    for name, text in UTTERANCES.items():
        wav = synth(name, text)
        clips.append((name, wav, duration_s(wav), text))
        print(f"素材 {name}: {clips[-1][2]:.2f}s")

    results = []
    for name, wav, dur, reference in clips:
        reset_peak_memory()

        def once() -> dict[str, float]:
            t0 = time.perf_counter()
            out = mlx_whisper.transcribe(
                str(wav),
                path_or_hf_repo=args.model,
                language="zh",
                initial_prompt=GLOSSARY if args.compare_glossary else None,
            )
            elapsed = time.perf_counter() - t0
            once.text = out["text"]
            return {"transcribe_s": round(elapsed, 4), "rtf": round(dur / elapsed, 2)}

        print(f"  测量 {name} ...", flush=True)
        stats = repeat(once, args.runs)
        entry = {
            "clip": name,
            "audio_seconds": round(dur, 2),
            "peak_memory_gb": peak_memory_gb(),
            "glossary_biased": args.compare_glossary,
            "reference_text": reference,
            "transcribed_text": getattr(once, "text", ""),
            **stats,
        }
        results.append(entry)
        m = stats["median"]
        print(f"    转写 {m['transcribe_s']}s | 实时率 {m['rtf']}x | 冷启动 {stats['cold']['transcribe_s']}s")
        print(f"    识别: {entry['transcribed_text'][:60]}")

    path = write_report(
        "asr",
        {"model": args.model, "runtime": "mlx-whisper", "results": results},
    )
    print(f"\n报告已写入 {path}")

    warm = [r["median"]["rtf"] for r in results]
    print(f"结论: 热态实时率 {min(warm)}x ~ {max(warm)}x")
    if min(warm) < 3:
        print("警告: 实时率低于 3x，I4 的流式转写很难压进 0.7s 预算，考虑换更小的模型")


if __name__ == "__main__":
    main()
