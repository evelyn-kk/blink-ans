"""本地语音转写基准：实时率与转写耗时。

architecture.md 第 6 节给转写留了 0.7 秒预算，但那是"语音结束到首段转写"，
对应的是流式分片场景。本脚本先测整段实时率（RTF = 音频时长 / 转写耗时），
RTF 必须显著大于 1 才有可能在 I4 做到流式低延迟。

遗留中文时延基线可继续用 macOS `say` 合成；**英文评测必须显式提供真人录音
manifest**，不允许用 TTS 结果伪装成英文术语识别率或时延。合成语音比真实口语
干净得多，现有中文数字只用于时延选型，准确率必须在 I4 用真实录音重新评估。

用法:
    # 复跑遗留中文合成时延基线
    python bench/bench_asr.py --language zh

    # 英文：录制后把音频和逐字参考写进 manifest；没有它会失败关闭
    python bench/bench_asr.py --language en --manifest bench/asr_real_manifest.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from common import Timer, peak_memory_gb, repeat, reset_peak_memory, write_report

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
REAL_MANIFEST_EXAMPLE = Path(__file__).resolve().parent / "asr_real_manifest.example.yaml"

# 中英混杂的技术口语，贴近真实提问方式
# 技术术语词表偏置：whisper 的 initial_prompt 会影响解码时的先验。
# 中英混杂的技术口语里，Kafka / PostgreSQL 这类词一旦识别错，检索必然失败，
# 因此词表偏置的收益直接体现在检索命中率上，而不只是转写准确率。
LANGUAGE_GLOSSARIES = {
"zh": (
    "以下是一段关于 Java 后端技术的讨论，涉及这些术语："
    "Kafka、RabbitMQ、Redis、PostgreSQL、MySQL、Oracle、Elasticsearch、"
    "Spring Boot、Spring Cloud、Hibernate、MyBatis、JPA、"
    "Kubernetes、Docker、Helm、Istio、OpenTelemetry、Prometheus、Grafana、"
    "Outbox、DLQ、offset、rebalance、幂等、预扣、超卖、扣减、对账、"
    "慢查询、执行计划、索引、事务、回滚、连接池、"
    "P95、P99、QPS、TPS、GC、JVM、OOM、CPU、liveness probe、readiness probe。"
),
"en": (
    "This is a discussion about Java backend operations. It uses these technical terms: "
    "Kafka, RabbitMQ, Redis, PostgreSQL, MySQL, Oracle, Elasticsearch, Spring Boot, "
    "Spring Cloud, Hibernate, MyBatis, JPA, Kubernetes, Docker, Helm, Istio, "
    "OpenTelemetry, Prometheus, Grafana, Outbox, DLQ, offset, rebalance, idempotency, "
    "inventory reservation, overselling, reconciliation, slow query, execution plan, "
    "index, transaction, rollback, connection pool, P95, P99, QPS, TPS, GC, JVM, OOM, "
    "CPU, liveness probe, and readiness probe."
),
}

ZH_SYNTHETIC_UTTERANCES = {
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


def transcribe_options(language: str, *, glossary: bool) -> dict[str, str | None]:
    """唯一的 ASR 语言/词表入口；不能让英文会话悄悄沿用中文偏置。"""
    if language not in LANGUAGE_GLOSSARIES:
        raise ValueError(f"不支持的 ASR 语言: {language}")
    return {
        "language": language,
        "initial_prompt": LANGUAGE_GLOSSARIES[language] if glossary else None,
    }


def load_real_manifest(path: Path, language: str) -> list[dict[str, Any]]:
    """加载真人录音清单，缺元数据或音频都失败关闭。

    参考文本是之后算术语命中/错误率的地面真值；没有音频或参考文本时，所谓
    “英文识别率”没有测量对象，不能退回 TTS 或只测时延。
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("kind") != "real_recording":
        raise ValueError(f"{path}: kind 必须为 real_recording，不能把合成语音当真人基准")
    clips = raw.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError(f"{path}: clips 不能为空；参见 {REAL_MANIFEST_EXAMPLE.name}")
    loaded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for clip in clips:
        if not isinstance(clip, dict):
            raise ValueError(f"{path}: clips 每项必须是对象")
        clip_id = clip.get("id")
        if not isinstance(clip_id, str) or not clip_id or clip_id in seen:
            raise ValueError(f"{path}: clip id 必须非空且唯一")
        seen.add(clip_id)
        if clip.get("language") != language:
            raise ValueError(f"{path}: {clip_id} 的 language 必须为 {language!r}")
        reference = clip.get("reference_text")
        reference_path = clip.get("reference_path")
        if reference is not None and reference_path is not None:
            raise ValueError(f"{path}: {clip_id} 只能登记 reference_text 或 reference_path 之一")
        if reference_path is not None:
            if not isinstance(reference_path, str) or not reference_path:
                raise ValueError(f"{path}: {clip_id} 的 reference_path 必须为非空路径")
            resolved_reference = (path.parent / reference_path).resolve()
            if not resolved_reference.is_file():
                raise ValueError(f"{path}: {clip_id} 逐字稿不存在: {resolved_reference}")
            reference = resolved_reference.read_text(encoding="utf-8").strip()
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"{path}: {clip_id} 缺 reference_text 或 reference_path")
        audio = clip.get("audio")
        if not isinstance(audio, str) or not audio:
            raise ValueError(f"{path}: {clip_id} 缺 audio")
        audio_path = (path.parent / audio).resolve()
        if not audio_path.is_file():
            raise ValueError(f"{path}: {clip_id} 音频不存在: {audio_path}")
        loaded.append({
            "id": clip_id, "wav": audio_path, "reference_text": reference,
            "reference_sha256": hashlib.sha256(reference.encode("utf-8")).hexdigest(),
        })
    return loaded


def _wer_words(text: str) -> list[str]:
    """英文 WER 的固定归一化：casefold、忽略标点、保留词内 apostrophe。"""
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.casefold().replace("’", "'"))


def word_error_rate(reference: str, hypothesis: str) -> dict[str, int | float]:
    """按词 Levenshtein 对齐；WER = (S + D + I) / 参考词数。"""
    ref, hyp = _wer_words(reference), _wer_words(hypothesis)
    if not ref:
        raise ValueError("WER 的 reference_text 归一化后为空")
    # cell = (distance, substitutions, deletions, insertions)
    rows = [[(0, 0, 0, 0)] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        rows[i][0] = (i, 0, i, 0)
    for j in range(1, len(hyp) + 1):
        rows[0][j] = (j, 0, 0, j)
    for i, ref_word in enumerate(ref, 1):
        for j, hyp_word in enumerate(hyp, 1):
            if ref_word == hyp_word:
                rows[i][j] = rows[i - 1][j - 1]
                continue
            sub = rows[i - 1][j - 1]
            delete = rows[i - 1][j]
            insert = rows[i][j - 1]
            options = [
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            ]
            rows[i][j] = min(options)
    distance, substitutions, deletions, insertions = rows[-1][-1]
    return {
        "reference_words": len(ref), "hypothesis_words": len(hyp),
        "substitutions": substitutions, "deletions": deletions, "insertions": insertions,
        "word_error_rate": round(distance / len(ref), 4),
    }


def initial_prompt_token_count(prompt: str | None, language: str) -> int:
    """用实际 Whisper multilingual tokenizer 计 initial_prompt，不拿字符数冒充 token。"""
    if not prompt:
        return 0
    from mlx_whisper.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(True, language=language, task="transcribe")
    return len(tokenizer.encode(prompt))


def public_report_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """生成可提交的 ASR 结果摘要，绝不携带用户原稿、转写或 prompt 文本。"""
    public_results = []
    for result in payload["results"]:
        samples = [{
            "transcribe_s": sample["transcribe_s"],
            "rtf": sample["rtf"],
            "word_error_rate": sample["word_error_rate"],
        } for sample in result["samples"]]
        public_results.append({
            "clip": result["clip"],
            "language": result["language"],
            "input_kind": result["input_kind"],
            "audio_seconds": result["audio_seconds"],
            "reference_sha256": result["reference_sha256"],
            "initial_prompt_tokens": result["initial_prompt_tokens"],
            "glossary_biased": result["glossary_biased"],
            "runs": result["runs"],
            "cold": result["cold"],
            "median": result["median"],
            "samples": samples,
        })
    return {
        "model": payload["model"],
        "runtime": payload["runtime"],
        "language": payload["language"],
        "input_kind": payload["input_kind"],
        "initial_prompt_tokens": payload["initial_prompt_tokens"],
        "results": public_results,
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


def _synthetic_zh_clips() -> list[dict[str, Any]]:
    """仅保留历史中文合成素材，明确不作为英文准确率基准。"""
    clips = []
    for name, text in ZH_SYNTHETIC_UTTERANCES.items():
        wav = synth(name, text)
        clips.append({"id": name, "wav": wav, "reference_text": text})
    return clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--language", choices=sorted(LANGUAGE_GLOSSARIES), default="zh",
                    help="会话选择传入 Whisper 的语言；英文必须配 --manifest 真人录音")
    ap.add_argument("--manifest", type=Path,
                    help="真人录音清单（格式见 bench/asr_real_manifest.example.yaml）")
    ap.add_argument("--clip", action="append", default=[], metavar="ID",
                    help="只跑 manifest 中指定 clip（可重复，用于可恢复地复跑长录音）")
    ap.add_argument("--public-summary", type=Path,
                    help="写入不含原稿、转写或 prompt 文本的可提交摘要 JSON")
    ap.add_argument("--summarize-report", type=Path,
                    help="从已有私有报告生成 --public-summary，不加载模型或音频")
    ap.add_argument(
        "--compare-glossary",
        action="store_true",
        help="对比开启/关闭技术术语词表偏置的转写差异",
    )
    args = ap.parse_args()

    if args.summarize_report:
        if not args.public_summary:
            raise SystemExit("--summarize-report 必须同时提供 --public-summary")
        try:
            private_payload = json.loads(args.summarize_report.read_text(encoding="utf-8"))
            summary = public_report_summary(private_payload)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SystemExit(f"私有 ASR 报告无效: {exc}") from exc
        args.public_summary.parent.mkdir(parents=True, exist_ok=True)
        args.public_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"公开摘要已写入 {args.public_summary}")
        return

    if args.manifest:
        try:
            clips = load_real_manifest(args.manifest, args.language)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise SystemExit(f"真人录音 manifest 无效: {exc}") from exc
        input_kind = "real_recording"
    elif args.language == "zh":
        clips = _synthetic_zh_clips()
        input_kind = "synthetic_legacy"
    else:
        raise SystemExit(
            "英文 ASR 基准需要 --manifest 真人录音；仓库没有英文真实音频，"
            f"请先按 {REAL_MANIFEST_EXAMPLE} 录制并登记。"
        )
    if args.clip:
        requested = set(args.clip)
        known = {clip["id"] for clip in clips}
        unknown = sorted(requested - known)
        if unknown:
            raise SystemExit(f"未知 clip id: {', '.join(unknown)}")
        clips = [clip for clip in clips if clip["id"] in requested]
    for clip in clips:
        clip["audio_seconds"] = duration_s(clip["wav"])
        print(f"素材 {clip['id']}: {clip['audio_seconds']:.2f}s")

    # 先验证输入资产再加载 Metal：没有英文真人录音时应给出可行动的失败信息，
    # 而不是在无 GPU 环境先报一个与语料无关的导入错误。
    import mlx_whisper

    results = []
    options = transcribe_options(args.language, glossary=args.compare_glossary)
    prompt_tokens = initial_prompt_token_count(options["initial_prompt"], args.language)
    for clip in clips:
        name, wav = clip["id"], clip["wav"]
        dur, reference = clip["audio_seconds"], clip["reference_text"]
        reset_peak_memory()
        transcription_samples: list[dict[str, Any]] = []

        def once() -> dict[str, float]:
            t0 = time.perf_counter()
            out = mlx_whisper.transcribe(
                str(wav),
                path_or_hf_repo=args.model,
                **options,
            )
            elapsed = time.perf_counter() - t0
            once.text = out["text"]
            wer = word_error_rate(reference, once.text)
            transcription_samples.append({"transcribed_text": once.text, **wer})
            return {
                "transcribe_s": round(elapsed, 4), "rtf": round(dur / elapsed, 2),
                "word_error_rate": wer["word_error_rate"],
            }

        print(f"  测量 {name} ...", flush=True)
        stats = repeat(once, args.runs)
        entry = {
            "clip": name,
            "language": args.language,
            "input_kind": input_kind,
            "audio_seconds": round(dur, 2),
            "reference_sha256": clip.get("reference_sha256"),
            "initial_prompt_tokens": prompt_tokens,
            "peak_memory_gb": peak_memory_gb(),
            "glossary_biased": args.compare_glossary,
            "reference_text": reference,
            "transcribed_text": getattr(once, "text", ""),
            "transcription_samples": transcription_samples,
            **stats,
        }
        results.append(entry)
        m = stats["median"]
        print(f"    转写 {m['transcribe_s']}s | 实时率 {m['rtf']}x | 冷启动 {stats['cold']['transcribe_s']}s")
        print(f"    识别: {entry['transcribed_text'][:60]}")

    report_payload = {
        "model": args.model, "runtime": "mlx-whisper", "language": args.language,
        "input_kind": input_kind, "initial_prompt": options["initial_prompt"],
        "initial_prompt_tokens": prompt_tokens, "results": results,
    }
    path = write_report("asr", report_payload)
    print(f"\n报告已写入 {path}")
    if args.public_summary:
        args.public_summary.parent.mkdir(parents=True, exist_ok=True)
        args.public_summary.write_text(
            json.dumps(public_report_summary(report_payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"公开摘要已写入 {args.public_summary}")

    warm = [r["median"]["rtf"] for r in results]
    print(f"结论: 热态实时率 {min(warm)}x ~ {max(warm)}x")
    if min(warm) < 3:
        print("警告: 实时率低于 3x，I4 的流式转写很难压进 0.7s 预算，考虑换更小的模型")


if __name__ == "__main__":
    main()
