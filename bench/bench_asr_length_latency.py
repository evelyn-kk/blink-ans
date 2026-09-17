"""转写耗时随话语长度的变化（T-008 R179）。

`architecture.md` §6.5 的 `vad_end→asr_final` 预算 0.7 s，依据写的是"转写实时率 4–16x"。
实时率 = 音频时长 / 转写耗时，而 Whisper 的编码器对任何 ≤30 s 的输入都按同一个 30 s 窗口
计算，所以耗时近似与长度无关，实时率就随长度线性变化——同一个模型在 7 s 话语上必然比在
20 s 话语上"实时率更低"。这个脚本按长度分档实测耗时，用来判断 0.7 s 这一行该怎么定。

素材用真实录音的语音段（避免纯静音那条更慢的路径，见 R178）。报告只含时长、耗时与词数。

用法：.venv/bin/python bench/bench_asr_length_latency.py --pcm clip=path.s16le ... \
        --language en --runs 3 --json bench/reports/asr-length.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.asr.stream import MAX_SEGMENT_SECONDS, SAMPLE_RATE  # noqa: E402
from services.asr.stream import mlx_whisper_transcriber  # noqa: E402

LENGTHS_S = (2, 5, 7, 10, 15, 20, 25, 29)
BLOCK = 2730
SPEECH_RMS = 0.015


def first_speech_offset(x: np.ndarray) -> int:
    n = len(x) // BLOCK
    rms = np.sqrt((x[: n * BLOCK].reshape(n, BLOCK) ** 2).mean(axis=1))
    speech = np.flatnonzero(rms >= SPEECH_RMS)
    return int(speech[0]) * BLOCK if len(speech) else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcm", action="append", required=True, help="clip_id=path (16 kHz mono s16le)")
    ap.add_argument("--language", required=True, choices=["zh", "en"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    clips = []
    for spec in args.pcm:
        clip, path = spec.split("=", 1)
        x = np.fromfile(path, dtype="<i2").astype(np.float32) / 32768
        clips.append((clip, x[first_speech_offset(x):]))

    # 预热：第一次调用要加载模型，不能混进分档样本。
    mlx_whisper_transcriber(clips[0][1][: SAMPLE_RATE * 3], language=args.language)

    rows = []
    for seconds in LENGTHS_S:
        if seconds > MAX_SEGMENT_SECONDS:
            continue
        for clip, x in clips:
            audio = x[: SAMPLE_RATE * seconds]
            if len(audio) < SAMPLE_RATE * seconds:
                continue
            samples_ms, words = [], []
            for _ in range(args.runs):
                started = time.perf_counter()
                text = mlx_whisper_transcriber(audio, language=args.language)
                samples_ms.append(round((time.perf_counter() - started) * 1000, 1))
                words.append(len(text.split()))
            rows.append({
                "clip": clip, "audio_s": seconds, "ms": samples_ms,
                "median_ms": statistics.median(samples_ms),
                "rtf": round(seconds / (statistics.median(samples_ms) / 1000), 2),
                "words": words,
            })
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    by_length = {}
    for row in rows:
        by_length.setdefault(row["audio_s"], []).extend(row["ms"])
    summary = {
        str(k): {
            "samples": len(v), "min_ms": min(v), "median_ms": statistics.median(v), "max_ms": max(v),
            "rtf_at_median": round(k / (statistics.median(v) / 1000), 2),
        }
        for k, v in sorted(by_length.items())
    }
    Path(args.json).write_text(json.dumps({
        "language": args.language, "runs_per_cell": args.runs, "model": "mlx-community/whisper-large-v3-turbo",
        "summary": summary, "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
