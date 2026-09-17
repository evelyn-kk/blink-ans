"""ASR final 尾部复用的质量实验（T-008 R178）。

问题：静音开始时先把语音尾巴作为 partial 转写掉，final 追加的若全是静音，就复用那次
partial 的文本而不再转写。代价是被复用的转写少看了尾部那段静音。这里量它与"现在的
final"（片段 + 1.2 s 尾部）的词级差异，**不量对人工参考的 WER**：参考稿按整段录音给出，
切不出与片段对齐的逐句参考。差异不等于变差，只回答"会不会改变输出"。

切段：块大小、RMS 阈值与 PWA 一致（2730 样本 / 0.015）。真实录音多为连续讲话，≥1.2 s
停顿极少，因此在 ≥2 个安静块的真实停顿处切 3~20 s 片段，再用同一录音里的真实安静块
补齐 VAD 停止所需的 8 个尾部块。报告只含位置、词数与编辑距离，不含任何文本。

用法：.venv/bin/python bench/bench_asr_tail_reuse.py --pcm clip=path.s16le ... \
        --language en --per-clip 40 --json bench/reports/asr-tail-reuse.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.bench_asr import word_error_rate  # noqa: E402
from services.asr.stream import mlx_whisper_transcriber  # noqa: E402

SAMPLE_RATE = 16_000
BLOCK = 2730            # apps/pwa ScriptProcessor 8192@48 kHz 的等时长
SPEECH_RMS = 0.015      # apps/pwa VAD_SPEECH_RMS
STOP_BLOCKS = 8         # 1.2 s 静音 = ceil(1200 / 170.6) 块
MIN_PAUSE_BLOCKS = 2
MIN_SEGMENT_S, MAX_SEGMENT_S = 3.0, 20.0
TAIL_KS = (0, 2, 5)


def block_rms(x: np.ndarray) -> np.ndarray:
    n = len(x) // BLOCK
    return np.sqrt((x[: n * BLOCK].reshape(n, BLOCK) ** 2).mean(axis=1))


def segments(rms: np.ndarray, limit: int) -> list[tuple[int, int]]:
    """返回 (起始块, 最后语音块)；片段后紧跟至少 MIN_PAUSE_BLOCKS 个真实安静块。"""
    speech = rms >= SPEECH_RMS
    found, start, i = [], None, 0
    while i < len(rms) and len(found) < limit:
        if start is None:
            if speech[i]:
                start = i
            i += 1
            continue
        length_s = (i - start) * BLOCK / SAMPLE_RATE
        if length_s > MAX_SEGMENT_S:
            start = None
            continue
        pause = speech[i : i + MIN_PAUSE_BLOCKS]
        if len(pause) == MIN_PAUSE_BLOCKS and not pause.any() and length_s >= MIN_SEGMENT_S and speech[i - 1]:
            found.append((start, i - 1))
            start = None
            i += MIN_PAUSE_BLOCKS
            continue
        i += 1
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcm", action="append", required=True, help="clip_id=path (16 kHz mono s16le)")
    ap.add_argument("--language", required=True, choices=["zh", "en"])
    ap.add_argument("--per-clip", type=int, default=40)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    rows = []
    for spec in args.pcm:
        clip, path = spec.split("=", 1)
        x = np.fromfile(path, dtype="<i2").astype(np.float32) / 32768
        rms = block_rms(x)
        quiet = [i for i in np.flatnonzero(rms < SPEECH_RMS)]
        if len(quiet) < STOP_BLOCKS:
            continue
        for seg_i, (start, last) in enumerate(segments(rms, args.per_clip)):
            # 尾部：先用片段后面真实的安静块，不足 8 块再依次取同一录音里的其他安静块。
            tail = []
            j = last + 1
            while j < len(rms) and rms[j] < SPEECH_RMS and len(tail) < STOP_BLOCKS:
                tail.append(j)
                j += 1
            filler = [q for q in quiet if q not in tail and not (start <= q <= last)]
            tail += filler[: STOP_BLOCKS - len(tail)]
            speech_audio = x[start * BLOCK : (last + 1) * BLOCK]
            tail_audio = [x[q * BLOCK : (q + 1) * BLOCK] for q in tail]

            def upto(k: int) -> np.ndarray:
                return np.concatenate([speech_audio, *tail_audio[:k]])

            started = time.perf_counter()
            baseline = mlx_whisper_transcriber(upto(STOP_BLOCKS), language=args.language)
            baseline_ms = round((time.perf_counter() - started) * 1000, 1)
            row = {
                "clip": clip, "segment": seg_i,
                "start_s": round(start * BLOCK / SAMPLE_RATE, 2),
                "speech_s": round((last + 1 - start) * BLOCK / SAMPLE_RATE, 2),
                "real_tail_blocks": sum(1 for q in tail if q > last and q <= last + STOP_BLOCKS),
                "baseline_ms": baseline_ms, "tails": {},
            }
            for k in TAIL_KS:
                started = time.perf_counter()
                hyp = mlx_whisper_transcriber(upto(k), language=args.language)
                elapsed = round((time.perf_counter() - started) * 1000, 1)
                try:
                    diff = word_error_rate(baseline, hyp)
                except ValueError:  # baseline 归一化后为空（例如只有标点）
                    diff = {"reference_words": 0, "hypothesis_words": len(hyp.split()),
                            "substitutions": 0, "deletions": 0, "insertions": 0, "word_error_rate": None}
                row["tails"][str(k)] = {**diff, "exact": hyp.strip() == baseline.strip(), "ms": elapsed}
            rows.append(row)
            print(json.dumps({"clip": clip, "segment": seg_i, **{k: v["word_error_rate"] for k, v in row["tails"].items()}}),
                  flush=True)

    summary = {}
    for k in map(str, TAIL_KS):
        ref_words = sum(r["tails"][k]["reference_words"] for r in rows)
        edits = sum(r["tails"][k]["substitutions"] + r["tails"][k]["deletions"] + r["tails"][k]["insertions"] for r in rows)
        summary[k] = {
            "segments": len(rows),
            "baseline_words": ref_words,
            "edit_distance": edits,
            "disagreement_rate": round(edits / ref_words, 4) if ref_words else None,
            "deletions": sum(r["tails"][k]["deletions"] for r in rows),
            "insertions": sum(r["tails"][k]["insertions"] for r in rows),
            "substitutions": sum(r["tails"][k]["substitutions"] for r in rows),
            "segments_with_any_word_diff": sum(
                1 for r in rows
                if r["tails"][k]["substitutions"] + r["tails"][k]["deletions"] + r["tails"][k]["insertions"]
            ),
            "exact_text_match": sum(1 for r in rows if r["tails"][k]["exact"]),
        }
    Path(args.json).write_text(json.dumps({
        "language": args.language, "block": BLOCK, "speech_rms": SPEECH_RMS, "stop_blocks": STOP_BLOCKS,
        "tail_ks": TAIL_KS, "summary": summary, "segments": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
