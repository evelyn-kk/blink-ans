"""投机检索的收益上限与代价（T-008 R180，CR-158 裁定的选项 3 前置证据）。

设想：`vad_end` 之前就用最近一次 partial 转写去跑检索，final 文本若与之相同就复用结果，
从而让检索不再串行加在 ASR 之后。这个脚本只回答两个能离线量的问题，**不改产品代码**：

1. **命中率上限**：partial 比 final 少说几个词时，检索 top-k 还是不是同一批块？少的词数按
   R177 的实际时序估计——最后一个 partial 在途约 1 s，这段时间说的话不在它里面。
2. **争用代价**：嵌入模型与 Whisper 共用同一块 GPU。投机检索与转写并发时，转写会不会变慢？
   省下的检索时间如果被转写吃掉，这条路就不成立。

用法：.venv/bin/python bench/bench_speculative_retrieval.py --json bench/reports/spec-retrieval.json \
        [--pcm clip=path.s16le] [--drop 1 2 3 4 5] [--runs 5]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.retrieval.embed import Embedder  # noqa: E402
from services.retrieval.search import hybrid_search  # noqa: E402
from services.retrieval.store import ChunkStore  # noqa: E402
from services.retrieval.tokenize import detect_technology  # noqa: E402

QUESTION_FILES = (
    ROOT / "knowledge" / "eval" / "basic_questions.yaml",
    ROOT / "knowledge" / "eval" / "scenario_questions.yaml",
)
TOP_K = 10          # 与 Orchestrator 一致：max_evidence(5) * 2
CANDIDATES = 30     # AnswerConfig.candidates


def questions() -> list[tuple[str, str, str]]:
    """(题号, 语言, 题面)；中英分别计，因为截断方式不同。

    两份题库的 schema 不同：`basic_questions.yaml` 是 `id` + `q_zh`/`q_en`，
    `scenario_questions.yaml` 只有中文 `q`、没有 `id`。两种都要收，漏掉一种会让
    "覆盖登记题面"这句话名不副实（R180 第一次跑就静默漏了后者）。
    """
    out = []
    for path in QUESTION_FILES:
        if not path.is_file():
            raise SystemExit(f"题库不存在：{path}")
        specs = yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]
        for i, spec in enumerate(specs):
            found = [(lang, spec[key]) for lang, key in (("zh", "q_zh"), ("en", "q_en"), ("zh", "q"))
                     if spec.get(key)]
            if not found:
                raise SystemExit(f"{path} 第 {i} 题没有可用题面字段：{sorted(spec)}")
            for lang, text in found:
                out.append((spec.get("id", f"{path.stem}-{i}"), lang, text))
    return out


def truncate(text: str, lang: str, drop: int) -> str:
    """去掉末尾 drop 个"词"：英文按空格，中文按字。"""
    if lang == "en":
        words = text.split()
        return " ".join(words[:-drop]) if drop < len(words) else ""
    stripped = re.sub(r"[?？。！!，,]+$", "", text)
    return stripped[:-drop] if drop < len(stripped) else ""


def top_rowids(store, embedder, text: str) -> list[int]:
    hits = hybrid_search(
        store, text, embedder.encode_one(text), limit=TOP_K,
        technology=detect_technology(text), candidates=CANDIDATES,
    )
    return [h.rowid for h in hits]


def agreement(store, embedder, drops: list[int]) -> dict:
    rows = []
    for qid, lang, text in questions():
        full = top_rowids(store, embedder, text)
        entry = {"id": qid, "lang": lang, "full_top1": full[0] if full else None, "drops": {}}
        for drop in drops:
            partial = truncate(text, lang, drop)
            if not partial.strip():
                continue
            got = top_rowids(store, embedder, partial)
            entry["drops"][str(drop)] = {
                "same_top1": bool(got and full and got[0] == full[0]),
                "same_top5_set": set(got[:5]) == set(full[:5]),
                "same_top10_order": got == full,
                "overlap5": len(set(got[:5]) & set(full[:5])),
            }
        rows.append(entry)
    summary = {}
    for drop in map(str, drops):
        cells = [r["drops"][drop] for r in rows if drop in r["drops"]]
        summary[drop] = {
            "questions": len(cells),
            "same_top1": sum(c["same_top1"] for c in cells),
            "same_top5_set": sum(c["same_top5_set"] for c in cells),
            "same_top10_order": sum(c["same_top10_order"] for c in cells),
            "mean_overlap5": round(statistics.mean(c["overlap5"] for c in cells), 2) if cells else None,
        }
    return {"summary": summary, "rows": rows}


def contention(store, embedder, pcm_path: str, runs: int) -> dict:
    """转写单独跑 vs 转写期间持续跑检索：只比较转写耗时。"""
    from services.asr.stream import SAMPLE_RATE, mlx_whisper_transcriber

    x = np.fromfile(pcm_path, dtype="<i2").astype(np.float32) / 32768
    audio = x[: SAMPLE_RATE * 7]
    query = "Kafka 消费者重复消费怎么排查"
    mlx_whisper_transcriber(audio, language="en")        # warm
    top_rowids(store, embedder, query)                   # warm

    alone = []
    for _ in range(runs):
        started = time.perf_counter()
        mlx_whisper_transcriber(audio, language="en")
        alone.append(round((time.perf_counter() - started) * 1000, 1))

    with_retrieval, retrieval_counts = [], []
    for _ in range(runs):
        stop = threading.Event()
        count = [0]

        def spin():
            while not stop.is_set():
                top_rowids(store, embedder, query)
                count[0] += 1

        worker = threading.Thread(target=spin)
        worker.start()
        started = time.perf_counter()
        mlx_whisper_transcriber(audio, language="en")
        with_retrieval.append(round((time.perf_counter() - started) * 1000, 1))
        stop.set()
        worker.join()
        retrieval_counts.append(count[0])

    solo_retrieval = []
    for _ in range(runs):
        started = time.perf_counter()
        top_rowids(store, embedder, query)
        solo_retrieval.append(round((time.perf_counter() - started) * 1000, 1))

    return {
        "audio_s": 7, "runs": runs,
        "transcribe_alone_ms": alone,
        "transcribe_with_concurrent_retrieval_ms": with_retrieval,
        "retrievals_completed_during_each_transcription": retrieval_counts,
        "retrieval_alone_ms": solo_retrieval,
        "median_transcribe_alone_ms": statistics.median(alone),
        "median_transcribe_with_retrieval_ms": statistics.median(with_retrieval),
        "median_retrieval_alone_ms": statistics.median(solo_retrieval),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--drop", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--pcm", help="clip_id=path，给争用测量用；省略则跳过")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    embedder = Embedder()
    embedder.load()
    if embedder.error:
        raise SystemExit(f"嵌入模型不可用：{embedder.error}")
    store = ChunkStore()
    try:
        payload = {"top_k": TOP_K, "candidates": CANDIDATES, "agreement": agreement(store, embedder, args.drop)}
        if args.pcm:
            payload["contention"] = contention(store, embedder, args.pcm.split("=", 1)[1], args.runs)
    finally:
        store.close()
    Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"agreement": payload["agreement"]["summary"],
                      "contention": {k: v for k, v in payload.get("contention", {}).items() if k.startswith("median")
                                     or k.startswith("retrievals")}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
