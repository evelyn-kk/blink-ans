"""顺序运行真人 ASR 基准；以子进程退出和去敏结果完整为完成判据。

终端输出可能在 Metal 子进程仍工作时先断开，不能据此认定“完成”或“失败”。本工具每个
clip/arm 单独启动 `bench_asr.py`，只有子进程 exit 0 且它生成的去敏摘要完整才以原子替换写入
结果；timeout 或不完整结果不会覆盖已存在的成功摘要。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from bench_asr import duration_s, load_real_manifest

BENCH_ASR = Path(__file__).with_name("bench_asr.py")
TOP_LEVEL_KEYS = {"model", "runtime", "language", "input_kind", "initial_prompt_tokens", "results"}
RESULT_KEYS = {
    "clip", "language", "input_kind", "audio_seconds", "reference_sha256", "initial_prompt_tokens",
    "glossary_biased", "runs", "cold", "median", "samples",
}
SAMPLE_KEYS = {"transcribe_s", "rtf", "word_error_rate"}


def default_timeout_s(audio_seconds: float, runs: int) -> int:
    """按保守 2x 实时（含 120 秒加载余量）给每个 clip/arm 明确超时。"""
    return max(180, math.ceil(audio_seconds * runs / 2 + 120))


def _assert_public_summary(path: Path, clip_id: str, glossary: bool, runs: int) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _exact_object(payload, TOP_LEVEL_KEYS, "顶层")
    for key in ("model", "runtime", "language", "input_kind"):
        _string(payload[key], f"顶层.{key}")
    _nonnegative_int(payload["initial_prompt_tokens"], "顶层.initial_prompt_tokens")
    if not isinstance(payload["results"], list) or len(payload["results"]) != 1:
        raise ValueError("公开摘要必须恰好包含一个 clip")
    result = payload["results"][0]
    _exact_object(result, RESULT_KEYS, "result")
    for key in ("clip", "language", "input_kind", "reference_sha256"):
        _string(result[key], f"result.{key}")
    for key in ("audio_seconds",):
        _nonnegative_number(result[key], f"result.{key}")
    _nonnegative_int(result["initial_prompt_tokens"], "result.initial_prompt_tokens")
    if type(result["glossary_biased"]) is not bool:
        raise ValueError("result.glossary_biased 必须为 bool")
    _positive_int(result["runs"], "result.runs")
    for key in ("cold", "median"):
        _sample(result[key], f"result.{key}")
    if not isinstance(result["samples"], list):
        raise ValueError("result.samples 必须为 list")
    for index, sample in enumerate(result["samples"]):
        _sample(sample, f"result.samples[{index}]")
    if result["clip"] != clip_id or result["glossary_biased"] is not glossary:
        raise ValueError("公开摘要的 clip 或 arm 与请求不一致")
    if result["runs"] != runs or len(result["samples"]) != runs:
        raise ValueError("公开摘要没有完整记录请求次数")


def _exact_object(value: Any, allowed: set[str], where: str) -> None:
    if not isinstance(value, dict) or set(value) != allowed:
        raise ValueError(f"{where} 字段不完整或含未登记字段")


def _string(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} 必须为非空字符串")


def _nonnegative_number(value: Any, where: str) -> None:
    if type(value) not in (int, float) or value < 0:
        raise ValueError(f"{where} 必须为非负数")


def _nonnegative_int(value: Any, where: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{where} 必须为非负整数")


def _positive_int(value: Any, where: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{where} 必须为正整数")


def _sample(value: Any, where: str) -> None:
    _exact_object(value, SAMPLE_KEYS, where)
    for key in SAMPLE_KEYS:
        _nonnegative_number(value[key], f"{where}.{key}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_case(
    *, manifest: Path, clip_id: str, glossary: bool, runs: int, timeout_s: int, output_path: Path,
    subprocess_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """跑一个 arm；只在 exit 0 + 完整去敏摘要时替换既有成功产物。"""
    candidate = output_path.with_name(f".{output_path.name}.candidate")
    candidate.unlink(missing_ok=True)
    command = [
        sys.executable, str(BENCH_ASR), "--language", "en", "--manifest", str(manifest),
        "--clip", clip_id, "--runs", str(runs), "--public-summary", str(candidate),
    ]
    if glossary:
        command.append("--compare-glossary")
    started = time.monotonic()
    base = {
        "clip": clip_id, "arm": "glossary" if glossary else "no_prompt", "runs_requested": runs,
        "timeout_s": timeout_s,
    }
    try:
        completed = subprocess_run(command, capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        candidate.unlink(missing_ok=True)
        return {**base, "status": "timeout", "elapsed_s": round(time.monotonic() - started, 4)}
    elapsed = round(time.monotonic() - started, 4)
    if completed.returncode != 0:
        candidate.unlink(missing_ok=True)
        return {**base, "status": "subprocess_exit", "exit_code": completed.returncode, "elapsed_s": elapsed}
    try:
        _assert_public_summary(candidate, clip_id, glossary, runs)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        candidate.unlink(missing_ok=True)
        return {**base, "status": "incomplete_public_summary", "elapsed_s": elapsed}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(candidate, output_path)
    return {**base, "status": "completed", "elapsed_s": elapsed, "summary": output_path.name}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=1, help="每个 clip/arm 的完整次数；R112 先用 1")
    ap.add_argument("--timeout-s", type=int, help="覆盖按音频长度计算的每 clip/arm timeout")
    args = ap.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs 必须至少为 1")
    try:
        clips = load_real_manifest(args.manifest, "en")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"真人录音 manifest 无效: {exc}") from exc

    index_path = args.output_dir / "index.json"
    index: dict[str, Any] = {"manifest": args.manifest.name, "runs_requested": args.runs, "cases": []}
    for clip in clips:
        audio_seconds = duration_s(clip["wav"])
        timeout_s = args.timeout_s or default_timeout_s(audio_seconds, args.runs)
        for glossary in (False, True):
            arm = "glossary" if glossary else "no_prompt"
            result = run_case(
                manifest=args.manifest, clip_id=clip["id"], glossary=glossary, runs=args.runs,
                timeout_s=timeout_s, output_path=args.output_dir / f"{clip['id']}--{arm}.json",
            )
            result["audio_seconds"] = round(audio_seconds, 2)
            result["reference_sha256"] = clip["reference_sha256"]
            index["cases"].append(result)
            _atomic_json(index_path, index)
            print(f"{clip['id']} {arm}: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
