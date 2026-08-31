"""基准脚本共用工具：机器信息采集、内存采样、报告落盘。

所有基准报告使用同一 JSON 结构，便于换模型后横向对比与回归。
"""

from __future__ import annotations

import json
import platform
import subprocess
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPORT_DIR = Path(__file__).resolve().parent / "reports"


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def machine_info() -> dict[str, Any]:
    mem = _sysctl("hw.memsize")
    return {
        "chip": _sysctl("machdep.cpu.brand_string"),
        "cores": _sysctl("hw.ncpu"),
        "memory_gb": round(int(mem) / 1024**3, 1) if mem else None,
        "os": f"{platform.system()} {platform.mac_ver()[0]}",
        "python": platform.python_version(),
    }


def peak_memory_gb() -> float | None:
    """读取 MLX 的 Metal 峰值内存；跨版本兼容 API 位置变化。"""
    try:
        import mlx.core as mx
    except ImportError:
        return None
    for getter in (
        getattr(mx, "get_peak_memory", None),
        getattr(getattr(mx, "metal", None), "get_peak_memory", None),
    ):
        if callable(getter):
            try:
                return round(getter() / 1024**3, 3)
            except Exception:
                continue
    return None


def reset_peak_memory() -> None:
    try:
        import mlx.core as mx
    except ImportError:
        return
    for resetter in (
        getattr(mx, "reset_peak_memory", None),
        getattr(getattr(mx, "metal", None), "reset_peak_memory", None),
    ):
        if callable(resetter):
            try:
                resetter()
                return
            except Exception:
                continue


def repeat(fn: Callable[[], dict[str, float]], runs: int) -> dict[str, Any]:
    """重复测量并按指标取中位数；同时保留冷启动（首次）和全部原始值。"""
    samples = [fn() for _ in range(runs)]
    keys = samples[0].keys()
    return {
        "runs": runs,
        "cold": samples[0],
        "median": {k: round(statistics.median(s[k] for s in samples), 4) for k in keys},
        "samples": samples,
    }


def write_report(component: str, payload: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "component": component,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "machine": machine_info(),
        **payload,
    }
    path = REPORT_DIR / f"{component}-{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
