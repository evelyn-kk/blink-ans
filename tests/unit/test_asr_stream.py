"""T-008：PCM 分片转写不落盘、上限与释放行为。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.asr.stream import (  # noqa: E402
    MAX_PCM_BYTES,
    TranscriptSession,
)


def _pcm(values: list[int]) -> bytes:
    return np.asarray(values, dtype="<i2").tobytes()


def test_partial_and_final_use_in_memory_pcm_and_final_releases_audio():
    observed: list[tuple[np.ndarray, str]] = []

    def fake(waveform, *, language):
        observed.append((waveform.copy(), language))
        return f"chunk-{len(observed)}"

    session = TranscriptSession("zh", fake)
    partial = session.append(_pcm([0, 16384]), final=False)
    final = session.append(_pcm([-16384]), final=True)

    assert partial.as_sse_event() == {
        "type": "transcript", "text": "chunk-1", "final": False, "sequence": 1,
    }
    assert final.as_sse_event() == {
        "type": "transcript", "text": "chunk-2", "final": True, "sequence": 2,
    }
    assert observed[0][1] == observed[1][1] == "zh"
    assert observed[1][0].tolist() == pytest.approx([0.0, 0.5, -0.5])
    assert session.buffered_pcm_bytes == 0
    assert session.finished is True
    with pytest.raises(RuntimeError, match="already final"):
        session.append(_pcm([1]))


def test_cancel_releases_pcm_without_calling_model_again():
    calls = 0

    def fake(_waveform, *, language):
        nonlocal calls
        calls += 1
        return language

    session = TranscriptSession("en", fake)
    session.append(_pcm([1]), final=False)
    session.cancel()

    assert calls == 1
    assert session.buffered_pcm_bytes == 0
    assert session.finished is True


def test_rejects_odd_pcm_and_overflow_before_transcribing():
    calls = 0

    def fake(_waveform, *, language):
        nonlocal calls
        calls += 1
        return language

    session = TranscriptSession("zh", fake)
    with pytest.raises(ValueError, match="whole 16-bit"):
        session.append(b"x")
    assert calls == 0

    with pytest.raises(ValueError, match="exceeds"):
        session.append(b"\0" * (MAX_PCM_BYTES + 2))
    assert calls == 0
