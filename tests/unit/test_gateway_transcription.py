"""T-008 转写 API：真实路由控制流，但永不加载 mlx-whisper。"""

from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import apps.gateway.main as gw  # noqa: E402
from services.asr.stream import TranscriptSession  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


async def _drain(response):
    return [chunk async for chunk in response.body_iterator]


@pytest.fixture(autouse=True)
def _isolated_transcripts(monkeypatch):
    monkeypatch.setattr(gw, "_transcripts", {})
    monkeypatch.setattr(gw, "_pending_transcript_events", {})
    monkeypatch.setattr(gw, "_voice_stage_marks", {})
    seen: list[TranscriptSession] = []

    def create(language):
        session = TranscriptSession(language, lambda wave, *, language: f"{language}:{len(wave)}")
        seen.append(session)
        return session

    monkeypatch.setattr(gw, "_new_transcript_session", create)
    return seen


def test_chunk_routes_in_memory_transcript_through_one_time_sse(_isolated_transcripts):
    created = _run(gw.create_transcription(gw.TranscriptCreateBody(language="zh")))
    tid = created["transcript_id"]
    pcm = base64.b64encode(b"\0\0\x00@").decode()
    written = _run(gw.append_transcript_chunk(tid, gw.TranscriptChunkBody(pcm_s16le_b64=pcm)))

    response = _run(gw.stream_transcript_event(tid, written["event_id"]))
    chunks = _run(_drain(response))
    assert 'event: transcript' in chunks[0]
    assert '"text": "zh:2"' in chunks[0]
    assert "pcm" not in chunks[0]
    assert _isolated_transcripts[0].buffered_pcm_bytes == 4

    with pytest.raises(HTTPException) as exc:
        _run(gw.stream_transcript_event(tid, written["event_id"]))
    assert exc.value.status_code == 404


def test_final_and_cancel_release_audio_and_reject_later_chunks(_isolated_transcripts):
    tid = _run(gw.create_transcription(gw.TranscriptCreateBody(language="en")))["transcript_id"]
    pcm = base64.b64encode(b"\0\0").decode()
    final = _run(gw.append_transcript_chunk(tid, gw.TranscriptChunkBody(pcm_s16le_b64=pcm, final=True)))
    assert _isolated_transcripts[0].buffered_pcm_bytes == 0

    with pytest.raises(HTTPException) as exc:
        _run(gw.append_transcript_chunk(tid, gw.TranscriptChunkBody(pcm_s16le_b64=pcm)))
    assert exc.value.status_code == 409
    _run(_drain(_run(gw.stream_transcript_event(tid, final["event_id"]))))
    assert tid not in gw._transcripts

    tid2 = _run(gw.create_transcription(gw.TranscriptCreateBody(language="en")))["transcript_id"]
    partial = _run(gw.append_transcript_chunk(tid2, gw.TranscriptChunkBody(pcm_s16le_b64=pcm)))
    _run(gw.cancel_transcription(tid2))
    assert _isolated_transcripts[1].buffered_pcm_bytes == 0
    with pytest.raises(HTTPException) as missing:
        _run(gw.append_transcript_chunk(tid2, gw.TranscriptChunkBody(pcm_s16le_b64=pcm)))
    assert missing.value.status_code == 404
    with pytest.raises(HTTPException) as abandoned:
        _run(gw.stream_transcript_event(tid2, partial["event_id"]))
    assert abandoned.value.status_code == 404


def test_final_transcriber_failure_is_503_and_discards_pcm_and_session(monkeypatch):
    """CR-151：final 模型失败不能伪装客户端状态冲突，更不能留住 PCM。"""
    sessions: list[TranscriptSession] = []
    monkeypatch.setattr(gw, "_transcripts", {})
    monkeypatch.setattr(gw, "_pending_transcript_events", {})

    def create(language):
        def fail(_waveform, *, language):
            raise RuntimeError(f"{language} model unavailable")

        session = TranscriptSession(language, fail)
        sessions.append(session)
        return session

    monkeypatch.setattr(gw, "_new_transcript_session", create)
    tid = _run(gw.create_transcription(gw.TranscriptCreateBody(language="zh")))["transcript_id"]
    pcm = base64.b64encode(b"\0\0").decode()

    with pytest.raises(HTTPException) as exc:
        _run(gw.append_transcript_chunk(tid, gw.TranscriptChunkBody(pcm_s16le_b64=pcm, final=True)))

    assert exc.value.status_code == 503
    assert sessions[0].buffered_pcm_bytes == 0
    assert sessions[0].finished is True
    assert tid not in gw._transcripts
    assert not gw._pending_transcript_events


def test_partial_failure_rolls_back_only_failed_chunk_for_safe_resend(monkeypatch):
    """CR-152：真实路由的同一 chunk 重送不得把波形/内存/sequence 加倍。"""
    sessions: list[TranscriptSession] = []
    waveforms: list[list[float]] = []
    attempts = {"n": 0}
    monkeypatch.setattr(gw, "_transcripts", {})
    monkeypatch.setattr(gw, "_pending_transcript_events", {})

    def create(language):
        def flaky(waveform, *, language):
            waveforms.append(waveform.tolist())
            attempts["n"] += 1
            if attempts["n"] == 2:
                raise RuntimeError("temporary local model error")
            return language

        session = TranscriptSession(language, flaky)
        sessions.append(session)
        return session

    monkeypatch.setattr(gw, "_new_transcript_session", create)
    tid = _run(gw.create_transcription(gw.TranscriptCreateBody(language="zh")))["transcript_id"]
    first = base64.b64encode(b"\x00@\x00 ").decode()  # two samples: 0.5, 0.25
    retry = base64.b64encode(b"\0\x40").decode()  # one sample: 0.5

    _run(gw.append_transcript_chunk(tid, gw.TranscriptChunkBody(pcm_s16le_b64=first)))
    with pytest.raises(HTTPException) as failed:
        _run(gw.append_transcript_chunk(tid, gw.TranscriptChunkBody(pcm_s16le_b64=retry)))
    assert failed.value.status_code == 503
    assert sessions[0].buffered_pcm_bytes == 4
    assert sessions[0].finished is False

    resent = _run(gw.append_transcript_chunk(tid, gw.TranscriptChunkBody(pcm_s16le_b64=retry)))
    # 失败尝试和安全重送都只看见“成功前缀 + 一个 retry chunk”，没有第三份 sample。
    assert waveforms[1] == pytest.approx([0.5, 0.25, 0.5])
    assert waveforms[2] == pytest.approx([0.5, 0.25, 0.5])
    assert sessions[0].buffered_pcm_bytes == 6
    assert gw._pending_transcript_events[resent["event_id"]][1]["sequence"] == 2


# ---------- T-008 阶段打点：语音侧时刻经 transcript_id 一次性交给答案请求 ----------

class ManualTime:
    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now


def _voice_answer_setup(monkeypatch):
    clock = ManualTime()
    monkeypatch.setattr(gw, "time", SimpleNamespace(perf_counter=clock.perf_counter))
    monkeypatch.setattr(gw, "_orchestrator", object())
    monkeypatch.setattr(gw, "_pending", {})

    def create(language):
        def transcribe(_waveform, *, language):
            clock.now += 2.5  # 模型推理耗时落在 final 到达与 asr_final 之间
            return "Kafka 怎么排查重复消费"
        return TranscriptSession(language, transcribe)

    monkeypatch.setattr(gw, "_new_transcript_session", create)
    return clock


def _answer(**kw):
    created = _run(gw.create_answer(gw.AskBody(question="Kafka 怎么排查重复消费", **kw)))
    return gw._pending[created["answer_id"]]


def test_final_transcript_marks_flow_once_into_the_voice_answer(monkeypatch):
    clock = _voice_answer_setup(monkeypatch)
    pcm = base64.b64encode(b"\0\0").decode()
    tid = _run(gw.create_transcription(gw.TranscriptCreateBody(language="zh")))["transcript_id"]

    clock.now = 5.0
    _run(gw.append_transcript_chunk(tid, gw.TranscriptChunkBody(pcm_s16le_b64=pcm)))
    assert gw._voice_stage_marks == {}, "partial chunks are not the end of speech"

    clock.now = 10.0
    final = _run(gw.append_transcript_chunk(tid, gw.TranscriptChunkBody(pcm_s16le_b64=pcm, final=True)))
    _run(_drain(_run(gw.stream_transcript_event(tid, final["event_id"]))))

    clock.now = 13.0
    req = _answer(transcript_id=tid)
    assert req.stage_marks == {
        "final_chunk_received": 10.0, "asr_final": 12.5, "answer_requested": 13.0,
    }

    clock.now = 20.0
    again = _answer(transcript_id=tid)
    assert again.stage_marks == {"answer_requested": 20.0}, "voice marks are consumed once"
    assert _answer().stage_marks == {"answer_requested": 20.0}


def test_unknown_transcript_id_still_answers_with_request_origin(monkeypatch):
    clock = _voice_answer_setup(monkeypatch)
    clock.now = 7.0
    assert _answer(transcript_id="missing").stage_marks == {"answer_requested": 7.0}


def test_cancel_and_final_failure_leave_no_voice_marks(monkeypatch):
    clock = _voice_answer_setup(monkeypatch)
    pcm = base64.b64encode(b"\0\0").decode()
    tid = _run(gw.create_transcription(gw.TranscriptCreateBody(language="en")))["transcript_id"]
    _run(gw.append_transcript_chunk(tid, gw.TranscriptChunkBody(pcm_s16le_b64=pcm, final=True)))
    assert tid in gw._voice_stage_marks
    _run(gw.cancel_transcription(tid))
    assert tid not in gw._voice_stage_marks

    def create_failing(language):
        def fail(_waveform, *, language):
            raise RuntimeError("model unavailable")
        return TranscriptSession(language, fail)

    monkeypatch.setattr(gw, "_new_transcript_session", create_failing)
    failed = _run(gw.create_transcription(gw.TranscriptCreateBody(language="en")))["transcript_id"]
    with pytest.raises(HTTPException):
        _run(gw.append_transcript_chunk(failed, gw.TranscriptChunkBody(pcm_s16le_b64=pcm, final=True)))
    assert gw._voice_stage_marks == {}
