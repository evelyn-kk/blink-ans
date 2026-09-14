"""T-024：按会话语言选择 Whisper initial_prompt，英文真人音频清单失败关闭。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))

import bench_asr  # noqa: E402


def _legacy_t024_transcribe_options(compare_glossary: bool) -> dict[str, str | None]:
    """R110 前的实际调用形状：不论会话语言都强制 zh + 中文词表。"""
    return {
        "language": "zh",
        "initial_prompt": bench_asr.LANGUAGE_GLOSSARIES["zh"] if compare_glossary else None,
    }


def test_english_main_passes_english_prompt_to_the_real_transcribe_call(monkeypatch, tmp_path):
    """判别性：旧硬编码会把同一英文录音传成 zh；新 main 实际传 en。"""
    audio = tmp_path / "speaker.wav"
    audio.write_bytes(b"not-decoded-in-this-fake")
    manifest = tmp_path / "english.yaml"
    manifest.write_text(yaml.safe_dump({
        "version": 1,
        "kind": "real_recording",
        "clips": [{
            "id": "kafka-offset", "language": "en", "audio": audio.name,
            "reference_text": "How should I inspect Kafka offset commits?",
        }],
    }), encoding="utf-8")
    calls: list[dict] = []
    payloads: list[dict] = []

    class FakeWhisper:
        @staticmethod
        def transcribe(_audio, **kwargs):
            calls.append(kwargs)
            return {"text": "How should I inspect Kafka offset commits?"}

    monkeypatch.setitem(sys.modules, "mlx_whisper", FakeWhisper)
    monkeypatch.setattr(bench_asr, "duration_s", lambda _path: 1.0)
    monkeypatch.setattr(bench_asr, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(bench_asr, "peak_memory_gb", lambda: None)
    monkeypatch.setattr(bench_asr, "initial_prompt_token_count", lambda _prompt, _language: 17)
    monkeypatch.setattr(bench_asr, "write_report", lambda _component, payload: payloads.append(payload) or tmp_path / "report.json")
    monkeypatch.setattr(sys, "argv", [
        "bench_asr.py", "--language", "en", "--manifest", str(manifest), "--runs", "1",
        "--compare-glossary",
    ])

    bench_asr.main()

    # The old implementation's behavior is observably wrong for the same English clip.
    assert _legacy_t024_transcribe_options(True)["language"] == "zh"
    assert calls == [{
        "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo",
        "language": "en",
        "initial_prompt": bench_asr.LANGUAGE_GLOSSARIES["en"],
    }]
    assert payloads[0]["language"] == "en"
    assert payloads[0]["input_kind"] == "real_recording"
    assert payloads[0]["initial_prompt"] == bench_asr.LANGUAGE_GLOSSARIES["en"]
    assert payloads[0]["initial_prompt_tokens"] == 17
    assert payloads[0]["results"][0]["transcription_samples"][0]["word_error_rate"] == 0.0


def test_real_manifest_rejects_missing_audio_instead_of_falling_back_to_synthetic(tmp_path):
    manifest = tmp_path / "english.yaml"
    manifest.write_text(yaml.safe_dump({
        "version": 1,
        "kind": "real_recording",
        "clips": [{
            "id": "missing", "language": "en", "audio": "missing.wav",
            "reference_text": "A real English utterance.",
        }],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="音频不存在"):
        bench_asr.load_real_manifest(manifest, "en")


def test_english_main_fails_for_missing_manifest_before_loading_metal(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench_asr.py", "--language", "en"])

    with pytest.raises(SystemExit, match="英文 ASR 基准需要 --manifest 真人录音"):
        bench_asr.main()


def test_main_rejects_an_unknown_clip_before_loading_metal(monkeypatch, tmp_path):
    audio = tmp_path / "speaker.wav"
    audio.write_bytes(b"audio")
    manifest = tmp_path / "english.yaml"
    manifest.write_text(yaml.safe_dump({
        "kind": "real_recording",
        "clips": [{"id": "known", "language": "en", "audio": audio.name,
                   "reference_text": "A real English utterance."}],
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "bench_asr.py", "--language", "en", "--manifest", str(manifest), "--clip", "unknown",
    ])

    with pytest.raises(SystemExit, match="未知 clip id: unknown"):
        bench_asr.main()


def test_example_manifest_cannot_be_mistaken_for_a_ready_english_benchmark():
    with pytest.raises(ValueError, match="clips 不能为空"):
        bench_asr.load_real_manifest(bench_asr.REAL_MANIFEST_EXAMPLE, "en")


def test_word_error_rate_records_word_level_substitution_deletion_and_insertion():
    wer = bench_asr.word_error_rate("Kafka offset commits", "Kafka retries commits extra")

    assert wer == {
        "reference_words": 3, "hypothesis_words": 4,
        "substitutions": 1, "deletions": 0, "insertions": 1, "word_error_rate": 0.6667,
    }
