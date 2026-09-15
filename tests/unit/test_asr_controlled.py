"""受控 ASR runner：不能把早输出当完成，也不能以失败覆盖已有成功产物。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))

import run_asr_controlled  # noqa: E402


def _public_payload(clip: str, glossary: bool, runs: int) -> dict:
    sample = {"transcribe_s": 1.0, "rtf": 4.0, "word_error_rate": 0.0}
    return {
        "model": "local", "runtime": "mlx-whisper", "language": "en", "input_kind": "real_recording",
        "initial_prompt_tokens": 153 if glossary else 0,
        "results": [{
            "clip": clip, "language": "en", "input_kind": "real_recording", "audio_seconds": 1.0,
            "reference_sha256": "hash", "initial_prompt_tokens": 153 if glossary else 0,
            "glossary_biased": glossary, "runs": runs, "cold": sample, "median": sample,
            "samples": [sample] * runs,
        }],
    }


def test_run_case_requires_process_exit_and_complete_summary_before_replacing_existing(monkeypatch, tmp_path):
    output = tmp_path / "existing.json"
    output.write_text('{"previous": "success"}\n', encoding="utf-8")

    def early_output_but_nonzero(command, **_kwargs):
        candidate = Path(command[command.index("--public-summary") + 1])
        candidate.write_text(json.dumps(_public_payload("clip", False, 1)), encoding="utf-8")
        return type("Result", (), {"returncode": 1})()

    result = run_asr_controlled.run_case(
        manifest=tmp_path / "private.yaml", clip_id="clip", glossary=False, runs=1, timeout_s=1,
        output_path=output, subprocess_run=early_output_but_nonzero,
    )

    assert result["status"] == "subprocess_exit"
    assert json.loads(output.read_text(encoding="utf-8")) == {"previous": "success"}


def test_run_case_atomically_replaces_only_a_complete_text_free_summary(tmp_path):
    output = tmp_path / "summary.json"

    def completed(command, **_kwargs):
        candidate = Path(command[command.index("--public-summary") + 1])
        candidate.write_text(json.dumps(_public_payload("clip", True, 2)), encoding="utf-8")
        return type("Result", (), {"returncode": 0})()

    result = run_asr_controlled.run_case(
        manifest=tmp_path / "private.yaml", clip_id="clip", glossary=True, runs=2, timeout_s=1,
        output_path=output, subprocess_run=completed,
    )

    assert result["status"] == "completed"
    assert json.loads(output.read_text(encoding="utf-8"))["results"][0]["runs"] == 2


def test_timeout_does_not_overwrite_a_prior_summary(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text('{"previous": "success"}\n', encoding="utf-8")

    def timeout(_command, **_kwargs):
        raise subprocess.TimeoutExpired("bench_asr.py", 1)

    import subprocess
    result = run_asr_controlled.run_case(
        manifest=tmp_path / "private.yaml", clip_id="clip", glossary=False, runs=1, timeout_s=1,
        output_path=output, subprocess_run=timeout,
    )

    assert result["status"] == "timeout"
    assert json.loads(output.read_text(encoding="utf-8")) == {"previous": "success"}


def test_exit_zero_nested_private_field_fails_closed_without_overwriting_sentinel(tmp_path):
    """行为回归：换名私有字段藏在 sample，exit 0 也绝不能发布。"""
    output = tmp_path / "existing.json"
    output.write_text('{"previous": "success"}\n', encoding="utf-8")

    def exit_zero_with_renamed_private_text(command, **_kwargs):
        candidate = Path(command[command.index("--public-summary") + 1])
        payload = _public_payload("clip", False, 1)
        payload["results"][0]["samples"][0]["private_note"] = "renamed transcript"
        candidate.write_text(json.dumps(payload), encoding="utf-8")
        return type("Result", (), {"returncode": 0})()

    result = run_asr_controlled.run_case(
        manifest=tmp_path / "private.yaml", clip_id="clip", glossary=False, runs=1, timeout_s=1,
        output_path=output, subprocess_run=exit_zero_with_renamed_private_text,
    )

    assert result["status"] == "incomplete_public_summary"
    assert json.loads(output.read_text(encoding="utf-8")) == {"previous": "success"}
