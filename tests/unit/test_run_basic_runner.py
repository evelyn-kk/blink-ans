"""Terminal provenance for the real child-process evaluation runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages" / "evaltools"))
import run_basic_runner as runner  # noqa: E402


def _configure(monkeypatch, tmp_path, body: str):
    child = tmp_path / "child.py"
    child.write_text(body, encoding="utf-8")
    reports = tmp_path / "bench" / "reports"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "RUN_BASIC", child)
    monkeypatch.setattr(runner, "REPORTS", reports)
    monkeypatch.setattr(runner, "STATUS_DIR", reports / "run-status")
    return reports


def test_runner_records_real_child_success_and_binds_its_unique_report(monkeypatch, tmp_path):
    reports = _configure(monkeypatch, tmp_path, '''
import json, os
from pathlib import Path
p = Path.cwd() / "bench/reports/eval-basic-child.json"; p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({"run_id": os.environ["BLINK_EVAL_RUN_ID"]}))
''')

    assert runner.main(["--offline"]) == 0
    status = json.loads(next((reports / "run-status").glob("*.json")).read_text())
    assert status["state"] == "succeeded"
    assert status["exit_code"] == 0
    assert status["report_path"] == "bench/reports/eval-basic-child.json"
    assert status["started_at_utc"] < status["finished_at_utc"]


def test_runner_nonzero_child_cannot_be_recorded_as_success(monkeypatch, tmp_path):
    reports = _configure(monkeypatch, tmp_path, "raise SystemExit(7)\n")

    assert runner.main([]) == 7
    status = json.loads(next((reports / "run-status").glob("*.json")).read_text())
    assert status["state"] == "failed"
    assert status["exit_code"] == 7
    assert "report_path" not in status


def test_runner_uses_distinct_status_files_for_repeated_runs(monkeypatch, tmp_path):
    reports = _configure(monkeypatch, tmp_path, '''
import json, os
from pathlib import Path
p = Path.cwd() / "bench/reports" / f"eval-basic-{os.environ['BLINK_EVAL_RUN_ID']}.json"; p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({"run_id": os.environ["BLINK_EVAL_RUN_ID"]}))
''')

    assert runner.main([]) == 0
    assert runner.main([]) == 0
    statuses = [json.loads(p.read_text()) for p in (reports / "run-status").glob("*.json")]
    assert len(statuses) == 2
    assert {s["run_id"] for s in statuses} == {p.stem for p in (reports / "run-status").glob("*.json")}
    assert all(s["state"] == "succeeded" for s in statuses)
