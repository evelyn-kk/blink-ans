"""Run run_basic in a child process and atomically record its terminal status.

The child report is not allowed to call itself successful: only this parent can
observe its exit code.  Each invocation gets a unique run id and an immutable
status file, so a failed or interrupted run cannot overwrite a prior success.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATUS_DIR = ROOT / "bench" / "reports" / "run-status"
REPORTS = ROOT / "bench" / "reports"
RUN_BASIC = ROOT / "packages" / "evaltools" / "run_basic.py"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _report_for(run_id: str) -> Path | None:
    """Find exactly the child report bound to this run, never by timestamp."""
    matches: list[Path] = []
    for path in REPORTS.glob("eval-basic-*.json"):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("run_id") == run_id:
                matches.append(path)
        except (OSError, json.JSONDecodeError):
            continue
    return matches[0] if len(matches) == 1 else None


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    run_id = uuid.uuid4().hex
    started = _utc()
    status = STATUS_DIR / f"{run_id}.json"
    base = {"run_id": run_id, "runner_pid": os.getpid(), "started_at_utc": started,
            "command": [sys.executable, str(RUN_BASIC), *args]}
    _atomic_write(status, {**base, "state": "running"})
    env = {**os.environ, "BLINK_EVAL_RUN_ID": run_id, "BLINK_EVAL_STARTED_AT": started}
    try:
        child = subprocess.run(base["command"], cwd=ROOT, env=env, check=False)
        report = _report_for(run_id) if child.returncode == 0 else None
        terminal = "succeeded" if report else "failed"
        exit_code = child.returncode if child.returncode else (0 if report else 2)
        payload = {**base, "state": terminal, "exit_code": exit_code, "finished_at_utc": _utc()}
        if report:
            payload["report_path"] = str(report.relative_to(ROOT))
        elif child.returncode == 0:
            payload["error"] = "child exited 0 without exactly one report bound to run_id"
        _atomic_write(status, payload)
        return exit_code
    except BaseException as exc:
        _atomic_write(status, {**base, "state": "interrupted", "error": repr(exc),
                               "finished_at_utc": _utc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
