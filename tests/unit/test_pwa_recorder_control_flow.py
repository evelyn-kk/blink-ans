"""Runs the browser-facade control-flow regression through the available Node runtime."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for PWA script regression")
def test_pwa_recorder_control_flow():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["node", "tests/pwa/test_recorder_control_flow.cjs"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
