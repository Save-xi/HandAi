import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_main_help_smoke_without_camera_or_hardware():
    result = subprocess.run(
        [sys.executable, "src/main.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        env={**os.environ, "PYTHONUTF8": "1"},
        check=False,
    )

    assert result.returncode == 0
    assert "单右手 AI" in result.stdout
    assert "--print-json" in result.stdout
    assert "--save-jsonl" in result.stdout
    assert "--no-gui" in result.stdout
    assert "--headless" in result.stdout
    assert "--preview-svh" in result.stdout
    assert "--input-mirrored" in result.stdout
    assert "--video-file" in result.stdout
    assert "--prediction-model" in result.stdout
