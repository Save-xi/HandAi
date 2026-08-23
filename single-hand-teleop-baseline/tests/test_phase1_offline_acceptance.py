from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acceptance_run_directory_refuses_overwrite(tmp_path):
    script = _load_script(
        "phase1_acceptance_script",
        PROJECT_ROOT / "experiments" / "freihand_eval" / "scripts" / "run_phase1_offline_acceptance.py",
    )

    run_dir = script._create_run_dir(tmp_path, "fixed_run")
    assert run_dir.is_dir()
    with pytest.raises(FileExistsError):
        script._create_run_dir(tmp_path, "fixed_run")


def test_acceptance_run_id_records_git_state(monkeypatch):
    script = _load_script(
        "phase1_acceptance_run_id_script",
        PROJECT_ROOT / "experiments" / "freihand_eval" / "scripts" / "run_phase1_offline_acceptance.py",
    )

    class _FixedDateTime:
        @classmethod
        def now(cls, tz):
            assert tz is timezone.utc
            return datetime(2026, 7, 26, 12, 34, 56, 123456, tzinfo=timezone.utc)

    monkeypatch.setattr(script, "datetime", _FixedDateTime)

    assert script._make_run_id("7f4e696abcdef", True) == "20260726T123456_123456Z_7f4e696_dirty"
    assert script._make_run_id("7f4e696abcdef", False).endswith("_7f4e696_clean")


def test_report_figures_use_supplied_metrics_instead_of_stale_numbers():
    figures = _load_script(
        "phase1_report_figures_script",
        PROJECT_ROOT / "experiments" / "freihand_eval" / "scripts" / "make_report_figures.py",
    )
    metrics = {
        "sample_counts": {"ground_truth_samples": 10, "evaluated_2d_samples": 9},
        "keypoint_complete_rate": 0.9,
        "pck_2d_at_thresholds_all_gt": {"20": 0.81234},
        "mpjpe_2d_px": 7.5,
        "latency_ms": {
            "mean": 12.345,
            "p95": 20.0,
            "threshold_ms": 50.0,
            "over_threshold_count": 0,
            "count": 10,
        },
    }

    dashboard = figures.render_dashboard(metrics)
    latency_card = figures.render_latency_card(metrics)

    assert "完整率 90.0%" in dashboard
    assert "全数据 PCK@20px 81.2%" in dashboard
    assert "12.345 ms/frame" in dashboard
    assert "85.7%" not in dashboard
    assert "约 19 ms" not in latency_card
    assert "P95 为 20.000 ms" in latency_card
