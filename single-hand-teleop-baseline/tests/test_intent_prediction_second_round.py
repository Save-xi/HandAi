from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "intent_prediction"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from intent_prediction.models import TORCH_AVAILABLE  # noqa: E402
from intent_prediction.second_round import run_second_round  # noqa: E402


def test_second_round_formal_config_has_explicit_unique_seeds_and_hold_fallback():
    config = json.loads(
        (EXPERIMENT_ROOT / "configs" / "h2o_second_round.json").read_text(encoding="utf-8")
    )
    labels = [candidate["label"] for candidate in config["candidates"]]
    seed_offsets = [candidate["seed_offset"] for candidate in config["candidates"]]
    assert len(labels) == len(set(labels))
    assert len(seed_offsets) == len(set(seed_offsets))
    assert {candidate["model"] for candidate in config["candidates"]} == {"gru", "residual_gru"}
    assert 0.0 in config["gate"]["alpha_candidates"]
    assert config["selection"]["minimum_objective_improvement"] >= 0.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="第二轮完整 smoke 需要独立 PyTorch 环境")
def test_second_round_smoke_freezes_validation_selection_before_test(tmp_path: Path):
    report_path = run_second_round(
        config_path=EXPERIMENT_ROOT / "configs" / "h2o_second_round.json",
        data_root=None,
        output_root=tmp_path / "runs",
        synthetic_smoke=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selection_path = Path(report["selection_path"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    event_names = [event["event"] for event in report["protocol"]["events"]]

    assert report["research_claims_allowed"] is False
    assert selection["selection_fit_split"] == "validation"
    assert selection["test_loaded"] is False
    assert selection["test_metrics_available"] is False
    assert hashlib.sha256(selection_path.read_bytes()).hexdigest() == report["selection_sha256"]
    assert event_names.index("selection_frozen_to_disk_before_test_load") < event_names.index(
        "test_loaded_after_selection_freeze"
    )
    assert report["protocol"]["test_based_reselection_performed"] is False
    assert len(report["validation_candidates"]) == len(
        json.loads((EXPERIMENT_ROOT / "configs" / "h2o_second_round.json").read_text(encoding="utf-8"))[
            "candidates"
        ]
    )
