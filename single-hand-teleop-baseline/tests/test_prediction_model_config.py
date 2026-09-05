import json
from pathlib import Path

import pytest

from prediction.model_loader import load_prediction_model
from utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_horizon_override_cannot_silently_relabel_a_model():
    cfg = load_config("configs/ai.yaml")
    cfg["prediction_shadow_horizon_ms"] = [20, 40, 60]
    with pytest.raises(ValueError, match="horizon_ms"):
        load_prediction_model(ROOT / "models/residual_motion4.json", cfg)


def test_invalid_sample_rate_is_rejected_before_loading_weights(tmp_path):
    spec = json.loads((ROOT / "models/residual_motion4.json").read_text(encoding="utf-8"))
    spec["target_fps"] = float("nan")
    path = tmp_path / "model.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="target_fps"):
        load_prediction_model(path, load_config("configs/ai.yaml"))
