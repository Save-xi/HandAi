from __future__ import annotations

import logging
from pathlib import Path

import pytest

from prediction.shadow_predictor import build_prediction_shadow
from svh.svh_layout import SVH_9CH_NAMES
from utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_checkpoint_runs_one_complete_shadow_window():
    pytest.importorskip("torch")
    checkpoint = (
        PROJECT_ROOT
        / "experiments"
        / "intent_prediction"
        / "outputs"
        / "second_round_v2_open_release"
        / "20260829T024400_143036Z"
        / "checkpoints"
        / "residual_motion4.pt"
    )
    if not checkpoint.exists():
        pytest.skip("本机未保留 ignored checkpoint；安全回退由纯单元测试覆盖")

    cfg = load_config("configs/svh_9ch_preview.yaml")
    cfg["prediction_shadow_enabled"] = True
    cfg["prediction_shadow_device"] = "auto"
    shadow = build_prediction_shadow(cfg, logger=logging.getLogger("test-shadow-checkpoint"))
    assert shadow is not None

    diagnostic = None
    for frame_index in range(30):
        diagnostic = shadow.observe(
            {
                "frame_index": frame_index,
                "timestamp": 1_000.0 + frame_index / 30.0,
                "svh_preview": {
                    "valid": True,
                    "target_positions": [
                        min(1.0, 0.1 + 0.04 * channel + 0.002 * frame_index)
                        for channel in range(9)
                    ],
                    "protocol_hint": {
                        "channel_layout": "svh_9ch",
                        "channel_order": ",".join(SVH_9CH_NAMES),
                    },
                },
            }
        )

    assert diagnostic is not None
    assert diagnostic["status"] == "predicted"
    assert diagnostic["ready"] is True
    assert diagnostic["model_label"] == "residual_motion4"
    assert diagnostic["raw_range_violation_count"] == 0
    assert len(diagnostic["gated_prediction"]) == 3
