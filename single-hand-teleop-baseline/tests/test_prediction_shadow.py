from __future__ import annotations

from copy import deepcopy
import logging

import numpy as np

from prediction.shadow_predictor import PredictionShadow, build_prediction_shadow
from svh.svh_layout import SVH_9CH_NAMES




def _payload(frame_index: int, *, valid: bool = True, timestamp: float | None = None) -> dict:
    positions = [0.10 + 0.02 * channel + 0.001 * frame_index for channel in range(9)]
    return {
        "frame_index": frame_index,
        "timestamp": 1_000.0 + frame_index / 30.0 if timestamp is None else timestamp,
        "svh_preview": {
            "valid": valid,
            "target_positions": positions if valid else [],
            "protocol_hint": {
                "channel_layout": "svh_9ch",
                "channel_order": ",".join(SVH_9CH_NAMES),
            },
        },
    }


def _shadow(predict_fn, *, history_frames: int = 4, max_gap_ms: float = 100.0) -> PredictionShadow:
    return PredictionShadow(
        predict_fn=predict_fn,
        history_frames=history_frames,
        horizon_ms=[50, 100, 150],
        gate_recent_frames=3,
        gate_threshold=0.001,
        gate_temperature=0.01,
        gate_alpha_by_horizon=[0.75, 0.75, 0.75],
        max_frame_gap_ms=max_gap_ms,
        device="cpu",
        model_label="test_residual",
    )


def test_shadow_warms_up_then_emits_hold_raw_and_gated_without_mutating_preview():
    def predict(history: np.ndarray) -> np.ndarray:
        return np.repeat(history[-1:, :], 3, axis=0) + 0.05

    shadow = _shadow(predict)
    statuses = []
    final = None
    for frame_index in range(4):
        payload = _payload(frame_index)
        preview_before = deepcopy(payload["svh_preview"])
        final = shadow.observe(payload)
        statuses.append(final["status"])
        assert payload["svh_preview"] == preview_before

    assert statuses == ["warming_up", "warming_up", "warming_up", "predicted"]
    assert final is not None
    assert final["ready"] is True
    assert final["history_frames_available"] == 4
    assert len(final["hold_last"]) == len(final["raw_prediction"]) == len(final["gated_prediction"]) == 3
    assert all(len(row) == 9 for row in final["gated_prediction"])
    assert 0.0 <= final["base_gate"] <= 1.0
    assert final["inference_ms"] >= 0.0
    assert final["gating_ms"] >= 0.0
    assert final["gating_completed_unix_ms"] >= final["inference_completed_unix_ms"]
    assert final["fallback_reason"] is None


def test_invalid_preview_and_large_gap_clear_history_instead_of_bridging_stale_frames():
    shadow = _shadow(lambda history: np.repeat(history[-1:, :], 3, axis=0), history_frames=3)

    assert shadow.observe(_payload(0))["history_frames_available"] == 1
    assert shadow.observe(_payload(1))["history_frames_available"] == 2
    invalid = shadow.observe(_payload(2, valid=False))
    assert invalid["status"] == "invalid_input"
    assert invalid["history_frames_available"] == 0
    assert shadow.observe(_payload(3))["history_frames_available"] == 1

    gap = shadow.observe(_payload(4, timestamp=1_001.0))
    assert gap["status"] == "warming_up"
    assert gap["history_frames_available"] == 1
    assert gap["fallback_reason"] == "history_reset_frame_gap"


def test_inference_exception_disables_predictor_for_the_rest_of_the_run():
    calls = 0

    def broken_predict(_history: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic inference failure")

    shadow = _shadow(broken_predict, history_frames=2)
    assert shadow.observe(_payload(0))["status"] == "warming_up"
    failed = shadow.observe(_payload(1))
    assert failed["status"] == "inference_error"
    assert "synthetic inference failure" in failed["fallback_reason"]
    again = shadow.observe(_payload(2))
    assert again["status"] == "inference_error"
    assert calls == 1


def test_default_off_factory_returns_none_without_loading_model_dependencies():
    assert build_prediction_shadow({}, logger=logging.getLogger("test-shadow")) is None


def test_missing_selection_becomes_initialization_diagnostic_instead_of_exception(tmp_path):
    shadow = build_prediction_shadow(
        {
            "prediction_shadow_enabled": True,
            "prediction_shadow_model_path": str(tmp_path / "missing-model.json"),
        },
        logger=logging.getLogger("test-shadow"),
    )

    assert shadow is not None
    diagnostic = shadow.observe(_payload(0))
    assert diagnostic["status"] == "initialization_error"
    assert diagnostic["ready"] is False
    assert "FileNotFoundError" in diagnostic["fallback_reason"]


def test_irregular_low_fps_history_is_resampled_to_training_rate():
    observed_history = None

    def predict(history: np.ndarray) -> np.ndarray:
        nonlocal observed_history
        observed_history = history.copy()
        return np.repeat(history[-1:, :], 3, axis=0)

    shadow = _shadow(predict, history_frames=30, max_gap_ms=100.0)
    diagnostic = None
    for frame_index in range(14):
        diagnostic = shadow.observe(
            _payload(frame_index, timestamp=1_000.0 + frame_index / 12.5)
        )

    assert diagnostic is not None
    assert diagnostic["status"] == "predicted"
    assert observed_history is not None
    assert observed_history.shape == (30, 9)
    assert abs(diagnostic["history_span_ms"] - (29 / 30 * 1000)) < 1e-6
    assert abs(diagnostic["observed_fps"] - 12.5) < 1e-6
