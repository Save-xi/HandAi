from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "intent_prediction"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from intent_prediction.delay_injection import (  # noqa: E402
    SequenceForecastTrace,
    build_runtime_jsonl_forecast_traces,
    evaluate_network_matrix,
)


def _linear_trace() -> SequenceForecastTrace:
    count = 12
    history_frames = 3
    timestamps = np.arange(count, dtype=np.float64) * 100.0
    current = np.repeat((np.arange(count, dtype=np.float32) / 20.0)[:, None], 9, axis=1)
    history = np.empty((count, history_frames, 9), dtype=np.float32)
    for index in range(count):
        start = max(0, index - history_frames + 1)
        values = current[start : index + 1]
        if len(values) < history_frames:
            values = np.concatenate([np.repeat(values[:1], history_frames - len(values), axis=0), values], axis=0)
        history[index] = values
    future = np.minimum(current + 0.05, 1.0)[:, None, :]
    motion = np.full(count, 0.05, dtype=np.float64)
    return SequenceForecastTrace(
        sequence_id="subject_test_linear",
        subject="subject_test",
        timestamps_ms=timestamps,
        frame_ids=np.arange(count, dtype=np.int64),
        history=history,
        current=current,
        raw_forecast=future,
        gated_forecast=future,
        prediction_available=np.ones(count, dtype=bool),
        motion_score=motion,
    )


def test_zero_delay_is_exact_current_command():
    scenarios, _, _ = evaluate_network_matrix(
        [_linear_trace()],
        horizon_ms=(100,),
        delays_ms=(0.0,),
        jitters_ms=(0.0,),
        loss_rates=(0.0,),
        seed=123,
        dynamic_threshold=0.01,
    )
    scenario = scenarios[0]
    assert scenario["methods"]["hold_last"]["rmse"] == 0.0
    assert scenario["methods"]["raw"]["rmse"] == 0.0
    assert scenario["methods"]["gated"]["rmse"] == 0.0
    assert scenario["improvement_percent_vs_hold"]["gated_rmse"] is None


def test_perfect_one_step_forecast_beats_hold_under_100ms_delay():
    scenarios, _, _ = evaluate_network_matrix(
        [_linear_trace()],
        horizon_ms=(100,),
        delays_ms=(100.0,),
        jitters_ms=(0.0,),
        loss_rates=(0.0,),
        seed=123,
        dynamic_threshold=0.01,
    )
    scenario = scenarios[0]
    assert scenario["methods"]["hold_last"]["rmse"] > 0.0
    assert scenario["methods"]["gated"]["rmse"] < 1e-7
    assert scenario["improvement_percent_vs_hold"]["gated_rmse"] > 99.99


def test_loss_and_jitter_are_deterministic_for_frozen_seed():
    kwargs = dict(
        traces=[_linear_trace()],
        horizon_ms=(100,),
        delays_ms=(50.0,),
        jitters_ms=(10.0,),
        loss_rates=(0.2,),
        seed=20260829,
        dynamic_threshold=0.01,
    )
    first, first_sequences, _ = evaluate_network_matrix(**kwargs)
    second, second_sequences, _ = evaluate_network_matrix(**kwargs)
    assert first == second
    assert first_sequences == second_sequences


class _FakeRuntimePredictor:
    history_frames = 3
    horizon_ms = (50, 100, 150)
    max_frame_gap_ms = 100.0
    gate_recent_frames = 3

    def __init__(self) -> None:
        self.valid_count = 0

    def observe(self, payload: dict) -> dict:
        preview = payload.get("svh_preview", {})
        if preview.get("valid") is not True:
            self.valid_count = 0
            return {"status": "invalid_input", "motion_score": None}
        self.valid_count += 1
        current = np.asarray(preview["target_positions"], dtype=np.float32)
        if self.valid_count < self.history_frames:
            return {"status": "warming_up", "motion_score": None}
        forecast = np.stack([current + 0.01, current + 0.02, current + 0.03]).clip(0.0, 1.0)
        return {
            "status": "predicted",
            "raw_prediction": forecast.tolist(),
            "gated_prediction": forecast.tolist(),
            "motion_score": 0.02,
        }


def test_runtime_jsonl_is_split_on_invalid_and_preserves_prediction_coverage(tmp_path):
    path = tmp_path / "camera.jsonl"
    rows = []
    for frame in range(7):
        valid = frame != 3
        value = 0.1 + 0.01 * frame
        rows.append(
            {
                "timestamp": 1000.0 + frame * 0.03,
                "frame_index": frame,
                "detected": valid,
                "control_ready": valid,
                "svh_preview": {
                    "enabled": True,
                    "valid": valid,
                    "target_positions": [value] * 9 if valid else [],
                },
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    group = build_runtime_jsonl_forecast_traces(
        path,
        predictor=_FakeRuntimePredictor(),
        recent_frames=3,
    )
    assert group.total_rows == 7
    assert group.valid_rows == 6
    assert group.predicted_rows == 2
    assert len(group.traces) == 2
    assert group.status_counts == {"warming_up": 4, "invalid_input": 1, "predicted": 2}
    assert [trace.prediction_available.tolist() for trace in group.traces] == [
        [False, False, True],
        [False, False, True],
    ]


def test_short_runtime_segment_without_arrival_is_counted_not_fatal():
    long_trace = _linear_trace()
    short_trace = SequenceForecastTrace(
        sequence_id="short",
        subject="runtime",
        timestamps_ms=long_trace.timestamps_ms[:2],
        frame_ids=long_trace.frame_ids[:2],
        history=long_trace.history[:2],
        current=long_trace.current[:2],
        raw_forecast=long_trace.raw_forecast[:2],
        gated_forecast=long_trace.gated_forecast[:2],
        prediction_available=long_trace.prediction_available[:2],
        motion_score=long_trace.motion_score[:2],
    )
    scenarios, sequences, _ = evaluate_network_matrix(
        [long_trace, short_trace],
        horizon_ms=(100,),
        delays_ms=(250.0,),
        jitters_ms=(0.0,),
        loss_rates=(0.0,),
        seed=123,
        dynamic_threshold=0.01,
    )
    short = next(item for item in sequences if item["sequence_id"] == "short")
    assert short["evaluated_ticks"] == 0
    assert short["methods"] is None
    assert scenarios[0]["sequence_count"] == 2
    assert scenarios[0]["evaluated_sequence_count"] == 1
