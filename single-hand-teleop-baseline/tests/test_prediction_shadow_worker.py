from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from prediction.shadow_predictor import PredictionShadow
from prediction.shadow_worker import PredictionShadowWorker


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _payload(frame_index: int) -> dict:
    rows = (
        PROJECT_ROOT / "examples" / "sample_session.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[1])
    payload["frame_index"] = frame_index
    payload["timestamp"] = 1_000.0 + frame_index / 30.0
    payload["svh_preview"]["target_positions"] = [
        min(1.0, value + 0.001 * frame_index)
        for value in payload["svh_preview"]["target_positions"]
    ]
    return payload


def test_worker_submit_does_not_wait_for_slow_inference_and_outputs_separate_frame():
    def slow_predict(history: np.ndarray) -> np.ndarray:
        time.sleep(0.05)
        return np.repeat(history[-1:, :], 3, axis=0)

    shadow = PredictionShadow(
        predict_fn=slow_predict,
        history_frames=2,
        horizon_ms=[50, 100, 150],
        gate_recent_frames=2,
        gate_threshold=0.0,
        gate_temperature=0.01,
        gate_alpha_by_horizon=[0.5, 0.5, 0.5],
        max_frame_gap_ms=100.0,
        device="cpu",
        model_label="slow-test",
        selection_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
    )
    worker = PredictionShadowWorker(shadow)
    try:
        worker.submit(_payload(0))
        deadline = time.perf_counter() + 1.0
        first_results = []
        while not first_results and time.perf_counter() < deadline:
            first_results = worker.drain_results()
            time.sleep(0.005)
        assert first_results[0]["prediction_diagnostics"]["status"] == "warming_up"

        source = _payload(1)
        preview_before = deepcopy(source["svh_preview"])
        start = time.perf_counter()
        assert worker.submit(source) is True
        submit_ms = (time.perf_counter() - start) * 1000.0
        assert submit_ms < 20.0
        assert source["svh_preview"] == preview_before
    finally:
        assert worker.close(timeout_s=1.0) is True

    results = worker.drain_results()
    assert results[-1]["frame_index"] == 1
    assert results[-1]["prediction_diagnostics"]["status"] == "predicted"
    assert results[-1]["svh_preview"] == source["svh_preview"]


def test_worker_input_queue_is_latest_only_and_bounded():
    shadow = PredictionShadow.unavailable("test-only")
    worker = PredictionShadowWorker(shadow, input_queue_size=1, result_queue_size=2)
    try:
        for frame_index in range(100):
            worker.submit(_payload(frame_index))
    finally:
        worker.close(timeout_s=1.0)

    assert worker._input.maxsize == 1
    assert worker._results.maxsize == 2
    assert worker.dropped_input_count > 0 or worker.completed_count == 100
