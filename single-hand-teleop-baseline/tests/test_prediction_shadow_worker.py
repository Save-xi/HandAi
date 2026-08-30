from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from threading import Event
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


def test_worker_preserves_invalid_frame_as_history_reset_when_queue_overwrites_it():
    inference_started = Event()
    release_inference = Event()

    def blocked_predict(history: np.ndarray) -> np.ndarray:
        inference_started.set()
        assert release_inference.wait(timeout=1.0)
        return np.repeat(history[-1:, :], 3, axis=0)

    shadow = PredictionShadow(
        predict_fn=blocked_predict,
        history_frames=2,
        horizon_ms=[50, 100, 150],
        gate_recent_frames=2,
        gate_threshold=0.0,
        gate_temperature=0.01,
        gate_alpha_by_horizon=[0.5, 0.5, 0.5],
        max_frame_gap_ms=100.0,
        device="cpu",
        model_label="invalid-barrier-test",
        selection_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
    )
    worker = PredictionShadowWorker(shadow, input_queue_size=1)
    try:
        assert worker.submit(_payload(0)) is True
        deadline = time.perf_counter() + 1.0
        while worker.completed_count < 1 and time.perf_counter() < deadline:
            time.sleep(0.005)
        assert worker.completed_count == 1

        assert worker.submit(_payload(1)) is True
        assert inference_started.wait(timeout=1.0)

        invalid = _payload(2)
        invalid["svh_preview"]["valid"] = False
        invalid["svh_preview"]["target_positions"] = []
        assert worker.submit(invalid) is True
        # worker 仍阻塞于 frame 1，因此 frame 3 必然覆盖等待中的 invalid。
        assert worker.submit(_payload(3)) is True
        release_inference.set()
    finally:
        release_inference.set()
        assert worker.close(timeout_s=1.0) is True

    by_frame = {item["frame_index"]: item for item in worker.drain_results()}
    assert 2 not in by_frame
    assert by_frame[3]["prediction_diagnostics"]["status"] == "warming_up"
    assert by_frame[3]["prediction_diagnostics"]["history_frames_available"] == 1
    assert worker.dropped_input_count >= 1
