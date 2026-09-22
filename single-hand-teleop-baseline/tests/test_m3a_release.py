from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_m3a import release_report  # noqa: E402
from pipeline import HandPipeline  # noqa: E402
from test_m3a_geometry import encoded  # noqa: E402
from utils.config import load_config  # noqa: E402


def test_release_micro_boundary_and_transition_budget():
    report = release_report()
    boundary = report["cases"]["curl_boundary"]
    assert boundary["legacy_release"]["max_jump"] == pytest.approx(0.470343, abs=1e-6)
    assert report["gates"]["microjump_at_most_0_05"]
    assert report["gates"]["extra_lag_at_most_one_source_period"]
    assert report["gates"]["loss_immediately_invalid"]
    assert boundary["continuous_release"]["trace"][-1]["gesture_stable"] == "unknown"
    assert boundary["continuous_release"]["trace"][-1]["valid"]


def test_fork_copies_every_state_and_does_not_advance_original(synthetic_hand_pose):
    pipeline = HandPipeline(load_config("configs/ai_m3a.yaml"))
    d = encoded(synthetic_hand_pose("open")[1])
    pipeline.process_detections([d], frame_index=0, timestamp=1)
    fork = pipeline.fork_mapping()
    original_gesture = deepcopy(vars(pipeline.stabilizer))
    original_release = deepcopy(vars(pipeline.release_state))
    assert vars(fork.stabilizer) == original_gesture
    assert vars(fork.release_state) == original_release
    assert fork._last_timestamp == pipeline._last_timestamp
    assert fork._last_image_size == pipeline._last_image_size
    future = fork.process_detections([d], frame_index=1, timestamp=1 + 1 / 30)
    assert vars(pipeline.stabilizer) == original_gesture
    assert vars(pipeline.release_state) == original_release
    actual = pipeline.process_detections([d], frame_index=1, timestamp=1 + 1 / 30)
    future.pop("latency_ms")
    actual.pop("latency_ms")
    assert future == actual
    fork.process_detections([], frame_index=2, timestamp=1 + 2 / 30)
    assert pipeline.stabilizer.stable_gesture == "open" and pipeline.release_state.weight == 1
