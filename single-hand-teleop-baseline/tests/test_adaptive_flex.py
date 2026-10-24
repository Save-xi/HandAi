"""检查新范围滤波的响应、逐指独立性，以及完整映射状态的生命周期。"""

from copy import deepcopy

import numpy as np
import pytest

from control.adaptive_flex import AdaptiveFlexState
from perception.base import HandDetection
from pipeline import HandPipeline
from test_mapping_repairs import POSE, controlled_geometry, single_finger_case
from utils.config import load_config, validate_config


def detection(points):
    raw = points.copy()
    raw[:, 1] *= POSE["width"] / POSE["height"]
    return HandDetection(raw[:, :2].tolist(), raw.tolist(), "Right", .98,
                         POSE["width"], POSE["height"], "mediapipe_image_xyz")


def test_filter_is_per_finger_and_adapts_cutoff_to_speed():
    cfg = load_config("configs/ai_mapping_m8.yaml")
    state = AdaptiveFlexState()
    assert state.update({"index": .2, "middle": .3}, 0, cfg) == {"index": .2, "middle": .3}
    changed = state.update({"index": .8, "middle": .3}, 1/30, cfg)
    assert changed["middle"] == .3
    assert .2 < changed["index"] < .8
    fixed = deepcopy(cfg)
    fixed["control_flex_filter_beta"] = 0
    slower = AdaptiveFlexState()
    slower.update({"index": .2, "middle": .3}, 0, fixed)
    assert changed["index"] > slower.update({"index": .8, "middle": .3}, 1/30, fixed)["index"]


@pytest.mark.parametrize("fps, n50, n90", [(30, 2, 6), (13, 1, 4)])
def test_filter_step_response_has_declared_lag(fps, n50, n90):
    cfg = load_config("configs/ai_mapping_m8.yaml")
    state = AdaptiveFlexState()
    state.update({"index": 0.0}, 0, cfg)
    steps = [state.update({"index": 1.0}, i/fps, cfg)["index"] for i in range(1, 12)]
    assert next(i+1 for i, x in enumerate(steps) if x >= .5) == n50
    assert next(i+1 for i, x in enumerate(steps) if x >= .9) == n90
    assert all(a < b <= 1 for a, b in zip([0] + steps[:-1], steps))


def test_invalid_frame_and_reset_clear_filter_instead_of_replaying_old_targets():
    cfg = load_config("configs/ai_mapping_m8.yaml")
    pipe = HandPipeline(cfg)
    opened = detection(controlled_geometry())
    changed = detection(single_finger_case("index70"))
    pipe.process_detections([opened], frame_index=0, timestamp=1)
    pipe.process_detections([changed], frame_index=1, timestamp=1+1/30)
    bad = pipe.process_detections([], frame_index=2, timestamp=1+2/30)
    assert not bad["control_ready"] and not bad["svh_preview"]["valid"]
    assert pipe.flex_state.timestamp is None
    recovered = pipe.process_detections([changed], frame_index=3, timestamp=1+3/30)
    fresh = HandPipeline(cfg).process_detections([changed], frame_index=3, timestamp=1+3/30)
    assert recovered["control_representation"] == fresh["control_representation"]
    pipe.reset()
    assert pipe.flex_state.timestamp is None and pipe.flex_state.filtered == {}


@pytest.mark.parametrize("interruption", ["gap", "reverse", "resize"])
def test_stream_discontinuity_discards_filter_history(interruption):
    cfg = load_config("configs/ai_mapping_m8.yaml")
    pipe = HandPipeline(cfg)
    pipe.process_detections([detection(controlled_geometry())], frame_index=0, timestamp=1)
    changed = detection(single_finger_case("index70"))
    timestamp = 1.5 if interruption == "gap" else .9 if interruption == "reverse" else 1+1/30
    if interruption == "resize":
        changed.image_width, changed.image_height = 1280, 720
    actual = pipe.process_detections([changed], frame_index=1, timestamp=timestamp)
    fresh = HandPipeline(cfg).process_detections([changed], frame_index=1, timestamp=timestamp)
    assert actual["input_diagnostics"]["state_reset"]
    assert actual["control_representation"] == fresh["control_representation"]


def test_fork_and_fractional_query_leave_live_filter_state_unchanged():
    pipe = HandPipeline(load_config("configs/ai_mapping_m8.yaml"))
    opened = detection(controlled_geometry())
    changed = detection(single_finger_case("index70"))
    pipe.process_detections([opened], frame_index=0, timestamp=1)
    before = deepcopy(pipe.flex_state)
    fork = pipe.fork_mapping()
    current = fork.process_detections([changed], frame_index=1, timestamp=1+1/30)
    same_time = fork.query_detections([changed], frame_index=1, timestamp=1+1/30)
    assert same_time["control_representation"] == current["control_representation"]
    assert pipe.flex_state == before
    query = pipe.query_detections([changed], frame_index=1, timestamp=1+1/60)
    assert pipe.flex_state == before
    assert query["svh_preview"]["valid"]
    expected = fork.query_detections([changed], frame_index=2, timestamp=1+1/20)
    assert np.isfinite(expected["svh_preview"]["target_positions"]).all()


@pytest.mark.parametrize("kind", ["index45", "index70", "thumb"])
@pytest.mark.parametrize("fps", [30, 13])
def test_warmed_stream_keeps_other_fingers_stable_during_single_finger_motion(kind, fps):
    pipe = HandPipeline(load_config("configs/ai_mapping_m8.yaml"))
    opened = detection(controlled_geometry())
    for i in range(5):
        base = pipe.process_detections([opened], frame_index=i, timestamp=1+i/fps)
    channels = list(range(4, 8)) if kind.startswith("index") else list(range(2, 8))
    for i in range(5, 20):
        out = pipe.process_detections([detection(single_finger_case(kind))], frame_index=i, timestamp=1+i/fps)
        delta = np.abs(np.subtract(out["svh_preview"]["target_positions"], base["svh_preview"]["target_positions"]))
        assert max(delta[channels]) < .05
        assert out["input_diagnostics"]["release_weight"] == 0
        if kind == "thumb":
            assert max(delta[:2]) > .05


@pytest.mark.parametrize("key, value", [
    ("control_flex_filter_min_cutoff_hz", 0), ("control_flex_filter_derivative_cutoff_hz", float("nan")),
    ("control_flex_filter_beta", -1), ("control_flex_filter_beta", True),
    ("control_flex_filter_mode", "fixed_window"),
])
def test_invalid_filter_parameters_fail_before_runtime(key, value):
    cfg = load_config("configs/ai_mapping_m8.yaml")
    cfg[key] = value
    with pytest.raises(ValueError, match="filter"):
        validate_config(cfg)


def test_legacy_version_rejects_filter_semantics_without_new_version():
    cfg = load_config("configs/ai_m3a.yaml")
    cfg["control_flex_filter_mode"] = "one_euro_v1"
    with pytest.raises(ValueError, match="camera_mapping_stage"):
        validate_config(cfg)
