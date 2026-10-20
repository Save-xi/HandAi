from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from control.open_release import OpenReleaseState  # noqa: E402
from evaluate_m3a import release_report  # noqa: E402
import pipeline as pipeline_module  # noqa: E402
from pipeline import HandPipeline  # noqa: E402
from prediction.camera_baselines import future_mapping  # noqa: E402
from test_m3a_geometry import encoded  # noqa: E402
from utils.config import load_config, validate_config  # noqa: E402


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


def test_fork_shares_checked_config_without_revalidating(synthetic_hand_pose, monkeypatch):
    class Detector:
        def detect(self, _frame):
            return []

        def close(self):
            pass

    pipeline = HandPipeline(load_config("configs/ai_m3a.yaml"), detector=Detector())
    d = encoded(synthetic_hand_pose("open")[1])
    current = pipeline.process_detections([d], frame_index=0, timestamp=1)
    cfg_before = deepcopy(pipeline.cfg)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("复制映射状态时不应重新校验配置或重建映射合同")

    # 预测热路径每次约复制 9 次；复制只带状态，计时不再包含配置校验。
    monkeypatch.setattr(pipeline_module, "validate_config", forbidden)
    monkeypatch.setattr(pipeline_module, "mapping_contract_payload", forbidden)
    fork = pipeline.fork_mapping()
    assert vars(fork).keys() == vars(pipeline).keys()  # __init__ 新增字段时须同步到 fork_mapping。
    assert fork.cfg is pipeline.cfg and fork.runtime is pipeline.runtime and fork.logger is pipeline.logger
    assert fork._mapping_version == pipeline._mapping_version and fork.detector is None
    assert fork.stabilizer is not pipeline.stabilizer and fork.release_state is not pipeline.release_state
    points = current["input_diagnostics"]["geometry_landmarks"]
    frame = {"frame_id": 0, "source_time_ms": 1000.0, "image_width": d.image_width, "image_height": d.image_height}
    result = future_mapping(pipeline, frame, [points] * 5, [points] * 3)
    assert all(positions is not None for positions in result["positions"])
    assert pipeline.cfg == cfg_before  # 副本共享只读 cfg，分支推进不得改写。


def test_release_full_ratio_above_one_is_reachable():
    cfg = load_config("configs/ai_m3a.yaml")
    cfg["control_open_release_full_ratio"] = 1.05
    validate_config(cfg)
    payload = {"hand_open_ratio": 1.0, "pinch_distance_norm": 1.0,
               "finger_curl": dict.fromkeys(("index", "middle", "ring", "little"), 0.0)}
    state = OpenReleaseState()
    assert 0 < state.update(payload, cfg) < 1  # full ≤ 1 时这里已完全释放。
    assert state.update({**payload, "hand_open_ratio": 1.06}, cfg) == 1.0
