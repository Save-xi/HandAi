"""M3-A 断流判定：慢而连续的源不重置，真实断流、倒序、换尺寸和坏帧仍立即重置。"""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/intent_prediction"))
from intent_prediction.camera_sequence import validate_sequence  # noqa: E402
from output.frame_payload_contract import validate_frame_payload  # noqa: E402
from pipeline import HandPipeline  # noqa: E402
from prediction.camera_baselines import observe_frame  # noqa: E402
from svh.mapping_contract import CAMERA_MAPPING_VERSION, mapping_contract_payload  # noqa: E402
from test_m3a_geometry import encoded  # noqa: E402
from utils.config import load_config, validate_config  # noqa: E402


def run(times, detections, cfg=None):
    pipeline = HandPipeline(cfg or load_config("configs/ai_m3a.yaml"))
    outputs = [pipeline.process_detections(d, frame_index=i, timestamp=t) for i, (t, d) in enumerate(zip(times, detections))]
    return pipeline, outputs


def steady(fps, count, start=1.0):
    return [start + i / fps for i in range(count)]


@pytest.mark.parametrize("fps,startup_resets", [(30.0, 0), (13.0, 0), (9.5, 1), (7.0, 1)])
def test_slow_but_continuous_source_keeps_confirmed_gesture(synthetic_hand_pose, fps, startup_resets):
    d = encoded(synthetic_hand_pose("open")[1])
    _, outputs = run(steady(fps, 30), [[d]] * 30)
    resets = [o["input_diagnostics"]["reset_reason"] for o in outputs]
    # 低于 10fps 时首个间隔尚无周期估计，按 100ms 下限重启一次；之后不再因计时重置。
    assert resets == [None] + ["frame_gap"] * startup_resets + [None] * (29 - startup_resets)
    gestures = [o["gesture_stable"] for o in outputs]
    assert gestures == ["unknown"] * (1 + startup_resets) + ["open"] * (29 - startup_resets)
    assert all(o["control_ready"] and validate_frame_payload(o) == [] for o in outputs)
    expected_limit = max(100.0, 2.5 * 1000 / fps)
    assert outputs[-1]["input_diagnostics"]["frame_gap_limit_ms"] == pytest.approx(expected_limit)


def test_fixed_100ms_rule_reproduces_the_live_regression(synthetic_hand_pose):
    """把上限压到下限即等同旧固定阈值：9.5fps 张手 30 帧全部 unknown。"""
    cfg = load_config("configs/ai_m3a.yaml")
    cfg["mapping_frame_gap_ceiling_ms"] = cfg["mapping_max_frame_gap_ms"]
    d = encoded(synthetic_hand_pose("open")[1])
    _, outputs = run(steady(9.5, 30), [[d]] * 30, cfg)
    assert sum(o["input_diagnostics"]["state_reset"] for o in outputs) == 29
    assert {o["gesture_stable"] for o in outputs} == {"unknown"}


@pytest.mark.parametrize("gap_ms,expect_reset", [(150.0, False), (190.0, False), (200.0, True), (400.0, True), (2000.0, True)])
def test_real_gap_after_slow_stream_still_restarts(synthetic_hand_pose, gap_ms, expect_reset):
    d = encoded(synthetic_hand_pose("open")[1])
    times = steady(13.0, 12)  # 13fps：阈值 = 2.5 × 76.9ms ≈ 192.3ms
    times += [times[-1] + gap_ms / 1000, times[-1] + gap_ms / 1000 + 1 / 13.0]
    _, outputs = run(times, [[d]] * len(times))
    gap, after = outputs[-2], outputs[-1]
    assert gap["input_diagnostics"]["frame_gap_limit_ms"] == pytest.approx(2.5 * 1000 / 13.0)
    assert gap["control_ready"]  # 当前观测仍有效，断流不等于丢手。
    if expect_reset:
        assert gap["input_diagnostics"]["reset_reason"] == "frame_gap" and gap["input_diagnostics"]["state_reset"]
        assert gap["gesture_stable"] == "unknown" and after["gesture_stable"] == "open"
    else:
        assert not gap["input_diagnostics"]["state_reset"] and gap["input_diagnostics"]["reset_reason"] is None
        assert gap["gesture_stable"] == "open"
    # 一次长间隔进入窗口中位数后不会把阈值抬高。
    assert after["input_diagnostics"]["frame_gap_limit_ms"] == pytest.approx(2.5 * 1000 / 13.0)


def test_ceiling_bounds_the_relative_threshold(synthetic_hand_pose):
    d = encoded(synthetic_hand_pose("open")[1])
    _, outputs = run(steady(0.8, 6), [[d]] * 6)  # 1250ms 一帧：超过 1000ms 上限，每帧都算断流。
    assert [o["input_diagnostics"]["reset_reason"] for o in outputs[1:]] == ["frame_gap"] * 5
    assert outputs[-1]["input_diagnostics"]["frame_gap_limit_ms"] == pytest.approx(1000.0)
    assert {o["gesture_stable"] for o in outputs} == {"unknown"}


def test_30fps_floor_dominates_and_matches_fixed_threshold(synthetic_hand_pose):
    d = encoded(synthetic_hand_pose("open")[1])
    times = steady(30.0, 10)
    times += [times[-1] + 0.099, times[-1] + 0.099 + 0.101]
    _, outputs = run(times, [[d]] * len(times))
    assert all(o["input_diagnostics"]["frame_gap_limit_ms"] == 100.0 for o in outputs)
    assert [o["input_diagnostics"]["reset_reason"] for o in outputs[-2:]] == [None, "frame_gap"]


@pytest.mark.parametrize("bad", ["missing", "nan"])
def test_invalid_input_resets_immediately_and_keeps_source_cadence(synthetic_hand_pose, bad):
    good = encoded(synthetic_hand_pose("open")[1])
    broken = deepcopy(good)
    broken.landmarks_xyz[3][2] = float("nan")
    lost = [] if bad == "missing" else [broken]
    detections = [[good]] * 6 + [lost] + [[good]] * 3
    pipeline, outputs = run(steady(9.5, len(detections)), detections)
    invalid = outputs[6]
    assert invalid["input_diagnostics"]["reset_reason"] == "invalid_input" and invalid["input_diagnostics"]["state_reset"]
    assert not invalid["control_ready"] and not invalid["svh_preview"]["valid"] and invalid["gesture_stable"] == "unknown"
    assert pipeline.stabilizer.stable_gesture == "open"  # 恢复后已重新确认
    assert validate_frame_payload(invalid) == []
    # 恢复：第一帧重新确认，第二帧确认；源周期未被坏帧清空，所以没有额外的计时重置。
    assert [o["gesture_stable"] for o in outputs[7:]] == ["unknown", "open", "open"]
    assert [o["input_diagnostics"]["reset_reason"] for o in outputs[7:]] == [None, None, None]


@pytest.mark.parametrize("change,reason", [("equal", "timestamp_not_increasing"), ("reverse", "timestamp_not_increasing"),
                                           ("resize", "image_size_changed")])
def test_reverse_and_resize_still_restart_with_reason(synthetic_hand_pose, change, reason):
    d = encoded(synthetic_hand_pose("open")[1])
    pipeline, outputs = run(steady(9.5, 4), [[d]] * 4)
    t = outputs[-1]["timestamp"] if change == "equal" else outputs[-1]["timestamp"] + 1 / 9.5
    if change == "reverse":
        t = 0.5
    elif change == "resize":
        d = deepcopy(d)
        d.image_width, d.image_height = d.image_width * 2, d.image_height * 2
    output = pipeline.process_detections([d], frame_index=4, timestamp=t)
    assert output["input_diagnostics"]["reset_reason"] == reason and output["gesture_stable"] == "unknown"
    assert output["control_ready"] and validate_frame_payload(output) == []


def test_fork_copies_cadence_and_queries_do_not_advance_it(synthetic_hand_pose):
    d = encoded(synthetic_hand_pose("open")[1])
    pipeline, _ = run(steady(9.5, 5), [[d]] * 5)
    before = (list(pipeline.cadence.intervals), pipeline.cadence.last_time_s)
    fork = pipeline.fork_mapping()
    assert fork.cadence is not pipeline.cadence and (list(fork.cadence.intervals), fork.cadence.last_time_s) == before
    pipeline.query_detections([d], frame_index=5, timestamp=before[1] + 0.05)
    fork.process_detections([d], frame_index=5, timestamp=before[1] + 1 / 9.5)
    assert (list(pipeline.cadence.intervals), pipeline.cadence.last_time_s) == before
    pipeline.reset()  # 显式中断只清映射状态，不清这条流的周期估计。
    assert (list(pipeline.cadence.intervals), pipeline.cadence.last_time_s) == before


def test_reset_reason_contract(synthetic_hand_pose):
    d = encoded(synthetic_hand_pose("open")[1])
    _, outputs = run(steady(30.0, 2), [[d]] * 2)
    output = outputs[-1]
    assert validate_frame_payload(output) == []
    for field, value, message in (("reset_reason", "camera_slow", "reset_reason"),
                                  ("reset_reason", "frame_gap", "state_reset"),
                                  ("frame_gap_limit_ms", 0.0, "frame_gap_limit_ms"),
                                  ("frame_gap_limit_ms", float("nan"), "frame_gap_limit_ms")):
        damaged = deepcopy(output)
        damaged["input_diagnostics"][field] = value
        assert any(message in error for error in validate_frame_payload(damaged))
    damaged = deepcopy(output)
    damaged["input_diagnostics"].update(state_reset=True, reset_reason="invalid_input")
    assert any("invalid_input" in error for error in validate_frame_payload(damaged))
    damaged["input_diagnostics"].pop("reset_reason")
    assert any("完整" in error for error in validate_frame_payload(damaged))


def test_legacy_geometry_reports_no_gap_limit(synthetic_hand_pose):
    cfg = load_config("configs/ai.yaml")
    pipeline = HandPipeline(cfg)
    d = encoded(synthetic_hand_pose("open")[1])
    good = pipeline.process_detections([d], frame_index=0, timestamp=1.0)
    lost = pipeline.process_detections([], frame_index=1, timestamp=5.0)
    assert good["input_diagnostics"]["frame_gap_limit_ms"] is None and good["input_diagnostics"]["reset_reason"] is None
    assert lost["input_diagnostics"]["reset_reason"] == "invalid_input"
    assert validate_frame_payload(good) == [] and validate_frame_payload(lost) == []


@pytest.mark.parametrize("key,value", [("mapping_frame_gap_period_factor", 0.5), ("mapping_frame_gap_period_factor", True),
                                       ("mapping_frame_gap_period_factor", float("inf")),
                                       ("mapping_frame_gap_period_window", 0), ("mapping_frame_gap_period_window", 2.5),
                                       ("mapping_frame_gap_ceiling_ms", -1.0), ("mapping_frame_gap_ceiling_ms", 50.0)])
def test_frame_gap_config_is_validated(key, value):
    cfg = load_config("configs/ai_m3a.yaml")
    cfg[key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(cfg)


def test_contract_records_cadence_parameters_and_new_version():
    contract = mapping_contract_payload(load_config("configs/ai_m3a.yaml"))
    assert contract["version"] == CAMERA_MAPPING_VERSION == "svh9-camera-v4-cadence-gap"
    assert {key: contract["parameters"][key] for key in ("mapping_max_frame_gap_ms", "mapping_frame_gap_period_factor",
            "mapping_frame_gap_period_window", "mapping_frame_gap_ceiling_ms")} == {
        "mapping_max_frame_gap_ms": 100.0, "mapping_frame_gap_period_factor": 2.5,
        "mapping_frame_gap_period_window": 9, "mapping_frame_gap_ceiling_ms": 1000.0}


def observations(synthetic_hand_pose, times, valid):
    """按 extract_sequence 的规则由流水线重置标出段号，生成缓存观测。"""
    cfg = load_config("configs/ai_m3a.yaml")
    d = encoded(synthetic_hand_pose("open")[1])
    points = np.array(d.landmarks_xyz)
    geometry = points.copy()
    geometry[:, 1] *= d.image_height / d.image_width
    pipeline, rows, segment, previous_valid = HandPipeline(cfg), [], -1, False
    for i, (t, ok) in enumerate(zip(times, valid)):
        payload = pipeline.process_detections([d] if ok else [], frame_index=i, timestamp=t / 1000)
        usable = bool(payload["control_ready"] and payload["svh_preview"]["valid"])
        if usable and (not previous_valid or payload["input_diagnostics"]["state_reset"]):
            segment += 1
        rows.append({"video_id": "fixture", "person_id": None, "session_id": None, "role": "development_validation",
                     "frame_id": i, "source_time_ms": t, "timebase": "media_pts_ms", "timestamp_source": "fixture_pts",
                     "image_width": d.image_width, "image_height": d.image_height, "input_valid": usable,
                     "reason": payload["input_diagnostics"]["reason"], "segment_id": segment if usable else -1,
                     "geometry_landmarks": geometry.tolist() if usable else [], "raw_landmarks_xyz": points.tolist() if usable else [],
                     "task_id": "camera_mediapipe_geometry_xyz_v1", "read_ms": 0.0, "detection_ms": 5.0,
                     "baseline_mapping_ms": 2.0})
        previous_valid = usable
    return cfg, rows


def test_cached_sequence_check_follows_cadence_and_replay_matches_extraction(synthetic_hand_pose):
    # 9.5fps 录像：中间一次 150ms 抖动（同段）、一帧丢手、一次 600ms 真实断流（新段）。
    times = [i * 1000 / 9.5 for i in range(8)]
    times += [times[-1] + 150.0] + [times[-1] + 150.0 + k * 1000 / 9.5 for k in range(1, 6)]
    times += [times[-1] + 600.0] + [times[-1] + 600.0 + k * 1000 / 9.5 for k in range(1, 4)]
    valid = [True] * len(times)
    valid[11] = False
    cfg, rows = observations(synthetic_hand_pose, times, valid)
    segments = [row["segment_id"] for row in rows]
    assert segments[1] == 1 and set(segments[1:11]) == {1}  # 启动后 150ms 抖动不断段
    assert segments[11] == -1 and segments[12] == 2 and segments[14] == 3  # 丢手与 600ms 断流各开新段
    validate_sequence(rows, cfg)
    reused = deepcopy(rows)
    for row in reused[14:]:
        row["segment_id"] = 2  # 真实断流后不能沿用旧段号
    with pytest.raises(ValueError, match="段号"):
        validate_sequence(reused, cfg)
    # M3-B/M3-C 重放的显式 reset() 不清周期估计，故与提取时的映射输出逐帧相同。
    pipeline, previous, replay = HandPipeline(cfg), None, []
    for row in rows:
        replay.append(observe_frame(pipeline, row, previous))
        previous = row
    _, extracted = run([t / 1000 for t in times], [[encoded(synthetic_hand_pose("open")[1])] if ok else [] for ok in valid])
    assert [o["gesture_stable"] for o in replay] == [o["gesture_stable"] for o in extracted]
    for a, b in zip(replay, extracted):
        assert a["svh_preview"]["valid"] == b["svh_preview"]["valid"]
        # 缓存几何还原原始点有浮点舍入；映射逐帧一致即可。
        np.testing.assert_allclose(a["svh_preview"]["target_positions"], b["svh_preview"]["target_positions"], atol=1e-9)
