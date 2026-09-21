"""M1 的数值、因果边界、掩码和训练加载闭环。"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/intent_prediction"))

from intent_prediction.baselines import predict_hold_last, predict_linear  # noqa: E402
from intent_prediction.keypoint_data import (TASK_ID, build_keypoint_windows, create_keypoint_smoke_dataset,
    denormalize, fit_normalization, make_sequence, prepare_history, sample_points, validate_manifest)  # noqa: E402
from intent_prediction.keypoint_metrics import compute_keypoint_metrics, query_trajectory  # noqa: E402
from intent_prediction.models import build_model  # noqa: E402
from intent_prediction.second_round import run_second_round  # noqa: E402
from intent_prediction.training import _training_loss  # noqa: E402
from prediction.keypoint_model_loader import load_keypoint_prediction_model  # noqa: E402
from prediction.model_loader import load_prediction_model  # noqa: E402


@pytest.fixture
def task():
    spec = json.loads((ROOT / "experiments/intent_prediction/configs/keypoint_m0.json").read_text(encoding="utf-8"))
    spec["time"]["history_frames"] = 4
    return spec


def moving_hand(times):
    times = np.asarray(times)
    base = np.zeros((21, 3))
    base[:, 0] = np.linspace(-0.03, 0.03, 21)
    base[0] = [0, 0, 0]
    base[9] = [0, 0.1, 0]
    base[5], base[17] = [-0.03, 0.05, 0], [0.03, 0.05, 0]
    velocity = np.array([0.1, -0.2, 0.05])
    return base[None] + np.array([0.01, -0.02, 0.5])[None, None] + times[:, None, None] * velocity


def test_irregular_time_interpolation_and_causal_anchor_transform(task):
    times = np.array([0, 0.021, 0.064, 0.099, 0.13, 0.167, 0.202, 0.235, 0.27, 0.305])
    points = moving_hand(times)
    seq = make_sequence("a", "subject1", times, points, max_gap_s=2 / 30)
    history, reason = prepare_history(seq, 4, task, 0.001)
    assert reason == "ok"
    wrist = points[4, 0]
    np.testing.assert_allclose(history["x"].reshape(4, 21, 3), (moving_hand(history["history_times"]) - wrist) / 0.1, atol=1e-7)
    assert (history["history_brackets"] <= times[4]).all()
    future, mask, brackets = sample_points(seq, np.array([0.18]), anchor=4, max_gap_s=2 / 30)
    np.testing.assert_allclose(future, moving_hand([0.18]))
    np.testing.assert_allclose(brackets, [[0.167, 0.202]])
    assert mask.all()
    normalized = ((future - wrist) / history["scale"])[None]
    restored = denormalize(normalized, wrist[None], np.array([history["scale"]]))[0]
    np.testing.assert_allclose(restored, future)
    assert not np.allclose(normalized[0, 0, 0], 0)  # 未来腕部位移得到保留。
    points[5:] += 500
    changed = make_sequence("a", "subject1", times, points, max_gap_s=2 / 30)
    other, _ = prepare_history(changed, 4, task, 0.001)
    np.testing.assert_array_equal(history["x"], other["x"])
    assert other["scale"] == history["scale"]


@pytest.mark.parametrize("failure", ["frame_gap", "invalid_frame", "large_time_gap"])
def test_never_bridge_invalid_segments(task, failure):
    times = np.arange(12) / 30
    ids = np.arange(12)
    valid = np.ones(12, dtype=bool)
    if failure == "frame_gap":
        ids[6:] += 1
    elif failure == "invalid_frame":
        valid[6] = False
    else:
        times[6:] += 0.1
    seq = make_sequence("a", "subject1", times, moving_hand(times), max_gap_s=2 / 30, frame_ids=ids, frame_valid=valid)
    _, mask, _ = sample_points(seq, times[[7]], anchor=4, max_gap_s=2 / 30)
    assert not mask.any()
    history, reason = prepare_history(seq, 7, task, 0.001)
    assert history is None and reason == "warmup_or_segment_boundary"


def test_partial_targets_and_input_missing_are_separate(task):
    times = np.arange(12) / 30
    mask = np.ones((12, 21), dtype=bool)
    mask[6, 4] = False
    seq = make_sequence("a", "subject1", times, moving_hand(times), max_gap_s=2 / 30, mask=mask)
    history, _ = prepare_history(seq, 4, task, 0.001)
    assert history is not None
    future, target_mask, _ = sample_points(seq, np.array([times[4] + 0.05]), anchor=4, max_gap_s=2 / 30)
    assert target_mask.sum() == 20 and not target_mask[0, 4]
    assert (future[0, 4] == 0).all()
    history, reason = prepare_history(seq, 7, task, 0.001)
    assert history is None and reason == "incomplete_history"


def test_scale_fallback_and_degenerate_anchor(task):
    times = np.arange(12) / 30
    xyz = moving_hand(times)
    xyz[:, 9] = xyz[:, 0]
    seq = make_sequence("a", "subject1", times, xyz, max_gap_s=2 / 30)
    history, _ = prepare_history(seq, 5, task, 0.001)
    assert history["width_fallback"]
    assert history["scale"] == pytest.approx(0.06)
    xyz[:, 17] = xyz[:, 5]
    seq = make_sequence("a", "subject1", times, xyz, max_gap_s=2 / 30)
    history, reason = prepare_history(seq, 5, task, 0.001)
    assert history is None and reason == "invalid_scale"


def test_nonfinite_point_and_nonmonotonic_time(task):
    times = np.arange(12) / 30
    xyz = moving_hand(times)
    xyz[4, 8, 0] = np.nan
    seq = make_sequence("a", "subject1", times, xyz, max_gap_s=2 / 30)
    history, reason = prepare_history(seq, 5, task, 0.001)
    assert history is None and reason == "incomplete_history"
    times[6] = times[5]
    with pytest.raises(ValueError, match="严格递增"):
        make_sequence("a", "subject1", times, xyz, max_gap_s=2 / 30)


def test_signed_model_and_linear_do_not_clip():
    torch = pytest.importorskip("torch")
    model = build_model("residual_gru", history_frames=4, horizon_count=5,
                        architecture={"residual_gru": {"hidden_size": 8, "layers": 1, "dropout": 0, "max_delta": None}},
                        input_size=63, output_size=63, output_bounds=None)
    history = np.broadcast_to(np.linspace(-2, 2, 63), (2, 4, 63)).copy().astype(np.float32)
    result = model(torch.from_numpy(history)).detach().numpy()
    np.testing.assert_array_equal(result, np.repeat(history[:, -1:, :], 5, axis=1))
    # 训练之后残差仍可为任意符号，不通过 sigmoid、tanh 或 [0,1] clamp。
    with torch.no_grad():
        model.head[-1].bias.fill_(-3)
    np.testing.assert_allclose(model(torch.from_numpy(history)).detach().numpy(), result - 3, atol=1e-6)
    linear = predict_linear(history, np.array([1.5, 3, 4.5]), feature_count=63, output_bounds=None)
    assert linear.min() == -2 and linear.max() == 2
    old = build_model("residual_gru", history_frames=4, horizon_count=3,
                      architecture={"residual_gru": {"hidden_size": 8, "layers": 1, "dropout": 0}})
    old_result = old(torch.tensor(history[:, :, :9])).detach().numpy()
    assert (old_result == 0).all()
    one_frame = np.zeros((1, 1, 9), dtype=np.float32)
    assert predict_hold_last(one_frame, np.array([1])).shape == (1, 1, 9)


def test_masked_loss_ignores_missing_targets_and_gradients():
    torch = pytest.importorskip("torch")
    prediction = torch.ones((1, 2, 63), requires_grad=True)
    truth = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask[:, 1, 3:6] = False
    truth[:, 1, 3:6] = 10000
    loss = _training_loss(prediction, truth, torch.zeros(1, 4, 63),
                          loss_config={"smooth_l1_beta": 1.0}, motion_reference=1, target_mask=mask)
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    assert torch.count_nonzero(prediction.grad[:, 1, 3:6]) == 0


def test_metrics_equal_sequence_units_and_finger_translation_invariance():
    grid = tuple(np.arange(1, 6) * 1000 / 30)
    prediction = np.zeros((3, 5, 21, 3))
    prediction[:, :, :, 0] = np.array([1, 3, 10])[:, None, None]
    split = SimpleNamespace(horizon_ms=grid, eval_horizon_ms=(50, 100, 150), eval_y=np.zeros((3, 3, 21, 3)),
        eval_mask=np.ones((3, 3, 21), dtype=bool), scales=np.full(3, 0.1), sequence_ids=np.array(["a", "a", "b"]),
        subjects=np.array(["s1", "s1", "s2"]), target_mask=np.ones((3, 5, 21), dtype=bool),
        anchor_frame_ids=np.array([30, 31, 30]), anchor_times=np.array([1, 31/30, 1]))
    result = compute_keypoint_metrics(split, prediction.reshape(3, 5, 63))
    assert result["selection_score"] == pytest.approx(6)
    row = result["per_horizon_ms"]["50"]
    assert row["normalized_mpjpe"]["frame_mean"] == pytest.approx(14/3)
    assert row["mpjpe_mm"]["sequence_equal_mean"] == pytest.approx(600)
    assert row["wrist_error_mm"]["sequence_equal_mean"] == pytest.approx(600)
    assert row["relative_finger_error_mm"]["sequence_equal_mean"] == 0
    with pytest.raises(ValueError, match="范围"):
        query_trajectory(prediction.reshape(3, 5, 63), grid, (200,))


def test_train_only_scale_and_tail_coverage(tmp_path, task):
    create_keypoint_smoke_dataset(tmp_path, task, frames=40)
    manifest = validate_manifest(tmp_path, task)
    norm = fit_normalization(tmp_path, manifest, task)
    for entry in manifest["sequences"]:
        if entry["split"] == "train":
            continue
        path = tmp_path / entry["path"]
        with np.load(path) as source:
            data = {key: source[key] for key in source.files}
        data["landmarks_3d"] *= 1000
        np.savez(path, **data)
    assert fit_normalization(tmp_path, manifest, task) == norm
    split = build_keypoint_windows(tmp_path, manifest, task, norm, split="train", seed=1)
    assert len(split.x) == 2 * (40 - 3)
    assert int((~split.target_mask.any(axis=(1, 2))).sum()) == 2
    assert split.stats["counts_before_window_subsampling"]["target_150ms_complete"] == 2 * (40 - 3 - 5)
    assert not split.eval_mask[split.anchor_frame_ids == 39].any()
    manifest["sequences"][0]["split"] = "test"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="跨切分"):
        validate_manifest(tmp_path, task)


def test_reject_other_coordinate_domain_before_training(tmp_path, task):
    create_keypoint_smoke_dataset(tmp_path, task, frames=40)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["task_id"] = "camera_mediapipe_image_xyz"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="坐标域"):
        validate_manifest(tmp_path, task)


def test_training_save_load_raw_api_and_task_rejection(tmp_path, task):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    config = json.loads((ROOT / "experiments/intent_prediction/configs/keypoint_m1.json").read_text(encoding="utf-8"))
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    config["task_spec"] = "task.json"
    config["linear_fit_frames_candidates"] = [2, 4]
    config["sampling"] = {"sequences_per_subject": 1, "windows_per_sequence": 20}
    config["architecture"]["residual_gru"]["hidden_size"] = 8
    config["training"].update(device="cpu", epochs=1, batch_size=16)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    report_path = run_second_round(config_path=config_path, data_root=None, output_root=tmp_path / "runs", synthetic_smoke=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed" and report["synthetic"]
    assert report["load_verification"]["raw_api_max_abs_delta_m"] < 1e-6
    model_path = report_path.parent / "model.json"
    loaded = load_keypoint_prediction_model(model_path, expected_task_id=TASK_ID, device="cpu")
    times = np.arange(12) / 30
    output = loaded.predict(moving_hand(times), times, input_task=task)
    assert output["landmarks_3d_m"].shape == (5, 21, 3)
    np.testing.assert_allclose(output["evaluation_timestamps_s"] - times[-1], [0.05, 0.1, 0.15])
    bad_task = deepcopy(task)
    bad_task["coordinates"]["raw_unit"] = "mm"
    with pytest.raises(ValueError, match="米坐标"):
        loaded.predict(moving_hand(times), times, input_task=bad_task)
    with pytest.raises(ValueError, match="任务"):
        load_keypoint_prediction_model(model_path, expected_task_id="camera_mediapipe_image_xyz", device="cpu")
    with pytest.raises(ValueError, match="版本"):
        load_prediction_model(model_path, {})
    with np.load(report_path.parent / "window_sample.npz") as sample:
        assert sample["x"].shape == (1, 4, 63)
        assert sample["x"].min() < 0
        assert (sample["history_brackets"] <= sample["anchor_times"][0]).all()
