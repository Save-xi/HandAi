from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "intent_prediction"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from intent_prediction.baselines import predict_hold_last, predict_kalman_cv, predict_linear  # noqa: E402
from intent_prediction.experiment_runner import NEURAL_MODEL_SEED_OFFSETS, run_first_round  # noqa: E402
from intent_prediction.gating import apply_motion_gate, fit_motion_gate  # noqa: E402
from intent_prediction.h2o_adapter import (  # noqa: E402
    CameraIntrinsics,
    canonicalize_h2o_camera_xy,
    h2o_frame_to_svh9,
    preprocess_h2o_pose_dataset,
    project_h2o_points_normalized,
    read_h2o_right_hand_pose,
)
from intent_prediction.metrics import (  # noqa: E402
    compute_forecast_metrics,
    compute_motion_stratified_metrics,
    observed_motion_score,
)
from intent_prediction.models import TORCH_AVAILABLE, build_model  # noqa: E402
from intent_prediction.sequence_data import build_window_split, create_synthetic_smoke_dataset  # noqa: E402
from utils.config import load_config  # noqa: E402


def _open_hand_xyz() -> np.ndarray:
    return np.asarray(
        [
            (0.0, 0.0, 10.0),
            (-1.0, 0.5, 10.0),
            (-1.4, 1.0, 10.0),
            (-1.8, 1.4, 10.0),
            (-2.2, 1.8, 10.0),
            (-0.8, 1.0, 10.0),
            (-0.8, 2.0, 10.0),
            (-0.8, 3.0, 10.0),
            (-0.8, 4.0, 10.0),
            (0.0, 1.0, 10.0),
            (0.0, 2.1, 10.0),
            (0.0, 3.2, 10.0),
            (0.0, 4.3, 10.0),
            (0.8, 1.0, 10.0),
            (0.8, 2.0, 10.0),
            (0.8, 3.0, 10.0),
            (0.8, 4.0, 10.0),
            (1.6, 1.0, 10.0),
            (1.6, 1.9, 10.0),
            (1.6, 2.8, 10.0),
            (1.6, 3.7, 10.0),
        ],
        dtype=np.float32,
    )


def _write_h2o_pose(path: Path, right_points: np.ndarray, *, right_valid: int = 1) -> None:
    values = np.concatenate(
        (
            np.asarray([0.0]),
            np.zeros(63, dtype=np.float64),
            np.asarray([float(right_valid)]),
            np.asarray(right_points, dtype=np.float64).reshape(-1),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(f"{value:.8f}" for value in values), encoding="utf-8")


def test_h2o_right_hand_parser_and_projection(tmp_path: Path):
    pose_path = tmp_path / "000000.txt"
    points = _open_hand_xyz()
    _write_h2o_pose(pose_path, points)

    valid, parsed = read_h2o_right_hand_pose(pose_path)
    assert valid is True
    np.testing.assert_allclose(parsed, points)

    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100.0, height=100.0)
    projected = project_h2o_points_normalized(parsed, intrinsics)
    assert projected.shape == (21, 2)
    np.testing.assert_allclose(projected[0], [0.5, 0.5])
    canonical = canonicalize_h2o_camera_xy(parsed)
    np.testing.assert_allclose(canonical[0], [0.0, 0.0])
    assert np.linalg.norm(canonical[9]) == pytest.approx(1.0)


def test_h2o_frame_reuses_current_svh9_mapping():
    points = _open_hand_xyz()
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100.0, height=100.0)
    projected = project_h2o_points_normalized(points, intrinsics)
    cfg = load_config(str(PROJECT_ROOT / "configs" / "svh_9ch_preview.yaml"))

    positions, summary = h2o_frame_to_svh9(points, projected, timestamp_s=0.0, mapping_cfg=cfg)

    assert positions.shape == (9,)
    assert np.all((positions >= 0.0) & (positions <= 1.0))
    assert summary["hand_open_ratio"] > 0.0


def test_h2o_fixture_preprocess_creates_hashed_sequence_manifest(tmp_path: Path):
    raw_root = tmp_path / "raw"
    cam_dir = raw_root / "subject1" / "h1" / "0" / "cam4"
    (cam_dir / "cam_intrinsics.txt").parent.mkdir(parents=True, exist_ok=True)
    (cam_dir / "cam_intrinsics.txt").write_text("100 100 50 50 100 100", encoding="utf-8")
    for frame_id in range(12):
        points = _open_hand_xyz().copy()
        points[:, 0] += 0.01 * frame_id
        _write_h2o_pose(cam_dir / "hand_pose" / f"{frame_id:06d}.txt", points)

    output_root = tmp_path / "processed"
    manifest_path = preprocess_h2o_pose_dataset(
        h2o_root=raw_root,
        output_root=output_root,
        mapping_config_path=PROJECT_ROOT / "configs" / "svh_9ch_preview.yaml",
        min_segment_frames=5,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"] == "h2o_pose_only_right_hand"
    assert manifest["split_counts"]["train"]["frames"] == 12
    assert len(manifest["sequences"][0]["sha256"]) == 64


def test_h2o_pose_only_without_intrinsics_uses_canonical_camera_plane(tmp_path: Path):
    raw_root = tmp_path / "raw"
    cam_dir = raw_root / "subject1" / "h1" / "0" / "cam4"
    for frame_id in range(8):
        _write_h2o_pose(cam_dir / "hand_pose" / f"{frame_id:06d}.txt", _open_hand_xyz())

    manifest_path = preprocess_h2o_pose_dataset(
        h2o_root=raw_root,
        output_root=tmp_path / "processed",
        mapping_config_path=PROJECT_ROOT / "configs" / "svh_9ch_preview.yaml",
        min_segment_frames=5,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["diagnostics"]["takes_with_canonical_xy"] == 1
    assert manifest["diagnostics"]["takes_with_intrinsics"] == 0


def test_window_builder_uses_disjoint_sequences_and_exact_horizon_interpolation(tmp_path: Path):
    dataset_root = tmp_path / "synthetic"
    create_synthetic_smoke_dataset(dataset_root)
    train = build_window_split(
        dataset_root,
        split="train",
        history_frames=30,
        horizon_ms=(50, 100, 150),
        stride=3,
        max_windows=100,
    )
    val = build_window_split(
        dataset_root,
        split="val",
        history_frames=30,
        horizon_ms=(50, 100, 150),
        stride=3,
        max_windows=50,
    )
    assert train.x.shape == (100, 30, 9)
    assert train.y.shape == (100, 3, 9)
    np.testing.assert_allclose(train.horizon_steps, [1.5, 3.0, 4.5])
    assert set(train.sequence_ids.tolist()).isdisjoint(set(val.sequence_ids.tolist()))


def test_classical_predictors_share_shape_and_linear_extrapolates():
    t = np.arange(10, dtype=np.float32)
    history = np.repeat((0.1 + 0.01 * t)[None, :, None], 9, axis=2)
    horizons = np.asarray([1.0, 2.0], dtype=np.float32)
    hold = predict_hold_last(history, horizons)
    linear = predict_linear(history, horizons, fit_frames=10)
    kalman = predict_kalman_cv(history, horizons)
    assert hold.shape == linear.shape == kalman.shape == (1, 2, 9)
    np.testing.assert_allclose(linear[0, :, 0], [0.20, 0.21], atol=1e-6)
    metrics = compute_forecast_metrics(linear, linear, horizon_ms=(50, 100))
    assert metrics["mae"] == pytest.approx(0.0)


def test_neural_model_seeds_do_not_depend_on_requested_model_order():
    assert NEURAL_MODEL_SEED_OFFSETS == {"gru": 103, "tcn": 104, "transformer": 105}
    assert len(set(NEURAL_MODEL_SEED_OFFSETS.values())) == 3


def test_motion_strata_use_observed_history_only():
    history = np.zeros((2, 4, 9), dtype=np.float32)
    history[1, :, :] = np.arange(4, dtype=np.float32)[None, :, None]
    scores = observed_motion_score(history, recent_frames=4)
    np.testing.assert_allclose(scores, [0.0, 1.0])
    truth = np.zeros((2, 1, 9), dtype=np.float32)
    prediction = truth.copy()
    result = compute_motion_stratified_metrics(
        truth,
        prediction,
        history=history,
        horizon_ms=(50,),
        thresholds={"moving": 0.5},
        recent_frames=4,
    )
    assert result["strata"]["moving"]["samples"] == 1
    assert result["strata"]["moving"]["metrics"]["mae"] == pytest.approx(0.0)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="baseline 环境不强制安装 PyTorch")
@pytest.mark.parametrize("name", ["gru", "residual_gru", "tcn", "transformer"])
def test_neural_models_preserve_forecast_shape(name: str):
    import torch

    model = build_model(
        name,
        history_frames=30,
        horizon_count=3,
        architecture={
            "gru": {"hidden_size": 16, "layers": 1, "dropout": 0.0},
            "residual_gru": {"hidden_size": 16, "layers": 1, "dropout": 0.0, "max_delta": 0.35},
            "tcn": {"channels": [16, 16], "kernel_size": 3, "dropout": 0.0},
            "transformer": {"d_model": 16, "nhead": 4, "layers": 1, "dim_feedforward": 32, "dropout": 0.0},
        },
    )
    output = model(torch.zeros(2, 30, 9))
    assert tuple(output.shape) == (2, 3, 9)
    assert torch.all((output >= 0.0) & (output <= 1.0))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="baseline 环境不强制安装 PyTorch")
def test_residual_gru_initializes_to_exact_hold_last():
    import torch

    model = build_model(
        "residual_gru",
        history_frames=6,
        horizon_count=3,
        architecture={"residual_gru": {"hidden_size": 16, "layers": 1, "dropout": 0.0, "max_delta": 0.35}},
    )
    history = torch.rand(4, 6, 9)
    expected = history[:, -1:, :].expand(-1, 3, -1)
    torch.testing.assert_close(model(history), expected)


def test_validation_motion_gate_can_preserve_static_and_use_dynamic_prediction():
    history = np.zeros((20, 4, 9), dtype=np.float32)
    history[:, :, :] = 0.4
    for index in range(10, 20):
        history[index, :, :] = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)[:, None]
    truth = np.repeat(history[:, -1:, :], 3, axis=1)
    truth[10:, :, :] += 0.1
    model_prediction = truth.copy()
    model_prediction[:10, :, :] += 0.1
    fit = fit_motion_gate(
        history,
        truth,
        model_prediction,
        horizon_ms=(50, 100, 150),
        config={
            "recent_frames": 4,
            "threshold_percentiles": [50],
            "temperature_fractions": [0.0],
            "alpha_candidates": [0.0, 1.0],
            "mae_weight": 0.5,
            "rmse_weight": 0.35,
            "p95_weight": 0.15,
            "allowed_mae_regression_ratio": 0.0,
            "mae_regression_penalty": 4.0,
        },
    )
    assert fit["best"]["alpha_by_horizon"] == [1.0, 1.0, 1.0]
    gated, _ = apply_motion_gate(
        history,
        model_prediction,
        threshold=fit["best"]["threshold"],
        temperature=fit["best"]["temperature"],
        alpha_by_horizon=fit["best"]["alpha_by_horizon"],
        recent_frames=4,
    )
    np.testing.assert_allclose(gated, truth, atol=1e-7)


def test_classical_synthetic_smoke_is_explicitly_non_claimable(tmp_path: Path):
    report_path = run_first_round(
        config_path=EXPERIMENT_ROOT / "configs" / "h2o_first_round.json",
        data_root=None,
        output_root=tmp_path / "runs",
        synthetic_smoke=True,
        model_names=["hold_last", "linear", "kalman"],
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["synthetic"] is True
    assert report["research_claims_allowed"] is False
    assert report["warning"]
