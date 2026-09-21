"""M2：共同目标、状态因果性、跨表示残差及多路线训练加载。"""
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

from gesture.rule_based_gesture import GestureStabilizer  # noqa: E402
from utils.config import load_config  # noqa: E402
from intent_prediction.keypoint_data import (build_keypoint_windows, create_keypoint_smoke_dataset,
    fit_normalization, validate_manifest)  # noqa: E402
from intent_prediction.models import build_model  # noqa: E402
from intent_prediction.representation_data import build_representation_batch, future_channels  # noqa: E402
from intent_prediction.representation_metrics import continuation_summary, control_metrics  # noqa: E402
from intent_prediction.second_round import run_second_round  # noqa: E402


@pytest.fixture
def batch(tmp_path):
    task = json.loads((ROOT / "experiments/intent_prediction/configs/keypoint_m0.json").read_text(encoding="utf-8"))
    task["time"]["history_frames"] = 4
    create_keypoint_smoke_dataset(tmp_path, task, frames=20, articulated=True)
    manifest = validate_manifest(tmp_path, task)
    norm = fit_normalization(tmp_path, manifest, task)
    points = build_keypoint_windows(tmp_path, manifest, task, norm, split="train", seed=7)
    return build_representation_batch(tmp_path, manifest, task, points,
                                     load_config(str(ROOT / "configs/svh_9ch_preview.yaml")))


def test_external_residual_uses_current_mapping_not_first_nine_coordinates():
    torch = pytest.importorskip("torch")
    model = build_model("residual_gru", history_frames=4, horizon_count=7,
        architecture={"residual_gru": {"hidden_size": 8, "layers": 1, "dropout": 0, "max_delta": None}},
        input_size=63, output_size=9, external_residual=True)
    history = torch.full((2, 4, 63), -2.0)
    base = torch.linspace(0.1, 0.9, 9).repeat(2, 1)
    result = model(history, base).detach().numpy()
    np.testing.assert_array_equal(result, np.repeat(base.numpy()[:, None], 7, axis=1))
    with pytest.raises(ValueError, match="显式提供"):
        model(history)
    with pytest.raises(ValueError, match="显式提供"):
        model(history, torch.zeros(2, 63))


def test_fractional_query_pose_first_and_state_only_on_grid():
    state = GestureStabilizer(confirm_frames=2)
    grid = tuple(np.arange(1, 6) * 1000 / 30)
    # 非线性映射可区分“插值姿态后映射”和“插值映射值”。
    dense = np.zeros((5, 21, 3))
    dense[:, 0, 0] = np.arange(1, 6) / 10
    queries = np.zeros((3, 21, 3))
    queries[:, 0, 0] = [0.15, 0.3, 0.45]
    calls = []

    def mapper(points, timestamp, local, cfg, *, advance):
        calls.append((timestamp, advance))
        if advance:
            local.update("open")
        values = np.full(9, points[0, 0] ** 2)
        values[1] = int(local.stable_gesture == "open")
        return values

    arguments = dict(anchor_time=0.0, anchor_state=state, grid_ms=grid,
                     query_ms=(50, 100, 150), cfg={}, fallback=np.zeros(9), mapper=mapper)
    result, valid = future_channels(dense, queries, **arguments)
    np.testing.assert_allclose(result[:, 0], [0.0225, 0.09, 0.2025])
    assert result[0, 0] != pytest.approx((0.1 ** 2 + 0.2 ** 2) / 2)
    assert result[:, 1].tolist() == [0, 1, 1]  # 50ms 仅一次确认。
    assert valid.all() and state.stable_gesture == "unknown" and state.candidate_count == 0
    assert sum(advance for _, advance in calls) == 5
    arguments["query_ms"] = (150, 50, 100)
    reordered, _ = future_channels(dense, queries[[2, 0, 1]], **arguments)
    np.testing.assert_array_equal(reordered, result[[2, 0, 1]])


def test_missing_intermediate_state_invalidates_only_dependent_queries():
    dense = np.zeros((5, 21, 3))
    queries = np.zeros((3, 21, 3))
    mapper = lambda points, timestamp, state, cfg, advance: np.ones(9)
    result, valid = future_channels(dense, queries, anchor_time=0,
        anchor_state=GestureStabilizer(), grid_ms=tuple(np.arange(1, 6) * 1000 / 30), query_ms=(50, 100, 150),
        cfg={}, fallback=np.full(9, 0.4), dense_valid=np.array([True, False, True, True, True]), mapper=mapper)
    assert valid.tolist() == [True, False, False]
    np.testing.assert_allclose(result[1:], 0.4)


def test_routes_share_targets_anchors_and_true_trajectory_mapping(batch):
    a, b, c = (batch.training_view(route) for route in "ABC")
    assert len(batch.query_ms) == 7
    assert np.ptp(batch.targets9[batch.target_valid]) > 0.1
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.target_mask, b.target_mask)
    np.testing.assert_array_equal(b.x, c.x)
    np.testing.assert_array_equal(b.residual_base, a.x[:, -1])
    mapped, valid = batch.map_predictions(c.y)
    np.testing.assert_allclose(mapped[batch.target_valid], batch.targets9[batch.target_valid], atol=2e-5)
    assert valid[batch.target_valid].all()
    assert (~batch.target_valid).any()  # 尾部留在覆盖率分母。


def test_prediction_mapping_cannot_read_future_labels_or_masks(batch):
    prediction = np.repeat(batch.points.x[:, -1:], 5, axis=1)
    before = batch.map_predictions(prediction)
    original_states = [vars(state).copy() for state in batch.states]
    batch.targets9[:] = 0.97
    batch.target_valid[:] = False
    batch.points.y[:] = 888
    batch.points.target_mask[:] = False
    batch.points.eval_y[:] = 777
    batch.reference_transition[:] = ~batch.reference_transition
    after = batch.map_predictions(prediction)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    assert [vars(state) for state in batch.states] == original_states


def test_invalid_geometry_falls_back_without_changing_scoring_denominator(batch):
    prediction = np.repeat(batch.points.x[:, -1:], 5, axis=1)
    prediction[:, :, 2::3] = -1000  # 全部落到相机后方。
    controls, valid = batch.map_predictions(prediction)
    assert not valid.any()
    np.testing.assert_allclose(controls, np.repeat(batch.history9[:, -1:, :], 7, axis=1))
    result = control_metrics(batch, controls, valid)
    for h in (50, 100, 150):
        index = np.flatnonzero(np.isclose(batch.query_ms, h))[0]
        row = result["per_horizon_ms"][str(h)]
        assert row["reference_valid_frames"] == batch.target_valid[:, index].sum()
        assert row["fallback_frames_with_reference"] == row["reference_valid_frames"]


def test_metrics_are_sequence_equal_and_bootstrap_groups_segments():
    fake = SimpleNamespace(targets9=np.zeros((4, 3, 9)), target_valid=np.ones((4, 3), dtype=bool),
        query_ms=(50, 100, 150), points=SimpleNamespace(eval_horizon_ms=(50, 100, 150),
        sequence_ids=np.array(["take1_segment0", "take1_segment0", "take1_segment1", "take2_segment0"]),
        anchor_frame_ids=np.array([1, 2, 3, 4])))
    prediction = np.broadcast_to(np.array([0.1, 0.3, 0.4, 0.8])[:, None, None], (4, 3, 9))
    baseline = control_metrics(fake, prediction)
    assert baseline["selection_score"] == pytest.approx((np.sqrt(0.05) + 0.4 + 0.8) / 3)
    improved = control_metrics(fake, prediction / 2)
    result = continuation_summary(baseline, [deepcopy(improved) for _ in range(3)], seed=1)
    assert result["paired_take_count"] == 2
    assert result["development_continue_criteria_passed"]
    assert result["improvement_fraction"] == pytest.approx(0.5)
    assert result["paired_take_bootstrap_95ci_absolute_improvement"][0] > 0
    single = continuation_summary(baseline, [improved], seed=1)
    assert not single["development_continue_criteria_passed"]


def test_three_route_training_save_load_and_common_reference(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    config = json.loads((ROOT / "experiments/intent_prediction/configs/representation_m2.json").read_text(encoding="utf-8"))
    task = json.loads((ROOT / "experiments/intent_prediction/configs/keypoint_m0.json").read_text(encoding="utf-8"))
    task["time"]["history_frames"] = 4
    (tmp_path / "task.json").write_text(json.dumps(task), encoding="utf-8")
    config["task_spec"] = "task.json"
    config["linear_fit_frames_candidates"] = [2, 4]
    config["sampling"] = {"sequences_per_subject": 1, "windows_per_sequence": 8}
    config["architecture"]["residual_gru"]["hidden_size"] = 8
    config["training"].update(device="cpu", epochs=1, batch_size=16)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    output = run_second_round(config_path=path, data_root=None, output_root=tmp_path / "runs", synthetic_smoke=True)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "completed" and report["synthetic"]
    assert len(report["query_ms"]) == 7
    assert [run["route"] for run in report["runs"]] == list("ABC")
    counts = []
    for run in report["runs"]:
        assert run["reload_max_abs_delta"] <= 1e-5
        assert run["training"]["history"][0]["train_loss"] > 0
        checkpoint = torch.load(run["checkpoint"], weights_only=True)
        assert checkpoint["data_contract"]["task_id"] == "h2o_m2_" + run["route"]
        assert checkpoint["horizon_count"] == (5 if run["route"] == "C" else 7)
        row = run["historical_reevaluation"]["per_horizon_ms"]["100"]
        counts.append(row["reference_valid_frames"])
        assert "low_motion" in row["strata"]
    assert len(set(counts)) == 1
    assert not any(c["development_continue_criteria_passed"] for c in report["continuation"].values())
