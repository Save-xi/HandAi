from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


FREIHAND_EVAL_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "freihand_eval"
if str(FREIHAND_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FREIHAND_EVAL_ROOT))

from freihand.io import load_config, load_split_annotations, resolve_path
from freihand.metrics import evaluate_predictions
from freihand.projection import project_xyz_to_uv


def test_project_xyz_to_uv_uses_pinhole_camera_formula():
    xyz = np.asarray([[[1.0, 2.0, 4.0], [0.0, 0.0, 0.0]]])
    K = np.asarray([[[100.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]]])

    uv = project_xyz_to_uv(xyz, K, z_epsilon=1.0e-8)

    assert uv[0, 0, 0] == 345.0
    assert uv[0, 0, 1] == 340.0
    assert math.isnan(uv[0, 1, 0])
    assert math.isnan(uv[0, 1, 1])


def test_evaluate_predictions_reports_core_2d_metrics_for_complete_samples():
    gt_2d = {"00000000": [[0.0, 0.0]] * 21}
    predictions = {
        "00000000": {
            "keypoints_2d": [[3.0, 4.0]] * 21,
            "latency_ms": 12.0,
        }
    }

    metrics = evaluate_predictions(
        predictions,
        gt_2d,
        pck_2d_thresholds_px=[5, 10],
    )

    assert metrics["keypoint_complete_rate"] == 1.0
    assert metrics["mpjpe_2d_px"] == 5.0
    assert metrics["pck_2d_at_thresholds"]["5"] == 1.0
    assert metrics["pck_2d_at_thresholds_all_gt"]["5"] == 1.0
    assert metrics["latency_ms"]["mean"] == 12.0
    assert metrics["sample_counts"]["evaluated_2d_samples"] == 1
    assert metrics["sample_counts"]["evaluated_2d_keypoints"] == 21


def test_keypoint_complete_rate_and_all_gt_pck_count_missing_samples_as_failures():
    complete_points_2d = [[0.0, 0.0]] * 21
    gt_2d = {"00000000": complete_points_2d, "00000001": complete_points_2d}
    predictions = {
        "00000000": {
            "keypoints_2d": complete_points_2d,
            "latency_ms": 10.0,
        }
    }

    metrics = evaluate_predictions(predictions, gt_2d, pck_2d_thresholds_px=[5])

    assert metrics["keypoint_complete_rate"] == 0.5
    assert metrics["sample_counts"]["missing_prediction_samples"] == 1
    assert metrics["pck_2d_at_thresholds"]["5"] == 1.0
    assert metrics["pck_2d_at_thresholds_all_gt"]["5"] == 0.5


def test_load_split_annotations_resolves_yaml_relative_paths_and_aligns_ids(tmp_path):
    dataset_dir = tmp_path / "dataset"
    config_dir = tmp_path / "configs"
    dataset_dir.mkdir()
    config_dir.mkdir()
    (dataset_dir / "evaluation_K.json").write_text(
        json.dumps(
            [
                [[100, 0, 320], [0, 100, 240], [0, 0, 1]],
                [[110, 0, 320], [0, 110, 240], [0, 0, 1]],
            ]
        ),
        encoding="utf-8",
    )
    (dataset_dir / "evaluation_xyz.json").write_text(
        json.dumps([[[0, 0, 1]] * 21, [[1, 1, 2]] * 21]),
        encoding="utf-8",
    )
    (dataset_dir / "evaluation_scale.json").write_text(json.dumps([1.0, 2.0]), encoding="utf-8")
    config_path = config_dir / "freihand_eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "annotations:",
                "  evaluation:",
                "    K: ../dataset/evaluation_K.json",
                "    xyz: ../dataset/evaluation_xyz.json",
                "    scale: ../dataset/evaluation_scale.json",
                "paths:",
                "  eval_metrics_json: ../reports/eval_metrics.json",
                "metrics:",
                "  joint_count: 21",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    split = load_split_annotations(config, "evaluation", require_xyz=True, require_K=True)

    assert resolve_path(config, "../reports/eval_metrics.json") == tmp_path / "reports" / "eval_metrics.json"
    assert split.sample_ids == ["00000000", "00000001"]
    assert split.K.shape == (2, 3, 3)
    assert split.xyz.shape == (2, 21, 3)
    assert split.scale.shape == (2,)
