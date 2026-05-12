from __future__ import annotations

import math
from typing import Any, Dict, Iterable

import numpy as np


JOINT_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
]


def _as_2d_keypoints(value: Any, joint_count: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.shape != (joint_count, 2):
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array


def _mean_or_none(values: Iterable[float]) -> float | None:
    items = [float(v) for v in values if math.isfinite(float(v))]
    if not items:
        return None
    return float(np.mean(items))


def _percentile_or_none(values: Iterable[float], q: float) -> float | None:
    items = [float(v) for v in values if math.isfinite(float(v))]
    if not items:
        return None
    return float(np.percentile(np.asarray(items, dtype=float), q))


def _threshold_key(threshold: float) -> str:
    return f"{float(threshold):g}"


def _pck_valid_predictions(errors: list[float], thresholds: list[float]) -> Dict[str, float | None]:
    if not errors:
        return {_threshold_key(threshold): None for threshold in thresholds}
    array = np.asarray(errors, dtype=float)
    return {_threshold_key(threshold): float(np.mean(array <= float(threshold))) for threshold in thresholds}


def _pck_all_ground_truth(errors: list[float], thresholds: list[float], denominator: int) -> Dict[str, float | None]:
    if denominator <= 0:
        return {_threshold_key(threshold): None for threshold in thresholds}
    array = np.asarray(errors, dtype=float)
    return {_threshold_key(threshold): float(np.sum(array <= float(threshold)) / denominator) for threshold in thresholds}


def _per_joint_mean(errors: list[list[float]], joint_count: int) -> Dict[str, float | None]:
    result: Dict[str, float | None] = {}
    for joint_idx in range(joint_count):
        name = JOINT_NAMES[joint_idx] if joint_idx < len(JOINT_NAMES) else f"joint_{joint_idx}"
        result[name] = _mean_or_none(row[joint_idx] for row in errors if len(row) > joint_idx)
    return result


def evaluate_predictions(
    predictions: Dict[str, Dict[str, Any]],
    gt_2d_by_id: Dict[str, Any],
    *,
    joint_count: int = 21,
    pck_2d_thresholds_px: list[float] | None = None,
) -> Dict[str, Any]:
    """对齐 sample id 后计算 2D FreiHAND 关键点评估指标。"""

    pck_2d_thresholds_px = pck_2d_thresholds_px or [5, 10, 20, 30]
    sample_ids = sorted(gt_2d_by_id.keys())
    predicted_ids = set(predictions.keys())

    complete_count = 0
    matched_count = 0
    missing_prediction_count = 0
    invalid_2d_prediction_count = 0
    evaluated_2d_samples = 0
    errors_2d_flat: list[float] = []
    errors_2d_by_sample: list[list[float]] = []
    latencies: list[float] = []

    for sid in sample_ids:
        pred = predictions.get(sid)
        if pred is None:
            missing_prediction_count += 1
            continue

        matched_count += 1
        latency = pred.get("latency_ms")
        if isinstance(latency, (int, float)) and math.isfinite(float(latency)):
            latencies.append(float(latency))

        pred_2d = _as_2d_keypoints(pred.get("keypoints_2d"), joint_count)
        gt_2d = _as_2d_keypoints(gt_2d_by_id.get(sid), joint_count)
        if pred_2d is None:
            invalid_2d_prediction_count += 1
            continue
        complete_count += 1

        if gt_2d is not None:
            sample_errors_2d = np.linalg.norm(pred_2d - gt_2d, axis=1).astype(float).tolist()
            evaluated_2d_samples += 1
            errors_2d_by_sample.append(sample_errors_2d)
            errors_2d_flat.extend(sample_errors_2d)

    total_gt_samples = len(sample_ids)
    complete_denominator = total_gt_samples if total_gt_samples else len(predictions)
    gt_keypoint_denominator = total_gt_samples * joint_count

    return {
        "sample_counts": {
            "ground_truth_samples": total_gt_samples,
            "prediction_samples": len(predictions),
            "matched_prediction_samples": matched_count,
            "missing_prediction_samples": missing_prediction_count,
            "extra_prediction_samples": len(predicted_ids - set(sample_ids)),
            "invalid_2d_prediction_samples": invalid_2d_prediction_count,
            "evaluated_2d_samples": evaluated_2d_samples,
            "evaluated_2d_keypoints": len(errors_2d_flat),
        },
        "keypoint_complete_rate": None
        if complete_denominator == 0
        else float(complete_count / complete_denominator),
        "mpjpe_2d_px": _mean_or_none(errors_2d_flat),
        "pck_2d_at_thresholds": _pck_valid_predictions(errors_2d_flat, pck_2d_thresholds_px),
        "pck_2d_at_thresholds_all_gt": _pck_all_ground_truth(
            errors_2d_flat,
            pck_2d_thresholds_px,
            gt_keypoint_denominator,
        ),
        "per_joint_error": {
            "2d_px": _per_joint_mean(errors_2d_by_sample, joint_count),
        },
        "latency_ms": {
            "mean": _mean_or_none(latencies),
            "p95": _percentile_or_none(latencies, 95),
            "count": len(latencies),
        },
    }
