from __future__ import annotations

"""统一的 9 通道未来控制预测指标。"""

from typing import Any

import numpy as np

from .sequence_data import SVH_CHANNEL_NAMES


def observed_motion_score(history: np.ndarray, *, recent_frames: int = 8) -> np.ndarray:
    """仅用已观测历史计算逐窗口运动强度，避免用未来真值定义动态样本。"""

    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != len(SVH_CHANNEL_NAMES):
        raise ValueError(f"history 必须为 [N, T, 9]，实际为 {values.shape}")
    count = min(max(2, int(recent_frames)), values.shape[1])
    return np.mean(np.abs(np.diff(values[:, -count:, :], axis=1)), axis=(1, 2))


def compute_motion_stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    history: np.ndarray,
    horizon_ms: tuple[int, ...],
    thresholds: dict[str, float],
    recent_frames: int = 8,
) -> dict[str, Any]:
    """按验证集预先确定的历史运动阈值，计算测试集动态子集指标。"""

    scores = observed_motion_score(history, recent_frames=recent_frames)
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    if len(scores) != len(truth) or len(truth) != len(prediction):
        raise ValueError("history、y_true 与 y_pred 的样本数必须一致")
    strata = {}
    for label, threshold in thresholds.items():
        mask = scores >= float(threshold)
        strata[label] = {
            "threshold": float(threshold),
            "samples": int(np.count_nonzero(mask)),
            "sample_fraction": float(np.mean(mask)),
            "metrics": compute_forecast_metrics(truth[mask], prediction[mask], horizon_ms=horizon_ms)
            if np.any(mask)
            else None,
        }
    return {
        "recent_frames": int(min(max(2, recent_frames), history.shape[1])),
        "score_percentiles": {
            str(percentile): float(np.percentile(scores, percentile)) for percentile in (0, 25, 50, 75, 90, 100)
        },
        "strata": strata,
    }


def compute_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    horizon_ms: tuple[int, ...],
) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 3 or truth.shape[-1] != len(SVH_CHANNEL_NAMES):
        raise ValueError(f"预测与真值必须同为 [N, H, 9]，实际为 {truth.shape} 和 {prediction.shape}")
    if truth.shape[1] != len(horizon_ms):
        raise ValueError("horizon_ms 数量与预测张量不一致")

    error = prediction - truth
    absolute = np.abs(error)
    squared = error**2
    per_horizon = {}
    for index, horizon in enumerate(horizon_ms):
        per_horizon[str(horizon)] = {
            "mae": float(np.mean(absolute[:, index, :])),
            "rmse": float(np.sqrt(np.mean(squared[:, index, :]))),
            "p95_abs_error": float(np.percentile(absolute[:, index, :], 95)),
        }
    per_channel = {
        name: {
            "mae": float(np.mean(absolute[:, :, index])),
            "rmse": float(np.sqrt(np.mean(squared[:, :, index]))),
        }
        for index, name in enumerate(SVH_CHANNEL_NAMES)
    }
    violation = np.logical_or(prediction < 0.0, prediction > 1.0)
    return {
        "samples": int(truth.shape[0]),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(squared))),
        "p95_abs_error": float(np.percentile(absolute, 95)),
        "range_violation_rate": float(np.mean(violation)),
        "per_horizon_ms": per_horizon,
        "per_channel": per_channel,
    }
