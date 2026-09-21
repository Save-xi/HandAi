"""关键点轨迹查询与带覆盖率的 MPJPE；不调用旧通道指标或门控。"""
from __future__ import annotations

import numpy as np

from .keypoint_data import KeypointWindows


def query_trajectory(prediction: np.ndarray, grid_ms: tuple[float, ...], horizons_ms: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(prediction)
    grid = np.asarray(grid_ms, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (len(grid), 63) or not np.isfinite(values).all():
        raise ValueError("关键点预测必须为有限的 [N,未来步数,63]")
    if not len(grid) or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("预测时间网格必须有限且严格递增")
    results = []
    for h in horizons_ms:
        if not np.isfinite(h) or h < grid[0] - 1e-7 or h > grid[-1] + 1e-7:
            raise ValueError("目标时距超出预测轨迹范围")
        right = min(int(np.searchsorted(grid, h)), len(grid) - 1)
        if np.isclose(grid[right], h, atol=1e-7, rtol=0):
            results.append(values[:, right])
        else:
            left = right - 1
            alpha = (h - grid[left]) / (grid[right] - grid[left])
            results.append(values[:, left] * (1 - alpha) + values[:, right] * alpha)
    return np.stack(results, axis=1).reshape(len(values), len(horizons_ms), 21, 3)


def _masked_frame_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    count = mask.sum(axis=-1)
    return np.divide((values * mask).sum(axis=-1), count, out=np.full(count.shape, np.nan, dtype=float), where=count > 0)


def _summarize(values: np.ndarray, sequence_ids: np.ndarray) -> dict:
    valid = np.isfinite(values)
    per_sequence = {str(seq): float(values[(sequence_ids == seq) & valid].mean())
                    for seq in np.unique(sequence_ids[valid])}
    return {"sequence_equal_mean": float(np.mean(list(per_sequence.values()))) if per_sequence else None,
            "frame_mean": float(values[valid].mean()) if valid.any() else None,
            "frame_p95": float(np.percentile(values[valid], 95)) if valid.any() else None,
            "frames": int(valid.sum()), "sequences": len(per_sequence), "per_sequence": per_sequence}


def compute_keypoint_metrics(split: KeypointWindows, prediction: np.ndarray) -> dict:
    pred = query_trajectory(prediction, split.horizon_ms, split.eval_horizon_ms).astype(np.float64)
    if pred.shape != split.eval_y.shape:
        raise ValueError("预测与监督样本数量不一致")
    target, masks = split.eval_y.astype(np.float64), split.eval_mask
    error = np.linalg.norm(pred - target, axis=-1)
    normalized = _masked_frame_mean(error, masks)
    metric_mm = normalized * split.scales[:, None] * 1000
    wrist = error[:, :, 0] * split.scales[:, None] * 1000
    wrist[~masks[:, :, 0]] = np.nan
    relative_error = np.linalg.norm((pred[:, :, 1:] - pred[:, :, :1]) -
                                    (target[:, :, 1:] - target[:, :, :1]), axis=-1)
    relative = _masked_frame_mean(relative_error, masks[:, :, 1:] & masks[:, :, :1]) * split.scales[:, None] * 1000
    per_horizon = {}
    for index, h in enumerate(split.eval_horizon_ms):
        mask = masks[:, index]
        per_horizon[str(h)] = {
            "normalized_mpjpe": _summarize(normalized[:, index], split.sequence_ids),
            "mpjpe_mm": _summarize(metric_mm[:, index], split.sequence_ids),
            "wrist_error_mm": _summarize(wrist[:, index], split.sequence_ids),
            "relative_finger_error_mm": _summarize(relative[:, index], split.sequence_ids),
            "coverage": {"eligible_input_windows": len(mask), "valid_target_points": int(mask.sum()),
                         "possible_target_points": int(mask.size), "point_fraction": float(mask.mean()),
                         "complete_target_frames": int(mask.all(axis=1).sum()),
                         "complete_target_fraction": float(mask.all(axis=1).mean())},
            "per_subject_normalized_mpjpe": {
                str(subject): _summarize(normalized[split.subjects == subject, index], split.sequence_ids[split.subjects == subject])["sequence_equal_mean"]
                for subject in np.unique(split.subjects)}
        }
    primary = [per_horizon[str(h)]["normalized_mpjpe"]["sequence_equal_mean"] for h in (50, 100) if str(h) in per_horizon]
    if len(primary) != 2 or any(value is None for value in primary):
        score = None
    else:
        score = float(np.mean(primary))
    finite_rows = np.argwhere(np.isfinite(normalized))
    ranked = sorted(finite_rows.tolist(), key=lambda row: normalized[tuple(row)], reverse=True)[:5]
    failures = [{"sequence_id": str(split.sequence_ids[n]), "anchor_frame_id": int(split.anchor_frame_ids[n]),
                 "anchor_time_s": float(split.anchor_times[n]), "horizon_ms": int(split.eval_horizon_ms[h]),
                 "normalized_mpjpe": float(normalized[n, h]), "mpjpe_mm": float(metric_mm[n, h]),
                 "valid_joints": int(masks[n, h].sum())} for n, h in ranked]
    return {"selection_metric": "sequence_equal_normalized_mpjpe_mean_50_100ms", "selection_score": score,
            "per_horizon_ms": per_horizon, "worst_anchors": failures,
            "coverage_scope": "sampled_history_valid_windows_including_missing_future_labels",
            "full_future_path_fraction": float(split.target_mask.all(axis=(1, 2)).mean())}
