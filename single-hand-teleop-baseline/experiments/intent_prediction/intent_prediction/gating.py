from __future__ import annotations

"""只使用已观测历史和验证集拟合的 persistence 安全门控。"""

from typing import Any

import numpy as np

from .metrics import compute_forecast_metrics, observed_motion_score


def _core_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "p95_abs_error": float(np.percentile(absolute, 95)),
    }


def _objective(
    metrics: dict[str, float],
    hold_metrics: dict[str, float],
    config: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    epsilon = 1e-12
    ratios = {
        key: float(metrics[key] / max(hold_metrics[key], epsilon))
        for key in ("mae", "rmse", "p95_abs_error")
    }
    weights = {
        "mae": float(config.get("mae_weight", 0.50)),
        "rmse": float(config.get("rmse_weight", 0.35)),
        "p95_abs_error": float(config.get("p95_weight", 0.15)),
    }
    weight_sum = sum(weights.values())
    if weight_sum <= 0.0:
        raise ValueError("门控目标权重之和必须为正数")
    weighted = sum(weights[key] * ratios[key] for key in weights) / weight_sum
    allowed = float(config.get("allowed_mae_regression_ratio", 0.0))
    penalty = float(config.get("mae_regression_penalty", 4.0)) * max(0.0, ratios["mae"] - 1.0 - allowed)
    return float(weighted + penalty), ratios


def _base_gate(scores: np.ndarray, *, threshold: float, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        return (scores >= threshold).astype(np.float64)
    z = np.clip((scores - threshold) / temperature, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def apply_motion_gate(
    history: np.ndarray,
    model_prediction: np.ndarray,
    *,
    threshold: float,
    temperature: float,
    alpha_by_horizon: list[float] | tuple[float, ...],
    recent_frames: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    prediction = np.asarray(model_prediction, dtype=np.float64)
    if prediction.ndim != 3 or prediction.shape[0] != len(history) or prediction.shape[-1] != 9:
        raise ValueError("model_prediction 必须为与 history 样本数一致的 [N, H, 9]")
    alpha = np.asarray(alpha_by_horizon, dtype=np.float64)
    if alpha.shape != (prediction.shape[1],) or np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise ValueError("alpha_by_horizon 必须与预测距离数量一致且位于 [0, 1]")
    scores = observed_motion_score(history, recent_frames=recent_frames)
    base_gate = _base_gate(scores, threshold=float(threshold), temperature=float(temperature))
    hold = np.repeat(np.asarray(history, dtype=np.float64)[:, -1:, :], prediction.shape[1], axis=1)
    effective_gate = base_gate[:, None, None] * alpha[None, :, None]
    blended = np.clip(hold + effective_gate * (prediction - hold), 0.0, 1.0).astype(np.float32)
    return blended, {
        "motion_score_min": float(np.min(scores)),
        "motion_score_max": float(np.max(scores)),
        "base_gate_mean": float(np.mean(base_gate)),
        "base_gate_active_fraction": float(np.mean(base_gate >= 0.5)),
        "effective_gate_mean_by_horizon": [float(np.mean(base_gate) * value) for value in alpha],
    }


def fit_motion_gate(
    history: np.ndarray,
    y_true: np.ndarray,
    model_prediction: np.ndarray,
    *,
    horizon_ms: tuple[int, ...],
    config: dict[str, Any],
) -> dict[str, Any]:
    """只在调用方提供的数据上拟合门控；正式流程必须传 validation split。"""

    prediction = np.asarray(model_prediction, dtype=np.float64)
    truth = np.asarray(y_true, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError("y_true 与 model_prediction 必须同为 [N, H, 9]")
    if prediction.shape[1] != len(horizon_ms):
        raise ValueError("horizon_ms 与预测距离数量不一致")
    recent_frames = int(config.get("recent_frames", 8))
    scores = observed_motion_score(history, recent_frames=recent_frames)
    percentiles = [float(value) for value in config.get("threshold_percentiles", [50, 75, 90, 95, 99])]
    temperature_fractions = [float(value) for value in config.get("temperature_fractions", [0.0, 0.25, 0.5])]
    alpha_candidates = [float(value) for value in config.get("alpha_candidates", [0.0, 0.25, 0.5, 0.75, 1.0])]
    if not percentiles or any(value < 0.0 or value > 100.0 for value in percentiles):
        raise ValueError("threshold_percentiles 必须位于 [0, 100]")
    if not alpha_candidates or any(value < 0.0 or value > 1.0 for value in alpha_candidates):
        raise ValueError("alpha_candidates 必须位于 [0, 1]")

    hold = np.repeat(np.asarray(history, dtype=np.float64)[:, -1:, :], truth.shape[1], axis=1)
    hold_core = _core_metrics(truth, hold)
    hold_full = compute_forecast_metrics(truth, hold, horizon_ms=horizon_ms)
    spread = max(float(np.percentile(scores, 90) - np.percentile(scores, 50)), 1e-8)
    search_results = []

    for percentile in percentiles:
        threshold = float(np.percentile(scores, percentile))
        for temperature_fraction in temperature_fractions:
            temperature = max(0.0, temperature_fraction * spread)
            base_gate = _base_gate(scores, threshold=threshold, temperature=temperature)
            chosen_alpha = []
            horizon_search = []
            for horizon_index in range(truth.shape[1]):
                hold_horizon = _core_metrics(truth[:, horizon_index : horizon_index + 1], hold[:, horizon_index : horizon_index + 1])
                best_horizon = None
                for alpha in alpha_candidates:
                    candidate = hold[:, horizon_index : horizon_index + 1] + (
                        base_gate[:, None, None]
                        * alpha
                        * (prediction[:, horizon_index : horizon_index + 1] - hold[:, horizon_index : horizon_index + 1])
                    )
                    candidate = np.clip(candidate, 0.0, 1.0)
                    core = _core_metrics(truth[:, horizon_index : horizon_index + 1], candidate)
                    score, ratios = _objective(core, hold_horizon, config)
                    record = {"alpha": alpha, "objective": score, "metrics": core, "ratios": ratios}
                    if best_horizon is None or score < best_horizon["objective"] - 1e-12:
                        best_horizon = record
                if best_horizon is None:
                    raise RuntimeError("门控 horizon 搜索没有产生候选")
                chosen_alpha.append(float(best_horizon["alpha"]))
                horizon_search.append(best_horizon)

            gated, gate_summary = apply_motion_gate(
                history,
                prediction,
                threshold=threshold,
                temperature=temperature,
                alpha_by_horizon=chosen_alpha,
                recent_frames=recent_frames,
            )
            core = _core_metrics(truth, gated)
            score, ratios = _objective(core, hold_core, config)
            search_results.append(
                {
                    "threshold_percentile": percentile,
                    "threshold": threshold,
                    "temperature_fraction": temperature_fraction,
                    "temperature": temperature,
                    "alpha_by_horizon": chosen_alpha,
                    "objective": score,
                    "metrics": core,
                    "ratios": ratios,
                    "gate_summary": gate_summary,
                    "horizon_selection": horizon_search,
                }
            )

    if not search_results:
        raise RuntimeError("门控搜索没有产生候选")
    search_results.sort(
        key=lambda item: (
            item["objective"],
            sum(item["alpha_by_horizon"]),
            item["gate_summary"]["base_gate_mean"],
        )
    )
    best = search_results[0]
    best_prediction, best_gate_summary = apply_motion_gate(
        history,
        prediction,
        threshold=float(best["threshold"]),
        temperature=float(best["temperature"]),
        alpha_by_horizon=list(best["alpha_by_horizon"]),
        recent_frames=recent_frames,
    )
    return {
        "fit_split_required": "validation",
        "uses_future_target_for_gate_at_inference": False,
        "recent_frames": recent_frames,
        "motion_score_percentiles": {
            str(value): float(np.percentile(scores, value)) for value in (0, 25, 50, 75, 90, 95, 99, 100)
        },
        "objective_config": {
            key: config.get(key)
            for key in (
                "mae_weight",
                "rmse_weight",
                "p95_weight",
                "allowed_mae_regression_ratio",
                "mae_regression_penalty",
            )
        },
        "hold_metrics": hold_full,
        "raw_model_metrics": compute_forecast_metrics(truth, prediction, horizon_ms=horizon_ms),
        "best": {
            **best,
            "gate_summary": best_gate_summary,
            "full_metrics": compute_forecast_metrics(truth, best_prediction, horizon_ms=horizon_ms),
        },
        "search_count": len(search_results),
        "top_candidates": search_results[: min(10, len(search_results))],
    }
