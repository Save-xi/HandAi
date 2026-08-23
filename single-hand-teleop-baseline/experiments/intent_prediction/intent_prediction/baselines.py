from __future__ import annotations

"""控制序列预测的非学习基线。"""

import numpy as np


def _validate(history: np.ndarray, horizon_steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(history, dtype=np.float32)
    horizons = np.asarray(horizon_steps, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != 9:
        raise ValueError(f"history 应为 [N, T, 9]，实际为 {x.shape}")
    if horizons.ndim != 1 or len(horizons) == 0 or np.any(horizons <= 0):
        raise ValueError("horizon_steps 必须是一维正数数组")
    return x, horizons


def predict_hold_last(history: np.ndarray, horizon_steps: np.ndarray) -> np.ndarray:
    x, horizons = _validate(history, horizon_steps)
    return np.repeat(x[:, -1:, :], len(horizons), axis=1)


def predict_linear(history: np.ndarray, horizon_steps: np.ndarray, *, fit_frames: int = 8) -> np.ndarray:
    """对最近若干帧做逐通道最小二乘直线拟合并外推。"""

    x, horizons = _validate(history, horizon_steps)
    count = min(max(2, int(fit_frames)), x.shape[1])
    recent = x[:, -count:, :].astype(np.float64)
    t = np.arange(count, dtype=np.float64)
    centered = t - np.mean(t)
    denominator = float(np.sum(centered**2))
    slope = np.sum(recent * centered[None, :, None], axis=1) / denominator
    prediction = recent[:, -1:, :] + slope[:, None, :] * horizons[None, :, None]
    return np.clip(prediction, 0.0, 1.0).astype(np.float32)


def predict_kalman_cv(
    history: np.ndarray,
    horizon_steps: np.ndarray,
    *,
    process_variance: float = 2e-3,
    measurement_variance: float = 8e-3,
) -> np.ndarray:
    """逐通道常速度 Kalman 滤波；协方差共享，状态对样本和通道向量化。"""

    x, horizons = _validate(history, horizon_steps)
    if process_variance <= 0 or measurement_variance <= 0:
        raise ValueError("Kalman 方差必须为正数")

    position = x[:, 0, :].astype(np.float64)
    velocity = np.zeros_like(position)
    p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0
    q = float(process_variance)
    r = float(measurement_variance)

    for frame in x[:, 1:, :].transpose(1, 0, 2):
        position = position + velocity
        p00_pred = p00 + p01 + p10 + p11 + 0.25 * q
        p01_pred = p01 + p11 + 0.5 * q
        p10_pred = p10 + p11 + 0.5 * q
        p11_pred = p11 + q

        innovation = frame.astype(np.float64) - position
        innovation_cov = p00_pred + r
        k0 = p00_pred / innovation_cov
        k1 = p10_pred / innovation_cov
        position = position + k0 * innovation
        velocity = velocity + k1 * innovation

        p00 = (1.0 - k0) * p00_pred
        p01 = (1.0 - k0) * p01_pred
        p10 = p10_pred - k1 * p00_pred
        p11 = p11_pred - k1 * p01_pred

    prediction = position[:, None, :] + velocity[:, None, :] * horizons[None, :, None]
    return np.clip(prediction, 0.0, 1.0).astype(np.float32)
