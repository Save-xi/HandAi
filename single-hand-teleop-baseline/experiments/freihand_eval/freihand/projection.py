from __future__ import annotations

from typing import Any

import numpy as np


def project_xyz_to_uv(xyz: np.ndarray, K: np.ndarray, *, z_epsilon: float = 1.0e-8) -> np.ndarray:
    """按针孔相机模型把 3D 点投影到 2D 像素坐标。

    xyz 支持形状 (21, 3) 或 (N, 21, 3)，K 支持 (3, 3) 或 (N, 3, 3)。
    Z 过小的点会输出 NaN，写 JSON 时再转成 None，避免无穷大污染评估。
    """

    xyz_array = np.asarray(xyz, dtype=float)
    K_array = np.asarray(K, dtype=float)

    single_sample = xyz_array.ndim == 2
    if single_sample:
        xyz_array = xyz_array[None, ...]
    if K_array.ndim == 2:
        K_array = np.broadcast_to(K_array, (xyz_array.shape[0], 3, 3))

    if xyz_array.ndim != 3 or xyz_array.shape[-1] != 3:
        raise ValueError(f"xyz must have shape (N, J, 3), got {xyz_array.shape}")
    if K_array.ndim != 3 or K_array.shape[-2:] != (3, 3):
        raise ValueError(f"K must have shape (N, 3, 3), got {K_array.shape}")
    if K_array.shape[0] != xyz_array.shape[0]:
        raise ValueError("xyz and K sample counts do not match")

    X = xyz_array[..., 0]
    Y = xyz_array[..., 1]
    Z = xyz_array[..., 2]
    valid_z = np.abs(Z) > float(z_epsilon)

    fx = K_array[:, 0, 0][:, None]
    fy = K_array[:, 1, 1][:, None]
    cx = K_array[:, 0, 2][:, None]
    cy = K_array[:, 1, 2][:, None]

    uv = np.full(xyz_array.shape[:-1] + (2,), np.nan, dtype=float)
    safe_Z = np.where(valid_z, Z, 1.0)
    uv[..., 0] = fx * X / safe_Z + cx
    uv[..., 1] = fy * Y / safe_Z + cy
    uv[~valid_z] = np.nan
    return uv[0] if single_sample else uv


def numpy_to_jsonable_points(points: np.ndarray) -> list[Any]:
    """把含 NaN 的 numpy 点数组转成 JSON 友好的 list/None。"""

    array = np.asarray(points, dtype=float)
    result: list[Any] = []
    for point in array:
        if np.asarray(point).ndim == 1:
            if not np.all(np.isfinite(point)):
                result.append([None for _ in range(point.shape[0])])
            else:
                result.append([float(value) for value in point])
        else:
            result.append(numpy_to_jsonable_points(point))
    return result
