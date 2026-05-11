from __future__ import annotations

"""手部几何计算的通用小工具。

这些函数都保持无状态、无业务含义，方便在 hand_features、control、
SVH preview 等模块里复用。
"""

import math
from typing import Sequence


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """计算 2D / 3D 点之间的欧氏距离。"""

    return math.dist(a, b)


def joint_angle(a: Sequence[float], b: Sequence[float], c: Sequence[float], default: float = math.pi) -> float:
    """计算以 b 为顶点的夹角。

    如果任一边长度过小，返回 default。默认 pi 表示“近似伸直”，
    这样退化点不会被误认为强弯曲。
    """

    ba = [x - y for x, y in zip(a, b)]
    bc = [x - y for x, y in zip(c, b)]
    norm_ba = math.sqrt(sum(v * v for v in ba))
    norm_bc = math.sqrt(sum(v * v for v in bc))
    if norm_ba <= 1e-6 or norm_bc <= 1e-6:
        return default
    cosine = sum(x * y for x, y in zip(ba, bc)) / (norm_ba * norm_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return math.acos(cosine)


def polyline_length(points: Sequence[Sequence[float]]) -> float:
    """计算一串点连接形成的折线长度。"""

    if len(points) < 2:
        return 0.0
    return sum(euclidean(points[i], points[i + 1]) for i in range(len(points) - 1))


def normalize_between(value: float, open_ref: float, closed_ref: float, default: float = 0.0) -> float:
    """把 value 按 open/closed 参考值归一化到 [0, 1]。

    同时支持 open_ref < closed_ref 和 open_ref > closed_ref 两种方向。
    """

    if abs(closed_ref - open_ref) <= 1e-6:
        return default
    if closed_ref > open_ref:
        return clamp01((value - open_ref) / (closed_ref - open_ref))
    return clamp01((open_ref - value) / (open_ref - closed_ref))


def clamp01(x: float) -> float:
    """夹到 [0, 1]，用于保护所有归一化控制量。"""

    return max(0.0, min(1.0, x))
