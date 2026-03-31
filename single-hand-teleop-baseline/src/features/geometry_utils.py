from __future__ import annotations

import math
from typing import Sequence


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.dist(a, b)


def joint_angle(a: Sequence[float], b: Sequence[float], c: Sequence[float], default: float = math.pi) -> float:
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
    if len(points) < 2:
        return 0.0
    return sum(euclidean(points[i], points[i + 1]) for i in range(len(points) - 1))


def normalize_between(value: float, open_ref: float, closed_ref: float, default: float = 0.0) -> float:
    if abs(closed_ref - open_ref) <= 1e-6:
        return default
    if closed_ref > open_ref:
        return clamp01((value - open_ref) / (closed_ref - open_ref))
    return clamp01((open_ref - value) / (open_ref - closed_ref))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
