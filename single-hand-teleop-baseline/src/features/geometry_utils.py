from __future__ import annotations

import math
from typing import Tuple


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.dist(a, b)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
