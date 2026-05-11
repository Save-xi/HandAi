from __future__ import annotations

"""控制前的 landmark 质量门控。

MediaPipe 有时会在手贴边、遮挡或快速运动时仍给出一组关键点。
这些关键点“看到了手”，但不一定适合生成控制命令。
本模块的作用就是在连续控制特征进入下游前，先拦住高风险帧。
"""

from typing import Dict, List, Tuple

from features.hand_features import INDEX_MCP, LITTLE_MCP, MIDDLE_MCP, RING_MCP, WRIST

# 掌心核心点比指尖更稳定，适合判断整只手是否贴边或大面积越界。
PALM_CORE_POINTS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, LITTLE_MCP]


def _mean_point(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def assess_control_readiness(landmarks_2d: List[Tuple[float, float]], cfg: Dict) -> Dict[str, float | bool]:
    """在局部缺失或越界的手影响控制特征前，先把它们拦下来。

    MediaPipe 的 landmark 可能会轻微漂到 [0, 1] 外面，因此掌心核心点
    检查允许一点容差。掌心中心边距则更严格，因为太贴近图像边缘的手
    往往会让 pinch / open 指标不稳定。
    """

    if not landmarks_2d:
        return {
            "control_ready": False,
            "in_bounds_ratio": 0.0,
            "palm_center_margin": 0.0,
        }

    min_in_bounds_ratio = float(cfg.get("control_ready_min_in_bounds_ratio", 0.90))
    palm_center_margin = float(cfg.get("control_ready_palm_center_margin", 0.08))
    palm_core_oob_tolerance = float(cfg.get("control_ready_palm_core_oob_tolerance", 0.02))

    # 大多数点在画面内，是“这帧能不能相信”的第一层粗检查。
    in_bounds_count = sum(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in landmarks_2d)
    in_bounds_ratio = in_bounds_count / len(landmarks_2d)

    palm_core = [landmarks_2d[idx] for idx in PALM_CORE_POINTS]
    palm_center = _mean_point(palm_core)
    # 掌心中心太靠边时，pinch_distance / hand_open_ratio 很容易被截断画面误导。
    palm_center_ok = (
        palm_center_margin <= palm_center[0] <= 1.0 - palm_center_margin
        and palm_center_margin <= palm_center[1] <= 1.0 - palm_center_margin
    )
    # 核心掌点允许轻微越界，是为了容忍 MediaPipe 的小幅数值漂移。
    palm_core_ok = all(
        -palm_core_oob_tolerance <= x <= 1.0 + palm_core_oob_tolerance
        and -palm_core_oob_tolerance <= y <= 1.0 + palm_core_oob_tolerance
        for x, y in palm_core
    )

    return {
        "control_ready": bool(in_bounds_ratio >= min_in_bounds_ratio and palm_center_ok and palm_core_ok),
        "in_bounds_ratio": float(in_bounds_ratio),
        "palm_center_margin": float(
            min(palm_center[0], 1.0 - palm_center[0], palm_center[1], 1.0 - palm_center[1])
        ),
    }
