from __future__ import annotations

from typing import Dict, List, Tuple

SVH_9CH_LAYOUT = "svh_9ch"
SVH_9CH_NAMES = [
    "thumb_flexion",
    "thumb_opposition",
    "index_finger_distal",
    "index_finger_proximal",
    "middle_finger_distal",
    "middle_finger_proximal",
    "ring_finger",
    "pinky",
    "finger_spread",
]

# 这些默认值来自 Unity / C# 参考实现里 SVH finger manager 的 home-setting
# 范围。它们仍然只是 preview 参考值，而不是最终完成设备标定后的编码器限位；
# 但在启用 9 通道布局时，它们相比单纯使用归一化 0..1 数值，更接近
# SVH 风格的尺度。
SVH_9CH_OPEN_TICKS = [
    -5000,
    -5000,
    -2000,
    2000,
    -2000,
    2000,
    -2000,
    -2000,
    -2000,
]
SVH_9CH_CLOSED_TICKS = [
    -175000,
    -150000,
    -47000,
    42000,
    -47000,
    42000,
    -47000,
    -47000,
    -47000,
]


def get_svh_9ch_tick_refs(cfg: Dict) -> Tuple[List[int], List[int]]:
    open_ticks = cfg.get("svh_9ch_open_ticks", SVH_9CH_OPEN_TICKS)
    closed_ticks = cfg.get("svh_9ch_closed_ticks", SVH_9CH_CLOSED_TICKS)
    open_values = [int(v) for v in open_ticks]
    closed_values = [int(v) for v in closed_ticks]
    if len(open_values) != len(SVH_9CH_NAMES) or len(closed_values) != len(SVH_9CH_NAMES):
        raise ValueError("svh_9ch_open_ticks 和 svh_9ch_closed_ticks 都必须包含 9 个值")
    return open_values, closed_values
