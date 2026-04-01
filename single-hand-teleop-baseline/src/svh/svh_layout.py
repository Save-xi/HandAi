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

# These defaults are derived from the Unity/C# SVH finger manager's home-setting
# ranges. They are still preview references rather than final, device-calibrated
# encoder limits, but they provide a much more SVH-like scale than normalized
# 0..1 values alone when the 9-channel layout is enabled.
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
        raise ValueError("svh_9ch_open_ticks and svh_9ch_closed_ticks must both contain 9 values")
    return open_values, closed_values
