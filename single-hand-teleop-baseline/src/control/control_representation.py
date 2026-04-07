from __future__ import annotations

from typing import Dict, Iterable

from features.geometry_utils import clamp01, normalize_between
from output.frame_payload_contract import get_stable_gesture

CONTROL_FINGERS = ["thumb", "index", "middle", "ring", "little"]
NON_THUMB_FINGERS = ["index", "middle", "ring", "little"]
SUPPORT_FINGERS = ["middle", "ring", "little"]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)

def empty_control_representation() -> Dict:
    return {
        "valid": False,
        "features_valid": False,
        "command_ready": False,
        "source": None,
        "gesture_context": None,
        "preferred_mapping": None,
        "grasp_close": None,
        "thumb_index_proximity": None,
        "effective_pinch_strength": None,
        "pinch_strength": None,
        "support_flex": None,
        "finger_flex": {name: None for name in CONTROL_FINGERS},
    }


def build_control_representation(payload: Dict, cfg: Dict) -> Dict:
    """把逐帧感知结果转换成面向控制的连续向量。

    这一层刻意保持与硬件无关。它保留手势标签作为上下文，
    同时输出 grasp / pinch 风格的连续量，方便后续 VR 或 SVH 集成
    在不直接依赖原始特征的前提下进行消费。

    语义说明：
    - features_valid：当前帧具备可用的连续特征
    - command_ready / valid：当前手势上下文已经稳定到足以选择映射
    - thumb_index_proximity：拇指与食指接近程度的原始线索
    - effective_pinch_strength：经过手势感知门控后的捏合强度
    """

    gesture = get_stable_gesture(payload)
    finger_curl = payload.get("finger_curl") or {}
    if (
        not payload.get("detected", False)
        or payload.get("hand_open_ratio") is None
        or payload.get("pinch_distance_norm") is None
        or any(finger_curl.get(name) is None for name in CONTROL_FINGERS)
    ):
        return empty_control_representation()

    finger_flex = {name: clamp01(float(finger_curl[name])) for name in CONTROL_FINGERS}
    mean_non_thumb_flex = _mean(finger_flex[name] for name in NON_THUMB_FINGERS)
    support_flex = _mean(finger_flex[name] for name in SUPPORT_FINGERS)

    grasp_from_flex = normalize_between(
        mean_non_thumb_flex,
        float(cfg.get("control_grasp_open_ref", cfg.get("svh_grasp_open_ref", 0.02))),
        float(cfg.get("control_grasp_closed_ref", cfg.get("svh_grasp_closed_ref", 0.55))),
    )
    grasp_from_open_ratio = normalize_between(
        float(payload["hand_open_ratio"]),
        float(cfg.get("control_hand_open_ratio_open_ref", cfg.get("svh_hand_open_ratio_open_ref", 0.95))),
        float(cfg.get("control_hand_open_ratio_closed_ref", cfg.get("svh_hand_open_ratio_closed_ref", 0.25))),
    )
    grasp_close = clamp01(0.60 * grasp_from_flex + 0.40 * grasp_from_open_ratio)

    pinch_from_distance = normalize_between(
        float(payload["pinch_distance_norm"]),
        float(cfg.get("control_pinch_open_ref", cfg.get("svh_pinch_open_ref", 0.45))),
        float(cfg.get("control_pinch_closed_ref", cfg.get("svh_pinch_closed_ref", 0.08))),
    )
    pinch_from_index_flex = normalize_between(
        finger_flex["index"],
        float(cfg.get("control_pinch_index_open_ref", cfg.get("svh_pinch_index_open_ref", 0.05))),
        float(cfg.get("control_pinch_index_closed_ref", cfg.get("svh_pinch_index_closed_ref", 0.35))),
    )
    thumb_index_proximity = clamp01(0.70 * pinch_from_distance + 0.30 * pinch_from_index_flex)

    preferred_mapping = None
    if gesture in {"open", "fist"}:
        preferred_mapping = "grasp"
    elif gesture == "pinch":
        preferred_mapping = "pinch"

    command_ready = preferred_mapping is not None
    effective_pinch_strength = thumb_index_proximity if preferred_mapping == "pinch" else 0.0

    return {
        "valid": command_ready,
        "features_valid": True,
        "command_ready": command_ready,
        "source": "features",
        "gesture_context": gesture,
        "preferred_mapping": preferred_mapping,
        "grasp_close": float(grasp_close),
        "thumb_index_proximity": float(thumb_index_proximity),
        "effective_pinch_strength": float(effective_pinch_strength),
        "pinch_strength": float(effective_pinch_strength),
        "support_flex": float(support_flex),
        "finger_flex": finger_flex,
    }
