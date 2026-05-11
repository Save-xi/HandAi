from __future__ import annotations

"""把视觉特征整理成硬件无关的控制中间层。

hand_features 输出的是“看见了什么”：捏合距离、张开比例、五指 curl。
control_representation 输出的是“下游控制可以怎么理解这些量”：
抓握闭合程度、捏合强度、支撑手指弯曲程度、推荐映射族。

这一层刻意不直接生成 SVH 通道值，是为了让 Unity、SVH 或其他下游
都能复用同一份连续控制语义。
"""

from typing import Dict, Iterable

from features.geometry_utils import clamp01, normalize_between
from output.frame_payload_contract import get_stable_gesture

CONTROL_FINGERS = ["thumb", "index", "middle", "ring", "little"]
"""控制层要求五根手指都具备有效 flex 值。"""

NON_THUMB_FINGERS = ["index", "middle", "ring", "little"]
"""抓握闭合更依赖四根长手指，拇指不直接主导 grasp_close。"""

SUPPORT_FINGERS = ["middle", "ring", "little"]
"""pinch 中的支撑手指。它们过度弯曲时，单纯的拇指-食指接近不应被当成强捏合。"""


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _resolve_effective_pinch_strength(thumb_index_proximity: float, support_flex: float, grasp_close: float) -> float:
    """把“拇指接近食指”修正成真正可用的捏合强度。

    仅凭拇指-食指距离很容易把紧凑拳头误当成 pinch。
    因此这里会扣除支撑手指弯曲和整体抓握闭合的影响。
    """

    # 21 个视觉关键点不会一一映射到 Unity 的 20 个关节。
    # 正确链路应该是：21 点 -> 连续手部特征 -> 9 个执行通道 -> Unity 20 个关节展开。
    # 之前的问题在于连续特征被稳定手势门控截断，导致大量中间姿态在进入 Unity 前就丢了。
    return clamp01(thumb_index_proximity - 0.45 * support_flex - 0.60 * grasp_close)

def empty_control_representation() -> Dict:
    """构造符合 contract 的空控制对象。

    即使 control 扩展关闭或当前帧不可控，payload 里也会保留这个对象。
    这样下游永远能读取同一套字段，只需要看 command_ready / valid。
    """

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

    # finger_flex 直接继承视觉侧 curl，但先夹到 [0, 1]，保护下游不受异常值影响。
    finger_flex = {name: clamp01(float(finger_curl[name])) for name in CONTROL_FINGERS}
    mean_non_thumb_flex = _mean(finger_flex[name] for name in NON_THUMB_FINGERS)
    support_flex = _mean(finger_flex[name] for name in SUPPORT_FINGERS)

    # grasp_close 同时看“长手指 curl”和“指尖离掌心距离”。
    # 这样比只用某一个指标更不容易受视角或单个手指抖动影响。
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

    # pinch 强度先由距离主导，再用食指弯曲补充。这样能区分“伸直靠近”和“食指参与捏合”。
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
    effective_pinch_strength = _resolve_effective_pinch_strength(
        thumb_index_proximity,
        support_flex,
        grasp_close,
    )

    if gesture == "pinch" or effective_pinch_strength >= 0.35:
        preferred_mapping = "pinch"
    else:
        preferred_mapping = "grasp"

    # 只要连续特征齐全，当前控制层就认为可以给下游消费。
    # 具体是否启用 SVH preview、是否发送 UDP，由后续扩展层决定。
    command_ready = True

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
