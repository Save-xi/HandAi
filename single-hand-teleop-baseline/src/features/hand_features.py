from __future__ import annotations

"""从 21 个手部关键点提取连续几何特征。

这一层是 CV 算法里最核心的“从点到量”步骤：
- 计算拇指-食指捏合距离；
- 计算整只手张开程度；
- 估计五根手指各自弯曲程度；
- 保留 2D / 3D landmark，供可视化、调试和下游记录使用。

后续 gesture / control / SVH 模块都尽量消费这些连续量，
而不是直接再次解析 21 个原始关键点。
"""

import math
from typing import Dict, List, Sequence, Tuple

from features.geometry_utils import clamp01, euclidean, joint_angle, polyline_length

# MediaPipe Hands 的固定 21 点索引。保留具名常量能避免后续代码里到处出现魔法数字。
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
LITTLE_MCP = 17
LITTLE_PIP = 18
LITTLE_DIP = 19
LITTLE_TIP = 20
PALM_CENTER_POINTS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, LITTLE_MCP]
# 低质量帧或空帧里，控制相关 finger_curl 会统一退化成这组空值。
EMPTY_FINGER_CURL = {"thumb": None, "index": None, "middle": None, "ring": None, "little": None}
HAND_LANDMARK_COUNT = LITTLE_TIP + 1


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    """安全除法：尺度过小时返回默认值，避免实时链路里炸出除零错误。"""

    if den <= 1e-6:
        return default
    return num / den


def _mean_point(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def _has_complete_landmarks_2d(landmarks_2d: Sequence[Sequence[float]]) -> bool:
    if len(landmarks_2d) != HAND_LANDMARK_COUNT:
        return False
    return all(len(point) >= 2 for point in landmarks_2d)


def _has_complete_landmarks_3d(
    landmarks_xyz: Sequence[Sequence[float]] | None,
    *,
    expected_len: int,
) -> bool:
    if not landmarks_xyz or len(landmarks_xyz) != expected_len:
        return False
    return all(len(point) >= 3 for point in landmarks_xyz)


def _as_xyz(landmarks_2d: List[Tuple[float, float]], landmarks_xyz: List[Tuple[float, float, float]] | None) -> List[Tuple[float, float, float]]:
    """优先使用 MediaPipe 3D-like 坐标；缺失时用 z=0 的 2D 退化版本。"""

    if _has_complete_landmarks_3d(landmarks_xyz, expected_len=len(landmarks_2d)):
        return [(float(x), float(y), float(z)) for x, y, z in landmarks_xyz]
    return [(x, y, 0.0) for x, y in landmarks_2d]


def _palm_size(landmarks: List[Tuple[float, float]]) -> float:
    # Wrist->middle MCP 形成的掌长尺度更稳定，
    # 相比 index-to-little MCP 的掌宽，更不容易受手指张开程度影响。
    primary = euclidean(landmarks[WRIST], landmarks[MIDDLE_MCP])
    if primary > 1e-6:
        return primary
    return euclidean(landmarks[INDEX_MCP], landmarks[LITTLE_MCP])


def _bend_from_angle(angle: float) -> float:
    # 关节越接近伸直，角度越接近 pi；弯曲越明显，角度越逼近 0。
    return clamp01((math.pi - angle) / math.pi)


def _chain_compression(landmarks_xyz: Sequence[Sequence[float]], joint_indices: Sequence[int]) -> float:
    """衡量一条手指骨架链相对“完全伸直”被压缩了多少。"""

    chain_points = [landmarks_xyz[idx] for idx in joint_indices]
    chain_len = polyline_length(chain_points)
    if chain_len <= 1e-6:
        return 0.0
    direct = euclidean(chain_points[0], chain_points[-1])
    return clamp01(1.0 - clamp01(direct / chain_len))


def _tip_to_palm_curl(
    landmarks_2d: Sequence[Sequence[float]],
    *,
    tip: int,
    palm_center: Tuple[float, float],
    palm_size: float,
) -> float:
    """用指尖靠近掌心的程度补充 curl 判断。

    真实摄像头里，三指抓握常会出现“关节角不夸张，但指尖已经收回掌心附近”的情况。
    单看 PIP/DIP 角度会偏保守，因此这里额外把 tip->palm 距离转成闭合线索。
    """

    tip_distance = euclidean(landmarks_2d[tip], palm_center)
    ratio = _safe_ratio(tip_distance, palm_size, default=10.0)
    open_ref = 1.45
    closed_ref = 0.70
    return clamp01((open_ref - ratio) / (open_ref - closed_ref))


def _long_finger_curl(
    landmarks_xyz: List[Tuple[float, float, float]],
    *,
    mcp: int,
    pip: int,
    dip: int,
    tip: int,
) -> float:
    # 这里使用混合 curl：
    # PIP/DIP 的弯曲反映手指折叠，
    # 链长压缩项则在 2D 投影看上去“假装很直”时补充稳定性。
    pip_bend = _bend_from_angle(joint_angle(landmarks_xyz[mcp], landmarks_xyz[pip], landmarks_xyz[dip]))
    dip_bend = _bend_from_angle(joint_angle(landmarks_xyz[pip], landmarks_xyz[dip], landmarks_xyz[tip]))
    compression = _chain_compression(landmarks_xyz, [mcp, pip, dip, tip])
    return clamp01(0.45 * pip_bend + 0.35 * dip_bend + 0.20 * compression)


def _thumb_curl(landmarks_xyz: List[Tuple[float, float, float]]) -> float:
    # 拇指的运动学和其余长手指不同，因此单独使用：
    # CMC->MCP->IP->TIP 这条链。
    # 权重略偏向两个弯曲角，压缩项则用于稳定部分对掌 / 屈曲姿态。
    mcp_bend = _bend_from_angle(joint_angle(landmarks_xyz[THUMB_CMC], landmarks_xyz[THUMB_MCP], landmarks_xyz[THUMB_IP]))
    ip_bend = _bend_from_angle(joint_angle(landmarks_xyz[THUMB_MCP], landmarks_xyz[THUMB_IP], landmarks_xyz[THUMB_TIP]))
    compression = _chain_compression(landmarks_xyz, [THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP])
    return clamp01(0.40 * mcp_bend + 0.40 * ip_bend + 0.20 * compression)


def _merge_curl_with_tip_proximity(angle_curl: float, tip_palm_curl: float, *, proximity_weight: float = 0.80) -> float:
    """融合关节弯曲和指尖回收两类线索。"""

    return clamp01(max(angle_curl, proximity_weight * tip_palm_curl))


def extract_hand_features(
    landmarks_2d: List[Tuple[float, float]],
    handedness: str,
    confidence: float | None,
    timestamp: float,
    landmarks_xyz: List[Tuple[float, float, float]] | None = None,
) -> Dict:
    """提取单只手的 baseline 特征。

    返回的字典还不是最终 frame payload，它只包含视觉侧基础字段。
    后续主流程会继续补上 gesture、control_representation、svh_preview、
    frame_index、fps、latency_ms 等字段。
    """

    # 让特征提取尽量保持“总是可返回”的风格，方便测试和非摄像头路径：
    # 当 landmark 列表损坏或不完整时，退化为空帧 payload，
    # 而不是在几何代码深处抛出索引错误。
    if not _has_complete_landmarks_2d(landmarks_2d):
        return empty_features(timestamp)

    landmarks_xyz = _as_xyz(landmarks_2d, landmarks_xyz)
    palm_size = _palm_size(landmarks_2d)
    palm_center = _mean_point([landmarks_2d[idx] for idx in PALM_CENTER_POINTS])
    # pinch_distance_norm 是后续 pinch 判断和捏合控制最重要的视觉线索。
    pinch_distance_raw = euclidean(landmarks_2d[THUMB_TIP], landmarks_2d[INDEX_TIP])
    pinch_distance_norm = _safe_ratio(pinch_distance_raw, palm_size, default=1.0)

    # hand_open_ratio 取的是指尖到掌心中心的平均距离，
    # 再用掌长归一化，降低它对相机远近的敏感性。
    hand_open_ratio = _safe_ratio(
        (
            euclidean(landmarks_2d[THUMB_TIP], palm_center)
            + euclidean(landmarks_2d[INDEX_TIP], palm_center)
            + euclidean(landmarks_2d[MIDDLE_TIP], palm_center)
            + euclidean(landmarks_2d[RING_TIP], palm_center)
            + euclidean(landmarks_2d[LITTLE_TIP], palm_center)
        )
        / 5.0,
        palm_size,
        default=0.0,
    )

    tip_palm_curl = {
        "thumb": _tip_to_palm_curl(landmarks_2d, tip=THUMB_TIP, palm_center=palm_center, palm_size=palm_size),
        "index": _tip_to_palm_curl(landmarks_2d, tip=INDEX_TIP, palm_center=palm_center, palm_size=palm_size),
        "middle": _tip_to_palm_curl(landmarks_2d, tip=MIDDLE_TIP, palm_center=palm_center, palm_size=palm_size),
        "ring": _tip_to_palm_curl(landmarks_2d, tip=RING_TIP, palm_center=palm_center, palm_size=palm_size),
        "little": _tip_to_palm_curl(landmarks_2d, tip=LITTLE_TIP, palm_center=palm_center, palm_size=palm_size),
    }

    finger_curl = {
        "thumb": _merge_curl_with_tip_proximity(_thumb_curl(landmarks_xyz), tip_palm_curl["thumb"], proximity_weight=0.70),
        "index": _merge_curl_with_tip_proximity(
            _long_finger_curl(landmarks_xyz, mcp=INDEX_MCP, pip=INDEX_PIP, dip=INDEX_DIP, tip=INDEX_TIP),
            tip_palm_curl["index"],
        ),
        "middle": _merge_curl_with_tip_proximity(
            _long_finger_curl(landmarks_xyz, mcp=MIDDLE_MCP, pip=MIDDLE_PIP, dip=MIDDLE_DIP, tip=MIDDLE_TIP),
            tip_palm_curl["middle"],
        ),
        "ring": _merge_curl_with_tip_proximity(
            _long_finger_curl(landmarks_xyz, mcp=RING_MCP, pip=RING_PIP, dip=RING_DIP, tip=RING_TIP),
            tip_palm_curl["ring"],
        ),
        "little": _merge_curl_with_tip_proximity(
            _long_finger_curl(landmarks_xyz, mcp=LITTLE_MCP, pip=LITTLE_PIP, dip=LITTLE_DIP, tip=LITTLE_TIP),
            tip_palm_curl["little"],
        ),
    }

    return {
        "timestamp": timestamp,
        "detected": True,
        "handedness": handedness,
        "confidence": float(confidence) if confidence is not None else None,
        "gesture_raw": None,
        "gesture_stable": None,
        "pinch_distance_norm": float(pinch_distance_norm),
        "hand_open_ratio": float(hand_open_ratio),
        "finger_curl": finger_curl,
        "landmarks_2d": [[float(x), float(y)] for x, y in landmarks_2d],
        "landmarks_3d": [[float(x), float(y), float(z)] for x, y, z in landmarks_xyz],
    }


def invalidate_control_features(features: Dict) -> Dict:
    """保留检测信息和 landmark，但在低质量帧里清空面向控制的几何量。"""
    degraded = dict(features)
    degraded["gesture_raw"] = None
    degraded["gesture_stable"] = None
    degraded["pinch_distance_norm"] = None
    degraded["hand_open_ratio"] = None
    degraded["finger_curl"] = dict(EMPTY_FINGER_CURL)
    return degraded


def empty_features(timestamp: float) -> Dict:
    """构造“当前没有可用右手”的规范基础特征。"""

    return {
        "timestamp": timestamp,
        "detected": False,
        "handedness": None,
        "confidence": None,
        "gesture_raw": "unknown",
        "gesture_stable": "unknown",
        "pinch_distance_norm": None,
        "hand_open_ratio": None,
        "finger_curl": dict(EMPTY_FINGER_CURL),
        "landmarks_2d": [],
        "landmarks_3d": [],
    }
