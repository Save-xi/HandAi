from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from features.geometry_utils import clamp01, euclidean, joint_angle, polyline_length

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
EMPTY_FINGER_CURL = {"thumb": None, "index": None, "middle": None, "ring": None, "little": None}


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    if den <= 1e-6:
        return default
    return num / den


def _mean_point(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def _has_complete_landmarks_2d(landmarks_2d: Sequence[Sequence[float]]) -> bool:
    if len(landmarks_2d) <= LITTLE_TIP:
        return False
    return all(len(point) >= 2 for point in landmarks_2d[: LITTLE_TIP + 1])


def _has_complete_landmarks_3d(
    landmarks_xyz: Sequence[Sequence[float]] | None,
    *,
    expected_len: int,
) -> bool:
    if not landmarks_xyz or len(landmarks_xyz) != expected_len:
        return False
    return all(len(point) >= 3 for point in landmarks_xyz)


def _as_xyz(landmarks_2d: List[Tuple[float, float]], landmarks_xyz: List[Tuple[float, float, float]] | None) -> List[Tuple[float, float, float]]:
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
    chain_points = [landmarks_xyz[idx] for idx in joint_indices]
    chain_len = polyline_length(chain_points)
    if chain_len <= 1e-6:
        return 0.0
    direct = euclidean(chain_points[0], chain_points[-1])
    return clamp01(1.0 - clamp01(direct / chain_len))


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


def extract_hand_features(
    landmarks_2d: List[Tuple[float, float]],
    handedness: str,
    confidence: float | None,
    timestamp: float,
    landmarks_xyz: List[Tuple[float, float, float]] | None = None,
) -> Dict:
    # 让特征提取尽量保持“总是可返回”的风格，方便测试和非摄像头路径：
    # 当 landmark 列表损坏或不完整时，退化为空帧 payload，
    # 而不是在几何代码深处抛出索引错误。
    if not _has_complete_landmarks_2d(landmarks_2d):
        return empty_features(timestamp)

    landmarks_xyz = _as_xyz(landmarks_2d, landmarks_xyz)
    palm_size = _palm_size(landmarks_2d)
    palm_center = _mean_point([landmarks_2d[idx] for idx in PALM_CENTER_POINTS])
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

    finger_curl = {
        "thumb": _thumb_curl(landmarks_xyz),
        "index": _long_finger_curl(landmarks_xyz, mcp=INDEX_MCP, pip=INDEX_PIP, dip=INDEX_DIP, tip=INDEX_TIP),
        "middle": _long_finger_curl(landmarks_xyz, mcp=MIDDLE_MCP, pip=MIDDLE_PIP, dip=MIDDLE_DIP, tip=MIDDLE_TIP),
        "ring": _long_finger_curl(landmarks_xyz, mcp=RING_MCP, pip=RING_PIP, dip=RING_DIP, tip=RING_TIP),
        "little": _long_finger_curl(landmarks_xyz, mcp=LITTLE_MCP, pip=LITTLE_PIP, dip=LITTLE_DIP, tip=LITTLE_TIP),
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
