from __future__ import annotations

from typing import Dict, List, Tuple

from features.geometry_utils import clamp01, euclidean

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
LITTLE_TIP = 20
INDEX_MCP = 5
PINKY_MCP = 17


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    if den <= 1e-6:
        return default
    return num / den


def _finger_curl(landmarks: List[Tuple[float, float]], tip: int, mcp: int) -> float:
    # Curl estimate: shorter tip->wrist distance means more curled.
    d_tip_wrist = euclidean(landmarks[tip], landmarks[WRIST])
    d_mcp_wrist = euclidean(landmarks[mcp], landmarks[WRIST])
    straightness = _safe_ratio(d_tip_wrist, d_mcp_wrist, default=1.0)
    curl = 1.0 - clamp01(straightness)
    return clamp01(curl)


def extract_hand_features(
    landmarks_2d: List[Tuple[float, float]], handedness: str, confidence: float, timestamp: float
) -> Dict:
    palm_width = euclidean(landmarks_2d[INDEX_MCP], landmarks_2d[PINKY_MCP])
    pinch_distance_raw = euclidean(landmarks_2d[THUMB_TIP], landmarks_2d[INDEX_TIP])
    pinch_distance = _safe_ratio(pinch_distance_raw, palm_width, default=1.0)

    avg_tip_wrist = (
        euclidean(landmarks_2d[THUMB_TIP], landmarks_2d[WRIST])
        + euclidean(landmarks_2d[INDEX_TIP], landmarks_2d[WRIST])
        + euclidean(landmarks_2d[MIDDLE_TIP], landmarks_2d[WRIST])
        + euclidean(landmarks_2d[RING_TIP], landmarks_2d[WRIST])
        + euclidean(landmarks_2d[LITTLE_TIP], landmarks_2d[WRIST])
    ) / 5.0
    hand_open_ratio = _safe_ratio(avg_tip_wrist, palm_width, default=0.0)

    finger_curl = {
        "thumb": _finger_curl(landmarks_2d, tip=THUMB_TIP, mcp=2),
        "index": _finger_curl(landmarks_2d, tip=INDEX_TIP, mcp=5),
        "middle": _finger_curl(landmarks_2d, tip=MIDDLE_TIP, mcp=9),
        "ring": _finger_curl(landmarks_2d, tip=RING_TIP, mcp=13),
        "little": _finger_curl(landmarks_2d, tip=LITTLE_TIP, mcp=17),
    }

    return {
        "timestamp": timestamp,
        "detected": True,
        "handedness": handedness,
        "detection_confidence": confidence,
        "pinch_distance": float(pinch_distance),
        "hand_open_ratio": float(hand_open_ratio),
        "finger_curl": finger_curl,
        "landmarks_2d": [[float(x), float(y)] for x, y in landmarks_2d],
    }


def empty_features(timestamp: float) -> Dict:
    return {
        "timestamp": timestamp,
        "detected": False,
        "handedness": "None",
        "detection_confidence": 0.0,
        "gesture": "unknown",
        "pinch_distance": 0.0,
        "hand_open_ratio": 0.0,
        "finger_curl": {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "little": 0.0},
        "landmarks_2d": [],
    }
