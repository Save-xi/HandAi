from __future__ import annotations

from typing import Dict, List, Tuple

from features.geometry_utils import clamp01, euclidean

WRIST = 0
THUMB_MCP = 2
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_MCP = 13
RING_TIP = 16
LITTLE_MCP = 17
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


def _palm_size(landmarks: List[Tuple[float, float]]) -> float:
    # Wrist->middle MCP is a stable longitudinal palm scale and is less affected
    # by finger spread than the index-to-little MCP width.
    primary = euclidean(landmarks[WRIST], landmarks[MIDDLE_MCP])
    if primary > 1e-6:
        return primary
    return euclidean(landmarks[INDEX_MCP], landmarks[LITTLE_MCP])


def _finger_curl(landmarks: List[Tuple[float, float]], tip: int, mcp: int) -> float:
    # Curl uses the tip->wrist distance relative to the finger base->wrist distance.
    # Extended fingers produce a ratio >= 1 and map near 0 curl; curled fingers
    # shorten the tip distance and push the estimate toward 1.
    d_tip_wrist = euclidean(landmarks[tip], landmarks[WRIST])
    d_mcp_wrist = euclidean(landmarks[mcp], landmarks[WRIST])
    straightness = _safe_ratio(d_tip_wrist, d_mcp_wrist, default=1.0)
    curl = 1.0 - clamp01(straightness)
    return clamp01(curl)


def extract_hand_features(
    landmarks_2d: List[Tuple[float, float]], handedness: str, confidence: float | None, timestamp: float
) -> Dict:
    palm_size = _palm_size(landmarks_2d)
    palm_center = _mean_point([landmarks_2d[idx] for idx in PALM_CENTER_POINTS])
    pinch_distance_raw = euclidean(landmarks_2d[THUMB_TIP], landmarks_2d[INDEX_TIP])
    pinch_distance_norm = _safe_ratio(pinch_distance_raw, palm_size, default=1.0)

    # Hand openness is the mean fingertip distance to the palm center, normalized
    # by palm size so the ratio is less sensitive to camera distance.
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
        "thumb": _finger_curl(landmarks_2d, tip=THUMB_TIP, mcp=THUMB_MCP),
        "index": _finger_curl(landmarks_2d, tip=INDEX_TIP, mcp=INDEX_MCP),
        "middle": _finger_curl(landmarks_2d, tip=MIDDLE_TIP, mcp=MIDDLE_MCP),
        "ring": _finger_curl(landmarks_2d, tip=RING_TIP, mcp=RING_MCP),
        "little": _finger_curl(landmarks_2d, tip=LITTLE_TIP, mcp=LITTLE_MCP),
    }

    return {
        "timestamp": timestamp,
        "detected": True,
        "handedness": handedness,
        "confidence": float(confidence) if confidence is not None else None,
        "gesture_raw": None,
        "gesture": None,
        "pinch_distance_norm": float(pinch_distance_norm),
        "hand_open_ratio": float(hand_open_ratio),
        "finger_curl": finger_curl,
        "landmarks_2d": [[float(x), float(y)] for x, y in landmarks_2d],
    }


def empty_features(timestamp: float) -> Dict:
    return {
        "timestamp": timestamp,
        "detected": False,
        "handedness": None,
        "confidence": None,
        "gesture_raw": "unknown",
        "gesture": "unknown",
        "pinch_distance_norm": None,
        "hand_open_ratio": None,
        "finger_curl": dict(EMPTY_FINGER_CURL),
        "landmarks_2d": [],
    }
