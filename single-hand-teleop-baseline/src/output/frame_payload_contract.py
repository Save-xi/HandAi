from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, TypedDict

FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
CONTROL_PREFERRED_MAPPINGS = ("grasp", "pinch")
SVH_PREVIEW_COMMAND_SOURCES = ("control_representation", "gesture_fallback")
FRAME_PAYLOAD_REQUIRED_FIELDS = (
    "timestamp",
    "frame_index",
    "detected",
    "handedness",
    "confidence",
    "control_ready",
    "gesture_raw",
    "gesture_stable",
    "pinch_distance_norm",
    "hand_open_ratio",
    "finger_curl",
    "landmarks_2d",
    "landmarks_3d",
    "control_representation",
    "svh_preview",
    "fps",
    "latency_ms",
)
CONTROL_REPRESENTATION_REQUIRED_FIELDS = (
    "valid",
    "features_valid",
    "command_ready",
    "source",
    "gesture_context",
    "preferred_mapping",
    "grasp_close",
    "thumb_index_proximity",
    "effective_pinch_strength",
    "pinch_strength",
    "support_flex",
    "finger_flex",
)
SVH_PREVIEW_REQUIRED_FIELDS = (
    "enabled",
    "mode",
    "valid",
    "command_source",
    "target_channels",
    "target_positions",
    "target_ticks_preview",
    "protocol_hint",
)
PROTOCOL_HINT_REQUIRED_FIELDS = (
    "set_control_state_addr",
    "set_all_channels_addr",
    "transport",
    "channel_layout",
    "channel_order",
    "position_units",
    "target_tick_units",
)
DEPRECATED_ALIASES = {
    "gesture": "gesture_stable",
    "svh": "svh_preview",
}


class FingerMap(TypedDict):
    thumb: float | None
    index: float | None
    middle: float | None
    ring: float | None
    little: float | None


class FramePayload(TypedDict, total=False):
    timestamp: float
    frame_index: int
    detected: bool
    handedness: str | None
    confidence: float | None
    control_ready: bool
    gesture_raw: str
    gesture_stable: str
    pinch_distance_norm: float | None
    hand_open_ratio: float | None
    finger_curl: FingerMap
    landmarks_2d: List[List[float]]
    landmarks_3d: List[List[float]]
    control_representation: Dict[str, Any]
    svh_preview: Dict[str, Any]
    fps: float
    latency_ms: float


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _clamp_unit_interval(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _normalize_finger_map(mapping: Any) -> FingerMap:
    mapping = mapping or {}
    return {
        name: _clamp_unit_interval(mapping.get(name))
        for name in FINGER_NAMES
    }


def _normalize_landmarks_2d(landmarks: Any) -> List[List[float]]:
    normalized: List[List[float]] = []
    for point in list(landmarks or []):
        normalized.append([float(point[0]), float(point[1])])
    return normalized


def _normalize_landmarks_3d(landmarks: Any, landmarks_2d: List[List[float]]) -> List[List[float]]:
    if not isinstance(landmarks, list) or len(landmarks) != len(landmarks_2d):
        return [[float(x), float(y), 0.0] for x, y in landmarks_2d]
    normalized: List[List[float]] = []
    for point in list(landmarks):
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            return [[float(x), float(y), 0.0] for x, y in landmarks_2d]
        normalized.append([float(point[0]), float(point[1]), float(point[2])])
    return normalized


def _normalize_control_representation(control: Any) -> Dict[str, Any]:
    normalized = dict(control or {})
    normalized["valid"] = bool(normalized.get("command_ready", normalized.get("valid", False)))
    normalized["features_valid"] = bool(normalized.get("features_valid", False))
    normalized["command_ready"] = bool(normalized.get("command_ready", normalized.get("valid", False)))
    normalized["source"] = None if normalized.get("source") is None else str(normalized.get("source"))
    normalized["gesture_context"] = None if normalized.get("gesture_context") is None else str(normalized.get("gesture_context"))
    preferred_mapping = normalized.get("preferred_mapping")
    normalized["preferred_mapping"] = None if preferred_mapping is None else str(preferred_mapping)
    normalized["grasp_close"] = _clamp_unit_interval(normalized.get("grasp_close"))
    normalized["thumb_index_proximity"] = _clamp_unit_interval(normalized.get("thumb_index_proximity"))
    normalized["support_flex"] = _clamp_unit_interval(normalized.get("support_flex"))
    normalized["finger_flex"] = _normalize_finger_map(normalized.get("finger_flex"))
    effective_pinch_strength = _clamp_unit_interval(
        normalized.get("effective_pinch_strength", normalized.get("pinch_strength"))
    )
    normalized["effective_pinch_strength"] = effective_pinch_strength
    normalized["pinch_strength"] = effective_pinch_strength
    if not normalized["command_ready"]:
        normalized["valid"] = False
    return normalized


def _normalize_svh_preview(preview: Any) -> Dict[str, Any]:
    normalized = dict(preview or {})
    normalized["enabled"] = bool(normalized.get("enabled", False))
    normalized["mode"] = str(normalized.get("mode", "preview" if normalized["enabled"] else "disabled"))
    normalized["valid"] = bool(normalized.get("valid", False))
    normalized["command_source"] = None if normalized.get("command_source") is None else str(normalized.get("command_source"))
    protocol_hint = dict(normalized.get("protocol_hint", {}))
    position_units = str(protocol_hint.get("position_units", "normalized_preview"))
    normalized["target_channels"] = [int(v) for v in list(normalized.get("target_channels", []))]
    if position_units == "normalized_preview":
        normalized["target_positions"] = [
            _clamp_unit_interval(v) or 0.0 for v in list(normalized.get("target_positions", []))
        ]
    else:
        normalized["target_positions"] = [float(v) for v in list(normalized.get("target_positions", []))]
    normalized["target_ticks_preview"] = [int(v) for v in list(normalized.get("target_ticks_preview", []))]
    normalized["protocol_hint"] = protocol_hint

    if not normalized["enabled"]:
        normalized["mode"] = "disabled"
        normalized["valid"] = False

    if not normalized["valid"]:
        normalized["command_source"] = None
        normalized["target_channels"] = []
        normalized["target_positions"] = []
        normalized["target_ticks_preview"] = []
        return normalized

    if len(normalized["target_channels"]) != len(normalized["target_positions"]):
        normalized["valid"] = False
        normalized["command_source"] = None
        normalized["target_channels"] = []
        normalized["target_positions"] = []
        normalized["target_ticks_preview"] = []
        return normalized

    target_tick_units = str(protocol_hint.get("target_tick_units", "none"))
    if target_tick_units != "none" and len(normalized["target_ticks_preview"]) != len(normalized["target_positions"]):
        normalized["valid"] = False
        normalized["command_source"] = None
        normalized["target_channels"] = []
        normalized["target_positions"] = []
        normalized["target_ticks_preview"] = []
    return normalized


def get_stable_gesture(payload: Dict[str, Any]) -> str:
    value = payload.get("gesture_stable", payload.get("gesture"))
    if value is None:
        return "unknown"
    return str(value)


def get_svh_preview(payload: Dict[str, Any]) -> Dict[str, Any]:
    preview = payload.get("svh_preview", payload.get("svh", {}))
    return dict(preview or {})


def normalize_frame_payload(
    payload: Dict[str, Any],
    *,
    include_deprecated_aliases: bool = False,
) -> FramePayload:
    normalized: Dict[str, Any] = deepcopy(payload)
    landmarks_2d = _normalize_landmarks_2d(normalized.get("landmarks_2d", []))
    normalized["gesture_raw"] = str(normalized.get("gesture_raw") or "unknown")
    normalized["gesture_stable"] = get_stable_gesture(normalized)
    normalized["finger_curl"] = _normalize_finger_map(normalized.get("finger_curl"))
    normalized["landmarks_2d"] = landmarks_2d
    normalized["landmarks_3d"] = _normalize_landmarks_3d(normalized.get("landmarks_3d"), landmarks_2d)
    normalized["control_representation"] = _normalize_control_representation(normalized.get("control_representation"))
    normalized["svh_preview"] = _normalize_svh_preview(get_svh_preview(normalized))
    normalized["control_ready"] = bool(
        normalized.get(
            "control_ready",
            normalized["control_representation"].get("command_ready", False),
        )
    )

    if include_deprecated_aliases:
        normalized["gesture"] = normalized["gesture_stable"]
        normalized["svh"] = deepcopy(normalized["svh_preview"])
    else:
        normalized.pop("gesture", None)
        normalized.pop("svh", None)

    return normalized  # type: ignore[return-value]


def _validate_finger_map(name: str, mapping: Any, errors: List[str]) -> None:
    if not isinstance(mapping, dict):
        errors.append(f"{name} 必须是一个包含五个命名手指的对象")
        return
    for finger in FINGER_NAMES:
        if finger not in mapping:
            errors.append(f"{name}.{finger} 是必填字段")
            continue
        value = mapping[finger]
        if value is not None and not _is_number(value):
            errors.append(f"{name}.{finger} 必须是数字或 null")


def _validate_landmarks(name: str, landmarks: Any, dims: int, errors: List[str]) -> None:
    if not isinstance(landmarks, list):
        errors.append(f"{name} 必须是列表")
        return
    for index, point in enumerate(landmarks):
        if not isinstance(point, list) or len(point) != dims:
            errors.append(f"{name}[{index}] 必须是一个长度为 {dims} 的列表")
            continue
        for axis, value in enumerate(point):
            if not _is_number(value):
                errors.append(f"{name}[{index}][{axis}] 必须是数字")


def _validate_unit_interval(name: str, value: Any, errors: List[str]) -> None:
    if value is None:
        return
    if not _is_number(value):
        errors.append(f"{name} 必须是数字或 null")
        return
    if not 0.0 <= float(value) <= 1.0:
        errors.append(f"{name} 必须位于 [0, 1] 区间内")


def validate_frame_payload(
    payload: Dict[str, Any],
    *,
    allow_deprecated_aliases: bool = False,
) -> List[str]:
    errors: List[str] = []

    for field in FRAME_PAYLOAD_REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"缺少必填字段：{field}")

    if not allow_deprecated_aliases:
        for alias in DEPRECATED_ALIASES:
            if alias in payload:
                errors.append(f"不应继续输出已弃用的别名字段：{alias}")
    else:
        for alias, canonical in DEPRECATED_ALIASES.items():
            if alias in payload and canonical in payload and payload[alias] != payload[canonical]:
                errors.append(f"已弃用别名 {alias} 必须与 {canonical} 保持一致")

    if "timestamp" in payload and not _is_number(payload["timestamp"]):
        errors.append("timestamp 必须是数字")
    if "frame_index" in payload:
        value = payload["frame_index"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append("frame_index 必须是非负整数")
    if "detected" in payload and not isinstance(payload["detected"], bool):
        errors.append("detected 必须是布尔值")
    if "handedness" in payload and payload["handedness"] is not None and not isinstance(payload["handedness"], str):
        errors.append("handedness 必须是字符串或 null")
    if "confidence" in payload and payload["confidence"] is not None and not _is_number(payload["confidence"]):
        errors.append("confidence 必须是数字或 null")
    if "confidence" in payload and payload["confidence"] is not None and _is_number(payload["confidence"]):
        if not 0.0 <= float(payload["confidence"]) <= 1.0:
            errors.append("confidence 必须位于 [0, 1] 区间内")
    if "control_ready" in payload and not isinstance(payload["control_ready"], bool):
        errors.append("control_ready 必须是布尔值")
    if "gesture_raw" in payload and not isinstance(payload["gesture_raw"], str):
        errors.append("gesture_raw 必须是字符串")
    if "gesture_stable" in payload and not isinstance(payload["gesture_stable"], str):
        errors.append("gesture_stable 必须是字符串")
    if "pinch_distance_norm" in payload and payload["pinch_distance_norm"] is not None and not _is_number(payload["pinch_distance_norm"]):
        errors.append("pinch_distance_norm 必须是数字或 null")
    if "hand_open_ratio" in payload and payload["hand_open_ratio"] is not None and not _is_number(payload["hand_open_ratio"]):
        errors.append("hand_open_ratio 必须是数字或 null")
    if "fps" in payload and not _is_number(payload["fps"]):
        errors.append("fps 必须是数字")
    if "latency_ms" in payload and not _is_number(payload["latency_ms"]):
        errors.append("latency_ms 必须是数字")

    if "finger_curl" in payload:
        _validate_finger_map("finger_curl", payload["finger_curl"], errors)
    if "landmarks_2d" in payload:
        _validate_landmarks("landmarks_2d", payload["landmarks_2d"], 2, errors)
    if "landmarks_3d" in payload:
        _validate_landmarks("landmarks_3d", payload["landmarks_3d"], 3, errors)
    if "landmarks_2d" in payload and "landmarks_3d" in payload:
        if isinstance(payload["landmarks_2d"], list) and isinstance(payload["landmarks_3d"], list):
            if len(payload["landmarks_2d"]) != len(payload["landmarks_3d"]):
                errors.append("landmarks_3d 的点数必须与 landmarks_2d 一致")

    control = payload.get("control_representation")
    if not isinstance(control, dict):
        errors.append("control_representation 必须是对象")
    else:
        for field in CONTROL_REPRESENTATION_REQUIRED_FIELDS:
            if field not in control:
                errors.append(f"control_representation.{field} 是必填字段")
        if "finger_flex" in control:
            _validate_finger_map("control_representation.finger_flex", control["finger_flex"], errors)
        _validate_unit_interval("control_representation.grasp_close", control.get("grasp_close"), errors)
        _validate_unit_interval("control_representation.thumb_index_proximity", control.get("thumb_index_proximity"), errors)
        _validate_unit_interval("control_representation.effective_pinch_strength", control.get("effective_pinch_strength"), errors)
        _validate_unit_interval("control_representation.pinch_strength", control.get("pinch_strength"), errors)
        _validate_unit_interval("control_representation.support_flex", control.get("support_flex"), errors)
        preferred_mapping = control.get("preferred_mapping")
        if preferred_mapping is not None and preferred_mapping not in CONTROL_PREFERRED_MAPPINGS:
            errors.append("control_representation.preferred_mapping 必须是 grasp、pinch 或 null")
        if "valid" in control and "command_ready" in control and control["valid"] != control["command_ready"]:
            errors.append("control_representation.valid 必须与 control_representation.command_ready 保持一致")
        if control.get("pinch_strength") != control.get("effective_pinch_strength"):
            errors.append("control_representation.pinch_strength 必须与 effective_pinch_strength 保持一致")
        if "control_ready" in payload and payload["control_ready"] != bool(control.get("command_ready", False)):
            errors.append("顶层 control_ready 必须与 control_representation.command_ready 保持一致")

    preview = payload.get("svh_preview")
    if not isinstance(preview, dict):
        errors.append("svh_preview 必须是对象")
    else:
        for field in SVH_PREVIEW_REQUIRED_FIELDS:
            if field not in preview:
                errors.append(f"svh_preview.{field} 是必填字段")
        if "target_channels" in preview:
            if not isinstance(preview["target_channels"], list):
                errors.append("svh_preview.target_channels 必须是列表")
            else:
                for index, value in enumerate(preview["target_channels"]):
                    if not isinstance(value, int) or isinstance(value, bool):
                        errors.append(f"svh_preview.target_channels[{index}] 必须是整数")
        if "target_positions" in preview:
            if not isinstance(preview["target_positions"], list):
                errors.append("svh_preview.target_positions 必须是列表")
            else:
                for index, value in enumerate(preview["target_positions"]):
                    if not _is_number(value):
                        errors.append(f"svh_preview.target_positions[{index}] 必须是数字")
                    elif not 0.0 <= float(value) <= 1.0:
                        errors.append(f"svh_preview.target_positions[{index}] 必须位于 [0, 1] 区间内")
        if "target_ticks_preview" in preview:
            if not isinstance(preview["target_ticks_preview"], list):
                errors.append("svh_preview.target_ticks_preview 必须是列表")
            else:
                for index, value in enumerate(preview["target_ticks_preview"]):
                    if not isinstance(value, int):
                        errors.append(f"svh_preview.target_ticks_preview[{index}] 必须是整数")
        protocol_hint = preview.get("protocol_hint")
        if not isinstance(protocol_hint, dict):
            errors.append("svh_preview.protocol_hint 必须是对象")
        else:
            for field in PROTOCOL_HINT_REQUIRED_FIELDS:
                if field not in protocol_hint:
                    errors.append(f"svh_preview.protocol_hint.{field} 是必填字段")
        if preview.get("enabled") is False and preview.get("valid") is True:
            errors.append("当 svh_preview.enabled 为 false 时，svh_preview.valid 不能为 true")
        if preview.get("valid") is False:
            if preview.get("target_channels"):
                errors.append("当 svh_preview.valid 为 false 时，svh_preview.target_channels 必须为空")
            if preview.get("target_positions"):
                errors.append("当 svh_preview.valid 为 false 时，svh_preview.target_positions 必须为空")
            if preview.get("target_ticks_preview"):
                errors.append("当 svh_preview.valid 为 false 时，svh_preview.target_ticks_preview 必须为空")
        if preview.get("valid") is True:
            if preview.get("command_source") not in SVH_PREVIEW_COMMAND_SOURCES:
                errors.append("当 svh_preview.valid 为 true 时，svh_preview.command_source 必须标明已知 preview 来源")
            if len(preview.get("target_channels", [])) != len(preview.get("target_positions", [])):
                errors.append("svh_preview.target_channels 与 target_positions 的长度必须一致")
            protocol_hint = preview.get("protocol_hint", {})
            if protocol_hint.get("target_tick_units") != "none":
                if len(preview.get("target_ticks_preview", [])) != len(preview.get("target_positions", [])):
                    errors.append("当启用 tick preview 时，svh_preview.target_ticks_preview 的长度必须与 target_positions 一致")

    return errors


def assert_valid_frame_payload(
    payload: Dict[str, Any],
    *,
    allow_deprecated_aliases: bool = False,
) -> None:
    errors = validate_frame_payload(payload, allow_deprecated_aliases=allow_deprecated_aliases)
    if errors:
        raise ValueError("frame payload contract 非法：" + "; ".join(errors))


def prepare_frame_payload(
    payload: Dict[str, Any],
    *,
    include_deprecated_aliases: bool = False,
) -> FramePayload:
    normalized = normalize_frame_payload(
        payload,
        include_deprecated_aliases=include_deprecated_aliases,
    )
    assert_valid_frame_payload(
        normalized,
        allow_deprecated_aliases=include_deprecated_aliases,
    )
    return normalized
