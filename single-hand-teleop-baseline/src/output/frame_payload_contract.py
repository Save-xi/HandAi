from __future__ import annotations

"""逐帧 payload 的规范化与校验。

这个模块是下游稳定消费的“合同层”：
- normalize_* 负责把运行期对象修成稳定形状和值域；
- validate_* 负责发现字段缺失、类型错误和值域错误；
- prepare_frame_payload 是导出前的统一入口。

如果未来要改 payload 字段，优先从这里和 schema 同步改起。
"""

from copy import deepcopy
import math
from typing import Any, Dict, List, TypedDict

FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
"""所有手指映射字段都应使用这一组固定键名。"""

HAND_LANDMARK_COUNT = 21
"""MediaPipe Hands 和当前单右手 contract 固定使用 21 个关键点。"""

CONTROL_PREFERRED_MAPPINGS = ("grasp", "pinch")
"""控制层当前只区分抓握和捏合两类映射。"""

SVH_PREVIEW_COMMAND_SOURCES = ("control_representation", "gesture_fallback")
"""SVH preview 的来源必须可追溯，避免下游误判命令可信度。"""

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
"""canonical frame payload 的顶层必填字段。"""

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
"""control_representation 必填字段；即使无效也必须保留同样形状。"""

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
"""svh_preview 必填字段；无效帧通过 valid=false 和空目标数组表达。"""

PROTOCOL_HINT_REQUIRED_FIELDS = (
    "transport",
    "channel_layout",
    "channel_order",
    "position_units",
    "target_tick_units",
)
"""protocol_hint 是 preview 元数据，不是真实硬件协议证明。"""

TIMING_REQUIRED_FIELDS = (
    "schema_version",
    "clock",
    "source_read_start_unix_ms",
    "source_read_end_unix_ms",
    "detection_end_unix_ms",
    "baseline_end_unix_ms",
    "preview_end_unix_ms",
    "payload_ready_unix_ms",
    "udp_send_attempt_unix_ms",
)
"""可选 timing v1 对象一旦出现，就必须使用完整字段集合。"""

TIMING_EPOCH_FIELDS = TIMING_REQUIRED_FIELDS[2:]
"""timing 中使用同一 Unix epoch 毫秒时钟的字段。"""

PREDICTION_DIAGNOSTICS_REQUIRED_FIELDS = (
    "schema_version",
    "mode",
    "enabled",
    "status",
    "ready",
    "source_frame_index",
    "source_timestamp_unix_ms",
    "history_frames_required",
    "history_frames_available",
    "history_span_ms",
    "observed_fps",
    "horizon_ms",
    "channel_order",
    "hold_last",
    "raw_prediction",
    "gated_prediction",
    "motion_score",
    "base_gate",
    "effective_gate_by_horizon",
    "inference_started_unix_ms",
    "inference_completed_unix_ms",
    "inference_ms",
    "gating_completed_unix_ms",
    "gating_ms",
    "raw_range_violation_count",
    "device",
    "model_label",
    "fallback_reason",
)
"""可选 prediction_diagnostics v1 一旦出现，就必须保持完整稳定形状。"""

PREDICTION_DIAGNOSTIC_STATUSES = (
    "initialization_error",
    "invalid_input",
    "warming_up",
    "predicted",
    "inference_error",
)

PREDICTION_CHANNEL_ORDER = (
    "thumb_flexion",
    "thumb_opposition",
    "index_finger_distal",
    "index_finger_proximal",
    "middle_finger_distal",
    "middle_finger_proximal",
    "ring_finger",
    "pinky",
    "finger_spread",
)
"""影子模型训练、preview 与诊断共用的固定 9 通道语义顺序。"""

DEPRECATED_ALIASES = {
    "gesture": "gesture_stable",
    "svh": "svh_preview",
}

FRAME_PAYLOAD_OPTIONAL_FIELDS = ("timing", "prediction_diagnostics")
"""canonical 顶层可选字段；其余未知键与 JSON Schema 一样必须拒绝。"""


class FingerMap(TypedDict):
    """五指数值映射。

    value 为 None 表示这一帧没有可靠控制特征；不是 0。
    """

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
    timing: Dict[str, Any]
    prediction_diagnostics: Dict[str, Any]


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


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
    source = dict(mapping or {})
    # 保留未知键，让 validate/prepare 明确报错；不能在 normalize 阶段静默吞掉，
    # 否则运行时 contract 会比 additionalProperties=false 的 JSON Schema 更宽松。
    normalized: Dict[str, Any] = dict(source)
    for name in FINGER_NAMES:
        normalized[name] = _clamp_unit_interval(source.get(name))
    return normalized  # type: ignore[return-value]


def _normalize_landmarks_2d(landmarks: Any) -> List[List[float]]:
    normalized: List[List[float]] = []
    for point in list(landmarks or []):
        normalized.append([float(point[0]), float(point[1])])
    return normalized


def _normalize_landmarks_3d(landmarks: Any, landmarks_2d: List[List[float]]) -> List[List[float]]:
    """保证 3D 点数与 2D 点数一致。

    MediaPipe 或测试数据缺失 3D 时，用 z=0 退化，避免下游同时处理两种长度。
    """

    if not isinstance(landmarks, list) or len(landmarks) != len(landmarks_2d):
        return [[float(x), float(y), 0.0] for x, y in landmarks_2d]
    normalized: List[List[float]] = []
    for point in list(landmarks):
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            return [[float(x), float(y), 0.0] for x, y in landmarks_2d]
        normalized.append([float(point[0]), float(point[1]), float(point[2])])
    return normalized


def _normalize_control_representation(control: Any) -> Dict[str, Any]:
    """规范化控制中间层。

    这里会把兼容字段拉齐，比如 valid 与 command_ready、pinch_strength 与
    effective_pinch_strength，避免同一 payload 内部自相矛盾。
    """

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
    """规范化 SVH preview 对象。

    重点保证：无效 preview 必须清空目标数组；有效 preview 的 channels、
    positions 和可选 ticks 长度必须匹配。
    """

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


def _normalize_timing(timing: Any) -> Dict[str, Any] | None:
    """规范化可选的阶段时戳对象，并保留未知键交给严格校验拒绝。"""

    if timing is None:
        return None
    source = dict(timing or {})
    normalized: Dict[str, Any] = dict(source)
    normalized["schema_version"] = int(source.get("schema_version", 1))
    normalized["clock"] = str(source.get("clock", "unix_epoch_ms"))
    for field in TIMING_EPOCH_FIELDS:
        normalized[field] = _maybe_float(source.get(field))
    return normalized


def _normalize_prediction_matrix(matrix: Any) -> List[List[float]]:
    return [
        [float(value) for value in list(row)]
        for row in list(matrix or [])
    ]


def _normalize_prediction_diagnostics(diagnostics: Any) -> Dict[str, Any] | None:
    """规范化可选影子诊断，并保留未知键交给 v1 严格校验拒绝。"""

    if diagnostics is None:
        return None
    source = dict(diagnostics or {})
    normalized: Dict[str, Any] = dict(source)
    normalized.update({
        "schema_version": int(source.get("schema_version", 1)),
        "mode": str(source.get("mode", "shadow")),
        "enabled": bool(source.get("enabled", True)),
        "status": str(source.get("status", "initialization_error")),
        "ready": bool(source.get("ready", False)),
        "source_frame_index": int(source.get("source_frame_index", -1)),
        "source_timestamp_unix_ms": _maybe_float(source.get("source_timestamp_unix_ms")),
        "history_frames_required": int(source.get("history_frames_required", 0)),
        "history_frames_available": int(source.get("history_frames_available", 0)),
        "history_span_ms": _maybe_float(source.get("history_span_ms")),
        "observed_fps": _maybe_float(source.get("observed_fps")),
        "horizon_ms": [int(value) for value in list(source.get("horizon_ms", []))],
        "channel_order": [str(value) for value in list(source.get("channel_order", []))],
        "hold_last": _normalize_prediction_matrix(source.get("hold_last")),
        "raw_prediction": _normalize_prediction_matrix(source.get("raw_prediction")),
        "gated_prediction": _normalize_prediction_matrix(source.get("gated_prediction")),
        "motion_score": _maybe_float(source.get("motion_score")),
        "base_gate": _maybe_float(source.get("base_gate")),
        "effective_gate_by_horizon": [
            float(value) for value in list(source.get("effective_gate_by_horizon", []))
        ],
        "inference_started_unix_ms": _maybe_float(source.get("inference_started_unix_ms")),
        "inference_completed_unix_ms": _maybe_float(source.get("inference_completed_unix_ms")),
        "inference_ms": _maybe_float(source.get("inference_ms")),
        "gating_completed_unix_ms": _maybe_float(source.get("gating_completed_unix_ms")),
        "gating_ms": _maybe_float(source.get("gating_ms")),
        "raw_range_violation_count": int(source.get("raw_range_violation_count", 0)),
        "device": None if source.get("device") is None else str(source.get("device")),
        "model_label": None if source.get("model_label") is None else str(source.get("model_label")),
        "fallback_reason": None if source.get("fallback_reason") is None else str(source.get("fallback_reason")),
    })
    return normalized


def get_stable_gesture(payload: Dict[str, Any]) -> str:
    """读取稳定手势，并兼容旧字段 gesture。"""

    value = payload.get("gesture_stable", payload.get("gesture"))
    if value is None:
        return "unknown"
    return str(value)


def get_svh_preview(payload: Dict[str, Any]) -> Dict[str, Any]:
    """读取 SVH preview，并兼容旧字段 svh。"""

    preview = payload.get("svh_preview", payload.get("svh", {}))
    return dict(preview or {})


def normalize_frame_payload(
    payload: Dict[str, Any],
    *,
    include_deprecated_aliases: bool = False,
) -> FramePayload:
    """把运行期 payload 规范化为 canonical 形状。

    normalize 不负责判断算法是否正确；它只负责让输出字段稳定、值域安全、
    弃用别名按配置处理。
    """

    normalized: Dict[str, Any] = deepcopy(payload)
    landmarks_2d = _normalize_landmarks_2d(normalized.get("landmarks_2d", []))
    normalized["gesture_raw"] = str(normalized.get("gesture_raw") or "unknown")
    normalized["gesture_stable"] = get_stable_gesture(normalized)
    normalized["finger_curl"] = _normalize_finger_map(normalized.get("finger_curl"))
    normalized["landmarks_2d"] = landmarks_2d
    normalized["landmarks_3d"] = _normalize_landmarks_3d(normalized.get("landmarks_3d"), landmarks_2d)
    normalized["control_representation"] = _normalize_control_representation(normalized.get("control_representation"))
    normalized["svh_preview"] = _normalize_svh_preview(get_svh_preview(normalized))
    timing = _normalize_timing(normalized.get("timing"))
    if timing is None:
        normalized.pop("timing", None)
    else:
        normalized["timing"] = timing
    prediction_diagnostics = _normalize_prediction_diagnostics(normalized.get("prediction_diagnostics"))
    if prediction_diagnostics is None:
        normalized.pop("prediction_diagnostics", None)
    else:
        normalized["prediction_diagnostics"] = prediction_diagnostics
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
    for field in mapping:
        if field not in FINGER_NAMES:
            errors.append(f"{name} 不允许未知字段：{field}")
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


def _validate_prediction_matrix(
    name: str,
    matrix: Any,
    *,
    expected_rows: int,
    errors: List[str],
) -> None:
    if not isinstance(matrix, list):
        errors.append(f"{name} 必须是列表")
        return
    if len(matrix) != expected_rows:
        errors.append(f"{name} 必须包含 {expected_rows} 个预测距离")
    for row_index, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != len(PREDICTION_CHANNEL_ORDER):
            errors.append(f"{name}[{row_index}] 必须包含 9 个通道")
            continue
        for channel_index, value in enumerate(row):
            if not _is_number(value):
                errors.append(f"{name}[{row_index}][{channel_index}] 必须是有限数字")
            elif not 0.0 <= float(value) <= 1.0:
                errors.append(f"{name}[{row_index}][{channel_index}] 必须位于 [0, 1]")


def _validate_optional_nonnegative_number(name: str, value: Any, errors: List[str]) -> None:
    if value is None:
        return
    if not _is_number(value):
        errors.append(f"{name} 必须是有限数字或 null")
    elif float(value) < 0.0:
        errors.append(f"{name} 不能为负数")


def _validate_prediction_diagnostics(
    payload: Dict[str, Any],
    diagnostics: Any,
    errors: List[str],
) -> None:
    if not isinstance(diagnostics, dict):
        errors.append("prediction_diagnostics 必须是对象")
        return

    for field in diagnostics:
        # 历史日志可携带旧身份字段；只读取，不作为推理或评测的前置条件。
        if field not in PREDICTION_DIAGNOSTICS_REQUIRED_FIELDS and field not in {"selection_sha256", "checkpoint_sha256"}:
            errors.append(f"prediction_diagnostics 不允许未知字段：{field}")
    for field in PREDICTION_DIAGNOSTICS_REQUIRED_FIELDS:
        if field not in diagnostics:
            errors.append(f"prediction_diagnostics.{field} 是必填字段")

    if diagnostics.get("schema_version") != 1:
        errors.append("prediction_diagnostics.schema_version 当前必须是 1")
    if diagnostics.get("mode") != "shadow":
        errors.append("prediction_diagnostics.mode 当前必须是 shadow")
    if diagnostics.get("enabled") is not True:
        errors.append("prediction_diagnostics.enabled 当前必须为 true")
    status = diagnostics.get("status")
    if status not in PREDICTION_DIAGNOSTIC_STATUSES:
        errors.append("prediction_diagnostics.status 不受支持")
    expected_ready = status == "predicted"
    if diagnostics.get("ready") is not expected_ready:
        errors.append("prediction_diagnostics.ready 必须且仅能在 predicted 状态为 true")

    source_frame_index = diagnostics.get("source_frame_index")
    if not isinstance(source_frame_index, int) or isinstance(source_frame_index, bool) or source_frame_index < 0:
        errors.append("prediction_diagnostics.source_frame_index 必须是非负整数")
    elif source_frame_index != payload.get("frame_index"):
        errors.append("prediction_diagnostics.source_frame_index 必须与顶层 frame_index 一致")
    source_timestamp = diagnostics.get("source_timestamp_unix_ms")
    if not _is_number(source_timestamp) or float(source_timestamp) < 0.0:
        errors.append("prediction_diagnostics.source_timestamp_unix_ms 必须是非负有限数字")
    elif _is_number(payload.get("timestamp")):
        expected_timestamp = float(payload["timestamp"]) * 1000.0
        if not math.isclose(float(source_timestamp), expected_timestamp, rel_tol=0.0, abs_tol=1e-6):
            errors.append("prediction_diagnostics.source_timestamp_unix_ms 必须与顶层 timestamp 对齐")

    required = diagnostics.get("history_frames_required")
    available = diagnostics.get("history_frames_available")
    if not isinstance(required, int) or isinstance(required, bool) or required < 2:
        errors.append("prediction_diagnostics.history_frames_required 必须是至少为 2 的整数")
    if not isinstance(available, int) or isinstance(available, bool) or available < 0:
        errors.append("prediction_diagnostics.history_frames_available 必须是非负整数")
    elif isinstance(required, int) and not isinstance(required, bool) and available > required:
        errors.append("prediction_diagnostics.history_frames_available 不能超过 required")

    _validate_optional_nonnegative_number(
        "prediction_diagnostics.history_span_ms", diagnostics.get("history_span_ms"), errors
    )
    _validate_optional_nonnegative_number(
        "prediction_diagnostics.observed_fps", diagnostics.get("observed_fps"), errors
    )
    horizons = diagnostics.get("horizon_ms")
    if not isinstance(horizons, list) or not horizons:
        errors.append("prediction_diagnostics.horizon_ms 必须是非空列表")
        horizon_count = 0
    else:
        horizon_count = len(horizons)
        for index, value in enumerate(horizons):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"prediction_diagnostics.horizon_ms[{index}] 必须是正整数")
    if diagnostics.get("channel_order") != list(PREDICTION_CHANNEL_ORDER):
        errors.append("prediction_diagnostics.channel_order 必须使用冻结的 SVH 9 通道顺序")

    matrix_fields = ("hold_last", "raw_prediction", "gated_prediction")
    if status == "predicted":
        if available != required:
            errors.append("predicted 状态必须具有完整历史窗口")
        preview = payload.get("svh_preview")
        preview_protocol = preview.get("protocol_hint", {}) if isinstance(preview, dict) else {}
        if (
            not isinstance(preview, dict)
            or preview.get("valid") is not True
            or len(preview.get("target_positions", [])) != len(PREDICTION_CHANNEL_ORDER)
            or not isinstance(preview_protocol, dict)
            or preview_protocol.get("channel_layout") != "svh_9ch"
        ):
            errors.append("predicted 状态必须来源于当前帧有效的 svh_9ch preview")
        for field in matrix_fields:
            _validate_prediction_matrix(
                f"prediction_diagnostics.{field}",
                diagnostics.get(field),
                expected_rows=horizon_count,
                errors=errors,
            )
        _validate_optional_nonnegative_number(
            "prediction_diagnostics.motion_score", diagnostics.get("motion_score"), errors
        )
        if diagnostics.get("motion_score") is None:
            errors.append("predicted 状态必须包含 motion_score")
        _validate_unit_interval("prediction_diagnostics.base_gate", diagnostics.get("base_gate"), errors)
        if diagnostics.get("base_gate") is None:
            errors.append("predicted 状态必须包含 base_gate")
        effective = diagnostics.get("effective_gate_by_horizon")
        if not isinstance(effective, list) or len(effective) != horizon_count:
            errors.append("prediction_diagnostics.effective_gate_by_horizon 长度必须与 horizon 一致")
        else:
            for index, value in enumerate(effective):
                _validate_unit_interval(
                    f"prediction_diagnostics.effective_gate_by_horizon[{index}]", value, errors
                )
        if diagnostics.get("fallback_reason") is not None:
            errors.append("predicted 状态的 fallback_reason 必须是 null")
    else:
        for field in matrix_fields:
            if diagnostics.get(field) != []:
                errors.append(f"非 predicted 状态的 prediction_diagnostics.{field} 必须为空")
        if diagnostics.get("motion_score") is not None or diagnostics.get("base_gate") is not None:
            errors.append("非 predicted 状态不能携带 motion_score 或 base_gate")
        if diagnostics.get("effective_gate_by_horizon") != []:
            errors.append("非 predicted 状态的 effective_gate_by_horizon 必须为空")
        if not isinstance(diagnostics.get("fallback_reason"), str) or not diagnostics.get("fallback_reason"):
            errors.append("非 predicted 状态必须给出 fallback_reason")

    inference_fields = (
        "inference_started_unix_ms",
        "inference_completed_unix_ms",
        "inference_ms",
    )
    for field in inference_fields:
        _validate_optional_nonnegative_number(
            f"prediction_diagnostics.{field}", diagnostics.get(field), errors
        )
    if status == "predicted":
        if any(diagnostics.get(field) is None for field in inference_fields):
            errors.append("predicted 状态必须包含完整推理时序")
    elif status != "inference_error" and any(diagnostics.get(field) is not None for field in inference_fields):
        errors.append("未执行推理的状态不能携带推理时序")
    elif status == "inference_error":
        present_count = sum(diagnostics.get(field) is not None for field in inference_fields)
        if present_count not in (0, len(inference_fields)):
            errors.append("inference_error 的推理时序必须全空或完整")

    gating_fields = ("gating_completed_unix_ms", "gating_ms")
    for field in gating_fields:
        _validate_optional_nonnegative_number(
            f"prediction_diagnostics.{field}", diagnostics.get(field), errors
        )
    if status == "predicted":
        if any(diagnostics.get(field) is None for field in gating_fields):
            errors.append("predicted 状态必须包含完整门控时序")
        if (
            _is_number(diagnostics.get("gating_completed_unix_ms"))
            and _is_number(diagnostics.get("inference_completed_unix_ms"))
            and float(diagnostics["gating_completed_unix_ms"])
            < float(diagnostics["inference_completed_unix_ms"])
        ):
            errors.append("门控完成时刻不能早于推理完成时刻")
    elif any(diagnostics.get(field) is not None for field in gating_fields):
        errors.append("非 predicted 状态不能携带门控时序")

    violation_count = diagnostics.get("raw_range_violation_count")
    if not isinstance(violation_count, int) or isinstance(violation_count, bool) or violation_count < 0:
        errors.append("prediction_diagnostics.raw_range_violation_count 必须是非负整数")
    for field in ("device", "model_label"):
        value = diagnostics.get(field)
        if value is not None and not isinstance(value, str):
            errors.append(f"prediction_diagnostics.{field} 必须是字符串或 null")
    if status == "predicted":
        for field in ("device", "model_label"):
            if not diagnostics.get(field):
                errors.append(f"predicted 状态必须包含 prediction_diagnostics.{field}")


def validate_frame_payload(
    payload: Dict[str, Any],
    *,
    allow_deprecated_aliases: bool = False,
) -> List[str]:
    """返回 payload contract 错误列表。

    这里不直接抛异常，是为了测试和调用方可以决定如何展示错误。
    assert_valid_frame_payload 会把错误列表转成异常。
    """

    errors: List[str] = []

    allowed_top_level = set(FRAME_PAYLOAD_REQUIRED_FIELDS) | set(FRAME_PAYLOAD_OPTIONAL_FIELDS)
    if allow_deprecated_aliases:
        allowed_top_level.update(DEPRECATED_ALIASES)
    for field in payload:
        if field not in allowed_top_level:
            errors.append(f"顶层 payload 不允许未知字段：{field}")

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
    timing = payload.get("timing")
    if timing is not None:
        if not isinstance(timing, dict):
            errors.append("timing 必须是对象")
        else:
            for field in timing:
                if field not in TIMING_REQUIRED_FIELDS:
                    errors.append(f"timing 不允许未知字段：{field}")
            for field in TIMING_REQUIRED_FIELDS:
                if field not in timing:
                    errors.append(f"timing.{field} 是必填字段")
            if timing.get("schema_version") != 1:
                errors.append("timing.schema_version 当前必须是 1")
            if timing.get("clock") != "unix_epoch_ms":
                errors.append("timing.clock 当前必须是 unix_epoch_ms")
            for field in TIMING_EPOCH_FIELDS:
                value = timing.get(field)
                if field == "udp_send_attempt_unix_ms" and value is None:
                    continue
                if not _is_number(value):
                    errors.append(f"timing.{field} 必须是有限数字")
                elif float(value) < 0.0:
                    errors.append(f"timing.{field} 不能为负数")
    if "prediction_diagnostics" in payload:
        _validate_prediction_diagnostics(payload, payload["prediction_diagnostics"], errors)

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
    detected = payload.get("detected")
    landmarks_2d = payload.get("landmarks_2d")
    landmarks_3d = payload.get("landmarks_3d")
    if detected is True:
        if payload.get("handedness") != "Right":
            errors.append("detected=true 时 handedness 必须是 Right")
        if not _is_number(payload.get("confidence")):
            errors.append("detected=true 时 confidence 必须是数字")
        if isinstance(landmarks_2d, list) and len(landmarks_2d) != HAND_LANDMARK_COUNT:
            errors.append(f"detected=true 时 landmarks_2d 必须恰好包含 {HAND_LANDMARK_COUNT} 个点")
        if isinstance(landmarks_3d, list) and len(landmarks_3d) != HAND_LANDMARK_COUNT:
            errors.append(f"detected=true 时 landmarks_3d 必须恰好包含 {HAND_LANDMARK_COUNT} 个点")
    elif detected is False:
        if payload.get("handedness") is not None:
            errors.append("detected=false 时 handedness 必须是 null")
        if payload.get("confidence") is not None:
            errors.append("detected=false 时 confidence 必须是 null")
        if isinstance(landmarks_2d, list) and landmarks_2d:
            errors.append("detected=false 时 landmarks_2d 必须为空")
        if isinstance(landmarks_3d, list) and landmarks_3d:
            errors.append("detected=false 时 landmarks_3d 必须为空")
        if payload.get("control_ready") is True:
            errors.append("detected=false 时 control_ready 不能为 true")

    control = payload.get("control_representation")
    if not isinstance(control, dict):
        errors.append("control_representation 必须是对象")
    else:
        for field in control:
            if field not in CONTROL_REPRESENTATION_REQUIRED_FIELDS:
                errors.append(f"control_representation 不允许未知字段：{field}")
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
        for field in preview:
            if field not in SVH_PREVIEW_REQUIRED_FIELDS:
                errors.append(f"svh_preview 不允许未知字段：{field}")
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
            for field in protocol_hint:
                if field not in PROTOCOL_HINT_REQUIRED_FIELDS and field not in {"set_control_state_addr", "set_all_channels_addr"}:
                    errors.append(f"svh_preview.protocol_hint 不允许未知字段：{field}")
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
            if detected is False:
                errors.append("detected=false 时 svh_preview.valid 不能为 true")
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
    """校验失败时抛出 ValueError，适合导出前的硬检查。"""

    errors = validate_frame_payload(payload, allow_deprecated_aliases=allow_deprecated_aliases)
    if errors:
        raise ValueError("frame payload contract 非法：" + "; ".join(errors))


def prepare_frame_payload(
    payload: Dict[str, Any],
    *,
    include_deprecated_aliases: bool = False,
) -> FramePayload:
    """导出前统一入口：先 normalize，再 validate。"""

    normalized = normalize_frame_payload(
        payload,
        include_deprecated_aliases=include_deprecated_aliases,
    )
    assert_valid_frame_payload(
        normalized,
        allow_deprecated_aliases=include_deprecated_aliases,
    )
    return normalized
