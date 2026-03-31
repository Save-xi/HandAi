from __future__ import annotations

from typing import Dict, List

from control.control_representation import build_control_representation
from features.geometry_utils import clamp01
from svh.svh_command import SvhCommandPreview
from svh.svh_protocol import SET_ALL_CHANNELS_ADDR, SET_CONTROL_STATE_ADDR


def _lerp(open_value: float, closed_value: float, alpha: float) -> float:
    return open_value + clamp01(alpha) * (closed_value - open_value)


def _float_list(values: List[float]) -> List[float]:
    return [float(v) for v in values]


def _protocol_hint(cfg: Dict) -> Dict[str, str]:
    return {
        "set_control_state_addr": f"0x{SET_CONTROL_STATE_ADDR:02X}",
        "set_all_channels_addr": f"0x{SET_ALL_CHANNELS_ADDR:02X}",
        "transport": str(cfg.get("svh_transport", "mock")),
    }


def _invalid_preview(enabled: bool, mode: str, cfg: Dict) -> Dict:
    return SvhCommandPreview(
        enabled=enabled,
        mode=mode,
        valid=False,
        command_source=None,
        target_channels=[],
        target_positions=[],
        protocol_hint=_protocol_hint(cfg),
    ).to_dict()


def _target_channels(count: int) -> List[int]:
    return list(range(max(0, count)))


def _resize_positions(values: List[float], count: int, fill_value: float) -> List[float]:
    if count <= len(values):
        return _float_list(values[:count])
    return _float_list(values + [fill_value] * (count - len(values)))


def _gesture_fallback_preview(gesture: str, cfg: Dict) -> Dict:
    # Gesture fallback is a demo-oriented safety net. It keeps the preview layer
    # visually responsive when continuous measurements are missing, but it is
    # intentionally configurable because a more hardware-facing pipeline should
    # usually refuse low-quality frames instead of synthesizing commands.
    channel_count = int(cfg.get("svh_preview_channel_count", 5))
    open_value = float(cfg.get("svh_position_open_value", 0.0))
    closed_value = float(cfg.get("svh_position_closed_value", 1.0))
    pinch_support_scale = float(cfg.get("svh_pinch_support_scale", 0.20))
    thumb_grasp_scale = float(cfg.get("svh_thumb_grasp_scale", 0.85))

    if gesture == "open":
        positions = _resize_positions([open_value] * 5, channel_count, open_value)
    elif gesture == "fist":
        positions = _resize_positions(
            [
                _lerp(open_value, closed_value, thumb_grasp_scale),
                closed_value,
                closed_value,
                closed_value,
                closed_value,
            ],
            channel_count,
            closed_value,
        )
    elif gesture == "pinch":
        support_value = _lerp(open_value, closed_value, pinch_support_scale)
        positions = _resize_positions(
            [closed_value, closed_value, support_value, support_value, support_value],
            channel_count,
            support_value,
        )
    else:
        return _invalid_preview(True, str(cfg.get("svh_preview_mode", "preview")), cfg)

    return SvhCommandPreview(
        enabled=True,
        mode=str(cfg.get("svh_preview_mode", "preview")),
        valid=True,
        command_source="gesture_fallback",
        target_channels=_target_channels(channel_count),
        target_positions=positions,
        protocol_hint=_protocol_hint(cfg),
    ).to_dict()


def _build_grasp_preview(control_representation: Dict, cfg: Dict) -> Dict:
    grasp_close = clamp01(float(control_representation["grasp_close"]))
    thumb_flex = clamp01(float(control_representation["finger_flex"]["thumb"]))
    channel_count = int(cfg.get("svh_preview_channel_count", 5))
    open_value = float(cfg.get("svh_position_open_value", 0.0))
    closed_value = float(cfg.get("svh_position_closed_value", 1.0))
    thumb_grasp_scale = float(cfg.get("svh_thumb_grasp_scale", 0.85))
    thumb_close = clamp01(0.75 * (grasp_close * thumb_grasp_scale) + 0.25 * thumb_flex)

    preview_values = [
        _lerp(open_value, closed_value, thumb_close),
        _lerp(open_value, closed_value, grasp_close),
        _lerp(open_value, closed_value, grasp_close),
        _lerp(open_value, closed_value, grasp_close),
        _lerp(open_value, closed_value, grasp_close),
    ]
    positions = _resize_positions(preview_values, channel_count, _lerp(open_value, closed_value, grasp_close))

    return SvhCommandPreview(
        enabled=True,
        mode=str(cfg.get("svh_preview_mode", "preview")),
        valid=True,
        command_source="control_representation",
        target_channels=_target_channels(channel_count),
        target_positions=positions,
        protocol_hint=_protocol_hint(cfg),
    ).to_dict()


def _build_pinch_preview(control_representation: Dict, cfg: Dict) -> Dict:
    pinch_close = clamp01(
        float(
            control_representation.get(
                "effective_pinch_strength",
                control_representation.get("pinch_strength", 0.0),
            )
        )
    )
    thumb_flex = clamp01(float(control_representation["finger_flex"]["thumb"]))
    index_flex = clamp01(float(control_representation["finger_flex"]["index"]))
    support_flex = clamp01(float(control_representation["support_flex"]))
    channel_count = int(cfg.get("svh_preview_channel_count", 5))
    open_value = float(cfg.get("svh_position_open_value", 0.0))
    closed_value = float(cfg.get("svh_position_closed_value", 1.0))
    pinch_support_scale = float(cfg.get("svh_pinch_support_scale", 0.20))

    thumb_value = _lerp(open_value, closed_value, clamp01(0.85 * pinch_close + 0.15 * thumb_flex))
    index_value = _lerp(open_value, closed_value, clamp01(0.75 * pinch_close + 0.25 * index_flex))
    support_value = _lerp(open_value, closed_value, max(support_flex, pinch_close * pinch_support_scale))
    preview_values = [
        thumb_value,
        index_value,
        support_value,
        support_value,
        support_value,
    ]
    positions = _resize_positions(preview_values, channel_count, support_value)

    return SvhCommandPreview(
        enabled=True,
        mode=str(cfg.get("svh_preview_mode", "preview")),
        valid=True,
        command_source="control_representation",
        target_channels=_target_channels(channel_count),
        target_positions=positions,
        protocol_hint=_protocol_hint(cfg),
    ).to_dict()


def build_svh_command_preview(payload: Dict, cfg: Dict) -> Dict:
    enabled = bool(cfg.get("svh_enable_preview", True))
    mode = str(cfg.get("svh_preview_mode", "preview" if enabled else "disabled"))
    if not enabled:
        return _invalid_preview(False, "disabled", cfg)

    gesture = payload.get("gesture") or "unknown"
    control_representation = payload.get("control_representation") or build_control_representation(payload, cfg)
    features_valid = bool(control_representation.get("features_valid", control_representation.get("valid", False)))
    command_ready = bool(control_representation.get("command_ready", control_representation.get("valid", False)))
    preferred_mapping = control_representation.get("preferred_mapping")
    enable_gesture_fallback = bool(cfg.get("svh_enable_gesture_fallback", False))

    if command_ready:
        if preferred_mapping == "grasp":
            return _build_grasp_preview(control_representation, cfg)
        if preferred_mapping == "pinch":
            return _build_pinch_preview(control_representation, cfg)
        return _invalid_preview(True, mode, cfg)

    if enable_gesture_fallback and not features_valid and gesture in {"open", "fist", "pinch"}:
        return _gesture_fallback_preview(gesture, cfg)

    return _invalid_preview(True, mode, cfg)
