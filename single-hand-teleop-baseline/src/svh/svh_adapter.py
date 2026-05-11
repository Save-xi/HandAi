from __future__ import annotations

"""把 control_representation 转换成 SVH / Unity 风格的 preview 命令。

这里仍然是 preview 层，不是真机驱动层。它做的是：
- 根据控制中间层选择 grasp 或 pinch 映射；
- 生成 compact5 或 svh_9ch 通道目标；
- 输出归一化 target_positions；
- 在 svh_9ch 下额外给出 target_ticks_preview 作为调试参考。

真实硬件发送、ACK、限位、homing、fault 处理都不在这个模块里。
"""

from typing import Dict, List

from control.control_representation import build_control_representation
from features.geometry_utils import clamp01
from output.frame_payload_contract import get_stable_gesture
from svh.svh_command import SvhCommandPreview
from svh.svh_layout import SVH_9CH_LAYOUT, SVH_9CH_NAMES, get_svh_9ch_tick_refs
from svh.svh_protocol import SET_ALL_CHANNELS_ADDR, SET_CONTROL_STATE_ADDR

COMPACT5_LAYOUT = "compact5"
"""轻量 5 通道预览：thumb/index/middle/ring/little。"""


def _lerp(open_value: float, closed_value: float, alpha: float) -> float:
    """把 [0, 1] 的闭合程度 alpha 映射到配置中的 open/closed 数值区间。"""

    return open_value + clamp01(alpha) * (closed_value - open_value)


def _float_list(values: List[float]) -> List[float]:
    return [float(v) for v in values]


def _blend_finger_follow(finger_flex: float, guide_close: float, *, flex_weight: float, guide_weight: float) -> float:
    """让单指既跟随自己的 curl，也跟随整体抓握引导。"""

    return clamp01(flex_weight * finger_flex + guide_weight * guide_close)


def _blend_support_follow(
    finger_flex: float,
    support_floor: float,
    grasp_close: float,
    *,
    flex_weight: float,
    support_weight: float,
    grasp_weight: float,
) -> float:
    """pinch 时支撑手指的混合策略。

    middle/ring/little 不应该像拇指食指那样完全闭合，但也不能完全不动。
    support_floor 给它们一个最小参与度，grasp_close 保留半握状态的影响。
    """

    return clamp01(flex_weight * finger_flex + support_weight * support_floor + grasp_weight * grasp_close)


def _protocol_hint(cfg: Dict) -> Dict[str, str]:
    """写入 preview 元数据，帮助下游解释通道布局和单位。

    这些字段是说明性 hint，不代表真实 SVH 协议已经完成验证。
    """

    layout = _layout(cfg)
    return {
        "set_control_state_addr": f"0x{SET_CONTROL_STATE_ADDR:02X}",
        "set_all_channels_addr": f"0x{SET_ALL_CHANNELS_ADDR:02X}",
        "transport": str(cfg.get("svh_transport", "mock")),
        "channel_layout": layout,
        "channel_order": ",".join(SVH_9CH_NAMES) if layout == SVH_9CH_LAYOUT else "thumb,index,middle,ring,little",
        "position_units": "normalized_preview",
        "target_tick_units": "encoder_ticks_preview" if layout == SVH_9CH_LAYOUT else "none",
    }


def _invalid_preview(enabled: bool, mode: str, cfg: Dict) -> Dict:
    """生成一个“扩展存在但当前没有可用命令”的安全对象。"""

    return SvhCommandPreview(
        enabled=enabled,
        mode=mode,
        valid=False,
        command_source=None,
        target_channels=[],
        target_positions=[],
        protocol_hint=_protocol_hint(cfg),
    ).to_dict()


def empty_svh_preview(cfg: Dict, *, enabled: bool = False, mode: str | None = None) -> Dict:
    resolved_mode = mode or ("preview" if enabled else "disabled")
    return _invalid_preview(enabled, resolved_mode, cfg)


def _target_channels(count: int) -> List[int]:
    return list(range(max(0, count)))


def _layout(cfg: Dict) -> str:
    return str(cfg.get("svh_preview_layout", COMPACT5_LAYOUT))


def _channel_count(cfg: Dict) -> int:
    default_count = 9 if _layout(cfg) == SVH_9CH_LAYOUT else 5
    return int(cfg.get("svh_preview_channel_count", default_count))


def _resize_positions(values: List[float], count: int, fill_value: float) -> List[float]:
    """让输出通道数和配置保持一致。

    count 小于默认映射时截断；count 大于默认映射时用 fill_value 补齐。
    这主要服务于 preview 实验，不建议真机阶段随意改通道数。
    """

    if count <= len(values):
        return _float_list(values[:count])
    return _float_list(values + [fill_value] * (count - len(values)))


def _svh9_positions_from_alphas(alphas: List[float], cfg: Dict) -> List[float]:
    open_value = float(cfg.get("svh_position_open_value", 0.0))
    closed_value = float(cfg.get("svh_position_closed_value", 1.0))
    return [_lerp(open_value, closed_value, alpha) for alpha in alphas]


def _position_alpha(position: float, cfg: Dict) -> float:
    open_value = float(cfg.get("svh_position_open_value", 0.0))
    closed_value = float(cfg.get("svh_position_closed_value", 1.0))
    denom = closed_value - open_value
    if abs(denom) < 1e-9:
        return 0.0
    return clamp01((float(position) - open_value) / denom)


def _target_ticks_preview(positions: List[float], cfg: Dict) -> List[int]:
    """把归一化 target_positions 换算成 preview ticks。

    这些 ticks 只用于调试和校准规划；真实硬件接入前必须重新确认零位、
    方向、上下限和单位。
    """

    if _layout(cfg) != SVH_9CH_LAYOUT:
        return []
    open_ticks, closed_ticks = get_svh_9ch_tick_refs(cfg)
    ticks: List[int] = []
    for position, open_tick, closed_tick in zip(positions, open_ticks, closed_ticks):
        alpha = _position_alpha(position, cfg)
        tick = round(open_tick + alpha * (closed_tick - open_tick))
        ticks.append(int(tick))
    return ticks


def _compact_gesture_fallback_preview(gesture: str, cfg: Dict) -> Dict:
    # 手势兜底是面向演示场景的安全网。
    # 当连续测量缺失时，它能让预览层保持可见响应；
    # 但它被刻意设计成可配置，因为更接近硬件的链路通常应该拒绝低质量帧，
    # 而不是主动合成命令。
    channel_count = _channel_count(cfg)
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
        return empty_svh_preview(cfg, enabled=True, mode=str(cfg.get("svh_preview_mode", "preview")))

    return SvhCommandPreview(
        enabled=True,
        mode=str(cfg.get("svh_preview_mode", "preview")),
        valid=True,
        command_source="gesture_fallback",
        target_channels=_target_channels(channel_count),
        target_positions=positions,
        target_ticks_preview=_target_ticks_preview(positions, cfg),
        protocol_hint=_protocol_hint(cfg),
    ).to_dict()


def _svh9_gesture_fallback_preview(gesture: str, cfg: Dict) -> Dict:
    channel_count = _channel_count(cfg)
    thumb_grasp_scale = float(cfg.get("svh_thumb_grasp_scale", 0.85))
    thumb_opposition_scale = float(cfg.get("svh_thumb_opposition_scale", 0.75))
    pinch_support_scale = float(cfg.get("svh_pinch_support_scale", 0.20))
    open_spread_scale = float(cfg.get("svh_open_spread_scale", 0.25))
    pinch_spread_scale = float(cfg.get("svh_pinch_spread_scale", 0.10))

    if gesture == "open":
        alphas = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, open_spread_scale]
        fill_value = _svh9_positions_from_alphas([0.0], cfg)[0]
    elif gesture == "fist":
        thumb_close = clamp01(thumb_grasp_scale)
        thumb_opp = clamp01(thumb_opposition_scale)
        alphas = [thumb_close, thumb_opp, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        fill_value = _svh9_positions_from_alphas([1.0], cfg)[0]
    elif gesture == "pinch":
        support = clamp01(pinch_support_scale)
        alphas = [1.0, 1.0, 1.0, 0.85, support, support, support, support, pinch_spread_scale]
        fill_value = _svh9_positions_from_alphas([support], cfg)[0]
    else:
        return empty_svh_preview(cfg, enabled=True, mode=str(cfg.get("svh_preview_mode", "preview")))

    positions = _resize_positions(_svh9_positions_from_alphas(alphas, cfg), channel_count, fill_value)
    return SvhCommandPreview(
        enabled=True,
        mode=str(cfg.get("svh_preview_mode", "preview")),
        valid=True,
        command_source="gesture_fallback",
        target_channels=_target_channels(channel_count),
        target_positions=positions,
        target_ticks_preview=_target_ticks_preview(positions, cfg),
        protocol_hint=_protocol_hint(cfg),
    ).to_dict()


def _gesture_fallback_preview(gesture: str, cfg: Dict) -> Dict:
    if _layout(cfg) == SVH_9CH_LAYOUT:
        return _svh9_gesture_fallback_preview(gesture, cfg)
    return _compact_gesture_fallback_preview(gesture, cfg)


def _build_compact_grasp_preview(control_representation: Dict, cfg: Dict) -> Dict:
    """生成 5 通道抓握预览。

    compact5 适合快速看五指开合趋势，不追求真实 SVH 关节结构。
    """

    grasp_close = clamp01(float(control_representation["grasp_close"]))
    finger_flex = control_representation["finger_flex"]
    thumb_flex = clamp01(float(finger_flex["thumb"]))
    index_flex = clamp01(float(finger_flex["index"]))
    middle_flex = clamp01(float(finger_flex["middle"]))
    ring_flex = clamp01(float(finger_flex["ring"]))
    little_flex = clamp01(float(finger_flex["little"]))
    channel_count = _channel_count(cfg)
    open_value = float(cfg.get("svh_position_open_value", 0.0))
    closed_value = float(cfg.get("svh_position_closed_value", 1.0))
    thumb_grasp_scale = float(cfg.get("svh_thumb_grasp_scale", 0.85))
    thumb_close = clamp01(0.75 * (grasp_close * thumb_grasp_scale) + 0.25 * thumb_flex)
    index_close = clamp01(index_flex + 0.25 * grasp_close)
    middle_close = clamp01(middle_flex + 0.25 * grasp_close)
    ring_close = clamp01(ring_flex + 0.20 * grasp_close)
    little_close = clamp01(little_flex + 0.15 * grasp_close)

    preview_values = [
        _lerp(open_value, closed_value, thumb_close),
        _lerp(open_value, closed_value, index_close),
        _lerp(open_value, closed_value, middle_close),
        _lerp(open_value, closed_value, ring_close),
        _lerp(open_value, closed_value, little_close),
    ]
    positions = _resize_positions(preview_values, channel_count, _lerp(open_value, closed_value, little_close))

    return SvhCommandPreview(
        enabled=True,
        mode=str(cfg.get("svh_preview_mode", "preview")),
        valid=True,
        command_source="control_representation",
        target_channels=_target_channels(channel_count),
        target_positions=positions,
        target_ticks_preview=_target_ticks_preview(positions, cfg),
        protocol_hint=_protocol_hint(cfg),
    ).to_dict()


def _build_svh9_grasp_preview(control_representation: Dict, cfg: Dict) -> Dict:
    """生成 9 通道抓握预览。

    svh_9ch 比 compact5 更接近 Unity / C# 参考实现：拇指 flexion 和 opposition
    分开，食指和中指拆成远端/近端，最后加 finger_spread。
    """

    grasp_close = clamp01(float(control_representation["grasp_close"]))
    finger_flex = control_representation["finger_flex"]
    thumb_flex = clamp01(float(finger_flex["thumb"]))
    index_flex = clamp01(float(finger_flex["index"]))
    middle_flex = clamp01(float(finger_flex["middle"]))
    ring_flex = clamp01(float(finger_flex["ring"]))
    little_flex = clamp01(float(finger_flex["little"]))
    channel_count = _channel_count(cfg)
    thumb_grasp_scale = float(cfg.get("svh_thumb_grasp_scale", 0.85))
    thumb_opposition_scale = float(cfg.get("svh_thumb_opposition_scale", 0.75))
    open_spread_scale = float(cfg.get("svh_open_spread_scale", 0.25))
    grasp_spread_scale = float(cfg.get("svh_grasp_spread_scale", 0.05))

    thumb_close = clamp01(0.75 * (grasp_close * thumb_grasp_scale) + 0.25 * thumb_flex)
    thumb_opp = clamp01(max(thumb_flex, grasp_close * thumb_opposition_scale))
    spread_alpha = clamp01((1.0 - grasp_close) * open_spread_scale + grasp_close * grasp_spread_scale)
    # alphas 的顺序必须和 SVH_9CH_NAMES 保持一致。
    alphas = [
        thumb_close,
        thumb_opp,
        max(_blend_finger_follow(index_flex, grasp_close, flex_weight=0.55, guide_weight=0.45), clamp01(grasp_close * 0.62)),
        max(_blend_finger_follow(index_flex, grasp_close, flex_weight=0.40, guide_weight=0.60), clamp01(grasp_close * 0.74)),
        max(_blend_finger_follow(middle_flex, grasp_close, flex_weight=0.72, guide_weight=0.28), clamp01(grasp_close * 0.66)),
        max(_blend_finger_follow(middle_flex, grasp_close, flex_weight=0.58, guide_weight=0.42), clamp01(grasp_close * 0.78)),
        max(_blend_finger_follow(ring_flex, grasp_close, flex_weight=0.80, guide_weight=0.20), clamp01(grasp_close * 0.74)),
        max(_blend_finger_follow(little_flex, grasp_close, flex_weight=0.84, guide_weight=0.16), clamp01(grasp_close * 0.72)),
        spread_alpha,
    ]
    grasp_fill_position = _svh9_positions_from_alphas([grasp_close], cfg)[0]
    positions = _resize_positions(_svh9_positions_from_alphas(alphas, cfg), channel_count, grasp_fill_position)

    return SvhCommandPreview(
        enabled=True,
        mode=str(cfg.get("svh_preview_mode", "preview")),
        valid=True,
        command_source="control_representation",
        target_channels=_target_channels(channel_count),
        target_positions=positions,
        target_ticks_preview=_target_ticks_preview(positions, cfg),
        protocol_hint=_protocol_hint(cfg),
    ).to_dict()


def _build_grasp_preview(control_representation: Dict, cfg: Dict) -> Dict:
    if _layout(cfg) == SVH_9CH_LAYOUT:
        return _build_svh9_grasp_preview(control_representation, cfg)
    return _build_compact_grasp_preview(control_representation, cfg)


def _build_compact_pinch_preview(control_representation: Dict, cfg: Dict) -> Dict:
    """生成 5 通道捏合预览。

    重点让拇指和食指跟随 pinch_close，其他手指只给较弱支撑参与。
    """

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
    middle_flex = clamp01(float(control_representation["finger_flex"]["middle"]))
    ring_flex = clamp01(float(control_representation["finger_flex"]["ring"]))
    little_flex = clamp01(float(control_representation["finger_flex"]["little"]))
    support_flex = clamp01(float(control_representation["support_flex"]))
    channel_count = _channel_count(cfg)
    open_value = float(cfg.get("svh_position_open_value", 0.0))
    closed_value = float(cfg.get("svh_position_closed_value", 1.0))
    pinch_support_scale = float(cfg.get("svh_pinch_support_scale", 0.20))

    thumb_value = _lerp(open_value, closed_value, clamp01(0.85 * pinch_close + 0.15 * thumb_flex))
    index_value = _lerp(open_value, closed_value, clamp01(0.75 * pinch_close + 0.25 * index_flex))
    support_floor = clamp01(max(support_flex, pinch_close * pinch_support_scale))
    middle_value = _lerp(
        open_value,
        closed_value,
        _blend_support_follow(middle_flex, support_floor, 0.0, flex_weight=0.90, support_weight=0.25, grasp_weight=0.0),
    )
    ring_value = _lerp(
        open_value,
        closed_value,
        _blend_support_follow(ring_flex, support_floor, 0.0, flex_weight=0.95, support_weight=0.20, grasp_weight=0.0),
    )
    little_value = _lerp(
        open_value,
        closed_value,
        _blend_support_follow(little_flex, support_floor, 0.0, flex_weight=1.00, support_weight=0.18, grasp_weight=0.0),
    )
    preview_values = [
        thumb_value,
        index_value,
        middle_value,
        ring_value,
        little_value,
    ]
    positions = _resize_positions(preview_values, channel_count, little_value)

    return SvhCommandPreview(
        enabled=True,
        mode=str(cfg.get("svh_preview_mode", "preview")),
        valid=True,
        command_source="control_representation",
        target_channels=_target_channels(channel_count),
        target_positions=positions,
        target_ticks_preview=_target_ticks_preview(positions, cfg),
        protocol_hint=_protocol_hint(cfg),
    ).to_dict()


def _build_svh9_pinch_preview(control_representation: Dict, cfg: Dict) -> Dict:
    """生成 9 通道捏合预览。

    与抓握不同，pinch 会突出 thumb/index 通道，同时让支撑手指保持低幅度跟随。
    """

    pinch_close = clamp01(
        float(
            control_representation.get(
                "effective_pinch_strength",
                control_representation.get("pinch_strength", 0.0),
            )
        )
    )
    grasp_close = clamp01(float(control_representation["grasp_close"]))
    finger_flex = control_representation["finger_flex"]
    thumb_flex = clamp01(float(finger_flex["thumb"]))
    index_flex = clamp01(float(finger_flex["index"]))
    middle_flex = clamp01(float(finger_flex["middle"]))
    ring_flex = clamp01(float(finger_flex["ring"]))
    little_flex = clamp01(float(finger_flex["little"]))
    support_flex = clamp01(float(control_representation["support_flex"]))
    channel_count = _channel_count(cfg)
    pinch_support_scale = float(cfg.get("svh_pinch_support_scale", 0.20))
    pinch_spread_scale = float(cfg.get("svh_pinch_spread_scale", 0.10))
    thumb_opposition_scale = float(cfg.get("svh_thumb_opposition_scale", 0.75))

    support_value = clamp01(max(support_flex, pinch_close * pinch_support_scale))
    # alphas 的顺序必须和 SVH_9CH_NAMES 保持一致。
    alphas = [
        clamp01(0.85 * pinch_close + 0.15 * thumb_flex),
        clamp01(0.75 * pinch_close + 0.15 * thumb_flex + 0.10 * thumb_opposition_scale * grasp_close),
        clamp01(0.80 * pinch_close + 0.20 * index_flex),
        clamp01(0.70 * pinch_close + 0.30 * index_flex),
        _blend_support_follow(middle_flex, support_value, grasp_close, flex_weight=0.90, support_weight=0.20, grasp_weight=0.10),
        _blend_support_follow(middle_flex, support_value, grasp_close, flex_weight=0.82, support_weight=0.20, grasp_weight=0.15),
        _blend_support_follow(ring_flex, support_value, grasp_close, flex_weight=0.95, support_weight=0.18, grasp_weight=0.10),
        _blend_support_follow(little_flex, support_value, grasp_close, flex_weight=1.00, support_weight=0.15, grasp_weight=0.10),
        clamp01(pinch_close * pinch_spread_scale),
    ]
    support_position = _svh9_positions_from_alphas([support_value], cfg)[0]
    positions = _resize_positions(_svh9_positions_from_alphas(alphas, cfg), channel_count, support_position)

    return SvhCommandPreview(
        enabled=True,
        mode=str(cfg.get("svh_preview_mode", "preview")),
        valid=True,
        command_source="control_representation",
        target_channels=_target_channels(channel_count),
        target_positions=positions,
        target_ticks_preview=_target_ticks_preview(positions, cfg),
        protocol_hint=_protocol_hint(cfg),
    ).to_dict()


def _build_pinch_preview(control_representation: Dict, cfg: Dict) -> Dict:
    if _layout(cfg) == SVH_9CH_LAYOUT:
        return _build_svh9_pinch_preview(control_representation, cfg)
    return _build_compact_pinch_preview(control_representation, cfg)


def build_svh_command_preview(payload: Dict, cfg: Dict) -> Dict:
    """从完整 payload 构造 SVH preview 对象。

    优先使用 control_representation。只有在显式启用 gesture_fallback 且连续特征
    不可用时，才会退回离散手势模板。真机阶段建议保持 fallback 关闭。
    """

    enabled = bool(cfg.get("svh_enable_preview", False))
    mode = str(cfg.get("svh_preview_mode", "preview" if enabled else "disabled"))
    if not enabled:
        return empty_svh_preview(cfg, enabled=False, mode="disabled")

    gesture = get_stable_gesture(payload)
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
        return empty_svh_preview(cfg, enabled=True, mode=mode)

    if enable_gesture_fallback and not features_valid and gesture in {"open", "fist", "pinch"}:
        return _gesture_fallback_preview(gesture, cfg)

    return empty_svh_preview(cfg, enabled=True, mode=mode)
