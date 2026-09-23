"""共享媒体 PTS 选择；源时间与处理时钟分开，回退来源必须可见。"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TimestampDecision:
    timestamp_ms: float
    source: str


def resolve_media_timestamp_ms(frame_index: int, *, raw_pts_ms: float | None, nominal_fps: float,
                               previous_timestamp_ms: float | None) -> TimestampDecision:
    if frame_index < 0:
        raise ValueError("frame_index 必须是非负整数")
    if not math.isfinite(float(nominal_fps)) or float(nominal_fps) <= 0:
        raise ValueError("nominal_fps 必须是有限正数")
    period_ms = 1000.0 / float(nominal_fps)
    raw_valid = raw_pts_ms is not None and math.isfinite(float(raw_pts_ms)) and float(raw_pts_ms) >= 0
    if raw_valid and (previous_timestamp_ms is None or float(raw_pts_ms) > previous_timestamp_ms + 1e-6):
        return TimestampDecision(float(raw_pts_ms), "container_pts_ms")
    fallback = frame_index * period_ms
    if previous_timestamp_ms is not None and fallback <= previous_timestamp_ms + 1e-6:
        return TimestampDecision(previous_timestamp_ms + period_ms, "continuity_fallback_period")
    return TimestampDecision(fallback, "frame_index_over_nominal_fps")
