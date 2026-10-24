"""共享媒体 PTS 选择与源节奏估计；源时间与处理时钟分开，回退来源必须可见。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from statistics import median


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


class SourceCadence:
    """按输入流自身的近期周期给出断流阈值；只读源时间，不看检测结果或手势。

    阈值 = min(上限, max(下限, 倍数 × 最近 window 个正间隔的中位数))，尚无间隔时取下限。
    每个到达的源帧（有效或失效）都记录一次间隔，当前帧的判定只用它之前的间隔；
    触发重置的长间隔也计入窗口，否则低于下限帧率的流会每帧重置、永远无法建立周期。
    时间单位为秒，与 HandPipeline 的 timestamp 一致；下限主导时与固定阈值逐帧相同。
    """

    def __init__(self, *, floor_ms: float, factor: float, window: int, ceiling_ms: float):
        self.floor_s = float(floor_ms) / 1000.0
        self.ceiling_s = float(ceiling_ms) / 1000.0
        self.factor = float(factor)
        self.intervals: deque[float] = deque(maxlen=int(window))
        self.last_time_s: float | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> SourceCadence:
        return cls(floor_ms=float(cfg.get("mapping_max_frame_gap_ms", 100.0)),
                   factor=float(cfg.get("mapping_frame_gap_period_factor", 2.5)),
                   window=int(cfg.get("mapping_frame_gap_period_window", 9)),
                   ceiling_ms=float(cfg.get("mapping_frame_gap_ceiling_ms", 1000.0)))

    def gap_limit_s(self) -> float:
        """相邻有效帧允许的最大间隔；超过即视为真实断流。"""
        if not self.intervals:
            return self.floor_s
        return min(self.ceiling_s, max(self.floor_s, self.factor * median(self.intervals)))

    def observe(self, time_s: float) -> None:
        """记录一个源帧；倒序或重复时刻不计入周期，只更新参考点。"""
        if self.last_time_s is not None and time_s > self.last_time_s:
            self.intervals.append(time_s - self.last_time_s)
        self.last_time_s = time_s
