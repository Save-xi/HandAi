from __future__ import annotations

"""逐指 One Euro 滤波；各输入流独占状态，不使用固定窗口平均。"""

from dataclasses import dataclass, field
import math


def _alpha(cutoff: float, interval: float) -> float:
    product = 2.0 * math.pi * cutoff * interval
    return product / (1.0 + product)


@dataclass
class AdaptiveFlexState:
    """速度越高，截止频率越高；首帧/中断后直接采用当前值。

    这里只处理控制用 flex，不改写原始 curl 或手势。查询计算当前时刻的
    假想输出但不写回状态；fork/reset 由 HandPipeline 与其他映射状态一同管理。
    """

    timestamp: float | None = None
    raw: dict[str, float] = field(default_factory=dict)
    filtered: dict[str, float] = field(default_factory=dict)
    derivative: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.timestamp = None
        self.raw.clear()
        self.filtered.clear()
        self.derivative.clear()

    def update(self, values: dict[str, float], timestamp: float, cfg: dict, *, advance: bool = True) -> dict[str, float]:
        if not advance and self.timestamp is not None and timestamp == self.timestamp:
            # 同时刻查询没有经过新的时间，不能绕过滤波直接返回原始值。
            return dict(self.filtered)
        if self.timestamp is None or timestamp <= self.timestamp:
            result = dict(values)
            slopes = dict.fromkeys(values, 0.0)
        else:
            interval = timestamp - self.timestamp
            slope_alpha = _alpha(float(cfg["control_flex_filter_derivative_cutoff_hz"]), interval)
            slopes, result = {}, {}
            for name, value in values.items():
                speed = (value - self.raw[name]) / interval
                slopes[name] = self.derivative[name] + slope_alpha * (speed - self.derivative[name])
                cutoff = float(cfg["control_flex_filter_min_cutoff_hz"]) + float(cfg["control_flex_filter_beta"]) * abs(slopes[name])
                alpha = _alpha(cutoff, interval)
                result[name] = self.filtered[name] + alpha * (value - self.filtered[name])
        if advance:
            self.timestamp = timestamp
            self.raw = dict(values)
            self.filtered = dict(result)
            self.derivative = slopes
        return result
