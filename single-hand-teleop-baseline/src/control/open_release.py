"""一个有界滞回的连续释放候选；每个真实源帧只更新一次。"""
from __future__ import annotations

from dataclasses import dataclass

from features.geometry_utils import clamp01


@dataclass
class OpenReleaseState:
    # 全部历史状态只有 weight；deepcopy 可供以后独立查询未来映射。
    weight: float = 0.0

    def reset(self) -> None:
        self.weight = 0.0

    def update(self, payload: dict, cfg: dict) -> float:
        if not cfg.get("control_open_release_enabled", False):
            self.reset()
            return self.weight
        start = float(cfg.get("control_open_release_start_ratio", 0.85))
        full = float(cfg.get("control_open_release_full_ratio", 0.95))
        ratio = clamp01((float(payload["hand_open_ratio"]) - start) / (full - start))
        mean_curl = sum(float(payload["finger_curl"][f]) for f in ("index", "middle", "ring", "little")) / 4
        half_width = float(cfg.get("control_release_curl_half_width", 0.02))
        curl = clamp01((float(cfg.get("open_mean_curl_max", 0.45)) + half_width - mean_curl) / (2 * half_width))
        pinch = clamp01(
            (float(payload["pinch_distance_norm"]) - float(cfg.get("pinch_distance_norm_threshold", 0.45)))
            / float(cfg.get("control_release_pinch_band", 0.10))
        )
        target = ratio * curl * pinch
        band = float(cfg.get("control_release_hysteresis", 0.01))
        # 幅度死区不添加时间平均；端点直接到达，确保确实张开/闭合时能释放/恢复。
        self.weight = target if target in (0.0, 1.0) else max(target - band, min(self.weight, target + band))
        return self.weight
