from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SvhCommandPreview:
    """面向未来 SVH 传输集成的 preview-only 命令对象。

    preview 层既可能输出紧凑的 5 通道抽象，也可能输出与论文 / C#
    参考实现更接近的 9 通道顺序。在真实传输、打包和硬件标定完成前，
    这两种形式都只应被视为 preview。
    """

    enabled: bool
    mode: str
    valid: bool
    command_source: str | None
    target_channels: List[int] = field(default_factory=list)
    target_positions: List[float] = field(default_factory=list)
    target_ticks_preview: List[int] = field(default_factory=list)
    protocol_hint: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        positions = [min(1.0, max(0.0, float(value))) for value in self.target_positions]
        channels = [int(value) for value in self.target_channels]
        ticks = [int(value) for value in self.target_ticks_preview]
        valid = bool(self.valid)
        enabled = bool(self.enabled)
        command_source = self.command_source

        if not enabled:
            valid = False
        if not valid:
            command_source = None
            channels = []
            positions = []
            ticks = []

        return {
            "enabled": enabled,
            "mode": self.mode,
            "valid": valid,
            "command_source": command_source,
            "target_channels": channels,
            "target_positions": positions,
            "target_ticks_preview": ticks,
            "protocol_hint": dict(self.protocol_hint),
        }
