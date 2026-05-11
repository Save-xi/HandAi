from __future__ import annotations

"""SVH preview 命令的数据结构。

这个模块只定义“命令对象长什么样”。具体如何从控制特征生成命令，
放在 svh_adapter.py；真实协议打包也不在这里完成。
"""

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
    """本次运行是否启用了 SVH preview 扩展。"""
    mode: str
    """preview 模式名，常见值是 preview / disabled。"""
    valid: bool
    """当前帧是否有可消费的目标数组。"""
    command_source: str | None
    """命令来源：control_representation 或 gesture_fallback；无效时为 None。"""
    target_channels: List[int] = field(default_factory=list)
    """目标通道索引，必须与 target_positions 一一对应。"""
    target_positions: List[float] = field(default_factory=list)
    """归一化 preview 位置，范围会被夹到 [0, 1]。"""
    target_ticks_preview: List[int] = field(default_factory=list)
    """仅供 preview / 调试的 encoder-like ticks，不是真机安全命令。"""
    protocol_hint: Dict[str, str] = field(default_factory=dict)
    """布局、单位和协议假设说明，帮助下游解释 target 数组。"""

    def to_dict(self) -> Dict:
        """导出成满足 frame payload contract 的字典。

        无效 preview 会主动清空目标数组，避免下游误用旧数据。
        """

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
