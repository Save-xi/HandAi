from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SvhCommandPreview:
    """Preview-only command object for future SVH transport integration.

    The preview layer may emit either a compact 5-channel abstraction or a
    paper/C#-aligned 9-channel ordering. Both remain preview-oriented until
    real transport, packing, and hardware calibration are implemented.
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
