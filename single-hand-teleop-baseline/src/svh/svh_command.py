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
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "valid": self.valid,
            "command_source": self.command_source,
            "target_channels": list(self.target_channels),
            "target_positions": list(self.target_positions),
            "target_ticks_preview": list(self.target_ticks_preview),
            "protocol_hint": dict(self.protocol_hint),
        }
