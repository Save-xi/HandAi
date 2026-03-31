from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SvhCommandPreview:
    """Preview-only command object for future SVH transport integration.

    The channel layout is intentionally abstract at this stage: it is a compact
    preview vector for grasp/pinch style commands, not a claimed one-to-one map
    to the final Schunk SVH motor indices.
    """

    enabled: bool
    mode: str
    valid: bool
    command_source: str | None
    target_channels: List[int] = field(default_factory=list)
    target_positions: List[float] = field(default_factory=list)
    protocol_hint: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "valid": self.valid,
            "command_source": self.command_source,
            "target_channels": list(self.target_channels),
            "target_positions": list(self.target_positions),
            "protocol_hint": dict(self.protocol_hint),
        }
