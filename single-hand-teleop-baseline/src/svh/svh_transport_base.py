from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict


class SvhTransportBase(ABC):
    """Base transport for future mock/TCP SVH senders."""

    @abstractmethod
    def send(self, command: Dict) -> Dict:
        """Send or record an SVH command preview."""
