from __future__ import annotations

import logging
from typing import Dict, List

from svh.svh_transport_base import SvhTransportBase


class MockSvhTransport(SvhTransportBase):
    """Preview-only transport.

    This keeps the adapter layer decoupled from real I/O. A future
    `svh_transport_tcp.py` can implement the same interface once the TCP/RS485
    bridge protocol is validated against hardware.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger
        self.last_command: Dict | None = None
        self.sent_commands: List[Dict] = []

    def send(self, command: Dict) -> Dict:
        self.last_command = dict(command)
        self.sent_commands.append(dict(command))
        if self.logger is not None:
            self.logger.debug("Mock SVH transport recorded command preview.")
        return {
            "transport": "mock",
            "accepted": True,
            "valid": bool(command.get("valid", False)),
            "recorded_count": len(self.sent_commands),
        }
