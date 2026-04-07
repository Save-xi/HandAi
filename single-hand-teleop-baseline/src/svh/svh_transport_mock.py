from __future__ import annotations

import logging
from typing import Dict, List

from svh.svh_transport_base import SvhTransportBase


class MockSvhTransport(SvhTransportBase):
    """仅用于 preview 的传输层。

    它让适配器层保持与真实 I/O 解耦。等到 TCP / RS485 桥接协议
    在真实硬件上验证完成后，未来可以再由 `svh_transport_tcp.py`
    实现同一套接口。
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger
        self.last_command: Dict | None = None
        self.sent_commands: List[Dict] = []

    def send(self, command: Dict) -> Dict:
        self.last_command = dict(command)
        self.sent_commands.append(dict(command))
        if self.logger is not None:
            self.logger.debug("Mock SVH transport 已记录一条命令 preview。")
        return {
            "transport": "mock",
            "accepted": True,
            "valid": bool(command.get("valid", False)),
            "recorded_count": len(self.sent_commands),
        }
