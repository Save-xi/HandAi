from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict

from svh.svh_transport_base import SvhTransportBase


class MockSvhTransport(SvhTransportBase):
    """仅用于 preview 的传输层。

    它让适配器层保持与真实 I/O 解耦。等到 TCP / RS485 桥接协议
    在真实硬件上验证完成后，未来可以再由 `svh_transport_tcp.py`
    实现同一套接口。
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        history_size: int = 32,
    ) -> None:
        self.logger = logger
        self.last_command: Dict | None = None
        self.history_size = max(1, int(history_size))
        # mock 只为调试保留一个有界尾窗；总计数独立累计，避免长时间摄像头
        # 运行时把每一帧完整 payload 永久留在内存中。
        self.sent_commands: Deque[Dict] = deque(maxlen=self.history_size)
        self._recorded_count = 0

    def send(self, command: Dict) -> Dict:
        self.last_command = dict(command)
        self.sent_commands.append(dict(command))
        self._recorded_count += 1
        if self.logger is not None:
            self.logger.debug("Mock SVH transport 已记录一条命令 preview。")
        return {
            "transport": "mock",
            "accepted": True,
            "valid": bool(command.get("valid", False)),
            "recorded_count": self._recorded_count,
            "retained_history_count": len(self.sent_commands),
        }
