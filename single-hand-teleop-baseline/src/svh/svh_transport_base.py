from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict


class SvhTransportBase(ABC):
    """为未来 mock / TCP 等 SVH 发送器预留的基础传输接口。"""

    @abstractmethod
    def send(self, command: Dict) -> Dict:
        """发送或记录一条 SVH 命令 preview。"""
