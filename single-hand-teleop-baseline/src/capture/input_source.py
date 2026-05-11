from __future__ import annotations

"""输入源抽象。

主循环只关心 InputSource.read() 能不能给出下一帧，不关心帧来自摄像头、
视频文件，还是未来可能新增的网络流/数据集回放。
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class InputSource(ABC):
    """为 webcam / video / 未来输入扩展预留的统一接口。"""

    @abstractmethod
    def is_opened(self) -> bool:
        """输入源是否已经成功打开。"""

        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读取下一帧。

        返回 (ok, frame)。ok=false 或 frame=None 表示输入源耗尽或读取失败。
        """

        pass

    @abstractmethod
    def release(self) -> None:
        """释放底层资源。"""

        pass
