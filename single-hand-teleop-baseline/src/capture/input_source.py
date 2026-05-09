from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class InputSource(ABC):
    """为未来 webcam / video 扩展预留的抽象输入源接口。"""

    @abstractmethod
    def is_opened(self) -> bool:
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        pass

    @abstractmethod
    def release(self) -> None:
        pass
