from __future__ import annotations

import time


class FrameTimer:
    def __init__(self) -> None:
        self.prev_time = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self.prev_time
        self.prev_time = now
        return dt


def now_ts() -> float:
    return time.time()
