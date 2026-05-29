from __future__ import annotations

from collections import deque


class FPSMeter:
    def __init__(self, window: int = 30) -> None:
        self.timestamps_ms: deque[int] = deque(maxlen=window)

    def update(self, timestamp_ms: int) -> float:
        self.timestamps_ms.append(timestamp_ms)
        if len(self.timestamps_ms) < 2:
            return 0.0
        elapsed_s = (self.timestamps_ms[-1] - self.timestamps_ms[0]) / 1000.0
        if elapsed_s <= 0:
            return 0.0
        return (len(self.timestamps_ms) - 1) / elapsed_s
