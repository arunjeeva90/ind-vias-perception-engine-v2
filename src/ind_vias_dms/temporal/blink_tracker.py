from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class BlinkStats:
    eye_closure_duration_ms: int = 0
    blink_rate_per_min: float = 0.0
    blink_count: int = 0


class BlinkTracker:
    def __init__(self, blink_min_duration_ms: int) -> None:
        self.blink_min_duration_ms = blink_min_duration_ms
        self.closed_since_ms: int | None = None
        self.blink_timestamps_ms: deque[int] = deque()

    def update(self, timestamp_ms: int, eyes_closed: bool) -> BlinkStats:
        if eyes_closed and self.closed_since_ms is None:
            self.closed_since_ms = timestamp_ms
        if not eyes_closed and self.closed_since_ms is not None:
            duration = timestamp_ms - self.closed_since_ms
            if duration >= self.blink_min_duration_ms:
                self.blink_timestamps_ms.append(timestamp_ms)
            self.closed_since_ms = None
        while self.blink_timestamps_ms and timestamp_ms - self.blink_timestamps_ms[0] > 60000:
            self.blink_timestamps_ms.popleft()
        closure = timestamp_ms - self.closed_since_ms if self.closed_since_ms is not None else 0
        return BlinkStats(
            eye_closure_duration_ms=max(0, int(closure)),
            blink_rate_per_min=float(len(self.blink_timestamps_ms)),
            blink_count=len(self.blink_timestamps_ms),
        )
