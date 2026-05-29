from __future__ import annotations

from collections import deque


class PERCLOSTracker:
    def __init__(self, window_s: float) -> None:
        self.window_ms = int(window_s * 1000)
        self.samples: deque[tuple[int, bool]] = deque()

    def update(self, timestamp_ms: int, eyes_closed: bool) -> float:
        self.samples.append((timestamp_ms, eyes_closed))
        while self.samples and timestamp_ms - self.samples[0][0] > self.window_ms:
            self.samples.popleft()
        if len(self.samples) < 2:
            return 1.0 if eyes_closed else 0.0
        closed_ms = 0
        total_ms = max(1, self.samples[-1][0] - self.samples[0][0])
        previous_t, previous_closed = self.samples[0]
        for current_t, current_closed in list(self.samples)[1:]:
            if previous_closed:
                closed_ms += current_t - previous_t
            previous_t, previous_closed = current_t, current_closed
        return min(1.0, max(0.0, closed_ms / total_ms))
