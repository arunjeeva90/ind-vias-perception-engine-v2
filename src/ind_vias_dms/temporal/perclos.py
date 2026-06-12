from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class PERCLOSResult:
    perclos: float = 0.0
    valid_time_ms: int = 0


class PERCLOSTracker:
    def __init__(self, window_s: float) -> None:
        self.window_ms = int(window_s * 1000)
        self.samples: deque[tuple[int, int, float, bool]] = deque()
        self.previous_timestamp_ms: int | None = None
        self.previous_weight: float = 0.0
        self.previous_valid: bool = False

    def update(self, timestamp_ms: int, eyes_closed: bool) -> float:
        weight = 1.0 if eyes_closed else 0.0
        return self.update_weighted(timestamp_ms, weight, True).perclos

    def update_weighted(
        self,
        timestamp_ms: int,
        closure_weight: float,
        valid: bool,
    ) -> PERCLOSResult:
        if self.previous_timestamp_ms is not None:
            dt_ms = max(0, timestamp_ms - self.previous_timestamp_ms)
            if dt_ms > 0:
                self.samples.append(
                    (
                        self.previous_timestamp_ms,
                        timestamp_ms,
                        self.previous_weight,
                        self.previous_valid,
                    )
                )
        self.previous_timestamp_ms = timestamp_ms
        self.previous_weight = max(0.0, min(1.0, closure_weight))
        self.previous_valid = valid
        return self.current(timestamp_ms)

    def pause(self, timestamp_ms: int) -> PERCLOSResult:
        return self.update_weighted(timestamp_ms, 0.0, False)

    def current(self, timestamp_ms: int | None = None) -> PERCLOSResult:
        if timestamp_ms is not None:
            self._trim(timestamp_ms)
        closed_ms = 0.0
        valid_ms = 0
        for start_ms, end_ms, weight, valid in self.samples:
            if not valid:
                continue
            dt_ms = max(0, end_ms - start_ms)
            valid_ms += dt_ms
            closed_ms += dt_ms * weight
        if valid_ms <= 0:
            return PERCLOSResult(0.0, 0)
        return PERCLOSResult(min(1.0, max(0.0, closed_ms / valid_ms)), valid_ms)

    def reset(self) -> None:
        self.samples.clear()
        self.previous_timestamp_ms = None
        self.previous_weight = 0.0
        self.previous_valid = False

    def _trim(self, timestamp_ms: int) -> None:
        cutoff = timestamp_ms - self.window_ms
        while self.samples and self.samples[0][1] <= cutoff:
            self.samples.popleft()
        if self.samples and self.samples[0][0] < cutoff:
            start_ms, end_ms, weight, valid = self.samples[0]
            self.samples[0] = (cutoff, end_ms, weight, valid)
