from __future__ import annotations

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import DrowsinessLevel


class DrowsinessFSM:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.level = DrowsinessLevel.NONE

    def update(
        self,
        perclos_short: float,
        perclos_long: float,
        eye_closure_duration_ms: int,
        blink_rate_per_min: float = 0.0,
        face_present: bool = True,
    ) -> DrowsinessLevel:
        if not face_present:
            self.level = DrowsinessLevel.UNKNOWN
            return self.level
        if eye_closure_duration_ms >= self.config.microsleep_duration_ms:
            self.level = DrowsinessLevel.MICROSLEEP
        elif perclos_short >= self.config.drowsiness_perclos_high or perclos_long >= self.config.drowsiness_perclos_high:
            self.level = DrowsinessLevel.HIGH
        elif perclos_short >= self.config.drowsiness_perclos_medium or perclos_long >= self.config.drowsiness_perclos_medium:
            self.level = DrowsinessLevel.MEDIUM
        elif blink_rate_per_min > 30:
            self.level = DrowsinessLevel.LOW
        elif self.level in {DrowsinessLevel.HIGH, DrowsinessLevel.MEDIUM} and perclos_short > 0.15:
            self.level = DrowsinessLevel.LOW
        else:
            self.level = DrowsinessLevel.NONE
        return self.level
