from __future__ import annotations

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import DrowsinessLevel


class DrowsinessFSM:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.level = DrowsinessLevel.NONE
        self.medium_since_ms: int | None = None
        self.high_since_ms: int | None = None
        self.release_since_ms: int | None = None

    def update(
        self,
        perclos_short: float,
        perclos_long: float,
        eye_closure_duration_ms: int,
        blink_rate_per_min: float = 0.0,
        face_present: bool = True,
        timestamp_ms: int | None = None,
    ) -> DrowsinessLevel:
        if not face_present:
            self.level = DrowsinessLevel.UNKNOWN
            self.medium_since_ms = None
            self.high_since_ms = None
            self.release_since_ms = None
            return self.level
        microsleep_ms = max(self.config.microsleep_duration_ms, self.config.microsleep_closure_ms)
        if eye_closure_duration_ms >= microsleep_ms:
            self.level = DrowsinessLevel.MICROSLEEP
            self.release_since_ms = None
            return self.level

        candidate = DrowsinessLevel.NONE
        if (
            perclos_short >= self.config.perclos_5s_high_threshold
            or perclos_long >= self.config.perclos_60s_high_threshold
        ):
            candidate = DrowsinessLevel.HIGH
        elif (
            perclos_short >= self.config.perclos_5s_medium_threshold
            or perclos_long >= self.config.perclos_60s_medium_threshold
        ):
            candidate = DrowsinessLevel.MEDIUM
        elif blink_rate_per_min > 30:
            candidate = DrowsinessLevel.LOW

        if (
            candidate in {DrowsinessLevel.MEDIUM, DrowsinessLevel.HIGH}
            and 0 < eye_closure_duration_ms < self.config.blink_max_duration_ms
        ):
            candidate = DrowsinessLevel.LOW

        if timestamp_ms is None:
            self.level = candidate
        elif candidate == DrowsinessLevel.HIGH:
            if self.high_since_ms is None:
                self.high_since_ms = timestamp_ms
            self.medium_since_ms = self.medium_since_ms or timestamp_ms
            self.release_since_ms = None
            high_sustain_ms = min(
                self.config.drowsiness_high_sustain_ms,
                self.config.drowsiness_high_sustain_override_ms,
            )
            if timestamp_ms - self.high_since_ms >= high_sustain_ms:
                self.level = DrowsinessLevel.HIGH
            elif timestamp_ms - self.medium_since_ms >= self.config.drowsiness_medium_sustain_ms:
                self.level = DrowsinessLevel.MEDIUM
            else:
                self.level = DrowsinessLevel.LOW
        elif candidate == DrowsinessLevel.MEDIUM:
            self.high_since_ms = None
            if self.medium_since_ms is None:
                self.medium_since_ms = timestamp_ms
            self.release_since_ms = None
            if timestamp_ms - self.medium_since_ms >= self.config.drowsiness_medium_sustain_ms:
                self.level = DrowsinessLevel.MEDIUM
            else:
                self.level = DrowsinessLevel.LOW
        elif candidate == DrowsinessLevel.LOW:
            self.high_since_ms = None
            self.medium_since_ms = None
            self.release_since_ms = None
            self.level = DrowsinessLevel.LOW
        elif self.level in {DrowsinessLevel.HIGH, DrowsinessLevel.MEDIUM} and perclos_short > 0.15:
            self.high_since_ms = None
            self.medium_since_ms = None
            if self.release_since_ms is None:
                self.release_since_ms = timestamp_ms
            if timestamp_ms - self.release_since_ms >= self.config.drowsiness_warning_release_ms:
                self.level = DrowsinessLevel.LOW
        else:
            self.high_since_ms = None
            self.medium_since_ms = None
            self.release_since_ms = None
            self.level = DrowsinessLevel.NONE
        return self.level
