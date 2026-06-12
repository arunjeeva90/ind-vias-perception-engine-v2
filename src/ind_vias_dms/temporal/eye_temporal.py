from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ind_vias_dms.core.config import DMSConfig


@dataclass
class EyeTemporalState:
    eye_state: str = "UNKNOWN"
    normalized_openness: float = 0.0
    calibration_state: str = "FALLBACK"
    closure_weight: float = 0.0
    valid_for_perclos: bool = False
    eye_closure_duration_ms: int = 0
    blink_rate_per_min: float = 0.0
    blink_count: int = 0


class EyeTemporalTracker:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.baseline_samples: deque[tuple[int, float]] = deque()
        self.open_eye_baseline: float | None = None
        self.previous_timestamp_ms: int | None = None
        self.current_closure_duration_ms = 0
        self.blink_timestamps_ms: deque[int] = deque()
        self.last_state = "UNKNOWN"

    def update(
        self,
        timestamp_ms: int,
        raw_openness: float,
        confidence: float,
        driver_face_present: bool,
        pause: bool = False,
        abs_yaw_deg: float = 0.0,
        abs_pitch_deg: float = 0.0,
    ) -> EyeTemporalState:
        dt_ms = 0 if self.previous_timestamp_ms is None else max(0, timestamp_ms - self.previous_timestamp_ms)
        self.previous_timestamp_ms = timestamp_ms
        if pause or not driver_face_present or confidence < self.config.eye_visibility_min_confidence:
            return self._state("UNKNOWN", raw_openness, confidence, valid=False)

        baseline_stable = self._baseline_update_allowed(
            confidence,
            abs_yaw_deg,
            abs_pitch_deg,
            raw_openness,
        )
        if baseline_stable:
            self._update_baseline(timestamp_ms, raw_openness)
        normalized = self._normalized(raw_openness)
        if self.open_eye_baseline is None:
            state = "CLOSED" if raw_openness < self.config.eye_closed_threshold else "OPEN"
            calibration = "WARMING_UP" if self.baseline_samples else "FALLBACK"
        elif normalized < self.config.normalized_eye_closed_threshold:
            state = "CLOSED"
            calibration = "CALIBRATED"
        elif normalized < self.config.normalized_eye_partial_threshold:
            state = "PARTIALLY_CLOSED"
            calibration = "CALIBRATED"
        else:
            state = "OPEN"
            calibration = "CALIBRATED"

        if state == "CLOSED" or (
            state == "PARTIALLY_CLOSED" and self.config.perclos_count_partial_closure
        ):
            self.current_closure_duration_ms += dt_ms
        elif state == "OPEN":
            if self.current_closure_duration_ms >= self.config.blink_min_duration_ms:
                self.blink_timestamps_ms.append(timestamp_ms)
            self.current_closure_duration_ms = 0
        while self.blink_timestamps_ms and timestamp_ms - self.blink_timestamps_ms[0] > 60000:
            self.blink_timestamps_ms.popleft()
        self.last_state = state
        closure_weight = self._closure_weight(state)
        return EyeTemporalState(
            eye_state=state,
            normalized_openness=normalized,
            calibration_state=calibration,
            closure_weight=closure_weight,
            valid_for_perclos=True,
            eye_closure_duration_ms=self.current_closure_duration_ms,
            blink_rate_per_min=float(len(self.blink_timestamps_ms)),
            blink_count=len(self.blink_timestamps_ms),
        )

    def _baseline_update_allowed(
        self,
        confidence: float,
        abs_yaw_deg: float,
        abs_pitch_deg: float,
        raw_openness: float,
    ) -> bool:
        if confidence < self.config.eye_baseline_update_min_visibility:
            return False
        if abs_yaw_deg > self.config.eye_baseline_update_max_abs_yaw_deg:
            return False
        if abs_pitch_deg > self.config.eye_baseline_update_max_abs_pitch_deg:
            return False
        if (
            self.config.eye_baseline_update_only_when_open
            and self.open_eye_baseline is not None
            and raw_openness / max(1e-6, self.open_eye_baseline)
            < self.config.normalized_eye_partial_threshold
        ):
            return False
        return True

    def pause(self, timestamp_ms: int) -> EyeTemporalState:
        self.previous_timestamp_ms = timestamp_ms
        return EyeTemporalState(
            eye_state="UNKNOWN",
            normalized_openness=0.0,
            calibration_state=self.calibration_state,
            valid_for_perclos=False,
            eye_closure_duration_ms=self.current_closure_duration_ms,
            blink_rate_per_min=float(len(self.blink_timestamps_ms)),
            blink_count=len(self.blink_timestamps_ms),
        )

    def reset(self) -> None:
        self.baseline_samples.clear()
        self.open_eye_baseline = None
        self.previous_timestamp_ms = None
        self.current_closure_duration_ms = 0
        self.blink_timestamps_ms.clear()
        self.last_state = "UNKNOWN"

    @property
    def calibration_state(self) -> str:
        if self.open_eye_baseline is not None:
            return "CALIBRATED"
        return "WARMING_UP" if self.baseline_samples else "FALLBACK"

    def _update_baseline(self, timestamp_ms: int, raw_openness: float) -> None:
        if raw_openness <= 0:
            return
        if self.open_eye_baseline is None or raw_openness >= self.open_eye_baseline * 0.8:
            self.baseline_samples.append((timestamp_ms, raw_openness))
        cutoff = timestamp_ms - int(self.config.eye_open_baseline_window_s * 1000)
        while self.baseline_samples and self.baseline_samples[0][0] < cutoff:
            self.baseline_samples.popleft()
        if self.baseline_samples:
            values = sorted(value for _, value in self.baseline_samples)
            top_count = max(1, len(values) // 3)
            top_values = values[-top_count:]
            self.open_eye_baseline = sum(top_values) / len(top_values)

    def _normalized(self, raw_openness: float) -> float:
        if self.open_eye_baseline is None or self.open_eye_baseline <= 1e-6:
            return 0.0
        return raw_openness / self.open_eye_baseline

    def _closure_weight(self, state: str) -> float:
        if state == "CLOSED":
            return 1.0
        if state == "PARTIALLY_CLOSED" and self.config.perclos_count_partial_closure:
            return self.config.perclos_partial_closure_weight
        return 0.0

    def _state(self, state: str, raw_openness: float, confidence: float, valid: bool) -> EyeTemporalState:
        return EyeTemporalState(
            eye_state=state,
            normalized_openness=self._normalized(raw_openness),
            calibration_state=self.calibration_state,
            valid_for_perclos=valid,
            eye_closure_duration_ms=self.current_closure_duration_ms,
            blink_rate_per_min=float(len(self.blink_timestamps_ms)),
            blink_count=len(self.blink_timestamps_ms),
        )
