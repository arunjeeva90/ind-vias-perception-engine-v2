from __future__ import annotations

from dataclasses import dataclass

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.vision.face_landmarks import FaceLandmarkResult


@dataclass
class DriverBodyState:
    state: str = "UNKNOWN"
    roi_norm: tuple[float, float, float, float] | None = None


class DriverBodyPresenceFallback:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.last_body_roi_norm: tuple[float, float, float, float] | None = None
        self.last_seen_timestamp_ms: int | None = None

    def update(
        self,
        face: FaceLandmarkResult,
        timestamp_ms: int,
        driver_session_held: bool,
    ) -> DriverBodyState:
        if not self.config.driver_body_fallback_enabled:
            return DriverBodyState("UNKNOWN")
        if face.face_found and face.box_norm is not None:
            self.last_body_roi_norm = self._expanded_roi(face.box_norm)
            self.last_seen_timestamp_ms = timestamp_ms
            return DriverBodyState("PRESENT", self.last_body_roi_norm)
        if (
            driver_session_held
            and self.last_seen_timestamp_ms is not None
            and timestamp_ms - self.last_seen_timestamp_ms <= self.config.driver_body_presence_hold_ms
        ):
            return DriverBodyState("PRESENT", self.last_body_roi_norm)
        if self.last_body_roi_norm is not None:
            return DriverBodyState("LOST", self.last_body_roi_norm)
        return DriverBodyState("UNKNOWN")

    def reset(self) -> None:
        self.last_body_roi_norm = None
        self.last_seen_timestamp_ms = None

    def _expanded_roi(
        self,
        box: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = (x2 - x1) * self.config.driver_body_roi_expand_x
        h = (y2 - y1) * self.config.driver_body_roi_expand_y
        return (
            max(0.0, cx - w / 2.0),
            max(0.0, cy - h * 0.35),
            min(1.0, cx + w / 2.0),
            min(1.0, cy + h * 0.65),
        )
