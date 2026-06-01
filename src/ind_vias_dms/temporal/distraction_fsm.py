from __future__ import annotations

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import DistractionLevel, DistractionType, GazeZone


class DistractionFSM:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.level = DistractionLevel.NONE
        self.type = DistractionType.NONE

    def update(
        self,
        gaze_zone: GazeZone,
        eyes_off_road_duration_ms: int,
        no_face_duration_ms: int = 0,
        phone_state: str = "UNKNOWN",
    ) -> tuple[DistractionLevel, DistractionType]:
        if no_face_duration_ms >= self.config.no_face_timeout_ms:
            self.level = DistractionLevel.UNKNOWN
            self.type = DistractionType.UNKNOWN
            return self.level, self.type
        if phone_state == "TEXTING_SUSPECTED":
            self.level = DistractionLevel.HIGH
            self.type = DistractionType.PHONE_SUSPECTED
        elif phone_state == "PHONE_TO_EAR_SUSPECTED":
            self.level = (
                DistractionLevel.HIGH
                if eyes_off_road_duration_ms >= self.config.eyes_off_road_warning_ms
                else DistractionLevel.MEDIUM
            )
            self.type = DistractionType.PHONE_SUSPECTED
        elif phone_state == "PHONE_DOWN_SUSPECTED":
            self.level = DistractionLevel.MEDIUM
            self.type = DistractionType.PHONE_SUSPECTED
        elif phone_state == "HAND_NEAR_FACE":
            self.level = DistractionLevel.LOW
            self.type = DistractionType.PHONE_SUSPECTED
        elif (
            gaze_zone in {GazeZone.DOWN, GazeZone.PHONE_DOWN}
            and eyes_off_road_duration_ms >= self.config.eyes_off_road_warning_ms
        ):
            self.level = DistractionLevel.HIGH
            self.type = DistractionType.PHONE_SUSPECTED
        elif eyes_off_road_duration_ms >= self.config.eyes_off_road_warning_ms * 2:
            self.level = DistractionLevel.HIGH
            self.type = DistractionType.VISUAL
        elif eyes_off_road_duration_ms >= self.config.eyes_off_road_warning_ms:
            self.level = DistractionLevel.MEDIUM
            self.type = DistractionType.VISUAL
        elif gaze_zone != GazeZone.ROAD and gaze_zone != GazeZone.UNKNOWN:
            self.level = DistractionLevel.LOW
            self.type = DistractionType.VISUAL
        else:
            self.level = DistractionLevel.NONE
            self.type = DistractionType.NONE
        return self.level, self.type
