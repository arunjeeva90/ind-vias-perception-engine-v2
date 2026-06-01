from __future__ import annotations

from dataclasses import dataclass

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import GazeZone
from ind_vias_dms.vision.head_pose import HeadPose


@dataclass
class GazeEstimate:
    zone: GazeZone = GazeZone.UNKNOWN
    confidence: float = 0.0


class GazeEstimator:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.down_since_ms: int | None = None
        self.yaw_offset_deg = config.road_center_yaw_offset_deg
        self.pitch_offset_deg = config.road_center_pitch_offset_deg

    def calibrate_road_center(self, yaw_deg: float, pitch_deg: float) -> None:
        self.yaw_offset_deg = yaw_deg
        self.pitch_offset_deg = pitch_deg

    def reset_road_center(self) -> None:
        self.yaw_offset_deg = self.config.road_center_yaw_offset_deg
        self.pitch_offset_deg = self.config.road_center_pitch_offset_deg

    def estimate(
        self,
        head_pose: HeadPose,
        timestamp_ms: int,
        face_present: bool = True,
    ) -> GazeEstimate:
        if not face_present or head_pose.confidence < self.config.head_pose_min_confidence:
            self.down_since_ms = None
            return GazeEstimate()
        if abs(head_pose.pitch_deg) > self.config.head_pose_outlier_threshold_deg:
            self.down_since_ms = None
            return GazeEstimate()
        relative_yaw = head_pose.yaw_deg
        relative_pitch = head_pose.pitch_deg
        if self.config.road_gaze_calibration_enabled:
            relative_yaw -= self.yaw_offset_deg
            relative_pitch -= self.pitch_offset_deg
        yaw_tolerance = self.config.road_yaw_tolerance_deg
        pitch_tolerance = self.config.road_pitch_tolerance_deg
        if relative_yaw < -yaw_tolerance:
            self.down_since_ms = None
            return GazeEstimate(GazeZone.LEFT, 0.75)
        if relative_yaw > yaw_tolerance:
            self.down_since_ms = None
            return GazeEstimate(GazeZone.RIGHT, 0.75)
        if relative_pitch > pitch_tolerance:
            if self.down_since_ms is None:
                self.down_since_ms = timestamp_ms
            if timestamp_ms - self.down_since_ms >= self.config.phone_down_sustain_ms:
                return GazeEstimate(GazeZone.PHONE_DOWN, 0.7)
            return GazeEstimate(GazeZone.DOWN, 0.7)
        self.down_since_ms = None
        if relative_pitch < -pitch_tolerance:
            return GazeEstimate(GazeZone.UP, 0.6)
        return GazeEstimate(GazeZone.ROAD, 0.8)
