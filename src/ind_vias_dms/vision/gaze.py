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

    def estimate(self, head_pose: HeadPose) -> GazeEstimate:
        if head_pose.confidence <= 0:
            return GazeEstimate()
        if head_pose.pitch_deg > self.config.head_pitch_down_threshold_deg + 10:
            return GazeEstimate(GazeZone.PHONE_DOWN, 0.65)
        if head_pose.pitch_deg > self.config.head_pitch_down_threshold_deg:
            return GazeEstimate(GazeZone.DOWN, 0.7)
        if head_pose.pitch_deg < self.config.head_pitch_up_threshold_deg:
            return GazeEstimate(GazeZone.UP, 0.6)
        if head_pose.yaw_deg < self.config.head_yaw_left_threshold_deg:
            return GazeEstimate(GazeZone.LEFT, 0.75)
        if head_pose.yaw_deg > self.config.head_yaw_right_threshold_deg:
            return GazeEstimate(GazeZone.RIGHT, 0.75)
        return GazeEstimate(GazeZone.ROAD, 0.8)
