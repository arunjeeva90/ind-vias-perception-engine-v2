from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, degrees, radians, sin, sqrt

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.vision.gaze import GazeEstimate
from ind_vias_dms.vision.head_pose import HeadPose, normalize_angle_deg


@dataclass
class RoadAxisReference:
    yaw_ref_deg: float = 0.0
    pitch_ref_deg: float = 0.0
    roll_ref_deg: float = 0.0
    calibration_source: str = "DEFAULT"
    timestamp_ms: int = 0
    confidence: float = 0.0
    calibrated: bool = False


@dataclass
class RoadAxisRelativePose:
    head_pose_raw_yaw_deg: float = 0.0
    head_pose_raw_pitch_deg: float = 0.0
    head_pose_raw_roll_deg: float = 0.0
    road_axis_yaw_ref_deg: float = 0.0
    road_axis_pitch_ref_deg: float = 0.0
    road_axis_roll_ref_deg: float = 0.0
    relative_yaw_deg: float = 0.0
    relative_pitch_deg: float = 0.0
    relative_roll_deg: float = 0.0
    road_axis_calibration_source: str = "DEFAULT"
    road_axis_calibration_confidence: float = 0.0
    road_axis_calibrated: bool = False
    side_glance_state: str = "ROAD_AXIS_NORMAL"
    side_glance_duration_ms: int = 0
    side_glance_recovery_ms: int = 0
    yaw_classifiable: bool = False
    side_profile_context_active: bool = False
    head_angle_from_road_deg: float = 0.0
    head_pose_vector_quality: float = 0.0


class RoadAxisHeadPoseReference:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.reference = RoadAxisReference()
        self._side_glance_since_ms: int | None = None
        self._recovery_since_ms: int | None = None
        self._last_side_glance_ms: int | None = None
        self._last_side_glance_state = "ROAD_AXIS_NORMAL"

    def calibrate(
        self,
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float,
        timestamp_ms: int,
        source: str,
        confidence: float = 1.0,
    ) -> RoadAxisReference:
        self.reference = RoadAxisReference(
            yaw_ref_deg=float(yaw_deg),
            pitch_ref_deg=float(pitch_deg),
            roll_ref_deg=float(roll_deg),
            calibration_source=source,
            timestamp_ms=int(timestamp_ms),
            confidence=max(0.0, min(1.0, float(confidence))),
            calibrated=True,
        )
        return self.reference

    def reset(self) -> None:
        self.reference = RoadAxisReference()
        self._side_glance_since_ms = None
        self._recovery_since_ms = None
        self._last_side_glance_ms = None
        self._last_side_glance_state = "ROAD_AXIS_NORMAL"

    def update(
        self,
        head_pose: HeadPose,
        timestamp_ms: int,
        face_present: bool,
        pose_reliable: bool,
        gaze_estimate: GazeEstimate,
    ) -> RoadAxisRelativePose:
        raw_yaw = float(head_pose.yaw_deg)
        raw_pitch = float(head_pose.pitch_deg)
        raw_roll = float(head_pose.roll_deg)
        ref = self.reference
        rel_yaw = normalize_angle_deg(raw_yaw - ref.yaw_ref_deg)
        rel_pitch = raw_pitch - ref.pitch_ref_deg
        rel_roll = normalize_angle_deg(raw_roll - ref.roll_ref_deg)
        head_angle_from_road_deg, vector_quality = self._head_angle_from_reference(
            raw_yaw,
            raw_pitch,
            raw_roll,
            ref,
            head_pose.confidence,
            pose_reliable,
        )
        yaw_classifiable = (
            bool(self.config.road_axis_head_pose_enabled)
            and ref.calibrated
            and face_present
            and pose_reliable
            and head_pose.confidence >= self.config.side_profile_min_pose_confidence
        )

        side_state = "ROAD_AXIS_NORMAL"
        side_active = yaw_classifiable and abs(rel_yaw) >= self.config.side_glance_monitor_deg
        if side_active:
            if self._side_glance_since_ms is None:
                self._side_glance_since_ms = timestamp_ms
            self._recovery_since_ms = None
            duration_ms = max(0, timestamp_ms - self._side_glance_since_ms)
            direction = "RIGHT" if rel_yaw > 0 else "LEFT"
            if abs(rel_yaw) >= self.config.side_glance_warning_deg and duration_ms >= self.config.side_glance_warning_ms:
                side_state = "SIDE_PROFILE_ATTENTION_LOSS"
            else:
                side_state = f"SIDE_GLANCE_{direction}"
            self._last_side_glance_ms = timestamp_ms
            self._last_side_glance_state = side_state
        else:
            held_side_context = (
                not yaw_classifiable
                and self._last_side_glance_ms is not None
                and self._side_glance_since_ms is not None
                and timestamp_ms - self._last_side_glance_ms <= self.config.side_profile_face_loss_grace_ms
            )
            if held_side_context:
                duration_ms = max(0, timestamp_ms - self._side_glance_since_ms)
                side_state = self._last_side_glance_state
            else:
                duration_ms = 0
                self._side_glance_since_ms = None
            if yaw_classifiable and abs(rel_yaw) <= self.config.road_yaw_normal_deg:
                if self._recovery_since_ms is None:
                    self._recovery_since_ms = timestamp_ms
                recovery_ms = max(0, timestamp_ms - self._recovery_since_ms)
                side_state = "SIDE_PROFILE_RECOVERY" if recovery_ms < self.config.side_glance_recovery_ms else "ROAD_AXIS_NORMAL"
            elif not held_side_context:
                recovery_ms = 0
                self._recovery_since_ms = None

        recovery_ms = 0 if self._recovery_since_ms is None else max(0, timestamp_ms - self._recovery_since_ms)
        side_profile_context = (
            self._last_side_glance_ms is not None
            and timestamp_ms - self._last_side_glance_ms <= self.config.side_profile_face_loss_grace_ms
            and self._last_side_glance_state != "ROAD_AXIS_NORMAL"
        )
        self._maybe_auto_update(ref, head_pose, timestamp_ms, yaw_classifiable, rel_yaw, rel_pitch, gaze_estimate)
        return RoadAxisRelativePose(
            head_pose_raw_yaw_deg=raw_yaw,
            head_pose_raw_pitch_deg=raw_pitch,
            head_pose_raw_roll_deg=raw_roll,
            road_axis_yaw_ref_deg=self.reference.yaw_ref_deg,
            road_axis_pitch_ref_deg=self.reference.pitch_ref_deg,
            road_axis_roll_ref_deg=self.reference.roll_ref_deg,
            relative_yaw_deg=rel_yaw,
            relative_pitch_deg=rel_pitch,
            relative_roll_deg=rel_roll,
            road_axis_calibration_source=self.reference.calibration_source,
            road_axis_calibration_confidence=self.reference.confidence,
            road_axis_calibrated=self.reference.calibrated,
            side_glance_state=side_state,
            side_glance_duration_ms=duration_ms,
            side_glance_recovery_ms=recovery_ms,
            yaw_classifiable=yaw_classifiable,
            side_profile_context_active=side_profile_context,
            head_angle_from_road_deg=head_angle_from_road_deg,
            head_pose_vector_quality=vector_quality,
        )

    def _maybe_auto_update(
        self,
        ref: RoadAxisReference,
        head_pose: HeadPose,
        timestamp_ms: int,
        yaw_classifiable: bool,
        relative_yaw_deg: float,
        relative_pitch_deg: float,
        gaze_estimate: GazeEstimate,
    ) -> None:
        if not (self.config.road_axis_auto_update_enabled and ref.calibrated and yaw_classifiable):
            return
        if self.config.road_axis_auto_update_only_when_road_gaze_stable:
            if gaze_estimate.zone.value != "ROAD" or gaze_estimate.confidence < self.config.road_axis_auto_update_min_confidence:
                return
        if abs(relative_yaw_deg) > self.config.road_axis_auto_update_max_abs_relative_yaw_deg:
            return
        if abs(relative_pitch_deg) > self.config.road_axis_auto_update_max_abs_relative_pitch_deg:
            return
        alpha = max(0.0, min(1.0, self.config.road_axis_auto_update_alpha))
        self.reference = RoadAxisReference(
            yaw_ref_deg=ref.yaw_ref_deg * (1.0 - alpha) + head_pose.yaw_deg * alpha,
            pitch_ref_deg=ref.pitch_ref_deg * (1.0 - alpha) + head_pose.pitch_deg * alpha,
            roll_ref_deg=ref.roll_ref_deg * (1.0 - alpha) + head_pose.roll_deg * alpha,
            calibration_source=ref.calibration_source,
            timestamp_ms=timestamp_ms,
            confidence=min(1.0, max(ref.confidence, gaze_estimate.confidence)),
            calibrated=True,
        )

    @staticmethod
    def _head_angle_from_reference(
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float,
        ref: RoadAxisReference,
        pose_confidence: float,
        pose_reliable: bool,
    ) -> tuple[float, float]:
        if not ref.calibrated or not pose_reliable:
            return 0.0, 0.0
        head = RoadAxisHeadPoseReference._forward_vector(yaw_deg, pitch_deg, roll_deg)
        road = RoadAxisHeadPoseReference._forward_vector(
            ref.yaw_ref_deg,
            ref.pitch_ref_deg,
            ref.roll_ref_deg,
        )
        dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(head, road))))
        quality = max(0.0, min(1.0, min(float(pose_confidence), ref.confidence)))
        return degrees(acos(dot)), quality

    @staticmethod
    def _forward_vector(yaw_deg: float, pitch_deg: float, roll_deg: float) -> tuple[float, float, float]:
        yaw = radians(yaw_deg)
        pitch = radians(pitch_deg)
        roll = radians(roll_deg)
        cy, sy = cos(yaw), sin(yaw)
        cp, sp = cos(pitch), sin(pitch)
        cr, sr = cos(roll), sin(roll)
        # Apply roll, pitch, then yaw to a camera-forward unit vector.
        x = cy * (sr * sp) + sy * cp
        y = cr * sp
        z = -sy * (sr * sp) + cy * cp
        norm = sqrt(x * x + y * y + z * z) or 1.0
        return x / norm, y / norm, z / norm
