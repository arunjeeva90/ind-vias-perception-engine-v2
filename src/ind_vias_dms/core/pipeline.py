from __future__ import annotations

import numpy as np

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.timing import FPSMeter
from ind_vias_dms.core.types import (
    AvailabilityState,
    CameraStatus,
    DMSHealth,
    DMSState,
    DistractionLevel,
    DistractionState,
    DriverAvailability,
    DriverPresence,
    DriverReadinessScore,
    DrowsinessLevel,
    DrowsinessState,
    GazeState,
    GazeZone,
    PresenceState,
    RiskLevel,
)
from ind_vias_dms.temporal.blink_tracker import BlinkTracker
from ind_vias_dms.temporal.distraction_fsm import DistractionFSM
from ind_vias_dms.temporal.drowsiness_fsm import DrowsinessFSM
from ind_vias_dms.temporal.perclos import PERCLOSTracker
from ind_vias_dms.vision.eye_state import EyeState, EyeStateEstimator
from ind_vias_dms.vision.face_landmarks import FaceLandmarkBackend, FaceLandmarkResult
from ind_vias_dms.vision.gaze import GazeEstimator
from ind_vias_dms.vision.head_pose import HeadPoseEstimator
from ind_vias_dms.vision.phone_detection import PhoneDetectionPlaceholder
from ind_vias_dms.vision.seatbelt import SeatbeltDetectionPlaceholder


class DMSPipeline:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.face_backend = FaceLandmarkBackend(config.face_backend)
        self.head_pose_estimator = HeadPoseEstimator()
        self.eye_state_estimator = EyeStateEstimator(config.eye_closed_threshold)
        self.gaze_estimator = GazeEstimator(config)
        self.blink_tracker = BlinkTracker(config.blink_min_duration_ms)
        self.perclos_short = PERCLOSTracker(config.perclos_short_window_s)
        self.perclos_long = PERCLOSTracker(config.perclos_long_window_s)
        self.drowsiness_fsm = DrowsinessFSM(config)
        self.distraction_fsm = DistractionFSM(config)
        self.phone_detector = PhoneDetectionPlaceholder()
        self.seatbelt_detector = SeatbeltDetectionPlaceholder()
        self.fps_meter = FPSMeter()
        self.no_face_since_ms: int | None = None
        self.eyes_off_road_since_ms: int | None = None

    def process(self, frame: np.ndarray, timestamp_ms: int, frame_id: int) -> tuple[DMSState, dict[str, object]]:
        face = self.face_backend.process(frame)
        if face.face_found:
            self.no_face_since_ms = None
        elif self.no_face_since_ms is None:
            self.no_face_since_ms = timestamp_ms

        no_face_duration_ms = (
            timestamp_ms - self.no_face_since_ms if self.no_face_since_ms is not None else 0
        )
        head_pose = self.head_pose_estimator.estimate(face.landmarks_px, frame.shape)
        eye_state = self.eye_state_estimator.estimate(face.landmarks_px)
        gaze_estimate = self.gaze_estimator.estimate(head_pose)
        self._update_eyes_off_road(timestamp_ms, gaze_estimate.zone)
        eyes_off_road_ms = (
            timestamp_ms - self.eyes_off_road_since_ms
            if self.eyes_off_road_since_ms is not None
            else 0
        )

        blink_stats = self.blink_tracker.update(timestamp_ms, eye_state.is_closed and face.face_found)
        perclos_5s = self.perclos_short.update(timestamp_ms, eye_state.is_closed and face.face_found)
        perclos_60s = self.perclos_long.update(timestamp_ms, eye_state.is_closed and face.face_found)
        drowsiness_level = self.drowsiness_fsm.update(
            perclos_5s,
            perclos_60s,
            blink_stats.eye_closure_duration_ms,
            blink_stats.blink_rate_per_min,
            face.face_found,
        )
        distraction_level, distraction_type = self.distraction_fsm.update(
            gaze_estimate.zone,
            eyes_off_road_ms,
            no_face_duration_ms,
        )

        availability = self._availability(
            face,
            eye_state,
            drowsiness_level,
            distraction_level,
            no_face_duration_ms,
        )
        readiness = self._readiness(
            face,
            eye_state,
            drowsiness_level,
            distraction_level,
            gaze_estimate.zone,
        )
        state = DMSState(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            dms_health=DMSHealth(
                camera_status=self._camera_status(face, no_face_duration_ms),
                face_visibility_score=face.confidence if face.face_found else 0.0,
                eye_visibility_score=eye_state.confidence,
                confidence=min(face.confidence, eye_state.confidence)
                if face.face_found
                else 0.0,
            ),
            driver_presence=DriverPresence(
                state=PresenceState.PRESENT if face.face_found else PresenceState.ABSENT,
                confidence=face.confidence if face.face_found else 0.0,
            ),
            driver_availability=availability,
            gaze=GazeState(
                zone=gaze_estimate.zone,
                eyes_off_road_duration_ms=int(eyes_off_road_ms),
                head_yaw_deg=head_pose.yaw_deg,
                head_pitch_deg=head_pose.pitch_deg,
                head_roll_deg=head_pose.roll_deg,
                confidence=gaze_estimate.confidence,
            ),
            drowsiness=DrowsinessState(
                level=drowsiness_level,
                perclos_5s=perclos_5s,
                perclos_60s=perclos_60s,
                eye_closure_duration_ms=blink_stats.eye_closure_duration_ms,
                blink_rate_per_min=blink_stats.blink_rate_per_min,
                confidence=eye_state.confidence if face.face_found else 0.0,
            ),
            distraction=DistractionState(
                level=distraction_level,
                type=distraction_type,
                duration_ms=int(eyes_off_road_ms),
                confidence=gaze_estimate.confidence,
            ),
            phone_use=self.phone_detector.process(frame),
            seatbelt_authenticity=self.seatbelt_detector.process(frame),
            driver_readiness_score=readiness,
        )
        context = {
            "face": face,
            "head_pose": head_pose,
            "eye_state": eye_state,
            "fps": self.fps_meter.update(timestamp_ms),
        }
        return state, context

    def close(self) -> None:
        self.face_backend.close()

    def _update_eyes_off_road(self, timestamp_ms: int, zone: GazeZone) -> None:
        if zone == GazeZone.ROAD:
            self.eyes_off_road_since_ms = None
        elif self.eyes_off_road_since_ms is None and zone != GazeZone.UNKNOWN:
            self.eyes_off_road_since_ms = timestamp_ms

    def _camera_status(self, face: FaceLandmarkResult, no_face_duration_ms: int) -> CameraStatus:
        if not face.face_found and no_face_duration_ms >= self.config.no_face_timeout_ms:
            return CameraStatus.NO_FACE
        if face.face_found and face.confidence < 0.5:
            return CameraStatus.LOW_CONFIDENCE
        return CameraStatus.OK if face.face_found else CameraStatus.NO_FACE

    def _availability(
        self,
        face: FaceLandmarkResult,
        eye_state: EyeState,
        drowsiness: DrowsinessLevel,
        distraction: DistractionLevel,
        no_face_duration_ms: int,
    ) -> DriverAvailability:
        reasons: list[str] = []
        if no_face_duration_ms >= self.config.no_face_timeout_ms:
            return DriverAvailability(AvailabilityState.UNAVAILABLE, 0.0, ["NO_FACE_TIMEOUT"])
        if drowsiness in {DrowsinessLevel.MICROSLEEP, DrowsinessLevel.HIGH}:
            reasons.append(drowsiness.value)
        if distraction == DistractionLevel.HIGH:
            reasons.append("HIGH_DISTRACTION")
        if reasons:
            return DriverAvailability(AvailabilityState.UNAVAILABLE, 0.15, reasons)
        if not face.face_found:
            return DriverAvailability(AvailabilityState.DEGRADED, 0.35, ["NO_FACE"])
        if eye_state.confidence < 0.5:
            return DriverAvailability(AvailabilityState.DEGRADED, 0.55, ["LOW_EYE_CONFIDENCE"])
        if drowsiness == DrowsinessLevel.MEDIUM or distraction == DistractionLevel.MEDIUM:
            return DriverAvailability(AvailabilityState.DEGRADED, 0.6, ["MODERATE_RISK"])
        return DriverAvailability(AvailabilityState.AVAILABLE, 0.95, [])

    def _readiness(
        self,
        face: FaceLandmarkResult,
        eye_state: EyeState,
        drowsiness: DrowsinessLevel,
        distraction: DistractionLevel,
        gaze_zone: GazeZone,
    ) -> DriverReadinessScore:
        score = 1.0
        if not face.face_found:
            score -= 0.6
        score -= max(0.0, 1.0 - eye_state.confidence) * 0.2
        score -= {
            DrowsinessLevel.NONE: 0.0,
            DrowsinessLevel.LOW: 0.1,
            DrowsinessLevel.MEDIUM: 0.25,
            DrowsinessLevel.HIGH: 0.45,
            DrowsinessLevel.MICROSLEEP: 0.75,
            DrowsinessLevel.UNKNOWN: 0.2,
        }[drowsiness]
        score -= {
            DistractionLevel.NONE: 0.0,
            DistractionLevel.LOW: 0.1,
            DistractionLevel.MEDIUM: 0.25,
            DistractionLevel.HIGH: 0.45,
            DistractionLevel.UNKNOWN: 0.15,
        }[distraction]
        if gaze_zone not in {GazeZone.ROAD, GazeZone.UNKNOWN}:
            score -= 0.1
        score = min(1.0, max(0.0, score))
        if score >= 0.75:
            risk = RiskLevel.LOW
        elif score >= 0.5:
            risk = RiskLevel.MEDIUM
        elif score >= 0.25:
            risk = RiskLevel.HIGH
        else:
            risk = RiskLevel.CRITICAL
        return DriverReadinessScore(score, risk)
