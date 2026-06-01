from __future__ import annotations

import numpy as np

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.pipeline import DMSPipeline
from ind_vias_dms.core.types import (
    AvailabilityState,
    DMSState,
    DistractionLevel,
    DistractionType,
    DrowsinessLevel,
    GazeZone,
    PlaceholderState,
)
from ind_vias_dms.temporal.distraction_fsm import DistractionFSM
from ind_vias_dms.temporal.drowsiness_fsm import DrowsinessFSM
from ind_vias_dms.vision.eye_state import EyeState
from ind_vias_dms.vision.face_landmarks import FaceLandmarkResult
from ind_vias_dms.vision.gaze import GazeEstimator
from ind_vias_dms.vision.head_pose import HeadPose, normalize_angle_deg
from ind_vias_dms.visualization.overlay import OverlayRenderer, clamp_endpoint


def test_drowsiness_fsm_enters_microsleep_after_threshold():
    config = DMSConfig(microsleep_duration_ms=1500)
    fsm = DrowsinessFSM(config)

    level = fsm.update(
        perclos_short=0.0,
        perclos_long=0.0,
        eye_closure_duration_ms=1600,
        face_present=True,
    )

    assert level == DrowsinessLevel.MICROSLEEP


def test_distraction_fsm_enters_high_after_eyes_off_road_threshold():
    config = DMSConfig(eyes_off_road_warning_ms=2000)
    fsm = DistractionFSM(config)

    level, kind = fsm.update(GazeZone.DOWN, eyes_off_road_duration_ms=2200)

    assert level == DistractionLevel.HIGH
    assert kind == DistractionType.PHONE_SUSPECTED


def test_distraction_fsm_no_face_handling_does_not_crash():
    config = DMSConfig(no_face_timeout_ms=1000)
    fsm = DistractionFSM(config)

    level, kind = fsm.update(
        GazeZone.UNKNOWN,
        eyes_off_road_duration_ms=0,
        no_face_duration_ms=1200,
    )

    assert level == DistractionLevel.UNKNOWN
    assert kind == DistractionType.UNKNOWN


def test_head_pose_angle_normalization_folds_around_180():
    assert normalize_angle_deg(190.0) == -170.0
    assert normalize_angle_deg(-190.0) == 170.0
    assert normalize_angle_deg(15.0) == 15.0


def test_gaze_does_not_enter_phone_down_from_one_frame():
    config = DMSConfig(phone_down_sustain_ms=1200, head_pitch_down_threshold_deg=18)
    gaze = GazeEstimator(config)

    first = gaze.estimate(HeadPose(pitch_deg=25.0, confidence=0.8), 1000, face_present=True)
    later = gaze.estimate(HeadPose(pitch_deg=25.0, confidence=0.8), 2300, face_present=True)

    assert first.zone == GazeZone.DOWN
    assert later.zone == GazeZone.PHONE_DOWN


def test_road_classification_uses_calibration_offsets():
    config = DMSConfig(
        road_center_yaw_offset_deg=15.0,
        road_center_pitch_offset_deg=5.0,
        road_yaw_tolerance_deg=10.0,
        road_pitch_tolerance_deg=8.0,
    )
    gaze = GazeEstimator(config)

    calibrated = gaze.estimate(HeadPose(yaw_deg=15.0, pitch_deg=5.0, confidence=0.8), 1000)
    raw_zero = gaze.estimate(HeadPose(yaw_deg=0.0, pitch_deg=0.0, confidence=0.8), 1100)

    assert calibrated.zone == GazeZone.ROAD
    assert raw_zero.zone == GazeZone.LEFT


def test_phone_to_ear_state_can_be_represented_and_serialized():
    state = DMSState(phone_use=PlaceholderState("PHONE_TO_EAR_SUSPECTED", 0.78))

    assert state.to_dict()["phone_use"]["state"] == "PHONE_TO_EAR_SUSPECTED"


def test_high_distraction_does_not_immediately_make_driver_unavailable():
    config = DMSConfig(high_distraction_unavailable_ms=5000)
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = config
    face = FaceLandmarkResult(face_found=True, confidence=0.9)
    eyes = EyeState(is_closed=False, confidence=0.9)

    short = pipeline._availability(
        face,
        eyes,
        DrowsinessLevel.NONE,
        DistractionLevel.HIGH,
        no_face_duration_ms=0,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=2000,
        gaze_zone=GazeZone.DOWN,
        phone_state="PHONE_DOWN_SUSPECTED",
    )
    sustained = pipeline._availability(
        face,
        eyes,
        DrowsinessLevel.NONE,
        DistractionLevel.HIGH,
        no_face_duration_ms=0,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=6000,
        gaze_zone=GazeZone.DOWN,
        phone_state="TEXTING_SUSPECTED",
    )

    assert short.state == AvailabilityState.DEGRADED
    assert sustained.state == AvailabilityState.UNAVAILABLE


def test_status_dashboard_render_returns_valid_image():
    image = OverlayRenderer().render_status_dashboard(DMSState(), fps=29.5, width=480, height=720)

    assert isinstance(image, np.ndarray)
    assert image.shape == (720, 480, 3)
    assert image.dtype == np.uint8


def test_vector_clamp_limits_endpoint_distance():
    endpoint = clamp_endpoint((10, 10), (300.0, 10.0), (100, 100, 3), max_length_px=50)

    assert endpoint == (60, 10)


def test_availability_reason_codes_do_not_use_severity_only_words():
    config = DMSConfig(high_distraction_unavailable_ms=5000)
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = config
    face = FaceLandmarkResult(face_found=True, confidence=0.9)
    eyes = EyeState(is_closed=False, confidence=0.9)

    availability = pipeline._availability(
        face,
        eyes,
        DrowsinessLevel.HIGH,
        DistractionLevel.NONE,
        no_face_duration_ms=0,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=0,
        gaze_zone=GazeZone.ROAD,
    )

    assert availability.reason_codes
    assert "HIGH" not in availability.reason_codes
    assert "MEDIUM" not in availability.reason_codes
    assert "DROWSINESS_HIGH" in availability.reason_codes
