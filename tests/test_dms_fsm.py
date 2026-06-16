from __future__ import annotations

import json

import numpy as np

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.driver_session import DriverSessionManager, DriverSessionState
from ind_vias_dms.core.occupant_manager import CabinOccupantManager, suppress_duplicate_faces
from ind_vias_dms.core.pipeline import DMSPipeline
from ind_vias_dms.core.road_axis import RoadAxisHeadPoseReference
from ind_vias_dms.core.road_calibration import load_road_calibration, save_road_calibration
from ind_vias_dms.core.types import (
    AttentionState,
    AttentionSubstate,
    AvailabilityState,
    CameraStatus,
    DMSConfidenceState,
    DMSHealth,
    DMSState,
    DMSV02Level,
    DistractionLevel,
    DistractionType,
    DriverObservability,
    DriverObservabilityState,
    DriverPresence,
    DrowsinessLevel,
    GazeZone,
    OccupantFace,
    OccupantsState,
    PlaceholderState,
    PresenceState,
)
from ind_vias_dms.temporal.attention_state import AttentionSignals, AttentionStateClassifier
from ind_vias_dms.temporal.distraction_fsm import DistractionFSM
from ind_vias_dms.temporal.drowsiness_fsm import DrowsinessFSM
from ind_vias_dms.temporal.dms_v02_decision import DMSV02DecisionMatrix, DMSV02Inputs
from ind_vias_dms.temporal.eye_temporal import EyeTemporalTracker
from ind_vias_dms.temporal.perclos import PERCLOSTracker
from ind_vias_dms.vision.eye_state import EyeState
from ind_vias_dms.utils.debug_trace import build_debug_record
from ind_vias_dms.vision.face_landmarks import FaceLandmarkResult, FaceQualityResult, evaluate_face_quality
from ind_vias_dms.vision.gaze import GazeEstimator
from ind_vias_dms.vision.gaze import GazeEstimate
from ind_vias_dms.vision.head_pose import HeadPose, normalize_angle_deg
from ind_vias_dms.vision.phone_detection import MobileDistractionEstimator
from ind_vias_dms.utils.learning_memory import LearningMemoryWriter
from ind_vias_dms.visualization.overlay import (
    OverlayRenderer,
    banner_decision,
    clamp_endpoint,
    occupant_label,
    status_dashboard_lines,
)


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
    config = DMSConfig(gaze_away_high_ms=2000)
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
        eyes_off_road_duration_ms=9000,
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


def _face(box: tuple[int, int, int, int], confidence: float = 0.8) -> FaceLandmarkResult:
    x1, y1, x2, y2 = box
    area_norm = ((x2 - x1) / 1000.0) * ((y2 - y1) / 1000.0)
    return FaceLandmarkResult(
        face_found=True,
        bbox=box,
        confidence=confidence,
        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        area=float((x2 - x1) * (y2 - y1)),
        box_norm=(x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0),
        quality=FaceQualityResult(
            proposal_confidence=confidence,
            landmark_count=160,
            landmark_coverage_score=0.85,
            face_box_area_norm=area_norm,
            face_aspect_ratio=(x2 - x1) / max(1.0, float(y2 - y1)),
            left_eye_visible=True,
            right_eye_visible=True,
            nose_visible=True,
            mouth_visible=True,
            chin_visible=True,
            both_eyes_available=True,
            face_completeness_score=0.95,
            is_valid_driver_face=True,
            validation_state="VALIDATED_FULL_FACE",
            rejection_reason_codes=[],
        ),
    )


def _proposal_face(box: tuple[int, int, int, int], confidence: float = 0.8) -> FaceLandmarkResult:
    x1, y1, x2, y2 = box
    return FaceLandmarkResult(
        face_found=True,
        bbox=box,
        confidence=confidence,
        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        area=float((x2 - x1) * (y2 - y1)),
        box_norm=(x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0),
    )


def _face_with_landmarks(
    box: tuple[int, int, int, int],
    confidence: float = 0.85,
    landmark_count: int = 160,
) -> FaceLandmarkResult:
    face = _face(box, confidence)
    x1, y1, x2, y2 = box
    face.landmarks_px = {
        idx: (
            x1 + (idx % 20) / 20.0 * max(1, x2 - x1),
            y1 + (idx // 20) / 10.0 * max(1, y2 - y1),
        )
        for idx in range(landmark_count)
    }
    return face


def test_driver_roi_selection_prefers_driver_over_larger_passenger():
    manager = CabinOccupantManager(DMSConfig(driver_image_side="RIGHT"))
    passenger = _face((80, 200, 430, 800), confidence=0.95)
    driver = _face((620, 240, 820, 720), confidence=0.75)

    selection = manager.update([passenger, driver], (1000, 1000, 3), timestamp_ms=0)

    assert selection.driver is not None
    assert selection.driver.observation is driver
    assert selection.driver.zone == "DRIVER"


def test_no_driver_face_with_passenger_visible_is_not_visible():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig()
    pipeline.occupants = CabinOccupantManager(pipeline.config)

    assert pipeline._presence_state(face_found=False, occupant_count=1) == PresenceState.NOT_VISIBLE


def test_no_driver_face_reason_code_is_driver_face_not_visible():
    config = DMSConfig(no_face_timeout_ms=1000)
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = config

    availability = pipeline._availability(
        FaceLandmarkResult(face_found=False),
        EyeState(confidence=0.0),
        DrowsinessLevel.UNKNOWN,
        DistractionLevel.UNKNOWN,
        no_face_duration_ms=1200,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=0,
        gaze_zone=GazeZone.UNKNOWN,
        occupant_count=1,
    )

    assert availability.state == AvailabilityState.UNAVAILABLE
    assert availability.reason_codes == ["DRIVER_FACE_NOT_VISIBLE"]


def test_passenger_phone_to_ear_does_not_set_driver_phone_state():
    from ind_vias_dms.vision.phone_detection import MobileDistractionEstimator

    class FakeEstimator(MobileDistractionEstimator):
        def process(self, frame, face_bbox, face_landmarks, gaze_zone, timestamp_ms):
            if face_bbox == (1, 1, 2, 2):
                return PlaceholderState("NO_PHONE", 0.6)
            return PlaceholderState("PHONE_TO_EAR_SUSPECTED", 0.8)

    estimator = FakeEstimator.__new__(FakeEstimator)
    estimator.config = DMSConfig()
    driver = FaceLandmarkResult(True, bbox=(1, 1, 2, 2))
    passenger = FaceLandmarkResult(True, bbox=(3, 1, 4, 2))

    driver_state, cabin_events = estimator.process_cabin(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [(1, "DRIVER", driver), (2, "FRONT_PASSENGER", passenger)],
        driver_track_id=1,
        gaze_zone=GazeZone.ROAD,
        timestamp_ms=0,
    )

    assert driver_state.state == "NO_PHONE"
    assert cabin_events == ["PASSENGER_PHONE_TO_EAR"]


def test_road_calibration_state_can_be_represented():
    gaze = GazeEstimator(DMSConfig(road_gaze_calibrated=False))

    assert gaze.road_gaze_calibrated is False
    gaze.calibrate_road_center(3.0, 4.0)
    assert gaze.road_gaze_calibrated is True


def test_visible_passenger_no_driver_has_cabin_ok_and_driver_not_visible():
    config = DMSConfig(driver_image_side="LEFT", auto_generate_rois_from_layout=True)
    manager = CabinOccupantManager(config)
    passenger = _face((650, 200, 900, 720), confidence=0.9)
    selection = manager.update([passenger], (1000, 1000, 3), timestamp_ms=0)
    state = DMSState(
        dms_health=DMSHealth(camera_status=CameraStatus.OK),
        driver_presence=DriverPresence(PresenceState.NOT_VISIBLE),
        occupants=OccupantsState(
            count=len(selection.faces),
            faces=[OccupantFace(1, "FRONT_PASSENGER", [0.65, 0.2, 0.9, 0.72])],
        ),
    )

    assert selection.driver is None
    assert state.dms_health.camera_status == CameraStatus.OK
    assert state.driver_presence.state == PresenceState.NOT_VISIBLE


def test_no_driver_face_eye_state_is_unknown():
    pipeline = DMSPipeline.__new__(DMSPipeline)

    assert pipeline._eye_state_label(False, EyeState(is_closed=False, confidence=0.0)) == "UNKNOWN"


def test_process_cabin_with_missing_driver_does_not_crash():
    from ind_vias_dms.vision.phone_detection import MobileDistractionEstimator

    estimator = MobileDistractionEstimator.__new__(MobileDistractionEstimator)
    estimator.config = DMSConfig()
    estimator._hands = None
    estimator.down_since_ms = None
    estimator.phone_to_ear_since_ms = None
    estimator.texting_since_ms = None

    driver_state, cabin_events = estimator.process_cabin(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [],
        driver_track_id=None,
        gaze_zone=GazeZone.UNKNOWN,
        timestamp_ms=0,
    )

    assert driver_state.state == "UNKNOWN"
    assert cabin_events == []


def test_driver_image_side_left_produces_left_driver_roi():
    manager = CabinOccupantManager(DMSConfig(driver_image_side="LEFT", auto_generate_rois_from_layout=True))

    assert manager._roi("driver_roi_norm")[0] == 0.0
    assert manager.driver_roi_state() == "AUTO_LEFT"


def test_driver_image_side_right_produces_right_driver_roi():
    manager = CabinOccupantManager(DMSConfig(driver_image_side="RIGHT", auto_generate_rois_from_layout=True))

    assert manager._roi("driver_roi_norm")[0] == 0.45
    assert manager.driver_roi_state() == "AUTO_RIGHT"


def test_auto_generated_roi_overrides_conflicting_explicit_roi():
    manager = CabinOccupantManager(
        DMSConfig(
            driver_image_side="LEFT",
            auto_generate_rois_from_layout=True,
            driver_roi_norm={"x_min": 0.45, "y_min": 0.1, "x_max": 1.0, "y_max": 0.95},
        )
    )

    assert manager._roi("driver_roi_norm") == (0.0, 0.10, 0.55, 0.95)


def test_overlay_label_formatting_uses_full_zone_and_track_id():
    assert occupant_label("FRONT_PASSENGER", 2) == "FRONT_PASSENGER T2"
    assert occupant_label("FRONT_PASSENGER", 2, selected_as_driver=True) == "DRIVER T2"
    assert occupant_label("DRIVER", 4, selected_as_driver=True, driver_session_id="D1") == "DRIVER D1"
    assert (
        occupant_label("DRIVER", 4, selected_as_driver=True, driver_session_id="D1", show_track_id=True)
        == "DRIVER D1 / T4"
    )


def test_status_dashboard_lines_split_cabin_camera_and_driver_face():
    state = DMSState(
        dms_health=DMSHealth(camera_status=CameraStatus.OK),
        driver_presence=DriverPresence(PresenceState.NOT_VISIBLE),
    )
    labels = [label for label, _ in status_dashboard_lines(state, fps=30.0)]

    assert "Camera health" in labels
    assert "Face detection" in labels
    assert "Driver face" in labels
    assert "Camera" not in labels


def test_duplicate_face_suppression_keeps_one_overlapping_face():
    strong = _face((100, 100, 300, 300), confidence=0.9)
    weak = _face((110, 110, 305, 305), confidence=0.7)

    kept = suppress_duplicate_faces([weak, strong], iou_threshold=0.45, center_distance_threshold=0.08)

    assert kept == [strong]


def test_driver_present_open_short_high_distraction_not_unavailable():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig(high_distraction_unavailable_ms=5000)
    availability = pipeline._availability(
        _face((100, 100, 300, 300)),
        EyeState(is_closed=False, confidence=0.9),
        DrowsinessLevel.NONE,
        DistractionLevel.HIGH,
        no_face_duration_ms=0,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=1000,
        gaze_zone=GazeZone.RIGHT,
        road_gaze_calibrated=True,
    )

    assert availability.state == AvailabilityState.DEGRADED


def test_sustained_high_distraction_without_adas_risk_stays_degraded():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig(high_distraction_unavailable_ms=5000)
    availability = pipeline._availability(
        _face((100, 100, 300, 300)),
        EyeState(is_closed=False, confidence=0.9),
        DrowsinessLevel.NONE,
        DistractionLevel.HIGH,
        no_face_duration_ms=0,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=6000,
        gaze_zone=GazeZone.RIGHT,
        road_gaze_calibrated=True,
    )

    assert availability.state == AvailabilityState.DEGRADED


def test_uncalibrated_road_gaze_adds_reason_without_high_distraction():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig()
    availability = pipeline._availability(
        _face((100, 100, 300, 300)),
        EyeState(is_closed=False, confidence=0.9),
        DrowsinessLevel.NONE,
        DistractionLevel.NONE,
        no_face_duration_ms=0,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=0,
        gaze_zone=GazeZone.RIGHT,
        road_gaze_calibrated=False,
    )

    assert availability.state == AvailabilityState.AVAILABLE
    assert "ROAD_GAZE_NOT_CALIBRATED" in availability.reason_codes


def test_driver_session_reassociation_keeps_session_after_short_loss():
    config = DMSConfig(driver_session_hold_ms=5000)
    manager = CabinOccupantManager(config)
    sessions = DriverSessionManager(config, manager)
    first = manager.update([_face((100, 200, 350, 700))], (1000, 1000, 3), 0)
    first_update = sessions.update(first.driver, 0)
    lost_update = sessions.update(None, 1000)
    second = manager.update([_face((110, 205, 360, 705))], (1000, 1000, 3), 1200)
    second.driver.track_id = 9
    second_update = sessions.update(second.driver, 1200)

    assert first_update.driver_session_id == "D1"
    assert lost_update.session_state == DriverSessionState.LOST_TEMP
    assert second_update.driver_session_id == "D1"
    assert second_update.driver_track_id == 9
    assert second_update.reassociated is True


def test_face_loss_shorter_than_hold_is_lost_temp():
    config = DMSConfig(driver_session_hold_ms=5000)
    manager = CabinOccupantManager(config)
    sessions = DriverSessionManager(config, manager)
    first = manager.update([_face((100, 200, 350, 700))], (1000, 1000, 3), 0)
    sessions.update(first.driver, 0)

    update = sessions.update(None, 3000)

    assert update.session_state == DriverSessionState.LOST_TEMP
    assert update.driver_session_id == "D1"


def test_long_absence_allows_new_driver_session():
    config = DMSConfig(driver_swap_absence_ms=15000)
    manager = CabinOccupantManager(config)
    sessions = DriverSessionManager(config, manager)
    first = manager.update([_face((100, 200, 350, 700))], (1000, 1000, 3), 0)
    sessions.update(first.driver, 0)
    sessions.update(None, 16000)
    second = manager.update([_face((120, 220, 380, 760))], (1000, 1000, 3), 17000)

    update = sessions.update(second.driver, 17000)

    assert update.driver_session_id == "D2"
    assert update.session_state == DriverSessionState.SWAPPED


def test_availability_degraded_during_lost_temp_session():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig()
    pipeline.last_driver_abs_yaw_deg = 30.0

    availability = pipeline._availability(
        FaceLandmarkResult(face_found=False),
        EyeState(confidence=0.0),
        DrowsinessLevel.UNKNOWN,
        DistractionLevel.UNKNOWN,
        no_face_duration_ms=1000,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=0,
        gaze_zone=GazeZone.UNKNOWN,
        session_state=DriverSessionState.LOST_TEMP.value,
    )

    assert availability.state == AvailabilityState.DEGRADED
    assert "DRIVER_FACE_LOST_TEMP" in availability.reason_codes
    assert "SIDE_PROFILE_FACE_LOST" in availability.reason_codes


def test_perclos_value_is_not_reset_during_lost_temp_pause():
    tracker = PERCLOSTracker(5)
    tracker.update(0, False)
    tracker.update(1000, True)
    tracker.update(2000, True)

    value = tracker.current(2000).perclos

    assert value > 0.0


def test_road_calibration_save_load_round_trip(tmp_path):
    path = tmp_path / "road.yaml"

    save_road_calibration(path, 12.5, -3.0)
    loaded = load_road_calibration(path)

    assert loaded.calibrated is True
    assert loaded.source == "FILE"
    assert loaded.yaw_offset_deg == 12.5
    assert loaded.pitch_offset_deg == -3.0


def test_missing_road_calibration_file_does_not_crash(tmp_path):
    loaded = load_road_calibration(tmp_path / "missing.yaml")

    assert loaded.calibrated is False
    assert loaded.source == "DEFAULT"


def test_synthetic_open_eye_sequence_gives_zero_perclos():
    tracker = PERCLOSTracker(5)
    tracker.update_weighted(0, 0.0, True)
    tracker.update_weighted(1000, 0.0, True)
    tracker.update_weighted(2000, 0.0, True)

    assert tracker.current(2000).perclos == 0.0


def test_synthetic_closed_eye_two_seconds_in_five_second_window():
    tracker = PERCLOSTracker(5)
    tracker.update_weighted(0, 1.0, True)
    tracker.update_weighted(2000, 0.0, True)
    tracker.update_weighted(5000, 0.0, True)

    assert 0.39 <= tracker.current(5000).perclos <= 0.41


def test_partial_closure_contributes_weighted_perclos():
    tracker = PERCLOSTracker(5)
    tracker.update_weighted(0, 0.5, True)
    tracker.update_weighted(1000, 0.0, True)

    assert 0.49 <= tracker.current(1000).perclos <= 0.51


def test_unknown_eye_state_pauses_perclos_not_open():
    tracker = PERCLOSTracker(5)
    tracker.update_weighted(0, 1.0, True)
    tracker.pause(1000)
    tracker.update_weighted(2000, 1.0, True)

    assert tracker.current(2000).valid_time_ms == 1000
    assert tracker.current(2000).perclos == 1.0


def test_session_reset_clears_perclos():
    tracker = PERCLOSTracker(5)
    tracker.update_weighted(0, 1.0, True)
    tracker.update_weighted(1000, 1.0, True)
    tracker.reset()

    assert tracker.current(1000).perclos == 0.0


def test_eye_closure_duration_increases_across_closed_frames():
    eye = EyeTemporalTracker(DMSConfig())
    eye.open_eye_baseline = 0.3

    eye.update(0, 0.1, 0.9, True)
    state = eye.update(1000, 0.1, 0.9, True)

    assert state.eye_state == "CLOSED"
    assert state.eye_closure_duration_ms == 1000


def test_adaptive_eye_baseline_calibrates_from_open_frames():
    eye = EyeTemporalTracker(DMSConfig())

    eye.update(0, 0.3, 0.9, True)
    state = eye.update(1000, 0.3, 0.9, True)

    assert state.calibration_state == "CALIBRATED"
    assert state.normalized_openness > 0.9


def test_perclos_does_not_remain_zero_for_closed_frames():
    tracker = PERCLOSTracker(5)
    tracker.update_weighted(0, 1.0, True)
    tracker.update_weighted(1000, 1.0, True)

    assert tracker.current(1000).perclos > 0.0


def test_partial_closure_weight_default_is_025():
    assert DMSConfig().perclos_partial_closure_weight == 0.25


def test_eye_baseline_does_not_update_during_large_yaw():
    eye = EyeTemporalTracker(DMSConfig())

    eye.update(0, 0.3, 0.9, True, abs_yaw_deg=35.0)

    assert eye.open_eye_baseline is None


def test_drowsiness_medium_requires_sustain_time():
    fsm = DrowsinessFSM(DMSConfig(drowsiness_medium_sustain_ms=1500))

    first = fsm.update(0.36, 0.0, 0, face_present=True, timestamp_ms=0)
    sustained = fsm.update(0.36, 0.0, 0, face_present=True, timestamp_ms=1600)

    assert first == DrowsinessLevel.LOW
    assert sustained == DrowsinessLevel.MEDIUM


def test_lost_temp_body_present_previous_yaw_adds_side_profile_reasons():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig()
    pipeline.last_driver_abs_yaw_deg = 35.0

    availability = pipeline._availability(
        FaceLandmarkResult(face_found=False),
        EyeState(confidence=0.0),
        DrowsinessLevel.UNKNOWN,
        DistractionLevel.UNKNOWN,
        no_face_duration_ms=1000,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=0,
        gaze_zone=GazeZone.UNKNOWN,
        session_state=DriverSessionState.LOST_TEMP.value,
        driver_body_state="PRESENT",
    )

    assert "SIDE_PROFILE_FACE_LOST" in availability.reason_codes
    assert "DRIVER_BODY_PRESENT_FACE_LOST" in availability.reason_codes
    assert "POSSIBLE_GAZE_AWAY_DURING_LOST_TEMP" in availability.reason_codes


def test_perclos_fields_are_near_eye_state_in_dashboard_order():
    labels = [label for label, _ in status_dashboard_lines(DMSState(), fps=30.0)]
    eyes_idx = labels.index("Effective eyes")
    perclos_idx = labels.index("PERCLOS 5s/60s")
    valid_idx = labels.index("PERCLOS valid")
    drowsiness_idx = labels.index("Drowsiness")

    assert eyes_idx < perclos_idx < valid_idx < drowsiness_idx


def test_loaded_calibration_can_mark_gaze_source_file(tmp_path):
    path = tmp_path / "road.yaml"
    save_road_calibration(path, 5.0, 2.0)
    loaded = load_road_calibration(path)
    gaze = GazeEstimator(DMSConfig())
    if loaded.calibrated:
        gaze.calibrate_road_center(loaded.yaw_offset_deg, loaded.pitch_offset_deg)

    assert gaze.road_gaze_calibrated is True
    assert loaded.source == "FILE"


def test_phone_to_ear_hand_near_driver_ear_escalates_after_sustain():
    from ind_vias_dms.vision.phone_detection import HandContext, MobileDistractionEstimator

    estimator = MobileDistractionEstimator.__new__(MobileDistractionEstimator)
    estimator.config = DMSConfig(phone_to_ear_sustain_ms=700)
    estimator.down_since_ms = None
    estimator.phone_to_ear_since_ms = None
    estimator.texting_since_ms = None

    estimator._update_timer("phone_to_ear_since_ms", 0, True)
    state = estimator._classify_from_context(HandContext(near_ear=True, confidence=0.85), False, 800)

    assert state.state == "PHONE_TO_EAR_SUSPECTED"
    assert state.confidence >= 0.85


def test_short_phone_to_ear_under_sustain_does_not_immediately_warn():
    from ind_vias_dms.vision.phone_detection import HandContext, MobileDistractionEstimator

    estimator = MobileDistractionEstimator.__new__(MobileDistractionEstimator)
    estimator.config = DMSConfig(phone_to_ear_sustain_ms=700)
    estimator.down_since_ms = None
    estimator.phone_to_ear_since_ms = None
    estimator.texting_since_ms = None

    estimator._update_timer("phone_to_ear_since_ms", 0, True)
    state = estimator._classify_from_context(HandContext(near_ear=True, near_face=True, confidence=0.85), False, 300)

    assert state.state == "HAND_NEAR_FACE"


def test_passenger_remote_hand_does_not_trigger_driver_phone_to_ear():
    from ind_vias_dms.vision.phone_detection import MobileDistractionEstimator

    class FakeEstimator(MobileDistractionEstimator):
        def process(self, frame, face_bbox, face_landmarks, gaze_zone, timestamp_ms):
            if face_bbox == (10, 10, 20, 20):
                return PlaceholderState("NO_PHONE", 0.6)
            return PlaceholderState("PHONE_TO_EAR_SUSPECTED", 0.9)

    estimator = FakeEstimator.__new__(FakeEstimator)
    estimator.config = DMSConfig()
    driver = FaceLandmarkResult(True, bbox=(10, 10, 20, 20))
    passenger = FaceLandmarkResult(True, bbox=(80, 10, 95, 30))

    driver_state, cabin_events = estimator.process_cabin(
        np.zeros((100, 100, 3), dtype=np.uint8),
        [(1, "DRIVER", driver), (2, "FRONT_PASSENGER", passenger)],
        driver_track_id=1,
        gaze_zone=GazeZone.ROAD,
        timestamp_ms=1000,
    )

    assert driver_state.state == "NO_PHONE"
    assert "PASSENGER_PHONE_TO_EAR" in cabin_events


def test_phone_to_ear_distraction_type_escalates_fast():
    fsm = DistractionFSM(DMSConfig(phone_to_ear_fast_escalation=True))

    level, kind = fsm.update(GazeZone.ROAD, 0, phone_state="PHONE_TO_EAR_SUSPECTED")

    assert level == DistractionLevel.HIGH
    assert kind == DistractionType.PHONE_TO_EAR


def test_blink_shorter_than_max_duration_does_not_trigger_drowsiness_warning():
    fsm = DrowsinessFSM(DMSConfig(blink_max_duration_ms=400))

    level = fsm.update(0.70, 0.0, 200, face_present=True, timestamp_ms=0)

    assert level == DrowsinessLevel.LOW


def test_sustained_perclos_crossing_required_for_drowsiness_high():
    fsm = DrowsinessFSM(DMSConfig(drowsiness_high_sustain_ms=2500, drowsiness_high_sustain_override_ms=2500))

    first = fsm.update(0.60, 0.0, 0, face_present=True, timestamp_ms=0)
    second = fsm.update(0.60, 0.0, 0, face_present=True, timestamp_ms=1000)
    sustained = fsm.update(0.60, 0.0, 0, face_present=True, timestamp_ms=2600)

    assert first == DrowsinessLevel.LOW
    assert second in {DrowsinessLevel.LOW, DrowsinessLevel.MEDIUM}
    assert sustained == DrowsinessLevel.HIGH


def test_microsleep_still_triggers_after_continuous_closure_threshold():
    fsm = DrowsinessFSM(DMSConfig(microsleep_closure_ms=1500, microsleep_duration_ms=1500))

    level = fsm.update(0.0, 0.0, 1500, face_present=True, timestamp_ms=1500)

    assert level == DrowsinessLevel.MICROSLEEP


def test_extreme_pose_adds_head_pose_unreliable_and_caps_gaze_confidence():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig()
    pipeline.road_calibration_source = "DEFAULT"

    assert pipeline._pose_unreliable(HeadPose(yaw_deg=72, pitch_deg=-40, roll_deg=35, confidence=0.8))
    availability = pipeline._availability(
        _face((100, 100, 300, 300)),
        EyeState(is_closed=False, confidence=0.9),
        DrowsinessLevel.NONE,
        DistractionLevel.NONE,
        no_face_duration_ms=0,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=0,
        gaze_zone=GazeZone.UNKNOWN,
        head_pose_unreliable=True,
    )

    assert availability.state == AvailabilityState.DEGRADED
    assert "HEAD_POSE_UNRELIABLE" in availability.reason_codes
    assert DMSConfig().pose_unreliable_gaze_confidence_cap == 0.30


def test_pose_unreliable_gaze_does_not_immediately_trigger_high_distraction():
    fsm = DistractionFSM(DMSConfig())

    level, kind = fsm.update(GazeZone.UNKNOWN, 0)

    assert level == DistractionLevel.NONE
    assert kind == DistractionType.NONE


def test_lost_temp_does_not_reset_road_calibration():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig()
    pipeline.road_calibration_source = "FILE"
    pipeline.last_driver_abs_yaw_deg = 35.0

    availability = pipeline._availability(
        FaceLandmarkResult(face_found=False),
        EyeState(confidence=0.0),
        DrowsinessLevel.UNKNOWN,
        DistractionLevel.UNKNOWN,
        no_face_duration_ms=500,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=0,
        gaze_zone=GazeZone.UNKNOWN,
        road_gaze_calibrated=True,
        session_state=DriverSessionState.LOST_TEMP.value,
        driver_body_state="PRESENT",
    )

    assert availability.state == AvailabilityState.DEGRADED
    assert pipeline.road_calibration_source == "FILE"


def test_calibration_file_load_reason_code_is_reported():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig()
    pipeline.road_calibration_source = "FILE"

    availability = pipeline._availability(
        _face((100, 100, 300, 300)),
        EyeState(is_closed=False, confidence=0.9),
        DrowsinessLevel.NONE,
        DistractionLevel.NONE,
        no_face_duration_ms=0,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=0,
        gaze_zone=GazeZone.ROAD,
        road_gaze_calibrated=True,
    )

    assert "ROAD_CALIBRATION_FILE_LOADED" in availability.reason_codes


def test_unknown_eye_state_pause_reason_can_be_reported():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig()
    pipeline.road_calibration_source = "DEFAULT"

    availability = pipeline._availability(
        _face((100, 100, 300, 300)),
        EyeState(is_closed=False, confidence=0.9),
        DrowsinessLevel.NONE,
        DistractionLevel.NONE,
        no_face_duration_ms=0,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=0,
        gaze_zone=GazeZone.ROAD,
        eye_state_label="UNKNOWN",
        perclos_pause_reason="PERCLOS_PAUSED_EYE_UNKNOWN",
    )

    assert "PERCLOS_PAUSED_EYE_UNKNOWN" in availability.reason_codes


def test_lost_temp_pause_reason_can_be_reported():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig()
    pipeline.last_driver_abs_yaw_deg = 35.0

    availability = pipeline._availability(
        FaceLandmarkResult(face_found=False),
        EyeState(confidence=0.0),
        DrowsinessLevel.UNKNOWN,
        DistractionLevel.UNKNOWN,
        no_face_duration_ms=500,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=0,
        gaze_zone=GazeZone.UNKNOWN,
        session_state=DriverSessionState.LOST_TEMP.value,
        driver_body_state="PRESENT",
        perclos_pause_reason="PERCLOS_PAUSED_FACE_LOST",
    )

    assert "PERCLOS_PAUSED_FACE_LOST" in availability.reason_codes


def test_nir_preprocessing_accepts_grayscale_and_bgr_frames():
    from ind_vias_dms.vision.nir_preprocess import preprocess_for_face_detection

    gray = np.full((40, 60), 80, dtype=np.uint8)
    gray_result = preprocess_for_face_detection(gray, DMSConfig())
    bgr_result = preprocess_for_face_detection(np.dstack([gray, gray, gray]), DMSConfig())

    assert gray_result.frame_bgr.shape == (40, 60, 3)
    assert bgr_result.frame_bgr.shape == (40, 60, 3)
    assert gray_result.grayscale_like is True


def test_face_proposal_nms_suppresses_duplicate_overlapping_boxes():
    from ind_vias_dms.vision.face_proposals import FaceProposal, suppress_duplicate_proposals

    strong = FaceProposal((100, 100, 220, 220), 0.9, "test")
    weak = FaceProposal((108, 108, 225, 225), 0.6, "test")

    kept = suppress_duplicate_proposals([weak, strong], 0.45, 0.08)

    assert kept == [strong]


def test_no_faces_from_frame_zero_driver_presence_absent_not_lost_temp():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig()
    pipeline.occupants = CabinOccupantManager(pipeline.config)

    state = pipeline._presence_state(False, 0, no_face_duration_ms=1200, session_state="UNKNOWN")

    assert state == PresenceState.ABSENT


def test_previous_driver_face_loss_reports_lost_temp_presence():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig()
    pipeline.occupants = CabinOccupantManager(pipeline.config)

    state = pipeline._presence_state(False, 0, no_face_duration_ms=500, session_state=DriverSessionState.LOST_TEMP.value)

    assert state == PresenceState.LOST_TEMP


def test_face_detection_no_face_does_not_mean_camera_failure():
    state = DMSState(
        dms_health=DMSHealth(
            camera_status=CameraStatus.OK,
            face_detection_status=CameraStatus.NO_FACE,
        )
    )

    assert state.dms_health.camera_status == CameraStatus.OK
    assert state.dms_health.face_detection_status == CameraStatus.NO_FACE


def test_status_dashboard_includes_face_backend_nir_and_proposals():
    labels = [label for label, _ in status_dashboard_lines(DMSState(), fps=30.0)]

    assert "Face backend" in labels
    assert "NIR mode" in labels
    assert "Face proposals" in labels


def test_downward_gaze_low_eye_openness_suppresses_effective_drowsiness():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig()

    result = pipeline._disambiguate_eye_gaze_phone(
        raw_eye_state="CLOSED",
        perclos_valid=True,
        closure_weight=1.0,
        eye_closure_duration_ms=300,
        eye_visibility=0.5,
        gaze_zone=GazeZone.DOWN,
        head_pitch_deg=25.0,
        face_found=True,
    )

    assert result["effective_eye_state"] == "UNKNOWN"
    assert result["perclos_valid"] is False
    assert result["closure_weight"] == 0.0
    assert "EYE_CLOSURE_SUPPRESSED_BY_DOWNWARD_GAZE" in result["reason_codes"]
    assert "POSSIBLE_PHONE_POSTURE" in result["reason_codes"]


def test_true_long_eye_closure_road_facing_remains_valid_for_drowsiness():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig(microsleep_closure_ms=1500)

    result = pipeline._disambiguate_eye_gaze_phone(
        raw_eye_state="CLOSED",
        perclos_valid=True,
        closure_weight=1.0,
        eye_closure_duration_ms=1600,
        eye_visibility=0.9,
        gaze_zone=GazeZone.ROAD,
        head_pitch_deg=0.0,
        face_found=True,
    )

    assert result["effective_eye_state"] == "CLOSED"
    assert result["perclos_valid"] is True
    assert result["closure_weight"] == 1.0


def test_phone_suspected_state_appears_when_downward_gaze_persists():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig(phone_gaze_offroad_suspect_ms=1200)

    state, reasons = pipeline._normalize_phone_state(
        "NO_PHONE",
        GazeZone.DOWN,
        head_pitch_deg=25.0,
        eyes_off_road_ms=1300,
        disambiguation_reasons=["POSSIBLE_PHONE_POSTURE"],
    )

    assert state in {"PHONE_DOWN_SUSPECTED", "PHONE_TEXTING_SCROLLING_SUSPECTED"}
    assert "POSSIBLE_PHONE_POSTURE" in reasons


def test_distraction_reason_codes_include_phone_suspected():
    reasons = DMSPipeline._distraction_reason_codes(
        DistractionType.PHONE_SUSPECTED,
        GazeZone.DOWN,
        "PHONE_SUSPECTED",
        ["POSSIBLE_PHONE_POSTURE"],
    )

    assert "PHONE_SUSPECTED" in reasons
    assert "POSSIBLE_PHONE_POSTURE" in reasons


def test_status_dashboard_includes_raw_effective_eye_and_phone_reason():
    labels = [label for label, _ in status_dashboard_lines(DMSState(), fps=30.0)]

    assert "Raw eyes" in labels
    assert "Effective eyes" in labels
    assert "PERCLOS reason" in labels
    assert "Phone reason" in labels


def _attention_signals(**overrides) -> AttentionSignals:
    values = {
        "timestamp_ms": 0,
        "driver_face_present": True,
        "driver_body_present": True,
        "session_state": "ACTIVE",
        "gaze_zone": GazeZone.ROAD,
        "gaze_confidence": 0.9,
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
        "eye_state": "OPEN",
        "eye_visibility": 0.9,
        "eye_closure_duration_ms": 0,
        "perclos_5s": 0.0,
        "perclos_60s": 0.0,
        "phone_state": "NO_PHONE",
        "distraction_level": DistractionLevel.NONE,
        "drowsiness_level": DrowsinessLevel.NONE,
        "head_pose_unreliable": False,
    }
    values.update(overrides)
    return AttentionSignals(**values)


def test_attention_head_down_eyes_open_is_phone_or_visual_not_microsleep():
    classifier = AttentionStateClassifier(DMSConfig())
    classifier.update(_attention_signals(timestamp_ms=0, gaze_zone=GazeZone.DOWN, pitch_deg=25.0))
    output = classifier.update(
        _attention_signals(timestamp_ms=2200, gaze_zone=GazeZone.DOWN, pitch_deg=25.0)
    )

    assert output.attention_state == AttentionState.ATTENTION_LOST
    assert output.attention_substate == AttentionSubstate.PHONE_SUSPECTED
    assert output.microsleep_candidate is False
    assert "HEAD_DOWN" in output.attention_reason_codes


def test_attention_head_down_closed_beyond_threshold_is_microsleep():
    classifier = AttentionStateClassifier(DMSConfig())

    output = classifier.update(
        _attention_signals(
            timestamp_ms=2000,
            gaze_zone=GazeZone.DOWN,
            pitch_deg=25.0,
            eye_state="CLOSED",
            eye_visibility=0.9,
            eye_closure_duration_ms=1600,
            perclos_5s=0.5,
        )
    )

    assert output.attention_state == AttentionState.ATTENTION_LOST
    assert output.attention_substate == AttentionSubstate.MICROSLEEP
    assert output.microsleep_candidate is True


def test_attention_head_down_poor_eye_visibility_is_ambiguous():
    classifier = AttentionStateClassifier(DMSConfig())
    classifier.update(
        _attention_signals(
            timestamp_ms=0,
            gaze_zone=GazeZone.DOWN,
            pitch_deg=25.0,
            eye_state="UNKNOWN",
            eye_visibility=0.2,
            head_pose_unreliable=True,
        )
    )
    output = classifier.update(
        _attention_signals(
            timestamp_ms=1200,
            gaze_zone=GazeZone.DOWN,
            pitch_deg=25.0,
            eye_state="UNKNOWN",
            eye_visibility=0.2,
            head_pose_unreliable=True,
        )
    )

    assert output.attention_substate == AttentionSubstate.AMBIGUOUS
    assert output.ambiguous_attention_loss is True
    assert "AMBIGUOUS_ATTENTION_LOSS" in output.attention_reason_codes


def test_attention_short_downward_glance_below_threshold_is_degraded_not_warning():
    classifier = AttentionStateClassifier(DMSConfig())

    output = classifier.update(_attention_signals(timestamp_ms=300, gaze_zone=GazeZone.DOWN, pitch_deg=25.0))

    assert output.attention_state == AttentionState.NORMAL
    assert output.attention_substate == AttentionSubstate.ROAD
    assert output.attention_lost_duration_ms == 0


def test_attention_enabled_false_preserves_existing_behavior():
    classifier = AttentionStateClassifier(DMSConfig(attention_state={"enabled": False}))

    output = classifier.update(_attention_signals(gaze_zone=GazeZone.DOWN, pitch_deg=30.0))

    assert output.attention_state == AttentionState.UNKNOWN
    assert output.attention_substate == AttentionSubstate.UNKNOWN
    assert output.attention_reason_codes == ["ATTENTION_STATE_DISABLED"]


def test_attention_side_profile_face_lost_with_body_adds_reason_codes():
    classifier = AttentionStateClassifier(DMSConfig())

    output = classifier.update(
        _attention_signals(
            driver_face_present=False,
            driver_body_present=True,
            session_state="LOST_TEMP",
            gaze_zone=GazeZone.UNKNOWN,
            eye_state="UNKNOWN",
            eye_visibility=0.0,
        )
    )

    assert output.attention_state == AttentionState.DEGRADED
    assert output.attention_substate == AttentionSubstate.FACE_LOST
    assert "SIDE_PROFILE_FACE_LOST" in output.attention_reason_codes
    assert "DRIVER_BODY_PRESENT_FACE_LOST" in output.attention_reason_codes


def test_attention_microsleep_overrides_phone_suspicion_in_availability():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig()
    pipeline.road_calibration_source = "DEFAULT"
    attention = AttentionStateClassifier(DMSConfig()).update(
        _attention_signals(
            gaze_zone=GazeZone.DOWN,
            pitch_deg=25.0,
            eye_state="CLOSED",
            eye_visibility=0.9,
            eye_closure_duration_ms=1600,
            perclos_5s=0.6,
            phone_state="PHONE_SUSPECTED",
        )
    )

    availability = pipeline._availability(
        _face((100, 100, 300, 300)),
        EyeState(is_closed=True, confidence=0.9),
        DrowsinessLevel.LOW,
        DistractionLevel.MEDIUM,
        no_face_duration_ms=0,
        eye_closure_duration_ms=1600,
        eyes_off_road_duration_ms=2000,
        gaze_zone=GazeZone.DOWN,
        phone_state="PHONE_SUSPECTED",
        attention=attention,
    )

    assert availability.state == AvailabilityState.UNAVAILABLE
    assert "MICROSLEEP_CANDIDATE" in availability.reason_codes


def test_status_dashboard_includes_attention_fields():
    labels = [label for label, _ in status_dashboard_lines(DMSState(), fps=30.0)]

    assert "Attention" in labels
    assert "Substate" in labels
    assert "Attn conf" in labels
    assert "Attn reason" in labels
    assert "Head-down uncertain" in labels
    assert "Pose reliable" in labels
    assert "Attn source" in labels


def test_head_down_reason_increments_head_down_duration():
    classifier = AttentionStateClassifier(DMSConfig())
    classifier.update(_attention_signals(timestamp_ms=0, gaze_zone=GazeZone.DOWN, pitch_deg=25.0))
    output = classifier.update(_attention_signals(timestamp_ms=1300, gaze_zone=GazeZone.DOWN, pitch_deg=25.0))

    assert "HEAD_DOWN" in output.attention_reason_codes
    assert output.head_down_duration_ms >= 1200


def test_head_pose_unreliable_down_posture_is_degraded_not_normal_road():
    classifier = AttentionStateClassifier(DMSConfig())
    output = classifier.update(
        _attention_signals(
            timestamp_ms=0,
            gaze_zone=GazeZone.UNKNOWN,
            pitch_deg=25.0,
            eye_visibility=0.2,
            head_pose_unreliable=True,
        )
    )

    assert output.attention_state == AttentionState.DEGRADED
    assert output.attention_substate in {
        AttentionSubstate.HEAD_POSE_UNRELIABLE,
        AttentionSubstate.VISUAL_OBSERVATION_LIMITED,
        AttentionSubstate.HEAD_DOWN_UNCERTAIN,
    }
    assert output.attention_substate != AttentionSubstate.ROAD


def test_sustained_head_down_uncertain_becomes_attention_lost():
    classifier = AttentionStateClassifier(DMSConfig(head_down_attention_lost_ms=2500))
    classifier.update(
        _attention_signals(
            timestamp_ms=0,
            pitch_deg=25.0,
            eye_visibility=0.2,
            head_pose_unreliable=True,
        )
    )
    output = classifier.update(
        _attention_signals(
            timestamp_ms=2600,
            pitch_deg=25.0,
            eye_visibility=0.2,
            head_pose_unreliable=True,
        )
    )

    assert output.attention_state == AttentionState.ATTENTION_LOST
    assert output.head_down_uncertain_duration_ms >= 2500


def test_sustained_gaze_offroad_without_adas_risk_stays_degraded_not_unavailable():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig(
        require_adas_risk_for_visual_unavailable=True,
        adas_risk_present_default=False,
        visual_distraction_unavailable_ms=10000,
    )
    pipeline.road_calibration_source = "DEFAULT"

    availability = pipeline._availability(
        _face((100, 100, 300, 300)),
        EyeState(is_closed=False, confidence=0.9),
        DrowsinessLevel.NONE,
        DistractionLevel.HIGH,
        no_face_duration_ms=0,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=12000,
        gaze_zone=GazeZone.RIGHT,
        phone_state="NO_PHONE",
        road_gaze_calibrated=True,
    )

    assert availability.state == AvailabilityState.DEGRADED


def test_visual_distraction_unavailable_requires_adas_risk_when_configured():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig(
        require_adas_risk_for_visual_unavailable=True,
        adas_risk_present_default=True,
        visual_distraction_unavailable_ms=10000,
    )
    pipeline.road_calibration_source = "DEFAULT"

    availability = pipeline._availability(
        _face((100, 100, 300, 300)),
        EyeState(is_closed=False, confidence=0.9),
        DrowsinessLevel.NONE,
        DistractionLevel.HIGH,
        no_face_duration_ms=0,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=12000,
        gaze_zone=GazeZone.RIGHT,
        phone_state="NO_PHONE",
        road_gaze_calibrated=True,
    )

    assert availability.state == AvailabilityState.UNAVAILABLE


def test_perclos_reason_does_not_contain_phone_reason():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig()

    result = pipeline._disambiguate_eye_gaze_phone(
        raw_eye_state="CLOSED",
        perclos_valid=True,
        closure_weight=1.0,
        eye_closure_duration_ms=300,
        eye_visibility=0.5,
        gaze_zone=GazeZone.DOWN,
        head_pitch_deg=25.0,
        face_found=True,
    )

    assert "POSSIBLE_PHONE_POSTURE" not in result["perclos_reason_codes"]
    assert "POSSIBLE_PHONE_POSTURE" in result["phone_reason_codes"]


def test_phone_down_suspected_triggers_after_sustained_head_down():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig(phone_down_suspect_ms=1200)

    state, reasons = pipeline._normalize_phone_state(
        "NO_PHONE",
        GazeZone.UNKNOWN,
        head_pitch_deg=0.0,
        eyes_off_road_ms=0,
        disambiguation_reasons=["POSSIBLE_PHONE_POSTURE"],
        head_down_ms=1300,
    )

    assert state in {"PHONE_DOWN_SUSPECTED", "PHONE_TEXTING_SCROLLING_SUSPECTED"}
    assert "POSSIBLE_PHONE_POSTURE" in reasons


def test_microsleep_attention_still_overrides_unavailable():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig()
    attention = AttentionStateClassifier(DMSConfig()).update(
        _attention_signals(
            eye_state="CLOSED",
            eye_visibility=0.9,
            eye_closure_duration_ms=1600,
            perclos_5s=0.6,
        )
    )

    availability = pipeline._availability(
        _face((100, 100, 300, 300)),
        EyeState(is_closed=True, confidence=0.9),
        DrowsinessLevel.MICROSLEEP,
        DistractionLevel.NONE,
        no_face_duration_ms=0,
        eye_closure_duration_ms=1600,
        eyes_off_road_duration_ms=0,
        gaze_zone=GazeZone.ROAD,
        attention=attention,
    )

    assert availability.state == AvailabilityState.UNAVAILABLE


def test_head_down_candidate_prevents_normal_attention():
    classifier = AttentionStateClassifier(DMSConfig(head_down_candidate_ms=500))
    classifier.update(
        _attention_signals(
            timestamp_ms=0,
            gaze_zone=GazeZone.UNKNOWN,
            phone_state="NO_PHONE",
            phone_reason_codes=["POSSIBLE_PHONE_POSTURE", "HEAD_DOWN", "GAZE_OFF_ROAD"],
        )
    )
    output = classifier.update(
        _attention_signals(
            timestamp_ms=650,
            gaze_zone=GazeZone.UNKNOWN,
            phone_state="NO_PHONE",
            phone_reason_codes=["POSSIBLE_PHONE_POSTURE", "HEAD_DOWN", "GAZE_OFF_ROAD"],
        )
    )

    assert output.attention_state != AttentionState.NORMAL
    assert output.attention_substate != AttentionSubstate.ROAD


def test_sustained_head_down_banner_is_distraction_warning():
    state = DMSState()
    state.attention.attention_state = AttentionState.ATTENTION_LOST
    state.attention.attention_substate = AttentionSubstate.HEAD_DOWN_DISTRACTION
    state.driver_availability.state = AvailabilityState.DEGRADED

    label, _ = banner_decision(state)

    assert label == "DISTRACTION WARNING"


def test_degraded_only_for_observation_issue_banner():
    state = DMSState()
    state.driver_availability.state = AvailabilityState.DEGRADED
    state.attention.attention_state = AttentionState.DEGRADED
    state.attention.attention_substate = AttentionSubstate.HEAD_POSE_UNRELIABLE

    label, _ = banner_decision(state)

    assert label == "DMS DEGRADED"


def test_distraction_overrides_degraded_banner():
    state = DMSState()
    state.driver_availability.state = AvailabilityState.DEGRADED
    state.attention.attention_state = AttentionState.ATTENTION_LOST
    state.attention.attention_substate = AttentionSubstate.PHONE_DOWN_SUSPECTED

    label, _ = banner_decision(state)

    assert label == "DISTRACTION WARNING"


def test_attention_unavailable_gating_without_adas_keeps_degraded():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig(
        require_adas_risk_for_attention_unavailable=True,
        adas_risk_present_default=False,
        attention_lost_unavailable_ms=10000,
    )
    attention = AttentionStateClassifier(DMSConfig()).update(
        _attention_signals(timestamp_ms=12000, gaze_zone=GazeZone.RIGHT)
    )
    attention.attention_state = AttentionState.ATTENTION_LOST
    attention.attention_substate = AttentionSubstate.VISUAL_DISTRACTION
    attention.attention_lost_duration_ms = 12000
    attention.attention_confidence = 0.9

    availability = pipeline._availability(
        _face((100, 100, 300, 300)),
        EyeState(is_closed=False, confidence=0.9),
        DrowsinessLevel.NONE,
        DistractionLevel.NONE,
        no_face_duration_ms=0,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=12000,
        gaze_zone=GazeZone.RIGHT,
        attention=attention,
    )

    assert availability.state == AvailabilityState.DEGRADED


def test_driver_absent_beyond_timeout_is_unavailable():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig(no_face_timeout_ms=1000)

    availability = pipeline._availability(
        FaceLandmarkResult(face_found=False),
        EyeState(confidence=0.0),
        DrowsinessLevel.UNKNOWN,
        DistractionLevel.UNKNOWN,
        no_face_duration_ms=1200,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=0,
        gaze_zone=GazeZone.UNKNOWN,
    )

    assert availability.state == AvailabilityState.UNAVAILABLE


def test_banner_hysteresis_holds_warning_before_clearing():
    renderer = OverlayRenderer(banner_min_hold_ms=700, state_clear_confirm_ms=800)
    warning = DMSState(timestamp_ms=0)
    warning.attention.attention_substate = AttentionSubstate.PHONE_DOWN_SUSPECTED
    warning.driver_availability.state = AvailabilityState.DEGRADED
    normal_soon = DMSState(timestamp_ms=300)
    normal_later = DMSState(timestamp_ms=1200)

    assert renderer._stable_banner(warning)[0] == "DISTRACTION WARNING"
    assert renderer._stable_banner(normal_soon)[0] == "DISTRACTION WARNING"
    assert renderer._stable_banner(normal_later)[0] == "NORMAL"


def test_return_to_road_clear_duration_returns_normal():
    classifier = AttentionStateClassifier(DMSConfig(state_clear_confirm_ms=800))
    classifier.update(_attention_signals(timestamp_ms=0, gaze_zone=GazeZone.DOWN, pitch_deg=25.0))
    classifier.update(_attention_signals(timestamp_ms=1300, gaze_zone=GazeZone.DOWN, pitch_deg=25.0))
    classifier.update(_attention_signals(timestamp_ms=2000, gaze_zone=GazeZone.ROAD, pitch_deg=0.0))
    output = classifier.update(_attention_signals(timestamp_ms=2900, gaze_zone=GazeZone.ROAD, pitch_deg=0.0))

    assert output.attention_state == AttentionState.NORMAL
    assert output.attention_substate == AttentionSubstate.ROAD


def _v02_inputs(**overrides) -> DMSV02Inputs:
    state = DMSState()
    values = {
        "timestamp_ms": 0,
        "health": state.dms_health,
        "availability": state.driver_availability,
        "drowsiness": state.drowsiness,
        "distraction_level": DistractionLevel.NONE,
        "distraction_type": DistractionType.NONE,
        "attention": state.attention,
        "phone_state": "NO_PHONE",
        "driver_present": True,
        "driver_body_present": True,
        "no_face_duration_ms": 0,
    }
    values["health"].camera_status = CameraStatus.OK
    values["health"].eye_visibility_score = 0.9
    values["availability"].state = AvailabilityState.AVAILABLE
    values.update(overrides)
    return DMSV02Inputs(**values)


def test_v02_short_head_down_maps_to_monitor_not_warning():
    attention = DMSState().attention
    attention.head_down_duration_ms = 900
    attention.attention_substate = AttentionSubstate.HEAD_DOWN_CANDIDATE

    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(_v02_inputs(attention=attention))

    assert decision.final_level == DMSV02Level.MONITOR
    assert decision.final_banner == "DMS MONITOR"


def test_v02_sustained_head_down_maps_to_distraction_warning():
    attention = DMSState().attention
    attention.head_down_duration_ms = 1700
    attention.attention_substate = AttentionSubstate.HEAD_DOWN_DISTRACTION

    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(_v02_inputs(attention=attention))

    assert decision.final_level == DMSV02Level.WARNING
    assert decision.final_banner == "DISTRACTION WARNING"


def test_v02_phone_down_danger_maps_to_danger():
    attention = DMSState().attention
    attention.phone_down_candidate_duration_ms = 3200

    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(
        _v02_inputs(attention=attention, phone_state="PHONE_DOWN_SUSPECTED")
    )

    assert decision.final_level == DMSV02Level.DANGER
    assert decision.final_banner == "DANGER"


def test_v02_low_eye_visibility_without_behavior_is_degraded():
    health = DMSHealth(camera_status=CameraStatus.OK, eye_visibility_score=0.1)

    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(_v02_inputs(health=health))

    assert decision.dms_confidence_state == DMSConfidenceState.LOW
    assert decision.final_level == DMSV02Level.DEGRADED


def test_phone_posture_sustained_head_down_produces_phone_down_suspected():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig(phone_down_suspect_ms=1500)

    state, reasons = pipeline._normalize_phone_state(
        "NO_PHONE",
        GazeZone.DOWN,
        head_pitch_deg=25.0,
        eyes_off_road_ms=1000,
        disambiguation_reasons=["POSSIBLE_PHONE_POSTURE"],
        head_down_ms=1600,
    )

    assert state in {"PHONE_DOWN_SUSPECTED", "PHONE_TEXTING_SCROLLING_SUSPECTED"}
    assert "POSSIBLE_PHONE_POSTURE" in reasons


def test_attention_phone_posture_prefers_distraction_substate():
    classifier = AttentionStateClassifier(DMSConfig())
    classifier.update(
        _attention_signals(
            timestamp_ms=0,
            gaze_zone=GazeZone.DOWN,
            pitch_deg=25.0,
            phone_state="PHONE_DOWN_SUSPECTED",
        )
    )
    output = classifier.update(
        _attention_signals(
            timestamp_ms=2600,
            gaze_zone=GazeZone.DOWN,
            pitch_deg=25.0,
            phone_state="PHONE_DOWN_SUSPECTED",
        )
    )

    assert output.attention_state == AttentionState.ATTENTION_LOST
    assert output.attention_substate == AttentionSubstate.PHONE_DOWN_SUSPECTED


def test_short_gaze_away_does_not_produce_attention_lost():
    classifier = AttentionStateClassifier(DMSConfig())

    output = classifier.update(_attention_signals(timestamp_ms=500, gaze_zone=GazeZone.RIGHT))

    assert output.attention_state == AttentionState.NORMAL


def test_sustained_gaze_away_produces_visual_distraction():
    classifier = AttentionStateClassifier(DMSConfig())
    classifier.update(_attention_signals(timestamp_ms=0, gaze_zone=GazeZone.RIGHT))
    output = classifier.update(_attention_signals(timestamp_ms=1800, gaze_zone=GazeZone.RIGHT))

    assert output.attention_state == AttentionState.ATTENTION_LOST
    assert output.attention_substate == AttentionSubstate.VISUAL_DISTRACTION


def test_non_driver_raw_proposal_without_landmarks_not_counted_as_occupant():
    manager = CabinOccupantManager(DMSConfig(driver_image_side="LEFT"))
    driver = _face_with_landmarks((100, 200, 350, 700), confidence=0.8)
    passenger_proposal = _proposal_face((650, 350, 720, 470), confidence=0.8)

    selection = manager.update([driver, passenger_proposal], (1000, 1000, 3), timestamp_ms=0)

    assert selection.driver is not None
    assert len(selection.faces) == 1
    assert selection.proposal_count == 2
    assert selection.unconfirmed_proposal_count == 1
    assert "FACE_PROPOSAL_NOT_VALIDATED" in selection.rejected_proposals[0]["reason_codes"]


def test_non_driver_proposal_under_confirm_frames_not_front_passenger():
    manager = CabinOccupantManager(DMSConfig(driver_image_side="LEFT", non_driver_confirm_frames=4))
    driver = _face_with_landmarks((100, 200, 350, 700), confidence=0.8)
    passenger = _face_with_landmarks((650, 220, 880, 760), confidence=0.9)

    selection = manager.update([driver, passenger], (1000, 1000, 3), timestamp_ms=0)

    assert len(selection.faces) == 1
    assert selection.unconfirmed_proposal_count == 1
    assert "TEMPORAL_CONFIRMATION_PENDING" in selection.rejected_proposals[0]["reason_codes"]


def test_false_headrest_like_small_box_is_rejected():
    manager = CabinOccupantManager(DMSConfig(driver_image_side="LEFT"))
    driver = _face_with_landmarks((100, 200, 350, 700), confidence=0.8)
    headrest = _face_with_landmarks((550, 350, 610, 470), confidence=0.9)

    selection = manager.update([driver, headrest], (1000, 1000, 3), timestamp_ms=500)

    assert len(selection.faces) == 1
    assert "HEADREST_LIKE_STATIC_BOX" in selection.rejected_proposals[0]["reason_codes"]


def test_occupants_state_counts_confirmed_faces_only_and_reports_proposals():
    manager = CabinOccupantManager(DMSConfig(driver_image_side="LEFT"))
    driver = _face_with_landmarks((100, 200, 350, 700), confidence=0.8)
    proposal = _proposal_face((650, 350, 720, 470), confidence=0.8)
    selection = manager.update([driver, proposal], (1000, 1000, 3), timestamp_ms=0)

    occupants = DMSPipeline._occupants_state(selection, proposal_count=2)

    assert occupants.face_count == 1
    assert occupants.confirmed_face_count == 1
    assert occupants.proposal_count == 2
    assert occupants.unconfirmed_proposal_count == 1


def test_driver_face_remains_tolerant_when_non_driver_confirmation_is_strict():
    manager = CabinOccupantManager(
        DMSConfig(driver_image_side="LEFT", non_driver_face_min_confidence=0.99, non_driver_require_landmarks=True)
    )
    driver_proposal = _face_with_landmarks((100, 200, 350, 700), confidence=0.55)
    selection = manager.update([driver_proposal], (1000, 1000, 3), timestamp_ms=0)

    assert selection.driver is not None
    assert selection.driver.zone == "DRIVER"


def test_dms4_style_false_front_passenger_proposal_is_suppressed():
    manager = CabinOccupantManager(DMSConfig(driver_image_side="LEFT"))
    driver = _face_with_landmarks((100, 200, 350, 700), confidence=0.8)
    false_passenger = _proposal_face((550, 350, 610, 470), confidence=0.85)

    selection = manager.update([driver, false_passenger], (1000, 1000, 3), timestamp_ms=500)

    assert len(selection.faces) == 1
    assert selection.faces[0].zone == "DRIVER"
    assert selection.unconfirmed_proposal_count == 1


def test_dms6_style_true_front_passenger_becomes_confirmed_after_persistence():
    config = DMSConfig(driver_image_side="LEFT", non_driver_confirm_frames=2, non_driver_confirm_time_ms=100)
    manager = CabinOccupantManager(config)
    driver = _face_with_landmarks((100, 200, 350, 700), confidence=0.8)
    passenger = _face_with_landmarks((650, 220, 880, 760), confidence=0.9)

    manager.update([driver, passenger], (1000, 1000, 3), timestamp_ms=0)
    selection = manager.update([driver, passenger], (1000, 1000, 3), timestamp_ms=150)

    assert len(selection.faces) == 2
    assert any(face.zone == "FRONT_PASSENGER" for face in selection.faces)


def test_raw_proposal_cannot_be_selected_as_driver():
    manager = CabinOccupantManager(DMSConfig(driver_image_side="LEFT"))
    proposal = _proposal_face((100, 200, 350, 700), confidence=0.95)

    selection = manager.update([proposal], (1000, 1000, 3), timestamp_ms=0)

    assert selection.driver is None
    assert selection.unconfirmed_proposal_count == 1
    assert "FACE_PROPOSAL_NOT_VALIDATED" in selection.rejected_proposals[0]["reason_codes"]


def test_eye_only_crop_is_rejected_by_face_quality_validation():
    face = _proposal_face((100, 200, 350, 700), confidence=0.95)
    face.landmarks_px = {33: (140, 320), 133: (180, 320), 263: (250, 320), 362: (290, 320)}

    quality = evaluate_face_quality(face, (1000, 1000, 3), DMSConfig())

    assert quality.is_valid_driver_face is False
    assert "EYE_ONLY_CROP_REJECTED" in quality.rejection_reason_codes


def test_partial_face_crop_is_rejected_by_face_quality_validation():
    face = _proposal_face((100, 200, 350, 700), confidence=0.95)
    face.landmarks_px = {1: (220, 360), 2: (220, 370), 4: (220, 380), 5: (220, 390)}

    quality = evaluate_face_quality(face, (1000, 1000, 3), DMSConfig())

    assert quality.is_valid_driver_face is False
    assert "PARTIAL_FACE_CROP" in quality.rejection_reason_codes


def test_rear_layer_face_inside_driver_roi_is_rejected_as_driver():
    config = DMSConfig(
        driver_image_side="LEFT",
        driver_min_face_area_norm=0.001,
        driver_min_candidate_score=0.2,
        rear_overlap_driver_reject_threshold=0.4,
    )
    manager = CabinOccupantManager(config)
    rear_like = _face((100, 120, 200, 220), confidence=0.95)

    selection = manager.update([rear_like], (1000, 1000, 3), timestamp_ms=0)

    assert selection.driver is None
    assert any("REAR_LAYER_REJECTED_AS_DRIVER" in item["reason_codes"] for item in selection.rejected_proposals)


def test_v02_distraction_warning_takes_priority_over_drowsiness_warning():
    attention = DMSState().attention
    attention.head_down_duration_ms = 1700
    attention.attention_substate = AttentionSubstate.HEAD_DOWN_DISTRACTION
    drowsiness = DMSState().drowsiness
    drowsiness.level = DrowsinessLevel.MEDIUM

    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(
        _v02_inputs(attention=attention, drowsiness=drowsiness)
    )

    assert decision.final_banner == "DISTRACTION WARNING"


def test_v022_phone_posture_sustained_becomes_distraction_warning_not_drowsiness():
    attention = DMSState().attention
    attention.head_down_duration_ms = 1400
    attention.gaze_offroad_duration_ms = 1400
    attention.phone_down_candidate_duration_ms = 1300
    attention.phone_texting_candidate_duration_ms = 1100
    attention.phone_suspicion_candidate = True
    attention.attention_reason_codes = [
        "HEAD_DOWN",
        "GAZE_OFF_ROAD",
        "POSSIBLE_PHONE_POSTURE",
        "PHONE_TEXTING_SCROLLING_SUSPECTED",
    ]
    drowsiness = DMSState().drowsiness
    drowsiness.level = DrowsinessLevel.MEDIUM
    drowsiness.effective_eye_state = "UNKNOWN"
    drowsiness.perclos_valid = False
    drowsiness.perclos_validity_reason_codes = ["LOW_EYE_VISIBILITY"]

    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(
        _v02_inputs(attention=attention, drowsiness=drowsiness, phone_state="PHONE_TEXTING_SCROLLING_SUSPECTED")
    )

    assert decision.final_banner == "DISTRACTION WARNING"
    assert decision.drowsiness_state != "DROWSY"
    assert "DROWSINESS_SUPPRESSED_POSSIBLE_PHONE" in decision.reason_codes
    assert "PHONE_TEXTING_SCROLLING_SUSPECTED" in decision.reason_codes


def test_v022_gaze_offroad_alone_cannot_trigger_drowsiness_warning():
    attention = DMSState().attention
    attention.gaze_offroad_duration_ms = 100
    attention.attention_reason_codes = ["GAZE_OFF_ROAD"]
    drowsiness = DMSState().drowsiness
    drowsiness.level = DrowsinessLevel.MEDIUM
    drowsiness.perclos_valid = False
    drowsiness.effective_eye_state = "UNKNOWN"
    drowsiness.perclos_validity_reason_codes = ["PERCLOS_PAUSED_EYE_UNKNOWN"]

    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(_v02_inputs(attention=attention, drowsiness=drowsiness))

    assert decision.final_banner != "DROWSINESS WARNING"
    assert decision.drowsiness_state != "DROWSY"
    assert "DROWSINESS_SUPPRESSED_NO_VALID_EYE_EVIDENCE" in decision.reason_codes


def test_v022_valid_eye_closure_still_triggers_drowsiness_warning():
    drowsiness = DMSState().drowsiness
    drowsiness.level = DrowsinessLevel.NONE
    drowsiness.effective_eye_state = "CLOSED"
    drowsiness.eye_visibility_score = 0.9
    drowsiness.perclos_valid = True
    drowsiness.perclos_valid_time_5s_ms = 2500
    drowsiness.perclos_5s = 0.3
    drowsiness.eye_closure_duration_ms = 900
    drowsiness.perclos_validity_reason_codes = ["VALID"]

    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(_v02_inputs(drowsiness=drowsiness))

    assert decision.final_banner == "DROWSINESS WARNING"
    assert decision.drowsiness_state == "DROWSY"
    assert "DROWSINESS_VALID_PERCLOS" in decision.reason_codes


def test_v022_unknown_eye_suppresses_drowsiness_escalation():
    drowsiness = DMSState().drowsiness
    drowsiness.level = DrowsinessLevel.HIGH
    drowsiness.effective_eye_state = "UNKNOWN"
    drowsiness.perclos_valid = False
    drowsiness.eye_visibility_score = 0.2
    drowsiness.perclos_validity_reason_codes = ["PERCLOS_PAUSED_EYE_UNKNOWN"]

    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(_v02_inputs(drowsiness=drowsiness))

    assert decision.drowsiness_state == "NONE"
    assert decision.final_banner != "DROWSINESS WARNING"


def test_v022_phone_to_ear_path_still_warns_as_distraction():
    attention = DMSState().attention
    attention.attention_substate = AttentionSubstate.PHONE_TO_EAR_SUSPECTED
    attention.phone_suspicion_candidate = True

    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(
        _v02_inputs(attention=attention, phone_state="PHONE_TO_EAR_SUSPECTED")
    )

    assert decision.final_banner == "DISTRACTION WARNING"
    assert decision.distraction_state == "PHONE_SUSPECTED"


def test_v022_normal_suppressed_when_attention_is_ambiguous():
    attention = DMSState().attention
    attention.attention_state = AttentionState.DEGRADED
    attention.attention_substate = AttentionSubstate.AMBIGUOUS
    attention.ambiguous_attention_loss = True
    attention.attention_lost_duration_ms = 600
    attention.attention_reason_codes = ["AMBIGUOUS_ATTENTION_LOSS"]

    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(_v02_inputs(attention=attention))

    assert decision.final_banner == "DMS MONITOR"
    assert "AMBIGUOUS_TO_MONITOR" in decision.reason_codes


def test_v022_low_head_motion_alone_does_not_trigger_degraded():
    attention = DMSState().attention
    attention.low_head_motion = True
    attention.attention_reason_codes = ["LOW_HEAD_MOTION"]

    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(_v02_inputs(attention=attention))

    assert decision.final_banner == "NORMAL"
    assert "LOW_HEAD_MOTION" in decision.reason_codes


def test_v022_debug_record_exposes_raw_and_classification_reason_namespaces():
    state = DMSState()
    state.dms_v02.raw_observation_codes = ["RAW_HEAD_DOWN"]
    state.dms_v02.classification_reason_codes = ["PHONE_DOWN_SUSPECTED"]
    state.dms_v02.reason_codes = ["PHONE_DOWN_SUSPECTED"]

    record = build_debug_record(state, {"face": FaceLandmarkResult(False)}, np.zeros((20, 20, 3), dtype=np.uint8))

    assert record["raw_observation_codes"] == ["RAW_HEAD_DOWN"]
    assert record["classification_reason_codes"] == ["PHONE_DOWN_SUSPECTED"]


def test_status_dashboard_includes_driver_face_validation_fields():
    labels = [label for label, _ in status_dashboard_lines(DMSState(), fps=30.0)]

    assert "Driver seat zone" in labels
    assert "Driver slot conf" in labels
    assert "Driver validation" in labels
    assert "Face quality" in labels
    assert "Face reject" in labels


def test_debug_record_flags_normal_with_active_phone_evidence():
    state = DMSState()
    state.dms_v02.final_banner = "NORMAL"
    state.phone_use.reason_codes = ["POSSIBLE_PHONE_POSTURE", "HEAD_DOWN"]

    record = build_debug_record(state, {"face": FaceLandmarkResult(False)}, np.zeros((20, 20, 3), dtype=np.uint8))

    assert "NORMAL_WITH_ACTIVE_DISTRACTION_EVIDENCE" in record["contradiction_flags"]


def test_face_loss_with_driver_body_present_is_not_unavailable():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig(no_face_timeout_ms=1000)
    pipeline.last_driver_abs_yaw_deg = 30.0

    availability = pipeline._availability(
        FaceLandmarkResult(face_found=False),
        EyeState(confidence=0.0),
        DrowsinessLevel.UNKNOWN,
        DistractionLevel.UNKNOWN,
        no_face_duration_ms=3000,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=0,
        gaze_zone=GazeZone.UNKNOWN,
        session_state=DriverSessionState.LOST_TEMP.value,
        driver_body_state="PRESENT",
        driver_observability=DriverObservabilityState.UNOBSERVABLE_TEMP.value,
    )

    assert availability.state == AvailabilityState.DEGRADED
    assert "FACE_LOSS_NOT_DRIVER_UNAVAILABLE" in availability.reason_codes
    assert "DRIVER_UNAVAILABLE_SUPPRESSED_BODY_PRESENT" in availability.reason_codes


def test_v02_body_present_unobservable_suppresses_driver_unavailable_banner():
    availability = DMSState().driver_availability
    availability.state = AvailabilityState.UNAVAILABLE
    availability.reason_codes = ["DRIVER_FACE_LOST_TEMP", "DRIVER_BODY_PRESENT_FACE_LOST"]

    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(
        _v02_inputs(
            availability=availability,
            driver_present=False,
            driver_body_present=True,
            no_face_duration_ms=5000,
            driver_observability=DriverObservabilityState.UNOBSERVABLE_TEMP.value,
        )
    )

    assert decision.driver_availability_state == "DEGRADED"


def test_v023_road_axis_reference_computes_relative_yaw():
    road_axis = RoadAxisHeadPoseReference(DMSConfig())
    road_axis.calibrate(10.0, 2.0, 1.0, timestamp_ms=0, source="RUNTIME", confidence=0.9)

    pose = road_axis.update(
        HeadPose(yaw_deg=52.0, pitch_deg=3.0, roll_deg=4.0, confidence=0.8),
        timestamp_ms=100,
        face_present=True,
        pose_reliable=True,
        gaze_estimate=GazeEstimate(GazeZone.LEFT, 0.8),
    )

    assert pose.relative_yaw_deg == 42.0
    assert pose.relative_pitch_deg == 1.0
    assert pose.relative_roll_deg == 3.0
    assert pose.yaw_classifiable is True
    assert pose.head_angle_from_road_deg > 35.0
    assert pose.head_pose_vector_quality == 0.8


def test_v023_status_dashboard_includes_head_angle_line():
    state = DMSState()
    state.gaze.head_pose_raw_yaw_deg = 45.0
    state.gaze.head_pose_raw_pitch_deg = 4.0
    state.gaze.head_pose_raw_roll_deg = 1.0
    state.gaze.relative_yaw_deg = 42.0
    state.gaze.relative_pitch_deg = 2.0
    state.gaze.relative_roll_deg = 0.5
    state.gaze.head_angle_from_road_deg = 42.2

    lines = dict(status_dashboard_lines(state, fps=30.0))

    assert "Head angle" in lines
    assert "raw yaw/pitch/roll" in lines["Head angle"]
    assert "road-relative yaw/pitch/roll" in lines["Head angle"]
    assert "road-vector angle" in lines["Head angle"]
    assert lines["Head vector quality"] == "0.00"


def test_v023_sustained_relative_side_yaw_becomes_warning():
    config = DMSConfig()
    road_axis = RoadAxisHeadPoseReference(config)
    road_axis.calibrate(0.0, 0.0, 0.0, timestamp_ms=0, source="RUNTIME", confidence=1.0)

    road_axis.update(
        HeadPose(yaw_deg=45.0, pitch_deg=0.0, roll_deg=0.0, confidence=0.8),
        timestamp_ms=0,
        face_present=True,
        pose_reliable=True,
        gaze_estimate=GazeEstimate(GazeZone.LEFT, 0.8),
    )
    pose = road_axis.update(
        HeadPose(yaw_deg=45.0, pitch_deg=0.0, roll_deg=0.0, confidence=0.8),
        timestamp_ms=1200,
        face_present=True,
        pose_reliable=True,
        gaze_estimate=GazeEstimate(GazeZone.LEFT, 0.8),
    )
    classifier = AttentionStateClassifier(config)
    attention = classifier.update(
        _attention_signals(
            timestamp_ms=1200,
            yaw_deg=pose.relative_yaw_deg,
            relative_yaw_deg=pose.relative_yaw_deg,
            side_glance_state=pose.side_glance_state,
            side_glance_duration_ms=pose.side_glance_duration_ms,
            yaw_classifiable=pose.yaw_classifiable,
        )
    )
    decision = DMSV02DecisionMatrix(config).evaluate(_v02_inputs(attention=attention))

    assert attention.attention_substate == AttentionSubstate.SIDE_PROFILE_ATTENTION_LOSS
    assert decision.final_banner == "DISTRACTION WARNING"
    assert "SIDE_GLANCE_DISTRACTION_WARNING" in decision.reason_codes


def test_v023_relative_side_yaw_monitor_suppresses_degraded():
    config = DMSConfig()
    attention = DMSState().attention
    attention.attention_state = AttentionState.DEGRADED
    attention.attention_substate = AttentionSubstate.SIDE_GLANCE_RIGHT
    attention.side_glance_duration_ms = 400
    attention.relative_yaw_deg = 37.0
    attention.yaw_classifiable = True
    attention.attention_reason_codes = ["RELATIVE_YAW_SIDE_GLANCE", "SIDE_GLANCE_MONITOR"]

    low_health = DMSHealth(camera_status=CameraStatus.OK, face_detection_status=CameraStatus.OK)
    low_health.eye_visibility_score = 0.2
    decision = DMSV02DecisionMatrix(config).evaluate(
        _v02_inputs(health=low_health, attention=attention)
    )

    assert decision.final_banner == "DMS MONITOR"
    assert decision.final_level == DMSV02Level.MONITOR


def test_v023_side_profile_context_is_not_face_lost_first():
    classifier = AttentionStateClassifier(DMSConfig())
    output = classifier.update(
        _attention_signals(
            timestamp_ms=1500,
            driver_face_present=False,
            driver_body_present=True,
            session_state="LOST_TEMP",
            gaze_zone=GazeZone.UNKNOWN,
            eye_state="UNKNOWN",
            eye_visibility=0.0,
            relative_yaw_deg=46.0,
            side_glance_state="SIDE_PROFILE_ATTENTION_LOSS",
            side_glance_duration_ms=1300,
            side_profile_context_active=True,
        )
    )

    assert output.attention_substate == AttentionSubstate.SIDE_PROFILE_ATTENTION_LOSS
    assert "SIDE_PROFILE_CLASSIFIED_AS_ATTENTION_NOT_FACE_LOSS" in output.attention_reason_codes
    assert "FACE_LOST" not in output.attention_substate.value


def test_v023_phone_object_model_missing_does_not_crash():
    estimator = MobileDistractionEstimator(
        DMSConfig(
            mobile_distraction_enabled=False,
            phone_object_detection={
                "enabled": True,
                "model_path": "models/weights/does_not_exist.onnx",
                "allow_missing_model": True,
            },
        )
    )

    state, events = estimator.process_cabin(np.zeros((32, 32, 3), dtype=np.uint8), [], None, GazeZone.UNKNOWN, 0)

    assert state.state == "UNKNOWN"
    assert events == []
    assert estimator.last_phone_object.backend_status == "MODEL_MISSING"


def test_v023_learning_memory_writes_review_event(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = LearningMemoryWriter(str(path), DMSConfig())
    state = DMSState(frame_id=7, timestamp_ms=250)
    state.driver_identity.driver_session_id = "D1"
    state.dms_v02.final_banner = "DMS DEGRADED"
    state.dms_v02.classification_reason_codes = ["DEGRADED_ACTIVE_OBSERVATION_FAILURE"]
    state.attention.attention_substate = AttentionSubstate.SIDE_PROFILE_TRACKED
    writer.write_frame(state, {}, np.zeros((20, 20, 3), dtype=np.uint8))
    writer.close()

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["schema_version"] == "dms_learning_event_v1"
    assert record["learning_memory_status"] == "EVENT_RECORDED"
    assert record["online_model_update_enabled"] is False
    assert record["session_id"] == "D1"
    assert record["event_type"] == "SIDE_PROFILE_FALSE_DEGRADED"


def test_v02_temp_unobservable_without_body_still_waits_before_unavailable():
    availability = DMSState().driver_availability
    availability.state = AvailabilityState.DEGRADED
    availability.reason_codes = ["DRIVER_OBSERVABILITY_TEMP_LOST"]

    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(
        _v02_inputs(
            availability=availability,
            driver_present=False,
            driver_body_present=False,
            no_face_duration_ms=2500,
            driver_observability=DriverObservabilityState.UNOBSERVABLE_TEMP.value,
        )
    )

    assert decision.driver_availability_state == "DEGRADED"
    assert decision.final_banner == "DMS DEGRADED"


def test_degraded_entry_and_exit_hysteresis_reduce_frame_flicker():
    matrix = DMSV02DecisionMatrix(
        DMSConfig(
            degraded_entry_sustain_ms=200,
            degraded_recovery_stable_ms=800,
            degraded_exit_hold_ms=700,
            min_banner_hold_ms=700,
        )
    )
    high = DMSHealth(camera_status=CameraStatus.OK, eye_visibility_score=0.9)
    low = DMSHealth(camera_status=CameraStatus.OK, eye_visibility_score=0.1)

    assert matrix.evaluate(_v02_inputs(timestamp_ms=0, health=high)).final_banner == "NORMAL"
    assert matrix.evaluate(_v02_inputs(timestamp_ms=33, health=low)).final_banner == "NORMAL"
    assert matrix.evaluate(_v02_inputs(timestamp_ms=260, health=low)).final_banner == "DMS DEGRADED"
    assert matrix.evaluate(_v02_inputs(timestamp_ms=400, health=high)).final_banner == "DMS DEGRADED"
    assert matrix.evaluate(_v02_inputs(timestamp_ms=1300, health=high)).final_banner == "NORMAL"


def test_degraded_recovery_clears_immediately_when_driver_observation_is_strong():
    matrix = DMSV02DecisionMatrix(
        DMSConfig(
            degraded_recovery_stable_ms=1000,
            degraded_exit_hold_ms=800,
            min_banner_hold_ms=800,
        )
    )
    low = DMSHealth(
        camera_status=CameraStatus.OK,
        face_detection_status=CameraStatus.NO_FACE,
        eye_visibility_score=0.0,
        face_visibility_score=0.0,
    )
    recovered = DMSHealth(
        camera_status=CameraStatus.OK,
        face_detection_status=CameraStatus.OK,
        eye_visibility_score=0.85,
        face_visibility_score=0.95,
    )
    drowsiness = DMSState().drowsiness
    drowsiness.effective_eye_state = "OPEN"
    attention = DMSState().attention
    attention.attention_state = AttentionState.NORMAL
    attention.attention_substate = AttentionSubstate.ROAD
    availability = DMSState().driver_availability
    availability.state = AvailabilityState.AVAILABLE

    assert matrix.evaluate(_v02_inputs(timestamp_ms=0, health=low)).final_banner == "DMS DEGRADED"
    decision = matrix.evaluate(
        _v02_inputs(
            timestamp_ms=100,
            health=recovered,
            drowsiness=drowsiness,
            attention=attention,
            availability=availability,
            driver_present=True,
            driver_observability=DriverObservabilityState.OBSERVABLE.value,
        )
    )

    assert decision.final_banner == "NORMAL"
    assert decision.final_decision_path == "NORMAL > road/available"


def test_debug_record_uses_final_classification_reasons_not_raw_attention_reasons():
    state = DMSState()
    state.attention.attention_reason_codes = ["GAZE_OFF_ROAD"]
    state.dms_v02.reason_codes = ["ROAD_GAZE_CONFIRMED"]
    state.dms_v02.classification_reason_codes = ["ROAD_GAZE_CONFIRMED"]
    state.dms_v02.raw_observation_codes = ["RAW_GAZE_OFF_ROAD"]

    record = build_debug_record(state, {"face": FaceLandmarkResult(False)}, np.zeros((20, 20, 3), dtype=np.uint8))

    assert record["reason_codes"] == ["ROAD_GAZE_CONFIRMED"]
    assert record["raw_observation_codes"] == ["RAW_GAZE_OFF_ROAD"]


def test_normal_final_decision_does_not_include_gaze_offroad_reason():
    attention = DMSState().attention
    attention.attention_reason_codes = ["GAZE_OFF_ROAD"]
    attention.attention_substate = AttentionSubstate.ROAD

    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(_v02_inputs(attention=attention))

    assert decision.final_banner == "NORMAL"
    assert "GAZE_OFF_ROAD" not in decision.reason_codes
    assert "ROAD_GAZE_CONFIRMED" in decision.reason_codes


def test_short_glance_uses_short_glance_reason_not_classification_gaze_offroad():
    attention = DMSState().attention
    attention.gaze_offroad_duration_ms = 1600
    attention.attention_reason_codes = ["GAZE_OFF_ROAD"]

    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(_v02_inputs(attention=attention))

    assert decision.final_banner == "DMS MONITOR"
    assert "SHORT_GLANCE_AWAY" in decision.reason_codes
    assert "GAZE_OFF_ROAD" not in decision.reason_codes


def test_sustained_gaze_away_uses_sustained_reason_and_warning():
    attention = DMSState().attention
    attention.gaze_offroad_duration_ms = 2300
    attention.attention_reason_codes = ["GAZE_OFF_ROAD"]

    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(_v02_inputs(attention=attention))

    assert decision.final_banner == "DISTRACTION WARNING"
    assert "GAZE_OFF_ROAD_SUSTAINED" in decision.reason_codes
    assert "VISUAL_ATTENTION_LOSS" in decision.reason_codes


def test_drowsiness_unknown_resolves_to_none_after_valid_open_eye_window():
    drowsiness = DMSState().drowsiness
    drowsiness.level = DrowsinessLevel.UNKNOWN
    drowsiness.effective_eye_state = "OPEN"
    drowsiness.eye_calibration_state = "CALIBRATED"
    drowsiness.eye_visibility_score = 0.85
    drowsiness.perclos_valid_time_5s_ms = 2500

    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(_v02_inputs(drowsiness=drowsiness))

    assert decision.drowsiness_state == "NONE"


def test_bgr_day_threshold_profile_fields_are_visible_in_status():
    state = DMSState()
    state.dms_health.nir_mode = "BGR"
    state.dms_health.input_color_mode = "BGR"
    state.dms_health.active_eye_threshold_profile = "BGR_DAY"
    state.dms_health.active_perclos_profile = "BGR_DAY"
    state.dms_health.nir_reason_codes = ["NIR_NOT_REQUIRED_BGR_INPUT", "BGR_DAY_PROFILE_ACTIVE"]

    lines = dict(status_dashboard_lines(state, fps=30.0))

    assert lines["Input mode"] == "BGR"
    assert lines["Threshold profile"] == "BGR_DAY"
    assert "BGR_DAY_PROFILE_ACTIVE" in lines["NIR reason"]


def test_glasses_low_eye_confidence_degrades_not_unavailable():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig()
    pipeline.road_calibration_source = "DEFAULT"

    availability = pipeline._availability(
        _face((100, 100, 300, 300)),
        EyeState(is_closed=False, confidence=0.2),
        DrowsinessLevel.UNKNOWN,
        DistractionLevel.NONE,
        no_face_duration_ms=0,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=0,
        gaze_zone=GazeZone.UNKNOWN,
    )

    assert availability.state == AvailabilityState.DEGRADED
    assert "GLASSES_REFLECTION_LOW_EYE_CONFIDENCE" in availability.reason_codes


def test_webcam_driver_zone_proposal_failed_landmarks_not_unavailable():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig()
    pipeline.road_calibration_source = "DEFAULT"
    pipeline.face_backend = type("Backend", (), {"last_nir_mode": "BGR"})()

    availability = pipeline._availability(
        FaceLandmarkResult(False),
        EyeState(is_closed=False, confidence=0.0),
        DrowsinessLevel.UNKNOWN,
        DistractionLevel.UNKNOWN,
        no_face_duration_ms=5000,
        eye_closure_duration_ms=0,
        eyes_off_road_duration_ms=0,
        gaze_zone=GazeZone.UNKNOWN,
        occupant_count=0,
        driver_body_state="UNKNOWN",
        driver_observability=DriverObservabilityState.PARTIALLY_OBSERVABLE.value,
        driver_proposal_visible=True,
        driver_proposal_reason_codes=["DRIVER_ZONE_PROPOSAL_PRESENT"],
        driver_proposal_visible_ms=800,
    )

    assert availability.state == AvailabilityState.DEGRADED
    assert "DRIVER_UNAVAILABLE_SUPPRESSED_PROPOSAL_PRESENT" in availability.reason_codes
    assert "PROPOSAL_ONLY_NOT_DRIVER_ABSENT" in availability.reason_codes


def test_webcam_proposal_visible_presence_state():
    pipeline = DMSPipeline.__new__(DMSPipeline)
    pipeline.config = DMSConfig()
    pipeline.occupants = type("Occupants", (), {"driver_last_seen_ms": None})()

    presence = pipeline._presence_state(
        face_found=False,
        occupant_count=0,
        no_face_duration_ms=2500,
        session_state="UNKNOWN",
        driver_proposal_visible=True,
    )

    assert presence == PresenceState.PROPOSAL_VISIBLE


def test_webcam_proposal_only_driver_maps_to_monitor_not_critical():
    availability = DMSState().driver_availability
    availability.state = AvailabilityState.DEGRADED
    availability.reason_codes = [
        "DRIVER_ZONE_PROPOSAL_PRESENT",
        "FACE_PROPOSAL_LANDMARK_FAILED",
        "DRIVER_UNAVAILABLE_SUPPRESSED_PROPOSAL_PRESENT",
    ]
    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(
        _v02_inputs(
            availability=availability,
            driver_present=False,
            driver_proposal_visible=True,
            driver_track_held=True,
            no_face_duration_ms=3000,
            driver_observability=DriverObservabilityState.PARTIALLY_OBSERVABLE.value,
        )
    )

    assert decision.final_banner == "DMS MONITOR"
    assert decision.driver_availability_state == "DEGRADED"
    assert "NORMAL_ALLOWED_PROPOSAL_VISIBLE_HELD" in decision.reason_codes


def test_webcam_normal_blocked_when_attention_degraded():
    attention = DMSState().attention
    attention.attention_state = AttentionState.DEGRADED
    decision = DMSV02DecisionMatrix(DMSConfig()).evaluate(_v02_inputs(attention=attention))

    assert decision.final_banner == "DMS MONITOR"


def test_webcam_raw_face_proposal_hidden_by_default():
    from ind_vias_dms.vision.face_proposals import FaceProposal

    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    proposal = FaceProposal((10, 65, 50, 95), 0.9, "unit", "REAR")
    rendered = OverlayRenderer().draw_video_overlay(
        frame,
        DMSState(),
        FaceLandmarkResult(False),
        HeadPose(confidence=0.0),
        fps=30.0,
        draw_panel=False,
        face_proposals=[proposal],
        show_debug_proposal_boxes=False,
    )

    assert np.array_equal(rendered[60:100], frame[60:100])


def test_webcam_debug_trace_flags_unavailable_with_proposal():
    state = DMSState()
    state.driver_availability.state = AvailabilityState.UNAVAILABLE
    state.driver_identity.driver_proposal_visible = True
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    record = build_debug_record(state, {"face": FaceLandmarkResult(False)}, frame)

    assert "DRIVER_UNAVAILABLE_WITH_PROPOSAL" in record["contradiction_flags"]
    assert "PROPOSAL_ONLY_DRIVER_FRAME" in record["contradiction_flags"]
