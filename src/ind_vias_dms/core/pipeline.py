from __future__ import annotations

import numpy as np

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.driver_body import DriverBodyPresenceFallback
from ind_vias_dms.core.driver_session import DriverSessionManager, DriverSessionState
from ind_vias_dms.core.occupant_manager import (
    CabinOccupantManager,
    OccupantSelection,
    suppress_duplicate_faces,
)
from ind_vias_dms.core.occupancy import CabinOccupancyManager
from ind_vias_dms.core.road_axis import RoadAxisHeadPoseReference
from ind_vias_dms.core.timing import FPSMeter
from ind_vias_dms.core.types import (
    AttentionOutput,
    AttentionState,
    AttentionSubstate,
    AvailabilityState,
    CameraStatus,
    DMSHealth,
    DMSState,
    DistractionLevel,
    DistractionState,
    DistractionType,
    DriverAvailability,
    DriverIdentityState,
    DriverObservability,
    DriverObservabilityState,
    DriverPresence,
    DriverReadinessScore,
    DrowsinessLevel,
    DrowsinessState,
    GazeState,
    GazeZone,
    OccupantFace,
    OccupantsState,
    PhoneUseState,
    PresenceState,
    RiskLevel,
)
from ind_vias_dms.temporal.blink_tracker import BlinkTracker
from ind_vias_dms.temporal.attention_state import AttentionSignals, AttentionStateClassifier
from ind_vias_dms.temporal.distraction_fsm import DistractionFSM
from ind_vias_dms.temporal.drowsiness_fsm import DrowsinessFSM
from ind_vias_dms.temporal.dms_v02_decision import DMSV02DecisionMatrix, DMSV02Inputs
from ind_vias_dms.temporal.eye_temporal import EyeTemporalTracker
from ind_vias_dms.temporal.head_pose_smoother import HeadPoseSmoother
from ind_vias_dms.temporal.perclos import PERCLOSTracker
from ind_vias_dms.vision.eye_state import EyeState, EyeStateEstimator
from ind_vias_dms.vision.face_landmarks import FaceLandmarkBackend, FaceLandmarkResult
from ind_vias_dms.vision.gaze import GazeEstimate, GazeEstimator
from ind_vias_dms.vision.head_pose import HeadPose, HeadPoseEstimator
from ind_vias_dms.vision.phone_detection import MobileDistractionEstimator
from ind_vias_dms.vision.seatbelt import SeatbeltDetectionPlaceholder


DOWNWARD_GAZE_ZONES = {GazeZone.DOWN, GazeZone.PHONE_DOWN}


class DMSPipeline:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.face_backend = FaceLandmarkBackend(config.face_backend, config.max_num_faces, config)
        self.occupants = CabinOccupantManager(config)
        self.driver_session = DriverSessionManager(config, self.occupants)
        self.occupancy = CabinOccupancyManager(config)
        self.driver_body = DriverBodyPresenceFallback(config)
        self.head_pose_estimator = HeadPoseEstimator()
        self.head_pose_smoother = HeadPoseSmoother(
            config.head_pose_smoothing_alpha,
            config.head_pose_outlier_threshold_deg,
            config.head_pose_min_confidence,
        )
        self.eye_state_estimator = EyeStateEstimator(config.eye_closed_threshold)
        self.eye_temporal = EyeTemporalTracker(config)
        self.gaze_estimator = GazeEstimator(config)
        self.road_axis = RoadAxisHeadPoseReference(config)
        self.blink_tracker = BlinkTracker(config.blink_min_duration_ms)
        self.perclos_short = PERCLOSTracker(config.perclos_short_window_s)
        self.perclos_long = PERCLOSTracker(config.perclos_long_window_s)
        self.drowsiness_fsm = DrowsinessFSM(config)
        self.distraction_fsm = DistractionFSM(config)
        self.attention_classifier = AttentionStateClassifier(config)
        self.v02_decision = DMSV02DecisionMatrix(config)
        self.phone_detector = MobileDistractionEstimator(config)
        self.seatbelt_detector = SeatbeltDetectionPlaceholder()
        self.fps_meter = FPSMeter()
        self.no_face_since_ms: int | None = None
        self.eyes_off_road_since_ms: int | None = None
        self.head_down_since_ms: int | None = None
        self.last_driver_abs_yaw_deg = 0.0
        self.last_reliable_gaze_away_ms: int | None = None
        self.road_calibration_source = "DEFAULT"
        self.valid_eye_observation_since_ms: int | None = None
        self.driver_proposal_visible_since_ms: int | None = None
        self._last_stable_head_pose: HeadPose | None = None
        self._last_stable_pose_ms: int | None = None
        self._pose_hold_until_ms: int | None = None
        self._pose_hold_reason_codes: list[str] = []

    def process(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
        frame_id: int,
    ) -> tuple[DMSState, dict[str, object]]:
        faces = suppress_duplicate_faces(
            self.face_backend.process_all(frame),
            self.config.duplicate_face_iou_threshold,
            self.config.duplicate_face_center_distance_threshold,
        )
        selection = self.occupants.update(faces, frame.shape, timestamp_ms)
        session = self.driver_session.update(selection.driver, timestamp_ms)
        driver_track_changed = selection.driver_track_changed
        face = selection.driver.observation if selection.driver is not None else FaceLandmarkResult(False)
        driver_proposal = self._driver_proposal_context(frame.shape)
        driver_proposal_visible = bool(driver_proposal["visible"])
        proposal_only_driver_visible = driver_proposal_visible and not bool(face.landmarks_px)
        if proposal_only_driver_visible:
            if self.driver_proposal_visible_since_ms is None:
                self.driver_proposal_visible_since_ms = timestamp_ms
        else:
            self.driver_proposal_visible_since_ms = None
        driver_proposal_visible_ms = (
            timestamp_ms - self.driver_proposal_visible_since_ms
            if self.driver_proposal_visible_since_ms is not None
            else 0
        )
        driver_session_held = session.session_state == DriverSessionState.LOST_TEMP
        body_state = self.driver_body.update(face, timestamp_ms, driver_session_held)
        if face.face_found:
            self.no_face_since_ms = None
        elif self.no_face_since_ms is None:
            self.no_face_since_ms = timestamp_ms

        no_face_duration_ms = (
            timestamp_ms - self.no_face_since_ms if self.no_face_since_ms is not None else 0
        )
        if no_face_duration_ms >= self.config.no_face_timeout_ms:
            self.head_pose_smoother.reset()
        if session.session_state in {DriverSessionState.LOST_LONG, DriverSessionState.SWAPPED}:
            self._reset_driver_temporal()
            self.driver_body.reset()
        elif driver_track_changed and not session.reassociated:
            self._reset_driver_temporal()
        raw_head_pose = self.head_pose_estimator.estimate(face.landmarks_px, frame.shape)
        head_pose = self.head_pose_smoother.update(raw_head_pose) if face.face_found else raw_head_pose
        eye_state = self.eye_state_estimator.estimate(face.landmarks_px)
        pose_unreliable = face.face_found and self._pose_unreliable(head_pose)
        pose_held = False
        pose_hold_codes: list[str] = []
        if self._should_hold_previous_pose(face, eye_state, head_pose, timestamp_ms):
            head_pose = self._last_stable_head_pose or head_pose
            pose_unreliable = False
            pose_held = True
            self._pose_hold_until_ms = timestamp_ms + self.config.pose_jump_hold_ms
            pose_hold_codes = [
                "HEAD_POSE_UNREALISTIC_JUMP",
                "HEAD_POSE_HELD_PREVIOUS_STABLE",
                "HEAD_POSE_DEGRADED_SUPPRESSED_VALID_FACE",
                "POSE_JUMP_FILTER_ACTIVE",
                "POSE_VISUAL_CONTRADICTION_VALID_FACE",
            ]
        elif self._pose_hold_until_ms is not None and timestamp_ms <= self._pose_hold_until_ms:
            if self._last_stable_head_pose is not None and self._valid_face_for_pose_hold(face, eye_state):
                head_pose = self._last_stable_head_pose
                pose_unreliable = False
                pose_held = True
                pose_hold_codes = ["HEAD_POSE_HELD_PREVIOUS_STABLE", "POSE_JUMP_FILTER_ACTIVE"]
        else:
            self._pose_hold_until_ms = None
            if pose_unreliable:
                pose_hold_codes = ["POSE_HOLD_EXPIRED"] if self._last_stable_head_pose is not None else []

        if (
            face.face_found
            and not pose_unreliable
            and head_pose.confidence >= self.config.head_pose_min_confidence
        ):
            self.last_driver_abs_yaw_deg = abs(head_pose.yaw_deg)
            if not pose_held:
                self._last_stable_head_pose = head_pose
                self._last_stable_pose_ms = timestamp_ms
        self._pose_hold_reason_codes = pose_hold_codes
        eye_temporal = self.eye_temporal.update(
            timestamp_ms,
            eye_state.openness,
            eye_state.confidence,
            face.face_found,
            pause=driver_session_held,
            abs_yaw_deg=abs(head_pose.yaw_deg),
            abs_pitch_deg=abs(head_pose.pitch_deg),
        )
        if pose_unreliable:
            gaze_estimate = GazeEstimate(
                GazeZone.UNKNOWN,
                self.config.pose_unreliable_gaze_confidence_cap,
            )
        else:
            gaze_estimate = self.gaze_estimator.estimate(head_pose, timestamp_ms, face.face_found)
        if not self.gaze_estimator.road_gaze_calibrated:
            gaze_estimate.confidence = min(gaze_estimate.confidence, 0.5)
        road_axis_pose = self.road_axis.update(
            head_pose,
            timestamp_ms,
            face.face_found,
            not pose_unreliable,
            gaze_estimate,
        )
        self._update_eyes_off_road(timestamp_ms, gaze_estimate.zone)
        self._update_head_down(timestamp_ms, head_pose.pitch_deg >= self.config.head_pitch_down_threshold_deg)
        if gaze_estimate.zone not in {GazeZone.ROAD, GazeZone.UNKNOWN} and not pose_unreliable:
            self.last_reliable_gaze_away_ms = timestamp_ms
        eyes_off_road_ms = (
            timestamp_ms - self.eyes_off_road_since_ms
            if self.eyes_off_road_since_ms is not None
            else 0
        )
        head_down_ms = timestamp_ms - self.head_down_since_ms if self.head_down_since_ms is not None else 0
        raw_eye_state = eye_temporal.eye_state
        disambiguation = self._disambiguate_eye_gaze_phone(
            raw_eye_state,
            eye_temporal.valid_for_perclos,
            eye_temporal.closure_weight,
            eye_temporal.eye_closure_duration_ms,
            eye_state.confidence,
            gaze_estimate.zone,
            head_pose.pitch_deg,
            face.face_found,
        )

        if driver_session_held:
            blink_stats = self.blink_tracker.update(timestamp_ms, False)
            perclos_short_result = self.perclos_short.pause(timestamp_ms)
            perclos_long_result = self.perclos_long.pause(timestamp_ms)
        elif (
            self.config.perclos_pause_on_low_eye_confidence
            and not disambiguation["perclos_valid"]
        ):
            blink_stats = self.blink_tracker.update(timestamp_ms, False)
            perclos_short_result = self.perclos_short.pause(timestamp_ms)
            perclos_long_result = self.perclos_long.pause(timestamp_ms)
        else:
            blink_stats = self.blink_tracker.update(
                timestamp_ms,
                disambiguation["effective_eye_state"] in {"CLOSED", "PARTIALLY_CLOSED"},
            )
            perclos_short_result = self.perclos_short.update_weighted(
                timestamp_ms,
                float(disambiguation["closure_weight"]),
                bool(disambiguation["perclos_valid"]),
            )
            perclos_long_result = self.perclos_long.update_weighted(
                timestamp_ms,
                float(disambiguation["closure_weight"]),
                bool(disambiguation["perclos_valid"]),
            )
        perclos_5s = perclos_short_result.perclos
        perclos_60s = perclos_long_result.perclos
        eye_closure_duration_ms = eye_temporal.eye_closure_duration_ms
        blink_rate_per_min = eye_temporal.blink_rate_per_min
        enough_eye_observation = (
            perclos_short_result.valid_time_ms >= self.config.perclos_min_valid_observation_ms
        )
        drowsiness_level = self.drowsiness_fsm.update(
            perclos_5s,
            perclos_60s,
            eye_closure_duration_ms,
            blink_rate_per_min,
            face.face_found
            and disambiguation["effective_eye_state"] != "UNKNOWN"
            and bool(disambiguation["perclos_valid"])
            and enough_eye_observation,
            timestamp_ms,
        )
        drowsiness_level = self._resolve_drowsiness_unknown(
            drowsiness_level,
            timestamp_ms,
            face.face_found,
            str(disambiguation["effective_eye_state"]),
            eye_state.confidence,
            eye_temporal.calibration_state,
            bool(disambiguation["perclos_valid"]),
        )
        phone_driver, cabin_events = self.phone_detector.process_cabin(
            frame,
            [(f.track_id, f.zone, f.observation) for f in selection.faces],
            selection.driver.track_id if selection.driver is not None else None,
            gaze_estimate.zone,
            timestamp_ms,
        )
        phone_state, phone_reason_codes = self._normalize_phone_state(
            phone_driver.state,
            gaze_estimate.zone,
            head_pose.pitch_deg,
            eyes_off_road_ms,
            disambiguation["phone_reason_codes"],
            head_down_ms=head_down_ms,
        )
        phone_object = self.phone_detector.last_phone_object
        if phone_object.backend_status == "MODEL_MISSING":
            phone_reason_codes = list(dict.fromkeys(phone_reason_codes + ["PHONE_OBJECT_MODEL_MISSING"]))
        phone_use = PhoneUseState(
            state=phone_state,
            confidence=phone_driver.confidence,
            driver_state=phone_state,
            cabin_events=cabin_events,
            reason_codes=phone_reason_codes,
            phone_object_detected=phone_object.detected,
            phone_object_bbox=list(phone_object.bbox_norm or []),
            phone_object_confidence=phone_object.confidence,
            phone_object_region=phone_object.region,
            phone_object_backend_status=phone_object.backend_status,
            phone_evidence_score=max(phone_driver.confidence, phone_object.confidence),
            phone_texting_candidate_ms=0,
            phone_down_candidate_ms=head_down_ms,
            phone_to_ear_candidate_ms=0,
            phone_final_state=phone_state,
        )
        distraction_level, distraction_type = self.distraction_fsm.update(
            gaze_estimate.zone,
            eyes_off_road_ms,
            no_face_duration_ms,
            phone_use.driver_state,
        )
        if not face.face_found:
            distraction_level = DistractionLevel.UNKNOWN
            distraction_type = DistractionType.UNKNOWN
        elif (
            not self.gaze_estimator.road_gaze_calibrated
            and distraction_type == DistractionType.VISUAL
            and distraction_level == DistractionLevel.HIGH
        ):
            distraction_level = DistractionLevel.MEDIUM
        attention = self.attention_classifier.update(
            AttentionSignals(
                timestamp_ms=timestamp_ms,
                driver_face_present=face.face_found,
                driver_body_present=body_state.state == "PRESENT",
                session_state=session.session_state.value,
                gaze_zone=gaze_estimate.zone,
                gaze_confidence=gaze_estimate.confidence,
                yaw_deg=road_axis_pose.relative_yaw_deg,
                pitch_deg=road_axis_pose.relative_pitch_deg,
                roll_deg=road_axis_pose.relative_roll_deg,
                eye_state=str(disambiguation["effective_eye_state"]),
                eye_visibility=eye_state.confidence,
                eye_closure_duration_ms=eye_closure_duration_ms,
                perclos_5s=perclos_5s,
                perclos_60s=perclos_60s,
                phone_state=phone_use.driver_state,
                phone_reason_codes=phone_use.reason_codes,
                distraction_level=distraction_level,
                drowsiness_level=drowsiness_level,
                head_pose_unreliable=pose_unreliable,
                relative_yaw_deg=road_axis_pose.relative_yaw_deg,
                relative_pitch_deg=road_axis_pose.relative_pitch_deg,
                relative_roll_deg=road_axis_pose.relative_roll_deg,
                side_glance_state=road_axis_pose.side_glance_state,
                side_glance_duration_ms=road_axis_pose.side_glance_duration_ms,
                side_glance_recovery_ms=road_axis_pose.side_glance_recovery_ms,
                yaw_classifiable=road_axis_pose.yaw_classifiable,
                side_profile_context_active=road_axis_pose.side_profile_context_active,
            )
        )
        if pose_hold_codes:
            attention.attention_reason_codes = list(
                dict.fromkeys(attention.attention_reason_codes + pose_hold_codes)
            )
            attention.final_decision_path = (
                "DMS_MONITOR > HEAD_POSE_HELD_PREVIOUS_STABLE"
                if attention.final_decision_path.startswith("DMS_DEGRADED")
                else attention.final_decision_path
            )
        if (
            phone_use.driver_state in {"NO_PHONE", "UNKNOWN"}
            and attention.phone_suspicion_candidate
            and attention.head_down_duration_ms >= self.config.phone_down_candidate_ms
            and not attention.microsleep_candidate
        ):
            texting_like = (
                attention.phone_texting_candidate_duration_ms >= self.config.phone_texting_warning_ms
                or (
                    attention.head_down_duration_ms >= self.config.phone_down_warning_ms
                    and attention.gaze_offroad_duration_ms > 0
                    and "POSSIBLE_PHONE_POSTURE" in phone_use.reason_codes
                )
            )
            inferred_phone_state = (
                "PHONE_TEXTING_SCROLLING_SUSPECTED"
                if texting_like
                else "PHONE_DOWN_SUSPECTED"
            )
            phone_use = PhoneUseState(
                state=inferred_phone_state,
                confidence=max(phone_use.confidence, attention.attention_confidence),
                driver_state=inferred_phone_state,
                cabin_events=phone_use.cabin_events,
                reason_codes=list(
                    dict.fromkeys(
                        phone_use.reason_codes
                        + [
                            "POSSIBLE_PHONE_POSTURE",
                            "PHONE_DOWN_SUSPECTED",
                            inferred_phone_state,
                            "HEAD_DOWN",
                            "GAZE_OFF_ROAD" if gaze_estimate.zone != GazeZone.UNKNOWN else "GAZE_UNKNOWN",
                            "POSSIBLE_PHONE_POSTURE_ACCUMULATING",
                            "PHONE_OBJECT_NOT_REQUIRED_POSTURE_BASED",
                        ]
                        + (["PHONE_TEXTING_SCROLLING_SUSPECTED", "PHONE_WARNING_FROM_POSTURE"] if texting_like else [])
                    )
                ),
                phone_object_detected=phone_use.phone_object_detected,
                phone_object_bbox=phone_use.phone_object_bbox,
                phone_object_confidence=phone_use.phone_object_confidence,
                phone_object_region=phone_use.phone_object_region,
                phone_object_backend_status=phone_use.phone_object_backend_status,
                phone_evidence_score=max(phone_use.phone_evidence_score, attention.attention_confidence),
                phone_texting_candidate_ms=attention.phone_texting_candidate_duration_ms,
                phone_down_candidate_ms=attention.phone_down_candidate_duration_ms,
                phone_to_ear_candidate_ms=0,
                phone_final_state=inferred_phone_state,
            )
            distraction_level, distraction_type = self.distraction_fsm.update(
                gaze_estimate.zone,
                max(eyes_off_road_ms, attention.head_down_duration_ms),
                no_face_duration_ms,
                phone_use.driver_state,
            )

        observability = self._driver_observability(
            face,
            eye_state,
            session.session_state.value,
            body_state.state,
            no_face_duration_ms,
            pose_unreliable,
            proposal_only_driver_visible,
        )
        availability = self._availability(
            face,
            eye_state,
            drowsiness_level,
            distraction_level,
            no_face_duration_ms,
            eye_closure_duration_ms,
            eyes_off_road_ms,
            gaze_estimate.zone,
            phone_use.state,
            driver_track_changed,
            len(selection.faces),
            self.gaze_estimator.road_gaze_calibrated,
            session.session_state.value,
            session.reason_codes or [],
            eye_temporal.eye_state,
            body_state.state,
            pose_unreliable,
            self._perclos_pause_reason(driver_session_held, eye_temporal.valid_for_perclos),
            attention,
            observability.state.value,
            proposal_only_driver_visible,
            list(driver_proposal["reason_codes"]),
            driver_proposal_visible_ms,
        )
        readiness = self._readiness(
            face,
            eye_state,
            drowsiness_level,
            distraction_level,
            gaze_estimate.zone,
        )
        occupancy = self.occupancy.update(selection, timestamp_ms)
        threshold_profile = "NIR_NIGHT" if self.face_backend.last_nir_mode == "NIR_PREPROCESSED" else "BGR_DAY"
        nir_reason_codes = (
            ["NIR_LIKE_FRAME_DETECTED", "NIR_PROFILE_ACTIVE", "NIR_THRESHOLD_PROFILE_SELECTED"]
            if self.face_backend.last_nir_mode == "NIR_PREPROCESSED"
            else ["NIR_NOT_REQUIRED_BGR_INPUT", "BGR_DAY_PROFILE_ACTIVE"]
        )
        health = DMSHealth(
            camera_status=CameraStatus.OK,
            face_detection_status=CameraStatus.OK if selection.faces or self.face_backend.last_proposals else CameraStatus.NO_FACE,
            face_visibility_score=face.confidence if face.face_found else 0.0,
            eye_visibility_score=eye_state.confidence,
            confidence=min(face.confidence, eye_state.confidence)
            if face.face_found
            else 0.0,
            face_backend=self.face_backend.last_backend_used,
            nir_mode=self.face_backend.last_nir_mode,
            nir_mode_detected=self.face_backend.last_nir_mode,
            input_color_mode="NIR" if self.face_backend.last_nir_mode == "NIR_PREPROCESSED" else "BGR",
            active_eye_threshold_profile=threshold_profile,
            active_perclos_profile=threshold_profile,
            nir_preprocessing_active=self.face_backend.last_nir_mode == "NIR_PREPROCESSED",
            nir_reason_codes=nir_reason_codes,
            face_proposals=len(self.face_backend.last_proposals),
            face_detection_confidence=max(
                [proposal.confidence for proposal in self.face_backend.last_proposals],
                default=face.confidence if face.face_found else 0.0,
            ),
        )
        drowsiness_state = DrowsinessState(
            level=drowsiness_level,
            eye_state=str(disambiguation["effective_eye_state"]),
            raw_eye_state=raw_eye_state,
            effective_eye_state=str(disambiguation["effective_eye_state"]),
            eye_openness_raw=eye_state.openness,
            eye_openness_normalized=eye_temporal.normalized_openness,
            eye_calibration_state=eye_temporal.calibration_state,
            eye_visibility_score=eye_state.confidence,
            perclos_valid=bool(disambiguation["perclos_valid"]),
            perclos_validity_reason_codes=list(disambiguation["perclos_reason_codes"]),
            perclos_valid_time_5s_ms=perclos_short_result.valid_time_ms,
            perclos_valid_time_60s_ms=perclos_long_result.valid_time_ms,
            perclos_5s=perclos_5s,
            perclos_60s=perclos_60s,
            eye_closure_duration_ms=eye_closure_duration_ms,
            blink_rate_per_min=blink_rate_per_min,
            confidence=eye_state.confidence if face.face_found else 0.0,
        )
        v02 = self.v02_decision.evaluate(
            DMSV02Inputs(
                timestamp_ms=timestamp_ms,
                health=health,
                availability=availability,
                drowsiness=drowsiness_state,
                distraction_level=distraction_level,
                distraction_type=distraction_type,
                attention=attention,
                phone_state=phone_use.driver_state,
                driver_present=face.face_found,
                driver_body_present=body_state.state == "PRESENT",
                no_face_duration_ms=no_face_duration_ms,
                driver_observability=observability.state.value,
                driver_proposal_visible=proposal_only_driver_visible,
                driver_track_held=proposal_only_driver_visible or driver_session_held,
            )
        )
        state = DMSState(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            dms_health=health,
            driver_presence=DriverPresence(
                state=self._presence_state(
                    face.face_found,
                    len(selection.faces),
                    no_face_duration_ms,
                    session.session_state.value,
                    proposal_only_driver_visible,
                ),
                confidence=face.confidence if face.face_found else float(driver_proposal["confidence"]),
            ),
            driver_observability=observability,
            driver_availability=availability,
            occupants=self._occupants_state(
                selection,
                body_state.state,
                proposal_count=len(self.face_backend.last_proposals) or selection.proposal_count,
            ),
            driver_identity=DriverIdentityState(
                driver_session_id=session.driver_session_id,
                driver_track_id=session.driver_track_id,
                session_state=session.session_state.value,
                reassociated=session.reassociated,
                time_since_seen_ms=session.time_since_seen_ms,
                driver_body_state=body_state.state,
                driver_candidate_score=selection.driver.driver_candidate_score if selection.driver else 0.0,
                driver_front_layer_score=selection.driver.front_layer_score if selection.driver else 0.0,
                driver_rear_layer_penalty=selection.driver.rear_layer_penalty if selection.driver else 0.0,
                driver_slot_assignment=selection.driver.seat_slot if selection.driver else "UNKNOWN",
                driver_slot_reason=selection.driver.slot_reason if selection.driver else "UNKNOWN",
                candidate_depth_layer=selection.driver.depth_layer if selection.driver else "UNKNOWN",
                candidate_seat_slot=selection.driver.zone if selection.driver else "UNKNOWN",
                rear_overlap_rejected_as_driver=(
                    selection.driver.slot_reason == "REAR_LAYER_REJECTED_AS_DRIVER"
                    if selection.driver
                    else False
                ),
                driver_validation_state=(
                    selection.driver.observation.quality.validation_state
                    if selection.driver and selection.driver.observation.quality
                    else "UNKNOWN"
                ),
                driver_validation_reasons=(
                    selection.driver.observation.quality.rejection_reason_codes or []
                    if selection.driver and selection.driver.observation.quality
                    else []
                ),
                driver_proposal_confidence=(
                    selection.driver.observation.quality.proposal_confidence
                    if selection.driver and selection.driver.observation.quality
                    else 0.0
                ),
                driver_face_completeness_score=(
                    selection.driver.observation.quality.face_completeness_score
                    if selection.driver and selection.driver.observation.quality
                    else 0.0
                ),
                driver_landmark_coverage_score=(
                    selection.driver.observation.quality.landmark_coverage_score
                    if selection.driver and selection.driver.observation.quality
                    else 0.0
                ),
                driver_landmark_count=(
                    selection.driver.observation.quality.landmark_count
                    if selection.driver and selection.driver.observation.quality
                    else 0
                ),
                driver_partial_face=(
                    selection.driver.observation.quality.is_partial_face
                    if selection.driver and selection.driver.observation.quality
                    else False
                ),
                face_proposal_state=str(driver_proposal["face_proposal_state"]),
                driver_face_state=self._driver_face_state(face, proposal_only_driver_visible),
                driver_proposal_visible=proposal_only_driver_visible,
                driver_proposal_bbox_norm=list(driver_proposal["bbox_norm"]),
                driver_track_hold_state="PROPOSAL_VISIBLE_HELD" if proposal_only_driver_visible else "NONE",
            ),
            gaze=GazeState(
                zone=gaze_estimate.zone,
                eyes_off_road_duration_ms=int(eyes_off_road_ms),
                head_yaw_deg=head_pose.yaw_deg,
                head_pitch_deg=head_pose.pitch_deg,
                head_roll_deg=head_pose.roll_deg,
                confidence=gaze_estimate.confidence,
                calibration_source=self.road_calibration_source,
                head_pose_raw_yaw_deg=road_axis_pose.head_pose_raw_yaw_deg,
                head_pose_raw_pitch_deg=road_axis_pose.head_pose_raw_pitch_deg,
                head_pose_raw_roll_deg=road_axis_pose.head_pose_raw_roll_deg,
                road_axis_yaw_ref_deg=road_axis_pose.road_axis_yaw_ref_deg,
                road_axis_pitch_ref_deg=road_axis_pose.road_axis_pitch_ref_deg,
                road_axis_roll_ref_deg=road_axis_pose.road_axis_roll_ref_deg,
                relative_yaw_deg=road_axis_pose.relative_yaw_deg,
                relative_pitch_deg=road_axis_pose.relative_pitch_deg,
                relative_roll_deg=road_axis_pose.relative_roll_deg,
                head_angle_from_road_deg=road_axis_pose.head_angle_from_road_deg,
                head_pose_vector_quality=road_axis_pose.head_pose_vector_quality,
                road_axis_calibration_source=road_axis_pose.road_axis_calibration_source,
                road_axis_calibration_confidence=road_axis_pose.road_axis_calibration_confidence,
            ),
            drowsiness=drowsiness_state,
            distraction=DistractionState(
                level=distraction_level,
                type=distraction_type,
                duration_ms=int(eyes_off_road_ms),
                confidence=gaze_estimate.confidence,
                reason_codes=self._distraction_reason_codes(
                    distraction_type,
                    gaze_estimate.zone,
                    phone_use.driver_state,
                    phone_use.reason_codes,
                ),
            ),
            phone_use=phone_use,
            attention=attention,
            dms_v02=v02,
            occupancy=occupancy,
            seatbelt_authenticity=self.seatbelt_detector.process(frame),
            driver_readiness_score=readiness,
        )
        context = {
            "face": face,
            "faces": selection.faces,
            "head_pose": head_pose,
            "road_axis_pose": road_axis_pose,
            "eye_state": eye_state,
            "fps": self.fps_meter.update(timestamp_ms),
            "driver_session": session,
            "driver_body": body_state,
            "face_proposals": self.face_backend.last_proposals,
            "driver_proposal_candidate": driver_proposal,
            "face_backend": self.face_backend.last_backend_used,
            "nir_mode": self.face_backend.last_nir_mode,
            "driver_roi_norm": self.occupants._roi("driver_roi_norm"),
        }
        return state, context

    def close(self) -> None:
        self.face_backend.close()
        self.phone_detector.close()

    def calibrate_road_gaze(
        self,
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float = 0.0,
        timestamp_ms: int = 0,
        source: str = "RUNTIME",
        confidence: float = 1.0,
    ) -> tuple[float, float]:
        self.gaze_estimator.calibrate_road_center(yaw_deg, pitch_deg)
        self.road_axis.calibrate(yaw_deg, pitch_deg, roll_deg, timestamp_ms, source, confidence)
        self.road_calibration_source = source
        return self.gaze_estimator.yaw_offset_deg, self.gaze_estimator.pitch_offset_deg

    def reset_road_gaze_calibration(self) -> tuple[float, float]:
        self.gaze_estimator.reset_road_center()
        self.road_axis.reset()
        self.road_calibration_source = "DEFAULT"
        return self.gaze_estimator.yaw_offset_deg, self.gaze_estimator.pitch_offset_deg

    def _driver_proposal_context(self, frame_shape: tuple[int, int, int]) -> dict[str, object]:
        height, width = frame_shape[:2]
        driver_roi = self.occupants._roi("driver_roi_norm")
        best: tuple[object, tuple[float, float, float, float], float] | None = None
        best_score = 0.0
        for proposal in self.face_backend.last_proposals:
            x1, y1, x2, y2 = proposal.bbox
            box_norm = (x1 / width, y1 / height, x2 / width, y2 / height)
            overlap = self._box_overlap_norm(box_norm, driver_roi)
            center_x = (box_norm[0] + box_norm[2]) / 2.0
            center_y = (box_norm[1] + box_norm[3]) / 2.0
            center_in_roi = driver_roi[0] <= center_x <= driver_roi[2] and driver_roi[1] <= center_y <= driver_roi[3]
            if proposal.confidence < self.config.face_proposal_min_confidence:
                continue
            if overlap <= 0.0 and not center_in_roi:
                continue
            score = proposal.confidence + overlap + (0.25 if center_in_roi else 0.0)
            if score > best_score:
                best = (proposal, box_norm, overlap)
                best_score = score
        if best is None:
            return {
                "visible": False,
                "face_proposal_state": "NO_PROPOSAL" if not self.face_backend.last_proposals else "PROPOSAL_PRESENT",
                "confidence": 0.0,
                "bbox_norm": [],
                "reason_codes": [],
            }
        proposal, box_norm, _overlap = best
        return {
            "visible": True,
            "face_proposal_state": "DRIVER_ZONE_PROPOSAL_PRESENT",
            "confidence": proposal.confidence,
            "bbox_norm": list(box_norm),
            "reason_codes": [
                "DRIVER_ZONE_PROPOSAL_PRESENT",
                "FACE_PROPOSAL_LANDMARK_FAILED",
                "DRIVER_NOT_VALIDATED_BUT_VISIBLE_PROPOSAL",
                "PROPOSAL_ONLY_NOT_DRIVER_ABSENT",
            ],
        }

    @staticmethod
    def _box_overlap_norm(
        box: tuple[float, float, float, float],
        roi: tuple[float, float, float, float],
    ) -> float:
        x1 = max(box[0], roi[0])
        y1 = max(box[1], roi[1])
        x2 = min(box[2], roi[2])
        y2 = min(box[3], roi[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        return inter / max(area, 1e-6)

    @staticmethod
    def _driver_face_state(face: FaceLandmarkResult, proposal_only_visible: bool = False) -> str:
        if face.landmarks_px:
            quality_state = face.quality.validation_state if face.quality else "VALIDATED"
            if quality_state in {"VALID", "VALIDATED", "FULL_FACE", "SIDE_PROFILE"}:
                return "VALIDATED"
            return "PARTIAL"
        if proposal_only_visible:
            return "LANDMARK_FAILED"
        if face.face_found:
            return "PROPOSAL_ONLY"
        return "NOT_VISIBLE"

    def _reset_driver_temporal(self) -> None:
        self.head_pose_smoother.reset()
        self.eye_temporal.reset()
        self.blink_tracker.closed_since_ms = None
        self.perclos_short.reset()
        self.perclos_long.reset()
        self.drowsiness_fsm.medium_since_ms = None
        self.drowsiness_fsm.high_since_ms = None
        self.drowsiness_fsm.release_since_ms = None
        self.valid_eye_observation_since_ms = None
        if hasattr(self, "attention_classifier"):
            self.attention_classifier.reset()

    def _driver_observability(
        self,
        face: FaceLandmarkResult,
        eye_state: EyeState,
        session_state: str,
        driver_body_state: str,
        no_face_duration_ms: int,
        head_pose_unreliable: bool,
        driver_proposal_visible: bool = False,
    ) -> DriverObservability:
        if face.face_found:
            reasons: list[str] = []
            partial = bool(face.quality and face.quality.is_partial_face)
            if partial or head_pose_unreliable or eye_state.confidence < self.config.eye_visibility_min_confidence:
                if partial:
                    reasons.append("PARTIAL_FACE_CROP")
                if head_pose_unreliable:
                    reasons.append("HEAD_POSE_UNRELIABLE")
                if eye_state.confidence < self.config.eye_visibility_min_confidence:
                    reasons.extend(["GLASSES_REFLECTION_LOW_EYE_CONFIDENCE", "OCCLUSION_GLASSES"])
                return DriverObservability(
                    DriverObservabilityState.PARTIALLY_OBSERVABLE,
                    max(0.35, min(face.confidence, max(eye_state.confidence, 0.35))),
                    list(dict.fromkeys(reasons)),
                )
            return DriverObservability(
                DriverObservabilityState.OBSERVABLE,
                min(1.0, max(face.confidence, eye_state.confidence)),
                ["DRIVER_OBSERVABLE"],
            )
        if driver_proposal_visible:
            return DriverObservability(
                DriverObservabilityState.PARTIALLY_OBSERVABLE,
                0.45,
                [
                    "DRIVER_ZONE_PROPOSAL_PRESENT",
                    "FACE_PROPOSAL_LANDMARK_FAILED",
                    "DRIVER_NOT_VALIDATED_BUT_VISIBLE_PROPOSAL",
                    "PROPOSAL_ONLY_NOT_DRIVER_ABSENT",
                    "DEGRADED_SUPPRESSED_PROPOSAL_VISIBLE",
                ],
            )
        if session_state == DriverSessionState.LOST_TEMP.value or driver_body_state == "PRESENT":
            reasons = [
                "DRIVER_OBSERVABILITY_TEMP_LOST",
                "FACE_LOSS_NOT_DRIVER_UNAVAILABLE",
            ]
            if driver_body_state == "PRESENT":
                reasons.append("DRIVER_BODY_PRESENT_FACE_UNOBSERVABLE")
                reasons.append("DRIVER_UNAVAILABLE_SUPPRESSED_BODY_PRESENT")
            if self.last_driver_abs_yaw_deg >= min(
                abs(self.config.head_yaw_left_threshold_deg),
                abs(self.config.head_yaw_right_threshold_deg),
            ):
                reasons.append("DRIVER_UNOBSERVABLE_SIDE_PROFILE")
            return DriverObservability(
                DriverObservabilityState.UNOBSERVABLE_TEMP,
                0.4,
                list(dict.fromkeys(reasons)),
            )
        if no_face_duration_ms >= self.config.driver_unobservable_long_ms:
            return DriverObservability(
                DriverObservabilityState.UNOBSERVABLE_LONG,
                0.0,
                ["DRIVER_FACE_NOT_VISIBLE"],
            )
        return DriverObservability(
            DriverObservabilityState.UNOBSERVABLE_TEMP,
            0.2,
            ["DRIVER_OBSERVABILITY_TEMP_LOST"],
        )

    def _resolve_drowsiness_unknown(
        self,
        level: DrowsinessLevel,
        timestamp_ms: int,
        face_found: bool,
        effective_eye_state: str,
        eye_confidence: float,
        calibration_state: str,
        perclos_valid: bool,
    ) -> DrowsinessLevel:
        valid_open_eye = (
            face_found
            and effective_eye_state in {"OPEN", "PARTIALLY_CLOSED"}
            and calibration_state in {"CALIBRATED", "FALLBACK"}
            and eye_confidence >= self.config.eye_visibility_min_confidence
            and perclos_valid
        )
        if valid_open_eye:
            if getattr(self, "valid_eye_observation_since_ms", None) is None:
                self.valid_eye_observation_since_ms = timestamp_ms
            valid_ms = timestamp_ms - self.valid_eye_observation_since_ms
            if (
                self.config.drowsiness_resolve_to_none_when_open
                and level == DrowsinessLevel.UNKNOWN
                and valid_ms >= self.config.drowsiness_min_valid_eye_ms
            ):
                return DrowsinessLevel.NONE
        else:
            self.valid_eye_observation_since_ms = None
        return level

    def _pose_unreliable(self, head_pose: object) -> bool:
        return (
            getattr(head_pose, "confidence", 0.0) < self.config.head_pose_min_confidence
            or abs(getattr(head_pose, "yaw_deg", 0.0)) > self.config.max_valid_yaw_deg
            or abs(getattr(head_pose, "pitch_deg", 0.0)) > self.config.max_valid_pitch_deg
            or abs(getattr(head_pose, "roll_deg", 0.0)) > self.config.max_valid_roll_deg
        )

    def _should_hold_previous_pose(
        self,
        face: FaceLandmarkResult,
        eye_state: EyeState,
        head_pose: HeadPose,
        timestamp_ms: int,
    ) -> bool:
        if not self.config.pose_plausibility_filter_enabled:
            return False
        if self._last_stable_head_pose is None or self._last_stable_pose_ms is None:
            return False
        if not self._valid_face_for_pose_hold(face, eye_state):
            return False
        dt_ms = max(1, timestamp_ms - self._last_stable_pose_ms)
        scale = max(1.0, dt_ms / 100.0)
        yaw_jump = abs(head_pose.yaw_deg - self._last_stable_head_pose.yaw_deg)
        pitch_jump = abs(head_pose.pitch_deg - self._last_stable_head_pose.pitch_deg)
        roll_jump = abs(head_pose.roll_deg - self._last_stable_head_pose.roll_deg)
        jump_unrealistic = (
            yaw_jump > self.config.max_yaw_jump_deg_per_100ms * scale
            or pitch_jump > self.config.max_pitch_jump_deg_per_100ms * scale
            or roll_jump > self.config.max_roll_jump_deg_per_100ms * scale
        )
        out_of_range = (
            abs(head_pose.pitch_deg) > self.config.max_plausible_pitch_deg
            or abs(head_pose.roll_deg) > self.config.max_plausible_roll_deg
            or head_pose.confidence < 0.4
        )
        return jump_unrealistic or out_of_range

    def _valid_face_for_pose_hold(self, face: FaceLandmarkResult, eye_state: EyeState) -> bool:
        if self.config.pose_hold_requires_valid_face and not face.face_found:
            return False
        if face.confidence < self.config.pose_hold_requires_face_confidence_min:
            return False
        if eye_state.confidence < self.config.pose_hold_requires_eye_visibility_min:
            return False
        quality = face.quality
        if quality is None:
            return bool(face.landmarks_px)
        return (
            quality.landmark_coverage_score >= self.config.pose_hold_requires_landmark_coverage_min
            and quality.landmark_count >= self.config.driver_min_landmark_count
        )

    @staticmethod
    def _perclos_pause_reason(driver_session_held: bool, eye_valid_for_perclos: bool) -> str | None:
        if driver_session_held:
            return "PERCLOS_PAUSED_FACE_LOST"
        if not eye_valid_for_perclos:
            return "PERCLOS_PAUSED_EYE_UNKNOWN"
        return None

    def _disambiguate_eye_gaze_phone(
        self,
        raw_eye_state: str,
        perclos_valid: bool,
        closure_weight: float,
        eye_closure_duration_ms: int,
        eye_visibility: float,
        gaze_zone: GazeZone,
        head_pitch_deg: float,
        face_found: bool,
    ) -> dict[str, object]:
        reasons: list[str] = []
        perclos_reasons: list[str] = []
        phone_reasons: list[str] = []
        effective_eye_state = raw_eye_state
        effective_closure_weight = closure_weight
        effective_perclos_valid = perclos_valid
        head_down = head_pitch_deg >= self.config.head_pitch_down_threshold_deg
        phone_posture = face_found and (head_down or gaze_zone in DOWNWARD_GAZE_ZONES)
        raw_closed_like = raw_eye_state in {"CLOSED", "PARTIALLY_CLOSED"}
        if phone_posture:
            reasons.append("POSSIBLE_PHONE_POSTURE")
            phone_reasons.append("POSSIBLE_PHONE_POSTURE")
            if head_down:
                phone_reasons.append("HEAD_DOWN")
            if gaze_zone in DOWNWARD_GAZE_ZONES:
                phone_reasons.append("GAZE_OFF_ROAD")
        if phone_posture and (raw_closed_like or eye_visibility < 0.65):
            effective_eye_state = "UNKNOWN"
            effective_closure_weight = 0.0
            effective_perclos_valid = False
            reasons.append("EYE_CLOSURE_SUPPRESSED_BY_DOWNWARD_GAZE")
            perclos_reasons.append("LOW_EYE_VISIBILITY")
            phone_reasons.append("EYE_CLOSURE_SUPPRESSED_BY_DOWNWARD_GAZE")
            if eye_visibility < 0.75 or head_down:
                reasons.append("LOW_EYE_VISIBILITY_DUE_TO_HEAD_POSE")
                perclos_reasons.append("LOW_EYE_VISIBILITY")
                phone_reasons.append("LOW_EYE_VISIBILITY_DUE_TO_HEAD_POSE")
        if not effective_perclos_valid and not reasons:
            reasons.append("PERCLOS_PAUSED_EYE_UNKNOWN")
            perclos_reasons.append("PERCLOS_PAUSED_EYE_UNKNOWN")
        if raw_closed_like and eye_closure_duration_ms >= self.config.microsleep_closure_ms:
            effective_eye_state = raw_eye_state
            effective_closure_weight = closure_weight
            effective_perclos_valid = perclos_valid
            reasons = [reason for reason in reasons if reason != "EYE_CLOSURE_SUPPRESSED_BY_DOWNWARD_GAZE"]
            perclos_reasons = []
        return {
            "effective_eye_state": effective_eye_state,
            "closure_weight": effective_closure_weight,
            "perclos_valid": effective_perclos_valid,
            "reason_codes": list(dict.fromkeys(reasons)),
            "perclos_reason_codes": list(dict.fromkeys(perclos_reasons)) or (["VALID"] if effective_perclos_valid else ["PERCLOS_PAUSED_EYE_UNKNOWN"]),
            "phone_reason_codes": list(dict.fromkeys(phone_reasons)),
        }

    def _normalize_phone_state(
        self,
        raw_phone_state: str,
        gaze_zone: GazeZone,
        head_pitch_deg: float,
        eyes_off_road_ms: int,
        disambiguation_reasons: object = None,
        head_down_ms: int = 0,
    ) -> tuple[str, list[str]]:
        reasons = list(disambiguation_reasons) if isinstance(disambiguation_reasons, list) else []
        if raw_phone_state in {
            "TEXTING_SUSPECTED",
            "PHONE_TO_EAR_SUSPECTED",
            "PHONE_TO_EAR_CONFIRMED",
            "PHONE_DOWN_SUSPECTED",
            "PHONE_DOWN_CONFIRMED",
            "PHONE_TEXTING_SCROLLING_SUSPECTED",
            "PHONE_TEXTING_SCROLLING_CONFIRMED",
        }:
            reasons.append(raw_phone_state)
            return raw_phone_state, list(dict.fromkeys(reasons))
        if raw_phone_state in {
            "SELF_TOUCH_TRANSIENT",
            "EAR_SCRATCH_GESTURE",
            "FACE_TOUCH_GROOMING",
        }:
            extra = [raw_phone_state]
            if raw_phone_state == "SELF_TOUCH_TRANSIENT":
                extra.extend([
                    "HAND_NEAR_EAR_RAW",
                    "SELF_TOUCH_TRANSIENT",
                    "PHONE_TO_EAR_SUPPRESSED_SELF_TOUCH",
                ])
            elif raw_phone_state == "EAR_SCRATCH_GESTURE":
                extra.extend([
                    "HAND_NEAR_EAR_RAW",
                    "EAR_SCRATCH_GESTURE",
                    "PHONE_TO_EAR_SUPPRESSED_SELF_TOUCH",
                    "PHONE_TO_EAR_SUPPRESSED_NO_PHONE_OBJECT",
                ])
            else:
                extra.extend(["FACE_TOUCH_GROOMING", "PHONE_TO_EAR_SUPPRESSED_SELF_TOUCH"])
            if gaze_zone == GazeZone.ROAD:
                extra.append("PHONE_TO_EAR_SUPPRESSED_ROAD_GAZE")
            return "NO_PHONE", list(dict.fromkeys(reasons + extra))
        if raw_phone_state == "PHONE_TO_EAR_CANDIDATE":
            extra = [
                "HAND_NEAR_EAR_RAW",
                "PHONE_TO_EAR_CANDIDATE",
                "PHONE_TO_EAR_SUPPRESSED_NO_PHONE_OBJECT",
            ]
            if gaze_zone == GazeZone.ROAD:
                extra.append("PHONE_TO_EAR_SUPPRESSED_ROAD_GAZE")
            return "PHONE_TO_EAR_CANDIDATE", list(dict.fromkeys(reasons + extra))
        if raw_phone_state == "PHONE_CONFIRMED":
            return "PHONE_CONFIRMED", list(dict.fromkeys(reasons + ["PHONE_CONFIRMED"]))
        if raw_phone_state == "HAND_NEAR_FACE":
            extra = [
                "HAND_NEAR_FACE",
                "FACE_TOUCH_GROOMING",
                "PHONE_TO_EAR_SUPPRESSED_SELF_TOUCH",
            ]
            if gaze_zone == GazeZone.ROAD:
                extra.append("PHONE_TO_EAR_SUPPRESSED_ROAD_GAZE")
            return "NO_PHONE", list(dict.fromkeys(reasons + extra))
        downward_phone_posture = (
            gaze_zone in DOWNWARD_GAZE_ZONES
            or head_pitch_deg >= self.config.head_pitch_down_threshold_deg
        )
        phone_posture_evidence = "POSSIBLE_PHONE_POSTURE" in reasons or downward_phone_posture
        if (
            phone_posture_evidence
            and (
                head_down_ms >= self.config.phone_down_candidate_ms
                or eyes_off_road_ms >= self.config.phone_gaze_offroad_suspect_ms
            )
        ):
            extra = ["POSSIBLE_PHONE_POSTURE", "PHONE_DOWN_SUSPECTED", "HEAD_DOWN"]
            extra.append("GAZE_OFF_ROAD" if gaze_zone != GazeZone.UNKNOWN else "GAZE_UNKNOWN")
            if head_down_ms >= self.config.phone_texting_warning_ms and gaze_zone in DOWNWARD_GAZE_ZONES:
                extra.extend([
                    "PHONE_TEXTING_SCROLLING_SUSPECTED",
                    "PHONE_OBJECT_NOT_REQUIRED_POSTURE_BASED",
                    "PHONE_WARNING_FROM_POSTURE",
                ])
                return "PHONE_TEXTING_SCROLLING_SUSPECTED", list(dict.fromkeys(reasons + extra))
            return "PHONE_DOWN_SUSPECTED", list(dict.fromkeys(reasons + extra))
        if raw_phone_state in {"NO_PHONE", "UNKNOWN"}:
            return raw_phone_state, list(dict.fromkeys(reasons))
        return raw_phone_state, list(dict.fromkeys(reasons))

    @staticmethod
    def _distraction_reason_codes(
        distraction_type: DistractionType,
        gaze_zone: GazeZone,
        phone_state: str,
        phone_reasons: list[str],
    ) -> list[str]:
        reasons = list(phone_reasons)
        if gaze_zone not in {GazeZone.ROAD, GazeZone.UNKNOWN}:
            reasons.append("GAZE_AWAY")
        if phone_state in {"PHONE_SUSPECTED", "PHONE_CONFIRMED"}:
            reasons.append(phone_state)
        if distraction_type != DistractionType.NONE:
            reasons.append(distraction_type.value)
        return list(dict.fromkeys(reasons))

    def _update_eyes_off_road(self, timestamp_ms: int, zone: GazeZone) -> None:
        if zone == GazeZone.ROAD:
            self.eyes_off_road_since_ms = None
        elif self.eyes_off_road_since_ms is None and zone != GazeZone.UNKNOWN:
            self.eyes_off_road_since_ms = timestamp_ms

    def _update_head_down(self, timestamp_ms: int, head_down: bool) -> None:
        if head_down and self.head_down_since_ms is None:
            self.head_down_since_ms = timestamp_ms
        elif not head_down:
            self.head_down_since_ms = None

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
        eye_closure_duration_ms: int,
        eyes_off_road_duration_ms: int,
        gaze_zone: GazeZone,
        phone_state: str = "UNKNOWN",
        driver_track_changed: bool = False,
        occupant_count: int = 0,
        road_gaze_calibrated: bool = True,
        session_state: str = "UNKNOWN",
        session_reason_codes: list[str] | None = None,
        eye_state_label: str = "OPEN",
        driver_body_state: str = "UNKNOWN",
        head_pose_unreliable: bool = False,
        perclos_pause_reason: str | None = None,
        attention: AttentionOutput | None = None,
        driver_observability: str = "UNKNOWN",
        driver_proposal_visible: bool = False,
        driver_proposal_reason_codes: list[str] | None = None,
        driver_proposal_visible_ms: int = 0,
    ) -> DriverAvailability:
        session_reasons = session_reason_codes or []
        attention_reasons = list(attention.attention_reason_codes) if attention is not None else []
        if attention is not None and attention.microsleep_candidate:
            return DriverAvailability(
                AvailabilityState.UNAVAILABLE,
                0.05,
                list(dict.fromkeys(attention_reasons + ["MICROSLEEP_CANDIDATE"])),
            )
        if driver_proposal_visible:
            reasons = list(
                dict.fromkeys(
                    (driver_proposal_reason_codes or [])
                    + [
                        "DRIVER_ZONE_PROPOSAL_PRESENT",
                        "FACE_PROPOSAL_LANDMARK_FAILED",
                        "DRIVER_NOT_VALIDATED_BUT_VISIBLE_PROPOSAL",
                        "DRIVER_UNAVAILABLE_SUPPRESSED_PROPOSAL_PRESENT",
                        "PROPOSAL_ONLY_NOT_DRIVER_ABSENT",
                        "PERCLOS_PAUSED_LANDMARK_FAILED",
                    ]
                    + (["NIGHT_UNAVAILABLE_SUPPRESSED_PROPOSAL_PRESENT"] if self.face_backend.last_nir_mode == "NIR_PREPROCESSED" else [])
                    + (["WEBCAM_HOLD_EXPIRED"] if driver_proposal_visible_ms > self.config.webcam_proposal_visible_hold_ms else ["WEBCAM_DRIVER_TRACK_HELD", "FACE_MESH_DROPOUT_HELD"])
                )
            )
            confidence = 0.45 if driver_proposal_visible_ms <= self.config.webcam_proposal_visible_hold_ms else 0.35
            return DriverAvailability(AvailabilityState.DEGRADED, confidence, reasons)
        if session_state == DriverSessionState.LOST_TEMP.value:
            reasons = [
                "DRIVER_FACE_LOST_TEMP",
                "DRIVER_SESSION_HELD",
                "DRIVER_OBSERVABILITY_TEMP_LOST",
                "FACE_LOSS_NOT_DRIVER_UNAVAILABLE",
            ]
            if driver_body_state == "PRESENT":
                reasons.append("DRIVER_BODY_PRESENT_FACE_LOST")
                reasons.append("DRIVER_BODY_PRESENT_FACE_UNOBSERVABLE")
                reasons.append("DRIVER_UNAVAILABLE_SUPPRESSED_BODY_PRESENT")
            if self.last_driver_abs_yaw_deg >= min(
                abs(self.config.head_yaw_left_threshold_deg),
                abs(self.config.head_yaw_right_threshold_deg),
            ):
                reasons.append("SIDE_PROFILE_FACE_LOST")
                reasons.append("DRIVER_UNOBSERVABLE_SIDE_PROFILE")
                reasons.append("POSSIBLE_GAZE_AWAY_DURING_LOST_TEMP")
            if session_reasons:
                reasons = list(dict.fromkeys(reasons + session_reasons))
            if perclos_pause_reason:
                reasons = list(dict.fromkeys(reasons + [perclos_pause_reason]))
            if attention_reasons:
                reasons = list(dict.fromkeys(reasons + attention_reasons))
            return DriverAvailability(AvailabilityState.DEGRADED, 0.45, reasons)
        if no_face_duration_ms >= self.config.no_face_timeout_ms:
            if driver_body_state == "PRESENT" or driver_observability in {
                DriverObservabilityState.UNOBSERVABLE_TEMP.value,
                DriverObservabilityState.PARTIALLY_OBSERVABLE.value,
            }:
                return DriverAvailability(
                    AvailabilityState.DEGRADED,
                    0.4,
                    [
                        "DRIVER_OBSERVABILITY_TEMP_LOST",
                        "DRIVER_BODY_PRESENT_FACE_UNOBSERVABLE",
                        "FACE_LOSS_NOT_DRIVER_UNAVAILABLE",
                        "DRIVER_UNAVAILABLE_SUPPRESSED_BODY_PRESENT",
                    ],
                )
            reason = "DRIVER_FACE_NOT_VISIBLE" if occupant_count > 0 else "NO_FACE"
            return DriverAvailability(AvailabilityState.UNAVAILABLE, 0.0, [reason])
        if drowsiness == DrowsinessLevel.MICROSLEEP:
            return DriverAvailability(AvailabilityState.UNAVAILABLE, 0.05, ["MICROSLEEP"])
        if eye_closure_duration_ms >= self.config.microsleep_duration_ms:
            return DriverAvailability(AvailabilityState.UNAVAILABLE, 0.05, ["EYE_CLOSED"])
        if (
            distraction == DistractionLevel.HIGH
        ):
            phone_related = phone_state in {
                "PHONE_TO_EAR_SUSPECTED",
                "PHONE_DOWN_SUSPECTED",
                "TEXTING_SUSPECTED",
                "HAND_NEAR_FACE",
                "PHONE_SUSPECTED",
                "PHONE_CONFIRMED",
            }
            unavailable_ms = self.config.phone_unavailable_ms if phone_related else self.config.visual_distraction_unavailable_ms
            if (
                not phone_related
                and self.config.require_adas_risk_for_visual_unavailable
                and not self.config.adas_risk_present_default
            ):
                return DriverAvailability(
                    AvailabilityState.DEGRADED,
                    0.6,
                    self._reason_codes(
                        face,
                        eye_state,
                        drowsiness,
                        distraction,
                        gaze_zone,
                        phone_state,
                        road_gaze_calibrated,
                        head_pose_unreliable,
                    ),
                )
            if eyes_off_road_duration_ms < unavailable_ms:
                return DriverAvailability(
                    AvailabilityState.DEGRADED,
                    0.6,
                    self._reason_codes(
                        face,
                        eye_state,
                        drowsiness,
                        distraction,
                        gaze_zone,
                        phone_state,
                        road_gaze_calibrated,
                        head_pose_unreliable,
                    ),
                )
            return DriverAvailability(
                AvailabilityState.UNAVAILABLE,
                0.2,
                self._reason_codes(
                    face,
                    eye_state,
                    drowsiness,
                    distraction,
                    gaze_zone,
                    phone_state,
                    road_gaze_calibrated,
                    head_pose_unreliable,
                ),
            )
        if (
            attention is not None
            and attention.attention_state == AttentionState.ATTENTION_LOST
            and attention.attention_substate
            in {
                AttentionSubstate.PHONE_SUSPECTED,
                AttentionSubstate.PHONE_DOWN_SUSPECTED,
                AttentionSubstate.PHONE_TO_EAR_SUSPECTED,
                AttentionSubstate.TEXTING_SUSPECTED,
                AttentionSubstate.PHONE_CONFIRMED,
            }
            and attention.attention_lost_duration_ms >= self.config.phone_unavailable_ms
            and attention.attention_confidence >= 0.7
        ):
            return DriverAvailability(
                AvailabilityState.UNAVAILABLE,
                0.25,
                list(
                    dict.fromkeys(
                        attention_reasons + [attention.driver_availability_reason or "PHONE_ATTENTION_LOST"]
                    )
                ),
            )
        if (
            attention is not None
            and attention.attention_state == AttentionState.ATTENTION_LOST
            and attention.attention_substate
            not in {
                AttentionSubstate.PHONE_SUSPECTED,
                AttentionSubstate.PHONE_DOWN_SUSPECTED,
                AttentionSubstate.PHONE_TO_EAR_SUSPECTED,
                AttentionSubstate.TEXTING_SUSPECTED,
                AttentionSubstate.PHONE_CONFIRMED,
            }
            and not (
                attention.attention_substate
                in {
                    AttentionSubstate.VISUAL_DISTRACTION,
                    AttentionSubstate.HEAD_DOWN,
                    AttentionSubstate.HEAD_DOWN_DISTRACTION,
                    AttentionSubstate.HEAD_DOWN_UNCERTAIN,
                }
                and self.config.require_adas_risk_for_attention_unavailable
                and not self.config.adas_risk_present_default
            )
            and attention.attention_lost_duration_ms >= self.config.attention_lost_unavailable_ms
            and attention.attention_confidence >= 0.7
        ):
            return DriverAvailability(
                AvailabilityState.UNAVAILABLE,
                0.25,
                list(
                    dict.fromkeys(
                        attention_reasons + [attention.driver_availability_reason or "ATTENTION_LOST"]
                    )
                ),
            )
        ambiguous_timeout_ms = int((self.config.attention_state or {}).get("ambiguous_timeout_ms", 1000))
        if (
            attention is not None
            and attention.attention_substate == AttentionSubstate.AMBIGUOUS
            and attention.attention_lost_duration_ms >= ambiguous_timeout_ms
        ):
            return DriverAvailability(
                AvailabilityState.DEGRADED,
                0.5,
                list(dict.fromkeys(attention_reasons + ["AMBIGUOUS_ATTENTION_LOSS"])),
            )
        if attention is not None and attention.attention_state in {
            AttentionState.ATTENTION_LOST,
            AttentionState.DEGRADED,
        }:
            return DriverAvailability(
                AvailabilityState.DEGRADED,
                0.6,
                list(
                    dict.fromkeys(
                        attention_reasons + [attention.driver_availability_reason or "ATTENTION_DEGRADED"]
                    )
                ),
            )
        if not face.face_found:
            return DriverAvailability(AvailabilityState.DEGRADED, 0.35, ["DRIVER_FACE_NOT_VISIBLE"])
        if head_pose_unreliable:
            reasons = self._reason_codes(
                face,
                eye_state,
                drowsiness,
                DistractionLevel.NONE,
                GazeZone.UNKNOWN,
                phone_state,
                road_gaze_calibrated,
                head_pose_unreliable,
            )
            if perclos_pause_reason:
                reasons.append(perclos_pause_reason)
            return DriverAvailability(
                AvailabilityState.DEGRADED,
                0.55,
                list(dict.fromkeys(reasons)),
            )
        if eye_state.confidence < 0.5:
            reasons = ["EYE_VISIBILITY_LOW", "GLASSES_REFLECTION_LOW_EYE_CONFIDENCE", "OCCLUSION_GLASSES"]
            if perclos_pause_reason:
                reasons.append(perclos_pause_reason)
            return DriverAvailability(AvailabilityState.DEGRADED, 0.55, reasons)
        if eye_state_label == "UNKNOWN":
            reasons = ["EYE_VISIBILITY_LOW", "GLASSES_REFLECTION_LOW_EYE_CONFIDENCE"]
            if perclos_pause_reason:
                reasons.append(perclos_pause_reason)
            return DriverAvailability(AvailabilityState.DEGRADED, 0.55, reasons)
        if driver_track_changed:
            return DriverAvailability(AvailabilityState.DEGRADED, 0.6, ["DRIVER_TRACK_CHANGED"])
        if session_reasons:
            base = self._reason_codes(
                face,
                eye_state,
                drowsiness,
                distraction,
                gaze_zone,
                phone_state,
                road_gaze_calibrated,
                head_pose_unreliable,
            )
            return DriverAvailability(
                AvailabilityState.DEGRADED if base else AvailabilityState.AVAILABLE,
                0.85,
                list(dict.fromkeys(session_reasons + base)),
            )
        if gaze_zone == GazeZone.UNKNOWN:
            return DriverAvailability(AvailabilityState.DEGRADED, 0.7, ["GAZE_UNKNOWN"])
        if drowsiness in {DrowsinessLevel.MEDIUM, DrowsinessLevel.HIGH}:
            return DriverAvailability(
                AvailabilityState.DEGRADED,
                0.6,
                self._reason_codes(
                    face,
                    eye_state,
                    drowsiness,
                    distraction,
                    gaze_zone,
                    phone_state,
                    road_gaze_calibrated,
                    head_pose_unreliable,
                ),
            )
        if distraction in {DistractionLevel.MEDIUM, DistractionLevel.HIGH}:
            return DriverAvailability(
                AvailabilityState.DEGRADED,
                0.65,
                self._reason_codes(
                    face,
                    eye_state,
                    drowsiness,
                    distraction,
                    gaze_zone,
                    phone_state,
                    road_gaze_calibrated,
                    head_pose_unreliable,
                ),
            )
        if face.face_found and not road_gaze_calibrated:
            return DriverAvailability(
                AvailabilityState.AVAILABLE,
                0.9,
                ["ROAD_GAZE_NOT_CALIBRATED"],
            )
        if face.face_found and getattr(self, "road_calibration_source", "DEFAULT") == "FILE":
            return DriverAvailability(
                AvailabilityState.AVAILABLE,
                0.95,
                ["ROAD_CALIBRATION_FILE_LOADED"],
            )
        return DriverAvailability(AvailabilityState.AVAILABLE, 0.95, [])

    def _reason_codes(
        self,
        face: FaceLandmarkResult,
        eye_state: EyeState,
        drowsiness: DrowsinessLevel,
        distraction: DistractionLevel,
        gaze_zone: GazeZone,
        phone_state: str,
        road_gaze_calibrated: bool = True,
        head_pose_unreliable: bool = False,
    ) -> list[str]:
        reasons: list[str] = []
        if road_gaze_calibrated and getattr(self, "road_calibration_source", "DEFAULT") == "FILE":
            reasons.append("ROAD_CALIBRATION_FILE_LOADED")
        if not road_gaze_calibrated:
            reasons.append("ROAD_GAZE_NOT_CALIBRATED")
        if head_pose_unreliable:
            reasons.append("HEAD_POSE_UNRELIABLE")
        if not face.face_found:
            reasons.append("DRIVER_FACE_NOT_VISIBLE")
        elif face.confidence < 0.5:
            reasons.append("LOW_FACE_CONFIDENCE")
        if eye_state.is_closed:
            reasons.append("EYE_CLOSED")
        if gaze_zone == GazeZone.UNKNOWN:
            reasons.append("GAZE_UNKNOWN")
        elif gaze_zone != GazeZone.ROAD:
            reasons.append("GAZE_AWAY")
        if phone_state in {
            "PHONE_TO_EAR_SUSPECTED",
            "PHONE_DOWN_SUSPECTED",
            "TEXTING_SUSPECTED",
            "HAND_NEAR_FACE",
            "PHONE_SUSPECTED",
            "PHONE_CONFIRMED",
        }:
            reasons.append(phone_state)
        if drowsiness == DrowsinessLevel.MICROSLEEP:
            reasons.append("MICROSLEEP")
        elif drowsiness == DrowsinessLevel.HIGH:
            reasons.append("DROWSINESS_HIGH")
        elif drowsiness == DrowsinessLevel.MEDIUM:
            reasons.append("DROWSINESS_MEDIUM")
        if (
            drowsiness in {DrowsinessLevel.HIGH, DrowsinessLevel.MEDIUM}
            and (not reasons or "DROWSINESS_HIGH" not in reasons)
        ):
            reasons.append("HIGH_PERCLOS")
        if distraction != DistractionLevel.NONE and not reasons:
            reasons.append("GAZE_AWAY")
        return reasons

    @staticmethod
    def _eye_state_label(face_found: bool, eye_state: EyeState) -> str:
        if not face_found or eye_state.confidence <= 0:
            return "UNKNOWN"
        return "CLOSED" if eye_state.is_closed else "OPEN"

    def _presence_state(
        self,
        face_found: bool,
        occupant_count: int,
        no_face_duration_ms: int = 0,
        session_state: str = "UNKNOWN",
        driver_proposal_visible: bool = False,
    ) -> PresenceState:
        if face_found:
            return PresenceState.PRESENT
        if driver_proposal_visible:
            return PresenceState.PROPOSAL_VISIBLE
        if session_state == DriverSessionState.LOST_TEMP.value:
            return PresenceState.LOST_TEMP
        if session_state == DriverSessionState.LOST_LONG.value:
            return PresenceState.LOST_LONG
        if occupant_count > 0:
            return PresenceState.NOT_VISIBLE
        if (
            self.occupants.driver_last_seen_ms is not None
            and no_face_duration_ms >= self.config.no_face_timeout_ms
        ):
            return PresenceState.LOST
        return PresenceState.ABSENT

    @staticmethod
    def _occupants_state(
        selection: OccupantSelection,
        driver_body_state: str = "UNKNOWN",
        proposal_count: int | None = None,
    ) -> OccupantsState:
        driver_id = selection.driver.track_id if selection.driver is not None else None
        return OccupantsState(
            count=len(selection.faces),
            face_count=len(selection.faces),
            proposal_count=selection.proposal_count if proposal_count is None else proposal_count,
            confirmed_face_count=len(selection.faces),
            unconfirmed_proposal_count=selection.unconfirmed_proposal_count,
            rejected_proposals=selection.rejected_proposals or [],
            driver_track_id=driver_id,
            driver_zone="DRIVER",
            driver_body_present=driver_body_state == "PRESENT",
            faces=[
                OccupantFace(
                    track_id=face.track_id,
                    zone=face.zone,
                    box_norm=list(face.observation.box_norm or (0.0, 0.0, 0.0, 0.0)),
                    selected_as_driver=face.selected_as_driver,
                )
                for face in selection.faces
            ],
        )

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
