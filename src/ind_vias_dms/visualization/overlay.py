from __future__ import annotations

import cv2
import numpy as np

from ind_vias_dms.core.occupant_manager import TrackedFace
from ind_vias_dms.core.types import (
    AttentionSubstate,
    AvailabilityState,
    DMSState,
    DistractionLevel,
    DrowsinessLevel,
)
from ind_vias_dms.vision.face_proposals import FaceProposal
from ind_vias_dms.vision.face_landmarks import FaceLandmarkResult
from ind_vias_dms.vision.head_pose import HeadPose
from ind_vias_dms.visualization.colors import BLACK, GRAY, GREEN, RED, WHITE, status_color


class OverlayRenderer:
    def __init__(
        self,
        banner_min_hold_ms: int = 700,
        normal_min_hold_ms: int = 300,
        state_clear_confirm_ms: int = 800,
    ) -> None:
        self.banner_min_hold_ms = banner_min_hold_ms
        self.normal_min_hold_ms = normal_min_hold_ms
        self.state_clear_confirm_ms = state_clear_confirm_ms
        self._banner_label = "NORMAL"
        self._banner_status: object = AvailabilityState.AVAILABLE
        self._banner_since_ms: int | None = None
        self._normal_candidate_since_ms: int | None = None

    def draw(
        self,
        frame: np.ndarray,
        state: DMSState,
        face: FaceLandmarkResult,
        head_pose: HeadPose,
        fps: float,
        telemetry_enabled: bool = True,
    ) -> np.ndarray:
        return self.draw_video_overlay(
            frame,
            state,
            face,
            head_pose,
            fps,
            draw_panel=telemetry_enabled,
        )

    def draw_video_overlay(
        self,
        frame: np.ndarray,
        state: DMSState,
        face: FaceLandmarkResult,
        head_pose: HeadPose,
        fps: float,
        draw_panel: bool = True,
        max_axis_length_px: int = 80,
        max_gaze_vector_length_px: int = 100,
        draw_pose_axes: bool = True,
        draw_gaze_vector: bool = True,
        faces: list[TrackedFace] | None = None,
        draw_all_faces: bool = False,
        show_track_id: bool = False,
        face_proposals: list[FaceProposal] | None = None,
        driver_roi_norm: tuple[float, float, float, float] | None = None,
        show_debug_proposal_boxes: bool = False,
    ) -> np.ndarray:
        out = frame.copy()
        if driver_roi_norm is not None:
            self._draw_norm_roi(out, driver_roi_norm)
        if face_proposals and show_debug_proposal_boxes:
            self._draw_face_proposals(out, face_proposals)
        if state.cabin_evidence.enabled and state.cabin_evidence.evidence_objects:
            self._draw_cabin_evidence(out, state)
        if draw_all_faces and faces:
            self._draw_occupant_boxes(out, faces, state, show_track_id)
        elif face.bbox is not None:
            cv2.rectangle(out, face.bbox[:2], face.bbox[2:], GREEN, 2)
        if face.landmarks_px:
            for idx, (x, y) in face.landmarks_px.items():
                if idx % 8 == 0:
                    cv2.circle(out, (int(x), int(y)), 1, WHITE, -1)
        if head_pose.confidence >= 0.3 and face.landmarks_px:
            if draw_pose_axes:
                self._draw_head_axis(out, face, head_pose, max_axis_length_px)
            if draw_gaze_vector:
                self._draw_gaze_hint(out, face, state, max_gaze_vector_length_px)
        if draw_panel:
            self._draw_panel(out, state, fps)
        self._draw_banner(out, state)
        return out

    def render_status_dashboard(
        self,
        state: DMSState,
        fps: float,
        width: int = 480,
        height: int = 720,
        road_yaw_offset_deg: float = 0.0,
        road_pitch_offset_deg: float = 0.0,
        road_calibrated: bool = False,
        vehicle_layout: str = "RHD",
        driver_image_side: str = "LEFT",
        camera_mount_position: str = "DASHBOARD_FRONT",
        camera_view_direction: str = "CABIN_REARWARD",
        driver_roi_state: str = "AUTO_LEFT",
    ) -> np.ndarray:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (24, 24, 24)
        cv2.rectangle(canvas, (0, 0), (width, 56), status_color(state.driver_availability.state), -1)
        cv2.putText(
            canvas,
            "IND-VIAS DualSight DMS",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            BLACK,
            2,
        )
        y = 70
        for label, value in status_dashboard_lines(
            state,
            fps,
            road_yaw_offset_deg,
            road_pitch_offset_deg,
            road_calibrated,
            vehicle_layout,
            driver_image_side,
            camera_mount_position,
            camera_view_direction,
            driver_roi_state,
        ):
            cv2.putText(canvas, label, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, GRAY, 1)
            cv2.putText(canvas, value[:36], (164, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)
            y += 20
        return canvas

    def render_vehicle_monitor(self, state: DMSState, width: int = 520, height: int = 260) -> np.ndarray:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (20, 22, 24)
        cv2.rectangle(canvas, (0, 0), (width, 48), (80, 120, 140), -1)
        cv2.putText(
            canvas,
            "IND-VIAS Vehicle Monitor",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            BLACK,
            2,
        )
        rows = vehicle_monitor_lines(state)
        y = 70
        for label, value in rows:
            cv2.putText(canvas, label, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, GRAY, 1)
            cv2.putText(canvas, value[:42], (168, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, WHITE, 1)
            y += 24
        return canvas

    def _draw_occupant_boxes(
        self,
        frame: np.ndarray,
        faces: list[TrackedFace],
        state: DMSState,
        show_track_id: bool,
    ) -> None:
        for tracked in faces:
            box = tracked.observation.bbox
            if box is None:
                continue
            if tracked.selected_as_driver:
                color = GREEN
            elif tracked.zone == "FRONT_PASSENGER":
                color = (0, 220, 255)
            else:
                color = (255, 180, 40)
            cv2.rectangle(frame, box[:2], box[2:], color, 2)
            cv2.putText(
                frame,
                occupant_label(
                    tracked.zone,
                    tracked.track_id,
                    tracked.selected_as_driver,
                    state.driver_identity.driver_session_id,
                    show_track_id,
                ),
                (box[0], max(20, box[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

    def _draw_norm_roi(
        self,
        frame: np.ndarray,
        roi: tuple[float, float, float, float],
    ) -> None:
        height, width = frame.shape[:2]
        x1, y1 = int(roi[0] * width), int(roi[1] * height)
        x2, y2 = int(roi[2] * width), int(roi[3] * height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 180, 255), 1)

    def _draw_face_proposals(
        self,
        frame: np.ndarray,
        proposals: list[FaceProposal],
    ) -> None:
        for proposal in proposals:
            color = (180, 160, 120)
            cv2.rectangle(frame, proposal.bbox[:2], proposal.bbox[2:], color, 1)
            cv2.putText(
                frame,
                "RAW PROPOSAL / NOT VALIDATED",
                (proposal.bbox[0], max(18, proposal.bbox[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )

    def _draw_cabin_evidence(self, frame: np.ndarray, state: DMSState) -> None:
        height, width = frame.shape[:2]
        if state.cabin_evidence.detector_backend == "synthetic" and state.cabin_evidence.synthetic_active:
            cv2.putText(
                frame,
                "CABIN EVIDENCE: SYNTHETIC TEST MODE",
                (18, max(28, height - 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (80, 220, 255),
                2,
                cv2.LINE_AA,
            )
        for obj in state.cabin_evidence.evidence_objects:
            if not obj.bbox:
                continue
            x1, y1, x2, y2 = _bbox_to_px(obj.bbox, width, height)
            color = (80, 220, 255)
            label = _cabin_evidence_label(
                obj.object_type.value,
                obj.state.value,
                obj.source,
                obj.relation_to_driver.value,
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                frame,
                label,
                (x1, max(18, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
            )

    def _draw_head_axis(
        self,
        frame: np.ndarray,
        face: FaceLandmarkResult,
        pose: HeadPose,
        max_axis_length_px: int,
    ) -> None:
        if pose.rvec is None or pose.tvec is None or pose.camera_matrix is None or pose.dist_coeffs is None:
            return
        axis = np.float32([[50, 0, 0], [0, 50, 0], [0, 0, 50]])
        points, _ = cv2.projectPoints(axis, pose.rvec, pose.tvec, pose.camera_matrix, pose.dist_coeffs)
        origin = tuple(int(v) for v in face.landmarks_px[1])
        for point, color in zip(points.reshape(-1, 2), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            endpoint = clamp_endpoint(origin, (float(point[0]), float(point[1])), frame.shape, max_axis_length_px)
            cv2.line(frame, origin, endpoint, color, 2)

    def _draw_gaze_hint(
        self,
        frame: np.ndarray,
        face: FaceLandmarkResult,
        state: DMSState,
        max_gaze_vector_length_px: int,
    ) -> None:
        if face.landmarks_px is None or 1 not in face.landmarks_px:
            return
        x, y = face.landmarks_px[1]
        origin = (int(x), int(y))
        endpoint = clamp_endpoint(
            origin,
            (x + int(state.gaze.head_yaw_deg * 2), y + int(state.gaze.head_pitch_deg * 2)),
            frame.shape,
            max_gaze_vector_length_px,
        )
        cv2.arrowedLine(frame, origin, endpoint, RED, 2)

    def _draw_panel(self, frame: np.ndarray, state: DMSState, fps: float) -> None:
        cv2.rectangle(frame, (0, 0), (330, 310), BLACK, -1)
        cv2.rectangle(frame, (0, 0), (330, 310), GRAY, 1)
        lines = [
            f"FPS: {fps:.1f}",
            f"Driver: {state.driver_presence.state.value}",
            f"Gaze: {state.gaze.zone.value}",
            f"Yaw/Pitch/Roll: {state.gaze.head_yaw_deg:.1f}/{state.gaze.head_pitch_deg:.1f}/{state.gaze.head_roll_deg:.1f}",
            f"Eyes: {state.drowsiness.eye_state}",
            f"PERCLOS: {state.drowsiness.perclos_5s:.2f}/{state.drowsiness.perclos_60s:.2f}",
            f"Attention: {state.attention.attention_state.value}",
            f"Drowsiness: {state.drowsiness.level.value}",
            f"Distraction: {state.distraction.level.value}",
            f"Availability: {state.driver_availability.state.value}",
            f"Readiness: {state.driver_readiness_score.score_0_to_1:.2f}",
        ]
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (12, 26 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    def _draw_banner(self, frame: np.ndarray, state: DMSState) -> None:
        label, status = self._stable_banner(state)
        color = status_color(status)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), color, -1)
        cv2.putText(frame, label, (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, BLACK, 2)

    def _stable_banner(self, state: DMSState) -> tuple[str, object]:
        label, status = banner_decision(state)
        now_ms = state.timestamp_ms
        if self._banner_since_ms is None:
            self._banner_since_ms = now_ms
            self._banner_label = label
            self._banner_status = status
            return label, status
        elapsed_ms = now_ms - self._banner_since_ms
        if label == self._banner_label:
            if label != "NORMAL":
                self._normal_candidate_since_ms = None
            return self._banner_label, self._banner_status
        if label == "NORMAL":
            if self._normal_candidate_since_ms is None:
                self._normal_candidate_since_ms = now_ms
            if (
                now_ms - self._normal_candidate_since_ms < self.state_clear_confirm_ms
                or elapsed_ms < self.banner_min_hold_ms
            ):
                return self._banner_label, self._banner_status
        elif self._banner_label == "NORMAL" and elapsed_ms < self.normal_min_hold_ms:
            return self._banner_label, self._banner_status
        self._banner_label = label
        self._banner_status = status
        self._banner_since_ms = now_ms
        self._normal_candidate_since_ms = None
        return label, status


def banner_decision(state: DMSState) -> tuple[str, object]:
        if state.dms_v02.final_banner and state.dms_v02.final_decision_path:
            banner = state.dms_v02.final_banner
            label = state.dms_v02.hmi_banner_text or banner
            if state.vehicle.dms_operational_mode in {"STARTUP_INITIALIZING", "STANDBY"}:
                return label, DistractionLevel.LOW
            if banner == "NORMAL":
                return label, AvailabilityState.AVAILABLE
            if banner == "DMS MONITOR":
                return label, DistractionLevel.LOW
            if banner == "DMS DEGRADED":
                return label, AvailabilityState.DEGRADED
            if banner == "DISTRACTION WARNING":
                return label, DistractionLevel.MEDIUM
            if banner == "DROWSINESS WARNING":
                return label, DrowsinessLevel.MEDIUM
            if banner == "DANGER":
                return label, DrowsinessLevel.HIGH
            if banner == "DRIVER UNAVAILABLE":
                return label, AvailabilityState.UNAVAILABLE
        label = "NORMAL"
        status = AvailabilityState.AVAILABLE
        if state.driver_availability.state == AvailabilityState.UNAVAILABLE:
            label = "DRIVER UNAVAILABLE"
            status = AvailabilityState.UNAVAILABLE
        elif state.drowsiness.level == DrowsinessLevel.MICROSLEEP:
            label = "MICROSLEEP WARNING"
            status = DrowsinessLevel.MICROSLEEP
        elif state.drowsiness.level in {DrowsinessLevel.HIGH, DrowsinessLevel.MEDIUM}:
            label = "DROWSINESS WARNING"
            status = state.drowsiness.level
        elif state.distraction.level in {DistractionLevel.HIGH, DistractionLevel.MEDIUM}:
            label = "DISTRACTION WARNING"
            status = state.distraction.level
        elif state.attention.attention_substate in {
            AttentionSubstate.VISUAL_DISTRACTION,
            AttentionSubstate.HEAD_DOWN_DISTRACTION,
            AttentionSubstate.PHONE_SUSPECTED,
            AttentionSubstate.PHONE_DOWN_SUSPECTED,
            AttentionSubstate.PHONE_TO_EAR_SUSPECTED,
            AttentionSubstate.TEXTING_SUSPECTED,
        }:
            label = "DISTRACTION WARNING"
            status = DistractionLevel.MEDIUM
        elif state.driver_availability.state == AvailabilityState.DEGRADED:
            label = "DMS DEGRADED"
            status = AvailabilityState.DEGRADED
        return label, status


def status_dashboard_lines(
    state: DMSState,
    fps: float,
    road_yaw_offset_deg: float = 0.0,
    road_pitch_offset_deg: float = 0.0,
    road_calibrated: bool = False,
    vehicle_layout: str = "RHD",
    driver_image_side: str = "LEFT",
    camera_mount_position: str = "DASHBOARD_FRONT",
    camera_view_direction: str = "CABIN_REARWARD",
    driver_roi_state: str = "AUTO_LEFT",
) -> list[tuple[str, str]]:
    return [
        ("FPS", f"{fps:.1f}"),
        ("Camera health", state.dms_health.camera_status.value),
        ("Face detection", state.dms_health.face_detection_status.value),
        ("Face backend", state.dms_health.face_backend),
        ("NIR mode", state.dms_health.nir_mode),
        ("Input mode", state.dms_health.input_color_mode),
        ("Threshold profile", state.dms_health.active_eye_threshold_profile),
        ("NIR active", "YES" if state.dms_health.nir_preprocessing_active else "NO"),
        ("NIR reason", ",".join(state.dms_health.nir_reason_codes) or "NONE"),
        ("Face proposals", str(state.dms_health.face_proposals)),
        ("Face det conf", f"{state.dms_health.face_detection_confidence:.2f}"),
        ("Driver face", state.driver_presence.state.value),
        ("Driver face state", state.driver_identity.driver_face_state),
        ("Proposal state", state.driver_identity.face_proposal_state),
        ("Track hold", state.driver_identity.driver_track_hold_state),
        ("Vehicle gate", state.vehicle.dms_speed_gate_state),
        ("Vehicle speed", f"{state.vehicle.ego_vehicle_speed_kph:.1f} km/h"),
        (
            "Indicators",
            f"L={'ON' if state.vehicle.left_indicator_on else 'OFF'} "
            f"R={'ON' if state.vehicle.right_indicator_on else 'OFF'}",
        ),
        ("Cabin backend", state.cabin_evidence.detector_backend),
        ("Cabin objects", str(state.cabin_evidence.cabin_evidence_count)),
        ("Cabin phone", state.cabin_evidence.phone_state.value),
        ("Cabin phone rel", state.cabin_evidence.phone_relation or "NONE"),
        ("Cabin belt", state.cabin_evidence.seatbelt_state.value),
        ("Cabin smoking", state.cabin_evidence.smoking_state.value),
        ("Cabin affect", "YES" if state.cabin_evidence.affect_final_dms_state else "NO"),
        ("HMI banner", state.dms_v02.hmi_banner_text or state.dms_v02.final_banner),
        (
            "Head angle",
            "Yaw "
            f"{_angle_label(state.gaze.relative_yaw_deg, 'L', 'R')} | Pitch "
            f"{_angle_label(state.gaze.relative_pitch_deg, 'U', 'D')} | Roll "
            f"{_angle_label(state.gaze.relative_roll_deg, 'L', 'R')} | Vector "
            f"{state.gaze.head_angle_from_road_deg:.0f} | Q "
            f"{state.gaze.head_pose_vector_quality:.2f}",
        ),
        (
            "Head raw/rel",
            f"raw {state.gaze.head_pose_raw_yaw_deg:.1f}/"
            f"{state.gaze.head_pose_raw_pitch_deg:.1f}/{state.gaze.head_pose_raw_roll_deg:.1f} | "
            f"rel {state.gaze.relative_yaw_deg:.1f}/"
            f"{state.gaze.relative_pitch_deg:.1f}/{state.gaze.relative_roll_deg:.1f}",
        ),
        ("Observability", state.driver_observability.state.value),
        ("Obs reason", ",".join(state.driver_observability.reason_codes) or "NONE"),
        ("Raw eyes", state.drowsiness.raw_eye_state),
        ("Effective eyes", state.drowsiness.effective_eye_state),
        ("Eye raw/norm", f"{state.drowsiness.eye_openness_raw:.2f}/{state.drowsiness.eye_openness_normalized:.2f}"),
        ("Eye calib", state.drowsiness.eye_calibration_state),
        ("Eye visibility", f"{state.drowsiness.eye_visibility_score:.2f}"),
        ("Closure ms", str(state.drowsiness.eye_closure_duration_ms)),
        ("PERCLOS usable", "YES" if state.drowsiness.perclos_valid else "NO"),
        (
            "PERCLOS reason",
            ",".join(state.drowsiness.perclos_validity_reason_codes) or "VALID",
        ),
        ("PERCLOS 5s/60s", f"{state.drowsiness.perclos_5s:.2f} / {state.drowsiness.perclos_60s:.2f}"),
        ("PERCLOS valid", f"{state.drowsiness.perclos_valid_time_5s_ms}/{state.drowsiness.perclos_valid_time_60s_ms}"),
        ("Drowsiness", state.drowsiness.level.value),
        ("Distraction", state.distraction.level.value),
        ("v0.2 level", state.dms_v02.final_level.value),
        ("v0.2 banner", state.dms_v02.final_banner),
        ("Drowsy head", state.dms_v02.drowsiness_state),
        ("Distract head", state.dms_v02.distraction_state),
        ("Avail head", state.dms_v02.driver_availability_state),
        ("Conf head", state.dms_v02.dms_confidence_state.value),
        ("Phone state", state.phone_use.driver_state),
        ("Phone reason", ",".join(state.phone_use.reason_codes) or "NONE"),
        ("Attention", state.attention.attention_state.value),
        ("Substate", state.attention.attention_substate.value),
        ("Attn conf", f"{state.attention.attention_confidence:.2f}"),
        ("Attn reason", ",".join(state.attention.attention_reason_codes) or "NONE"),
        ("Head-down ms", str(state.attention.head_down_duration_ms)),
        ("Head-down uncertain", str(state.attention.head_down_uncertain_duration_ms)),
        ("Gaze-offroad ms", str(state.attention.gaze_offroad_duration_ms)),
        ("Phone-down ms", str(state.attention.phone_down_candidate_duration_ms)),
        ("Pose reliable", "YES" if state.attention.pose_reliable else "NO"),
        ("Attn source", state.attention.effective_attention_source),
        ("Final path", state.attention.final_decision_path or "NONE"),
        ("v0.2 path", state.dms_v02.final_decision_path or "NONE"),
        ("Availability", state.driver_availability.state.value),
        (
            "Reason codes",
            ", ".join(state.driver_availability.reason_codes)
            if state.driver_availability.reason_codes
            else "NONE",
        ),
        ("Driver session", str(state.driver_identity.driver_session_id)),
        ("Driver track", f"T{state.driver_identity.driver_track_id}" if state.driver_identity.driver_track_id is not None else "None"),
        ("Session state", state.driver_identity.session_state),
        ("Reassociated", "YES" if state.driver_identity.reassociated else "NO"),
        ("Time since seen", str(state.driver_identity.time_since_seen_ms)),
        ("Driver body", state.driver_identity.driver_body_state),
        ("Driver seat zone", state.driver_identity.driver_slot_assignment),
        ("Driver slot conf", f"{state.driver_identity.driver_candidate_score:.2f}"),
        ("Driver front score", f"{state.driver_identity.driver_front_layer_score:.2f}"),
        ("Driver depth", state.driver_identity.candidate_depth_layer),
        ("Driver slot reason", state.driver_identity.driver_slot_reason or "NONE"),
        ("Driver validation", state.driver_identity.driver_validation_state),
        ("Proposal conf", f"{state.driver_identity.driver_proposal_confidence:.2f}"),
        (
            "Face quality",
            f"comp={state.driver_identity.driver_face_completeness_score:.2f} "
            f"cov={state.driver_identity.driver_landmark_coverage_score:.2f}",
        ),
        ("Landmarks", str(state.driver_identity.driver_landmark_count)),
        ("Partial face", "YES" if state.driver_identity.driver_partial_face else "NO"),
        (
            "Face reject",
            ",".join(state.driver_identity.driver_validation_reasons) or "NONE",
        ),
        (
            "Occupants",
            f"faces={state.occupants.face_count} proposals={state.occupants.proposal_count} "
            f"pending={state.occupants.unconfirmed_proposal_count} body={state.occupants.driver_body_present}",
        ),
        ("Occupancy", str(state.occupancy.cabin_occupant_count)),
        ("Driver seat", state.occupancy.seats.get("driver", None).occupied if state.occupancy.seats.get("driver") else "unknown"),
        (
            "Front passenger",
            state.occupancy.seats.get("front_passenger", None).occupied
            if state.occupancy.seats.get("front_passenger")
            else "unknown",
        ),
        (
            "Rear L/C/R",
            f"{state.occupancy.rear_left_present}/{state.occupancy.rear_center_present}/{state.occupancy.rear_right_present}",
        ),
        ("Occ conf", f"{state.occupancy.occupancy_confidence:.2f}"),
        ("Occ reason", ",".join(state.occupancy.occupancy_reason_codes) or "NONE"),
        ("Gaze", state.gaze.zone.value),
        ("Gaze confidence", f"{state.gaze.confidence:.2f}"),
        (
            "Yaw/Pitch/Roll",
            f"{state.gaze.head_yaw_deg:.1f} / "
            f"{state.gaze.head_pitch_deg:.1f} / {state.gaze.head_roll_deg:.1f}",
        ),
        (
            "Head angle raw",
            f"raw yaw/pitch/roll = {state.gaze.head_pose_raw_yaw_deg:.1f}/"
            f"{state.gaze.head_pose_raw_pitch_deg:.1f}/{state.gaze.head_pose_raw_roll_deg:.1f} deg | "
            f"road-relative yaw/pitch/roll = {state.gaze.relative_yaw_deg:.1f}/"
            f"{state.gaze.relative_pitch_deg:.1f}/{state.gaze.relative_roll_deg:.1f} deg",
        ),
        ("Head vector quality", f"{state.gaze.head_pose_vector_quality:.2f}"),
        ("Side glance", f"{state.attention.side_glance_state} {state.attention.side_glance_duration_ms}ms"),
        ("Road offsets", f"{road_yaw_offset_deg:.1f} / {road_pitch_offset_deg:.1f}"),
        ("Road calib", "CALIBRATED" if road_calibrated else "NOT_CALIBRATED"),
        ("Road source", getattr(state.gaze, "calibration_source", "DEFAULT")),
        ("Vehicle layout", vehicle_layout),
        ("Driver image side", driver_image_side),
        ("Camera mount", camera_mount_position),
        ("View direction", camera_view_direction),
        ("Driver ROI", driver_roi_state),
        ("Cabin mobile", ", ".join(state.phone_use.cabin_events) or "NONE"),
        ("Readiness", f"{state.driver_readiness_score.score_0_to_1:.2f}"),
        ("Risk", state.driver_readiness_score.risk_level.value),
    ]


def vehicle_monitor_lines(state: DMSState) -> list[tuple[str, str]]:
    return [
        ("Speed", f"{state.vehicle.ego_vehicle_speed_kph:.1f} km/h ({state.vehicle.ego_vehicle_speed_source})"),
        ("Gate", state.vehicle.dms_speed_gate_state),
        ("Alerts", "ENABLED" if state.vehicle.dms_alerts_enabled else "SUPPRESSED"),
        ("Suppress reason", state.vehicle.dms_alert_suppression_reason),
        ("Indicators", f"L={'ON' if state.vehicle.left_indicator_on else 'OFF'} R={'ON' if state.vehicle.right_indicator_on else 'OFF'}"),
        ("Sanctioned task", state.vehicle.sanctioned_task_state),
        ("Vehicle reason", ",".join(state.vehicle.vehicle_speed_reason_codes) or "NONE"),
        ("Indicator reason", ",".join(state.vehicle.indicator_reason_codes) or "NONE"),
        ("HMI", state.vehicle.hmi_banner_text or state.dms_v02.final_banner),
        ("Timing", f"fps={state.vehicle.live_output_fps:.1f} proc={state.vehicle.processing_time_ms:.1f}ms"),
    ]


def _angle_label(value: float, negative_label: str, positive_label: str, limit: float = 90.0) -> str:
    if abs(value) > limit:
        return "POSE_OUT_OF_HUMAN_RANGE"
    if round(abs(value)) == 0:
        return "0"
    direction = positive_label if value >= 0 else negative_label
    return f"{direction}{abs(value):02.0f}"


def _bbox_to_px(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(0, min(width - 1, int(round(x2)))),
        max(0, min(height - 1, int(round(y2)))),
    )


def _cabin_evidence_label(
    object_type: str,
    lifecycle: str,
    source: str = "",
    relation: str = "",
) -> str:
    prefix = "SYNTH " if source == "synthetic" else ""
    if object_type == "PHONE":
        if prefix:
            return f"{prefix}PHONE / {relation or lifecycle}"
        if lifecycle in {"SUSPECTED", "CONFIRMED"}:
            return f"PHONE {lifecycle}"
        return "PHONE CANDIDATE"
    if object_type == "SEATBELT":
        if prefix:
            return f"{prefix}SEATBELT / {relation or lifecycle}"
        return "SEATBELT WORN" if lifecycle == "CONFIRMED" else "SEATBELT UNKNOWN"
    if object_type in {"CIGARETTE", "HAND"}:
        if prefix:
            return f"{prefix}{object_type} / {relation or lifecycle}"
        return "SMOKING CANDIDATE" if lifecycle == "CANDIDATE" else f"SMOKING {lifecycle}"
    return f"{prefix}CABIN EVIDENCE"


def occupant_label(
    zone: str,
    track_id: int,
    selected_as_driver: bool = False,
    driver_session_id: str | None = None,
    show_track_id: bool = False,
) -> str:
    if selected_as_driver and driver_session_id:
        return f"DRIVER {driver_session_id} / T{track_id}" if show_track_id else f"DRIVER {driver_session_id}"
    label_zone = "DRIVER" if selected_as_driver else zone
    return f"{label_zone} T{track_id}"


def clamp_endpoint(
    origin: tuple[int, int],
    endpoint: tuple[float, float],
    frame_shape: tuple[int, ...],
    max_length_px: int,
) -> tuple[int, int]:
    height, width = frame_shape[:2]
    ox, oy = origin
    ex, ey = endpoint
    if not np.isfinite(ex) or not np.isfinite(ey):
        return origin
    dx = ex - ox
    dy = ey - oy
    length = float(np.hypot(dx, dy))
    if length > max_length_px and length > 0:
        scale = max_length_px / length
        ex = ox + dx * scale
        ey = oy + dy * scale
    ex = min(max(0, int(round(ex))), width - 1)
    ey = min(max(0, int(round(ey))), height - 1)
    return ex, ey
