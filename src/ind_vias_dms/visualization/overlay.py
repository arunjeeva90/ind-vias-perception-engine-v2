from __future__ import annotations

import cv2
import numpy as np

from ind_vias_dms.core.occupant_manager import TrackedFace
from ind_vias_dms.core.types import (
    AttentionSubstate,
    AvailabilityState,
    CabinEvidenceObject,
    DMSState,
    DistractionLevel,
    DrowsinessLevel,
)
from ind_vias_dms.vision.face_proposals import FaceProposal
from ind_vias_dms.vision.face_landmarks import FaceLandmarkResult
from ind_vias_dms.vision.head_pose import HeadPose
from ind_vias_dms.visualization.colors import BLACK, GRAY, GREEN, RED, WHITE, status_color


VIDEO_HEADER_HEIGHT = 46
VIDEO_SIDEBAR_WIDTH = 190


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
        if draw_panel:
            self._draw_panel(out, state, fps)
        # Pose/gaze vectors are rendered in the dedicated instrument strip so
        # they never obscure the driver's eyes or facial landmarks.
        out = self._add_pose_instrument_strip(
            out,
            state,
            head_pose,
            draw_pose_axes=draw_pose_axes,
            draw_gaze_vector=draw_gaze_vector,
        )
        return self._add_video_header(out, state)

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
        eye_runtime_source: str = "LANDMARK_EAR",
        eye_model_status: str = "DISABLED",
        landmark_106_status: str = "DISABLED",
        runtime_metrics: dict[str, object] | None = None,
        compute_backend: str = "CPU",
        npu_active: bool = False,
    ) -> np.ndarray:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (18, 21, 27)
        ui_scale = min(width / 720.0, height / 1000.0)
        ui_scale = max(1.0, ui_scale)
        header_color = status_color(state.driver_availability.state)
        cv2.rectangle(
            canvas,
            (0, 0),
            (width, int(round(68 * ui_scale))),
            header_color,
            -1,
        )
        cv2.circle(
            canvas,
            (int(round(28 * ui_scale)), int(round(34 * ui_scale))),
            int(round(12 * ui_scale)),
            (20, 24, 30),
            -1,
        )
        cv2.circle(
            canvas,
            (int(round(28 * ui_scale)), int(round(34 * ui_scale))),
            int(round(6 * ui_scale)),
            WHITE,
            max(2, int(round(2 * ui_scale))),
        )
        cv2.putText(
            canvas,
            "IND-VIAS DualSight DMS",
            (int(round(52 * ui_scale)), int(round(42 * ui_scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78 * ui_scale,
            BLACK,
            max(2, int(round(2 * ui_scale))),
            cv2.LINE_AA,
        )
        rows = status_dashboard_lines(
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
            eye_runtime_source,
            eye_model_status,
            landmark_106_status,
            runtime_metrics,
            compute_backend,
            npu_active,
        )
        if width < 650 or height < 850:
            _draw_compact_dashboard(canvas, _prioritize_status_dashboard_lines(rows), top=82)
            return canvas

        values = dict(rows)
        metrics = dict(runtime_metrics or {})
        feature_latency = _metric_text(
            metrics.get("feature_latency_ms", metrics.get("frame_latency_ms")),
            "ms",
        )
        inference_latency = _metric_text(metrics.get("inference_time_ms"), "ms")
        capture_fps = _metric_text(metrics.get("capture_fps"), " FPS")
        processing_fps = _metric_text(
            metrics.get("processing_fps", fps),
            " FPS",
        )
        inference_fps = _metric_text(metrics.get("inference_fps_actual"), " FPS")
        cpu_ram = _cpu_ram_text(metrics)
        npu_text = "ACTIVE" if npu_active else "NOT ACTIVE"
        npu_tops = _npu_tops_text(metrics, npu_active)

        margin = 14
        gap = 12
        logical_width = int(round(width / ui_scale))
        column_width = (logical_width - margin * 2 - gap) // 2
        left_x = margin
        right_x = margin + column_width + gap

        _draw_dashboard_card(
            canvas,
            "RUNTIME",
            [
                ("Processing", processing_fps),
                ("Capture", capture_fps),
                ("Inference", inference_fps),
                ("Feature latency", feature_latency),
                ("Model latency", inference_latency),
                ("CPU / RAM", cpu_ram),
                ("Compute", compute_backend),
                ("NPU", npu_text),
                ("NPU TOPS", npu_tops),
            ],
            (left_x, 84, column_width, 294),
            (255, 184, 72),
            ui_scale=ui_scale,
        )
        _draw_dashboard_card(
            canvas,
            "DRIVER & EYES",
            [
                ("Driver", values.get("Driver face", "UNKNOWN")),
                ("Face state", values.get("Driver face state", "UNKNOWN")),
                ("Track", values.get("Driver track", "NONE")),
                ("Face backend", values.get("Face backend", "UNKNOWN")),
                ("Eye source", values.get("Eye runtime", "UNKNOWN")),
                ("Eye CNN", values.get("Eye CNN", "DISABLED")),
                ("106 evidence", values.get("106 geometry", "DISABLED")),
                (
                    "106 latency",
                    _metric_text(metrics.get("landmark_106_inference_ms"), "ms"),
                ),
                ("Raw / effective", f"{values.get('Raw eyes', 'UNKNOWN')} / {values.get('Effective eyes', 'UNKNOWN')}"),
                ("Openness", values.get("Eye raw/norm", "0.00 / 0.00")),
                ("Visibility", values.get("Eye visibility", "0.00")),
                ("Closure", f"{values.get('Closure ms', '0')} ms"),
                ("PERCLOS", values.get("PERCLOS 5s/60s", "0.00 / 0.00")),
                ("Drowsiness", values.get("Drowsiness", "UNKNOWN")),
            ],
            (right_x, 84, column_width, 366),
            (78, 214, 160),
            ui_scale=ui_scale,
        )
        _draw_dashboard_card(
            canvas,
            "HEAD & ROAD",
            [
                ("Head angle", values.get("Head angle", "UNKNOWN")),
                ("Raw / relative", values.get("Head raw/rel", "UNKNOWN")),
                ("Gaze", values.get("Gaze", "UNKNOWN")),
                ("Confidence", values.get("Gaze confidence", "0.00")),
                ("Road calibration", values.get("Road calib", "NOT_CALIBRATED")),
                ("Road source", values.get("Road source", "DEFAULT")),
                ("Road offsets", values.get("Road offsets", "0.0 / 0.0")),
                ("Attention", values.get("Attention", "UNKNOWN")),
                ("Distraction", values.get("Distraction", "UNKNOWN")),
            ],
            (left_x, 464, logical_width - margin * 2, 230),
            (104, 164, 255),
            value_fraction=0.78,
            ui_scale=ui_scale,
        )
        _draw_dashboard_card(
            canvas,
            "VEHICLE & CABIN",
            [
                ("Speed", values.get("Vehicle speed", "0.0 km/h")),
                ("DMS gate", values.get("Vehicle gate", "UNKNOWN")),
                ("Indicators", values.get("Indicators", "L=OFF R=OFF")),
                ("Cabin backend", values.get("Cabin backend", "dummy")),
                ("Cabin status", values.get("Cabin status", "UNKNOWN")),
                ("Occupants", values.get("Occupancy", "0")),
                ("Phone", values.get("Cabin phone obs", "NO")),
                ("Seat belt", values.get("Cabin belt", "UNKNOWN")),
                ("Smoking", values.get("Cabin smoking", "UNKNOWN")),
            ],
            (left_x, 708, logical_width - margin * 2, 144),
            (190, 122, 255),
            columns=2,
            ui_scale=ui_scale,
        )
        _draw_dashboard_card(
            canvas,
            "LIVE DECISION",
            [
                ("HMI", values.get("HMI banner", "DMS MONITOR")),
                ("Availability", values.get("Availability", "UNKNOWN")),
                ("Attention", values.get("Attention", "UNKNOWN")),
                ("Risk", values.get("Risk", "UNKNOWN")),
            ],
            (left_x, 866, logical_width - margin * 2, 120),
            header_color,
            value_fraction=0.77,
            ui_scale=ui_scale,
        )
        return canvas

    def render_vehicle_monitor(
        self,
        state: DMSState,
        width: int = 520,
        height: int = 320,
        runtime_metrics: dict[str, object] | None = None,
        compute_backend: str = "CPU",
        npu_active: bool = False,
    ) -> np.ndarray:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (18, 21, 27)
        ui_scale = max(0.66, min(width / 780.0, height / 390.0))
        logical_width = int(round(width / ui_scale))
        cv2.rectangle(
            canvas,
            (0, 0),
            (width, int(round(58 * ui_scale))),
            (128, 111, 72),
            -1,
        )
        cv2.putText(
            canvas,
            "IND-VIAS Vehicle Monitor",
            (int(round(20 * ui_scale)), int(round(38 * ui_scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72 * ui_scale,
            WHITE,
            max(1, int(round(2 * ui_scale))),
            cv2.LINE_AA,
        )
        metrics = dict(runtime_metrics or {})
        left_rows = [
            ("Speed", f"{state.vehicle.ego_vehicle_speed_kph:.1f} km/h"),
            ("DMS gate", state.vehicle.dms_speed_gate_state),
            ("Alerts", "ENABLED" if state.vehicle.dms_alerts_enabled else "SUPPRESSED"),
            ("Indicators", f"L={'ON' if state.vehicle.left_indicator_on else 'OFF'} R={'ON' if state.vehicle.right_indicator_on else 'OFF'}"),
        ]
        right_rows = [
            ("Feature latency", _metric_text(metrics.get("feature_latency_ms", metrics.get("frame_latency_ms")), "ms")),
            ("Processing", _metric_text(metrics.get("processing_fps"), " FPS")),
            ("Compute", compute_backend),
            ("NPU / TOPS", f"{'ACTIVE' if npu_active else 'OFF'} / {_npu_tops_text(metrics, npu_active)}"),
        ]
        half = (logical_width - 36) // 2
        _draw_dashboard_card(
            canvas,
            "VEHICLE",
            left_rows,
            (12, 70, half, 142),
            (255, 184, 72),
            ui_scale=ui_scale,
        )
        _draw_dashboard_card(
            canvas,
            "PERFORMANCE",
            right_rows,
            (24 + half, 70, half, 142),
            (104, 164, 255),
            ui_scale=ui_scale,
        )
        _draw_dashboard_card(
            canvas,
            "HMI",
            [("Decision", state.vehicle.hmi_banner_text or state.dms_v02.final_banner)],
            (12, 222, logical_width - 24, 156),
            status_color(state.driver_availability.state),
            value_fraction=0.80,
            ui_scale=ui_scale,
        )
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
            else:
                color = (255, 190, 60)
            cv2.rectangle(frame, box[:2], box[2:], color, 2)
            label = occupant_label(
                tracked.zone,
                tracked.track_id,
                tracked.selected_as_driver,
                state.driver_identity.driver_session_id,
                show_track_id,
            )
            self._draw_box_label(frame, box, label, color)

    @staticmethod
    def _draw_box_label(
        frame: np.ndarray,
        box: tuple[int, int, int, int],
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.50
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
        x = max(0, box[0])
        label_bottom = max(text_h + baseline + 8, box[1])
        label_top = max(0, label_bottom - text_h - baseline - 8)
        cv2.rectangle(
            frame,
            (x, label_top),
            (min(frame.shape[1] - 1, x + text_w + 12), label_bottom),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x + 6, label_bottom - baseline - 4),
            font,
            scale,
            BLACK,
            thickness,
            cv2.LINE_AA,
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
        visible_objects, hidden_count = _visible_cabin_evidence_objects(state)
        state.cabin_evidence.overlay_phone_hidden_duplicate_count = hidden_count
        state.cabin_evidence.overlay_phone_suppressed_duplicates = hidden_count
        drawn_labels: list[str] = []
        drawn_boxes: list[list[float]] = []
        for obj in visible_objects:
            if not obj.bbox:
                continue
            x1, y1, x2, y2 = _bbox_to_px(obj.bbox, width, height)
            semantic_level = _phone_overlay_semantic_level(obj, state)
            color = _cabin_evidence_color(obj, semantic_level)
            label = _cabin_evidence_label(
                obj.object_type.value,
                obj.state.value,
                obj.source,
                obj.relation_to_driver.value,
                semantic_level,
                state.cabin_evidence.ignored_phone_reasons,
            )
            drawn_labels.append(label)
            drawn_boxes.append([float(v) for v in obj.bbox[:4]])
            if obj.object_type.value == "PHONE" and semantic_level == "HELD":
                state.cabin_evidence.overlay_phone_track_label = label
                state.cabin_evidence.overlay_phone_track_is_held = True
            if obj.object_type.value == "PHONE" and semantic_level not in {"IGNORED", "HELD"}:
                mask = frame.copy()
                cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 220, 255), -1)
                cv2.addWeighted(mask, 0.40, frame, 0.60, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if obj.object_type.value == "PHONE" else 1)
            cv2.putText(
                frame,
                label,
                (x1, max(18, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
            )
        state.cabin_evidence.overlay_phone_drawn_count = len(drawn_labels)
        state.cabin_evidence.overlay_phone_drawn_labels = drawn_labels
        state.cabin_evidence.overlay_phone_drawn_boxes = drawn_boxes
        state.cabin_evidence.phone_overlay_drawn = any(label.startswith("PHONE") for label in drawn_labels)

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

    def _add_pose_instrument_strip(
        self,
        frame: np.ndarray,
        state: DMSState,
        head_pose: HeadPose,
        *,
        draw_pose_axes: bool,
        draw_gaze_vector: bool,
    ) -> np.ndarray:
        """Put head orientation beside the camera image, not on the face."""

        height, width = frame.shape[:2]
        canvas = np.zeros(
            (height, width + VIDEO_SIDEBAR_WIDTH, 3),
            dtype=np.uint8,
        )
        canvas[:] = (18, 21, 27)
        canvas[:, VIDEO_SIDEBAR_WIDTH:] = frame
        cv2.rectangle(
            canvas,
            (VIDEO_SIDEBAR_WIDTH - 2, 0),
            (VIDEO_SIDEBAR_WIDTH - 1, height - 1),
            (67, 76, 90),
            -1,
        )
        cv2.putText(
            canvas,
            "3D HEAD POSE",
            (14, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (230, 235, 241),
            1,
            cv2.LINE_AA,
        )

        if draw_pose_axes:
            self._draw_pose_instrument_axis(canvas, head_pose)
        else:
            cv2.putText(
                canvas,
                "AXIS DISABLED",
                (28, 104),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                GRAY,
                1,
                cv2.LINE_AA,
            )

        text_y = min(height - 154, 176)
        text_y = max(126, text_y)
        values = (
            f"Yaw    {head_pose.yaw_deg:+6.1f} deg",
            f"Pitch  {head_pose.pitch_deg:+6.1f} deg",
            f"Roll   {head_pose.roll_deg:+6.1f} deg",
        )
        for index, text in enumerate(values):
            cv2.putText(
                canvas,
                text,
                (14, text_y + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (229, 234, 241),
                1,
                cv2.LINE_AA,
            )

        legend_y = text_y + 78
        legend = (
            ("R / X", "LATERAL", (0, 70, 255)),
            ("G / Y", "VERTICAL", (40, 220, 70)),
            ("B / Z", "DEPTH", (255, 120, 40)),
        )
        for index, (axis, meaning, color) in enumerate(legend):
            y = legend_y + index * 21
            if y >= height - 20:
                break
            cv2.putText(
                canvas,
                axis,
                (14, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                meaning,
                (70, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (192, 201, 213),
                1,
                cv2.LINE_AA,
            )

        if draw_gaze_vector and height >= 344:
            cv2.putText(
                canvas,
                f"GAZE  {state.gaze.zone.value}",
                (14, height - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 205, 80),
                1,
                cv2.LINE_AA,
            )
        return canvas

    @staticmethod
    def _draw_pose_instrument_axis(
        canvas: np.ndarray,
        head_pose: HeadPose,
    ) -> None:
        origin = np.asarray((VIDEO_SIDEBAR_WIDTH // 2, 100), dtype=np.float64)
        cv2.circle(
            canvas,
            (int(origin[0]), int(origin[1])),
            48,
            (47, 54, 65),
            1,
            cv2.LINE_AA,
        )
        if head_pose.confidence < 0.3 or head_pose.rvec is None:
            cv2.putText(
                canvas,
                "NO POSE",
                (62, 104),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                GRAY,
                1,
                cv2.LINE_AA,
            )
            return

        rotation, _ = cv2.Rodrigues(head_pose.rvec)
        axes = (
            (rotation @ np.asarray((1.0, 0.0, 0.0)), (0, 70, 255), "X"),
            (rotation @ np.asarray((0.0, 1.0, 0.0)), (40, 220, 70), "Y"),
            (rotation @ np.asarray((0.0, 0.0, 1.0)), (255, 120, 40), "Z"),
        )
        for vector, color, label in axes:
            # Orthographic instrument view with a small depth component. This
            # visualizes the same solvePnP rotation without using face pixels.
            endpoint = origin + np.asarray(
                (
                    vector[0] * 44.0 + vector[2] * 13.0,
                    -vector[1] * 44.0 + vector[2] * 13.0,
                )
            )
            end = (int(round(endpoint[0])), int(round(endpoint[1])))
            cv2.arrowedLine(
                canvas,
                (int(origin[0]), int(origin[1])),
                end,
                color,
                2,
                cv2.LINE_AA,
                tipLength=0.18,
            )
            cv2.putText(
                canvas,
                label,
                (end[0] + 3, end[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.37,
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_banner(self, frame: np.ndarray, state: DMSState) -> None:
        label, status = self._stable_banner(state)
        color = status_color(status)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), color, -1)
        cv2.putText(frame, label, (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, BLACK, 2)

    def _add_video_header(self, frame: np.ndarray, state: DMSState) -> np.ndarray:
        """Render the HMI above the camera image without hiding video pixels."""

        label, status = self._stable_banner(state)
        color = status_color(status)
        height, width = frame.shape[:2]
        canvas = np.zeros((height + VIDEO_HEADER_HEIGHT, width, 3), dtype=np.uint8)
        canvas[:VIDEO_HEADER_HEIGHT] = color
        canvas[VIDEO_HEADER_HEIGHT:] = frame
        cv2.rectangle(
            canvas,
            (0, VIDEO_HEADER_HEIGHT - 2),
            (width - 1, VIDEO_HEADER_HEIGHT - 1),
            (28, 32, 38),
            -1,
        )
        cv2.circle(canvas, (20, VIDEO_HEADER_HEIGHT // 2), 7, (24, 28, 34), -1)
        cv2.circle(canvas, (20, VIDEO_HEADER_HEIGHT // 2), 3, WHITE, -1)
        fitted_label = _fit_text(label, max(40, width - 58), 0.66, 2)
        cv2.putText(
            canvas,
            fitted_label,
            (38, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            BLACK,
            2,
            cv2.LINE_AA,
        )
        return canvas

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
    eye_runtime_source: str = "LANDMARK_EAR",
    eye_model_status: str = "DISABLED",
    landmark_106_status: str = "DISABLED",
    runtime_metrics: dict[str, object] | None = None,
    compute_backend: str = "CPU",
    npu_active: bool = False,
) -> list[tuple[str, str]]:
    metrics = dict(runtime_metrics or {})
    return [
        ("FPS", f"{fps:.1f}"),
        ("Compute", compute_backend),
        ("NPU", "ACTIVE" if npu_active else "NOT ACTIVE"),
        ("NPU TOPS", _npu_tops_text(metrics, npu_active)),
        (
            "Feature latency",
            _metric_text(
                metrics.get("feature_latency_ms", metrics.get("frame_latency_ms")),
                "ms",
            ),
        ),
        ("Inference latency", _metric_text(metrics.get("inference_time_ms"), "ms")),
        ("Camera health", state.dms_health.camera_status.value),
        ("Face detection", state.dms_health.face_detection_status.value),
        ("Face backend", state.dms_health.face_backend),
        ("NIR mode", state.dms_health.nir_mode),
        ("Input mode", state.dms_health.input_color_mode),
        ("Threshold profile", state.dms_health.active_eye_threshold_profile),
        ("Eye runtime", eye_runtime_source),
        ("Eye CNN", eye_model_status),
        ("106 geometry", landmark_106_status),
        ("NIR active", "YES" if state.dms_health.nir_preprocessing_active else "NO"),
        ("NIR reason", ",".join(state.dms_health.nir_reason_codes) or "NONE"),
        ("Face proposals", str(state.dms_health.face_proposals)),
        ("Face det conf", f"{state.dms_health.face_detection_confidence:.2f}"),
        ("Driver face", state.driver_presence.state.value),
        ("Driver face state", state.driver_identity.driver_face_state),
        ("Proposal state", state.driver_identity.face_proposal_state),
        ("Track hold", state.driver_identity.driver_track_hold_state),
        ("Status page", f"{state.cabin_evidence.status_page_index}/2"),
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
        ("Vehicle gate", state.vehicle.dms_speed_gate_state),
        ("Vehicle speed", f"{state.vehicle.ego_vehicle_speed_kph:.1f} km/h"),
        (
            "Indicators",
            f"L={'ON' if state.vehicle.left_indicator_on else 'OFF'} "
            f"R={'ON' if state.vehicle.right_indicator_on else 'OFF'}",
        ),
        ("Cabin backend", state.cabin_evidence.detector_backend),
        ("Cabin status", state.cabin_evidence.backend_status),
        ("Cabin objects", str(state.cabin_evidence.cabin_evidence_count)),
        ("Cabin phone obs", "YES" if state.cabin_evidence.cabin_phone_observed else "NO"),
        ("Cabin phone regs", "/".join(state.cabin_evidence.cabin_phone_observed_regions) or "NONE"),
        ("Driver ROI phone", "YES" if state.cabin_evidence.phone_inside_driver_roi else "NO"),
        ("Phone scenario", state.cabin_evidence.phone_scenario or "NONE"),
        ("Phone confidence", f"{state.cabin_evidence.phone_track_confidence_smoothed:.2f}"),
        ("Phone track age", f"{state.cabin_evidence.driver_phone_track_age_ms}ms"),
        ("Phone raw/fresh", f"{'YES' if state.cabin_evidence.phone_raw_detected_this_frame else 'NO'}/{'YES' if state.cabin_evidence.phone_track_fresh_this_frame else 'NO'}"),
        ("Ignored phone", str(state.cabin_evidence.ignored_phone_count)),
        ("Cabin belt", state.cabin_evidence.seatbelt_state.value),
        ("Cabin smoking", state.cabin_evidence.smoking_state.value),
        ("Cabin affect", "YES" if state.cabin_evidence.affect_final_dms_state else "NO"),
        ("HMI banner", state.dms_v02.hmi_banner_text or state.dms_v02.final_banner),
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


def _draw_dashboard_card(
    canvas: np.ndarray,
    title: str,
    rows: list[tuple[str, str]],
    rect: tuple[int, int, int, int],
    accent: tuple[int, int, int],
    *,
    value_fraction: float = 0.62,
    columns: int = 1,
    ui_scale: float = 1.0,
) -> None:
    x, y, width, height = (
        int(round(value * ui_scale))
        for value in rect
    )
    x2 = min(canvas.shape[1] - 1, x + width)
    y2 = min(canvas.shape[0] - 1, y + height)
    cv2.rectangle(canvas, (x, y), (x2, y2), (27, 32, 40), -1)
    cv2.rectangle(canvas, (x, y), (x2, y2), (54, 63, 76), 1)
    cv2.rectangle(canvas, (x, y), (x + 4, y2), accent, -1)
    cv2.putText(
        canvas,
        title,
        (x + int(round(16 * ui_scale)), y + int(round(26 * ui_scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53 * ui_scale,
        accent,
        max(1, int(round(ui_scale))),
        cv2.LINE_AA,
    )
    if not rows:
        return
    columns = max(1, min(int(columns), len(rows)))
    rows_per_column = int(np.ceil(len(rows) / columns))
    available = max(
        1,
        y2 - (y + int(round(44 * ui_scale))) - int(round(8 * ui_scale)),
    )
    line_height = min(
        int(round(24 * ui_scale)),
        max(int(round(18 * ui_scale)), available // rows_per_column),
    )
    inner_width = max(1, width - int(round(20 * ui_scale)))
    column_width = inner_width // columns
    for index, (label, value) in enumerate(rows):
        column_index = index // rows_per_column
        row_index = index % rows_per_column
        column_x = x + int(round(10 * ui_scale)) + column_index * column_width
        label_width = max(
            int(round(88 * ui_scale)),
            int(column_width * (1.0 - value_fraction)),
        )
        value_x = column_x + label_width
        value_width = max(
            int(round(20 * ui_scale)),
            column_x + column_width - value_x - int(round(8 * ui_scale)),
        )
        row_y = y + int(round(50 * ui_scale)) + row_index * line_height
        if row_y > y2 - 7:
            continue
        cv2.putText(
            canvas,
            str(label),
            (column_x + int(round(6 * ui_scale)), row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40 * ui_scale,
            (164, 174, 188),
            max(1, int(round(ui_scale))),
            cv2.LINE_AA,
        )
        value_text = _fit_text(
            str(value),
            value_width,
            0.43 * ui_scale,
            max(1, int(round(ui_scale))),
        )
        cv2.putText(
            canvas,
            value_text,
            (value_x, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43 * ui_scale,
            (242, 246, 250),
            max(1, int(round(ui_scale))),
            cv2.LINE_AA,
        )


def _draw_compact_dashboard(
    canvas: np.ndarray,
    rows: list[tuple[str, str]],
    *,
    top: int,
) -> None:
    y = top
    label_x = 16
    value_x = max(158, int(canvas.shape[1] * 0.36))
    max_value_width = max(40, canvas.shape[1] - value_x - 14)
    for label, value in rows:
        if y > canvas.shape[0] - 12:
            break
        cv2.putText(
            canvas,
            label,
            (label_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (166, 176, 190),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            _fit_text(str(value), max_value_width, 0.46, 1),
            (value_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            WHITE,
            1,
            cv2.LINE_AA,
        )
        y += 23


def _fit_text(text: str, max_width: int, scale: float, thickness: int) -> str:
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] <= max_width:
        return text
    suffix = "..."
    candidate = text
    while candidate:
        candidate = candidate[:-1]
        clipped = candidate.rstrip() + suffix
        if cv2.getTextSize(clipped, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] <= max_width:
            return clipped
    return suffix


def _metric_text(value: object, suffix: str = "") -> str:
    if value is None:
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "--"
    precision = 1 if abs(number) < 100.0 else 0
    return f"{number:.{precision}f}{suffix}"


def _cpu_ram_text(metrics: dict[str, object]) -> str:
    cpu = _metric_text(metrics.get("cpu_percent"), "%")
    ram = _metric_text(metrics.get("ram_mb"), " MB")
    return f"{cpu} / {ram}"


def _npu_tops_text(metrics: dict[str, object], npu_active: bool) -> str:
    if not npu_active:
        return "0.00 (inactive)"
    actual = metrics.get("npu_tops_utilized")
    if actual is None:
        return "UNAVAILABLE"
    return _metric_text(actual, " TOPS")


_STATUS_DASHBOARD_PRIORITY = (
    "FPS",
    "Compute",
    "NPU",
    "NPU TOPS",
    "Feature latency",
    "Inference latency",
    "Camera health",
    "Face detection",
    "Face backend",
    "NIR mode",
    "Input mode",
    "Threshold profile",
    "NIR active",
    "NIR reason",
    "Face proposals",
    "Face det conf",
    "Driver face",
    "Driver face state",
    "Proposal state",
    "Track hold",
    "Eye runtime",
    "Eye CNN",
    "106 geometry",
    "Raw eyes",
    "Effective eyes",
    "Eye raw/norm",
    "Eye visibility",
    "Closure ms",
    "PERCLOS usable",
    "PERCLOS 5s/60s",
    "Drowsiness",
    "Head angle",
    "Head raw/rel",
    "Gaze",
    "Gaze confidence",
    "Road calib",
    "Road source",
    "Road offsets",
    "Vehicle gate",
    "Vehicle speed",
    "Indicators",
    "Cabin backend",
    "Cabin status",
    "Cabin objects",
    "Cabin phone obs",
    "Phone scenario",
    "Cabin belt",
    "Cabin smoking",
    "HMI banner",
)


def _prioritize_status_dashboard_lines(
    rows: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Put vehicle-test safety signals above diagnostic detail.

    The complete row collection is retained after the visible priority block,
    so callers that render a taller canvas do not lose any telemetry.
    """

    by_label = {label: (label, value) for label, value in rows}
    priority = [
        by_label[label]
        for label in _STATUS_DASHBOARD_PRIORITY
        if label in by_label
    ]
    priority_labels = set(_STATUS_DASHBOARD_PRIORITY)
    return priority + [row for row in rows if row[0] not in priority_labels]


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


def _visible_cabin_evidence_objects(state: DMSState) -> tuple[list[CabinEvidenceObject], int]:
    visible: list[CabinEvidenceObject] = []
    hidden = 0
    for obj in state.cabin_evidence.evidence_objects:
        if obj.object_type.value != "PHONE":
            visible.append(obj)
            continue
        if state.cabin_evidence.driver_phone_track_held and obj.relation_to_driver.value == state.cabin_evidence.driver_phone_relation:
            hidden += 1
            continue
        duplicate_index = next(
            (
                index for index, kept in enumerate(visible)
                if kept.object_type.value == "PHONE" and _same_physical_phone(obj, kept)
            ),
            None,
        )
        if duplicate_index is None:
            visible.append(obj)
            continue
        kept = visible[duplicate_index]
        obj_priority = _phone_overlay_priority(obj, state)
        kept_priority = _phone_overlay_priority(kept, state)
        if obj_priority > kept_priority or (obj_priority == kept_priority and obj.confidence > kept.confidence):
            visible[duplicate_index] = obj
        hidden += 1
    return visible, hidden


def _same_physical_phone(a: CabinEvidenceObject, b: CabinEvidenceObject) -> bool:
    return _overlay_iou(a.bbox, b.bbox) > 0.35 or _overlay_center_distance(a.bbox, b.bbox) < 0.08


def _phone_overlay_semantic_level(obj: CabinEvidenceObject, state: DMSState) -> str:
    if obj.object_type.value != "PHONE":
        return obj.state.value
    if obj.state.value == "REJECTED":
        return "IGNORED"
    if state.cabin_evidence.driver_phone_track_held and obj.relation_to_driver.value in {
        state.cabin_evidence.driver_phone_relation,
        "NEAR_EAR",
        "NEAR_HAND",
        "NEAR_LAP",
    }:
        return "HELD"
    if state.cabin_evidence.driver_phone_state.value == "PHONE_CONFIRMED" and obj.relation_to_driver.value == state.cabin_evidence.driver_phone_relation:
        return "CONFIRMED"
    if state.cabin_evidence.driver_phone_state.value in {
        "PHONE_TO_EAR_SUSPECTED",
        "PHONE_DISTRACTION",
        "PHONE_IN_HAND_SUSPECTED",
        "PHONE_DOWN_TEXTING_SUSPECTED",
    } and obj.relation_to_driver.value == state.cabin_evidence.driver_phone_relation:
        return "SUSPECTED"
    if state.cabin_evidence.driver_phone_state.value != "NO_PHONE" and obj.relation_to_driver.value == state.cabin_evidence.driver_phone_relation:
        return "CANDIDATE"
    return "PENDING"


def _phone_overlay_priority(obj: CabinEvidenceObject, state: DMSState) -> int:
    level = _phone_overlay_semantic_level(obj, state)
    return {"IGNORED": 0, "PENDING": 1, "HELD": 2, "CANDIDATE": 3, "SUSPECTED": 4, "CONFIRMED": 5}.get(level, 0)


def _cabin_evidence_color(obj: CabinEvidenceObject, semantic_level: str) -> tuple[int, int, int]:
    if semantic_level == "IGNORED":
        return (130, 130, 130)
    if semantic_level == "PENDING":
        return (170, 190, 120)
    if semantic_level == "HELD":
        return (120, 170, 220)
    if semantic_level == "SUSPECTED":
        return (0, 190, 255)
    return (80, 220, 255)


def _cabin_evidence_label(
    object_type: str,
    lifecycle: str,
    source: str = "",
    relation: str = "",
    semantic_level: str = "",
    ignored_reasons: list[str] | None = None,
) -> str:
    prefix = "SYNTH " if source == "synthetic" else ("DET " if source == "onnx" else "")
    if object_type == "PHONE":
        if semantic_level == "IGNORED" or lifecycle == "REJECTED":
            reason = _short_phone_ignore_reason(ignored_reasons or [])
            if reason == "OUTSIDE_DRIVER_ROI":
                return "PHONE OUTSIDE DRIVER ROI / IGNORED"
            return "PHONE / IGNORED"
        if semantic_level == "HELD":
            return "PHONE TRACK / HELD"
        if semantic_level == "SUSPECTED" and relation == "NEAR_EAR":
            return "PHONE TO EAR / SUSPECTED"
        if semantic_level == "SUSPECTED":
            return "PHONE DISTRACTION"
        if semantic_level == "CONFIRMED":
            return "PHONE DISTRACTION"
        if semantic_level == "CANDIDATE":
            return "PHONE DISTRACTION"
        return "PHONE IN DRIVER ROI / PENDING"
    if object_type == "SEATBELT":
        if prefix:
            return f"{prefix}SEATBELT / {relation or lifecycle}"
        return "SEATBELT WORN" if lifecycle == "CONFIRMED" else "SEATBELT UNKNOWN"
    if object_type in {"CIGARETTE", "HAND"}:
        if prefix:
            return f"{prefix}{object_type} / {relation or lifecycle}"
        return "SMOKING CANDIDATE" if lifecycle == "CANDIDATE" else f"SMOKING {lifecycle}"
    return f"{prefix}CABIN EVIDENCE"


def _short_phone_ignore_reason(reasons: list[str]) -> str:
    if "PHONE_OUTSIDE_DRIVER_INTERACTION_ROI" in reasons:
        return "OUTSIDE_DRIVER_ROI"
    if any(reason.startswith("PHONE_BBOX") or reason == "PHONE_LARGE_SQUARE_LOW_CONFIDENCE" for reason in reasons):
        return "IMPLAUSIBLE_SIZE"
    if "DRIVER_PHONE_LOW_CONFIDENCE" in reasons:
        return "LOW_CONF"
    if "PHONE_UNSTABLE_TRACK" in reasons:
        return "UNSTABLE_TRACK"
    return "IGNORED"


def _overlay_center_distance(a: list[float], b: list[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 1.0
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def _overlay_iou(a: list[float], b: list[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0

def occupant_label(
    zone: str,
    track_id: int,
    selected_as_driver: bool = False,
    driver_session_id: str | None = None,
    show_track_id: bool = False,
) -> str:
    if selected_as_driver:
        label = f"DRIVER {driver_session_id}" if driver_session_id else "DRIVER"
    else:
        label = "PASSENGER"
    return f"{label} / T{track_id}" if show_track_id else label


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
