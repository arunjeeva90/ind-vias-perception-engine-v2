from __future__ import annotations

import cv2
import numpy as np

from ind_vias_dms.core.types import AvailabilityState, DMSState, DistractionLevel, DrowsinessLevel
from ind_vias_dms.vision.face_landmarks import FaceLandmarkResult
from ind_vias_dms.vision.head_pose import HeadPose
from ind_vias_dms.visualization.colors import BLACK, GRAY, GREEN, RED, WHITE, status_color


class OverlayRenderer:
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
    ) -> np.ndarray:
        out = frame.copy()
        if face.bbox is not None:
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
        lines = [
            ("FPS", f"{fps:.1f}"),
            ("Presence", state.driver_presence.state.value),
            ("Camera", state.dms_health.camera_status.value),
            ("Gaze", state.gaze.zone.value),
            ("Gaze confidence", f"{state.gaze.confidence:.2f}"),
            (
                "Yaw/Pitch/Roll",
                f"{state.gaze.head_yaw_deg:.1f} / "
                f"{state.gaze.head_pitch_deg:.1f} / {state.gaze.head_roll_deg:.1f}",
            ),
            ("Road offsets", f"{road_yaw_offset_deg:.1f} / {road_pitch_offset_deg:.1f}"),
            ("Eyes", "CLOSED" if state.drowsiness.eye_closure_duration_ms else "OPEN"),
            ("PERCLOS 5s/60s", f"{state.drowsiness.perclos_5s:.2f} / {state.drowsiness.perclos_60s:.2f}"),
            ("Drowsiness", state.drowsiness.level.value),
            ("Distraction", state.distraction.level.value),
            ("Mobile", state.phone_use.state),
            ("Availability", state.driver_availability.state.value),
            ("Readiness", f"{state.driver_readiness_score.score_0_to_1:.2f}"),
            ("Risk", state.driver_readiness_score.risk_level.value),
            (
                "Reason codes",
                ", ".join(state.driver_availability.reason_codes)
                if state.driver_availability.reason_codes
                else "NONE",
            ),
        ]
        y = 88
        for label, value in lines:
            cv2.putText(canvas, label, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY, 1)
            cv2.putText(canvas, value, (190, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, WHITE, 1)
            y += 38
        return canvas

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
        dx = int(state.gaze.head_yaw_deg * 2)
        dy = int(state.gaze.head_pitch_deg * 2)
        origin = (int(x), int(y))
        endpoint = clamp_endpoint(origin, (x + dx, y + dy), frame.shape, max_gaze_vector_length_px)
        cv2.arrowedLine(frame, origin, endpoint, RED, 2)

    def _draw_panel(self, frame: np.ndarray, state: DMSState, fps: float) -> None:
        cv2.rectangle(frame, (0, 0), (330, 310), BLACK, -1)
        cv2.rectangle(frame, (0, 0), (330, 310), GRAY, 1)
        lines = [
            f"FPS: {fps:.1f}",
            f"Presence: {state.driver_presence.state.value}",
            f"Gaze: {state.gaze.zone.value}",
            f"Yaw/Pitch/Roll: {state.gaze.head_yaw_deg:.1f}/{state.gaze.head_pitch_deg:.1f}/{state.gaze.head_roll_deg:.1f}",
            f"Eyes: {'CLOSED' if state.drowsiness.eye_closure_duration_ms else 'OPEN'}",
            f"PERCLOS: {state.drowsiness.perclos_5s:.2f}/{state.drowsiness.perclos_60s:.2f}",
            f"Drowsiness: {state.drowsiness.level.value}",
            f"Distraction: {state.distraction.level.value}",
            f"Availability: {state.driver_availability.state.value}",
            f"Readiness: {state.driver_readiness_score.score_0_to_1:.2f}",
        ]
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (12, 26 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    def _draw_banner(self, frame: np.ndarray, state: DMSState) -> None:
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
        elif state.driver_availability.state == AvailabilityState.DEGRADED:
            label = "DMS DEGRADED"
            status = AvailabilityState.DEGRADED
        color = status_color(status)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), color, -1)
        cv2.putText(frame, label, (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, BLACK, 2)


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
