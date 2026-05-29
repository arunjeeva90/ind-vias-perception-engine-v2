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
        out = frame.copy()
        if face.bbox is not None:
            cv2.rectangle(out, face.bbox[:2], face.bbox[2:], GREEN, 2)
        if face.landmarks_px:
            for idx, (x, y) in face.landmarks_px.items():
                if idx % 8 == 0:
                    cv2.circle(out, (int(x), int(y)), 1, WHITE, -1)
        if head_pose.confidence > 0 and face.landmarks_px:
            self._draw_head_axis(out, face, head_pose)
            self._draw_gaze_hint(out, face, state)
        if telemetry_enabled:
            self._draw_panel(out, state, fps)
        self._draw_banner(out, state)
        return out

    def _draw_head_axis(self, frame: np.ndarray, face: FaceLandmarkResult, pose: HeadPose) -> None:
        if pose.rvec is None or pose.tvec is None or pose.camera_matrix is None or pose.dist_coeffs is None:
            return
        axis = np.float32([[50, 0, 0], [0, 50, 0], [0, 0, 50]])
        points, _ = cv2.projectPoints(axis, pose.rvec, pose.tvec, pose.camera_matrix, pose.dist_coeffs)
        origin = tuple(int(v) for v in face.landmarks_px[1])
        for point, color in zip(points.reshape(-1, 2), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv2.line(frame, origin, (int(point[0]), int(point[1])), color, 2)

    def _draw_gaze_hint(self, frame: np.ndarray, face: FaceLandmarkResult, state: DMSState) -> None:
        if face.landmarks_px is None or 1 not in face.landmarks_px:
            return
        x, y = face.landmarks_px[1]
        dx = int(state.gaze.head_yaw_deg * 2)
        dy = int(state.gaze.head_pitch_deg * 2)
        cv2.arrowedLine(frame, (int(x), int(y)), (int(x + dx), int(y + dy)), RED, 2)

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
