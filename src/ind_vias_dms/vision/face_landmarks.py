from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


MEDIAPIPE_REQUIRED_MESSAGE = (
    "MediaPipe is required for the default DMS v0.1 face backend. "
    "Install using: pip install mediapipe"
)


@dataclass
class FaceLandmarkResult:
    face_found: bool
    bbox: tuple[int, int, int, int] | None = None
    landmarks_px: dict[int, tuple[float, float]] | None = None
    confidence: float = 0.0


class FaceLandmarkBackend:
    def __init__(self, backend: str = "mediapipe") -> None:
        self.backend = backend
        self._face_mesh: Any | None = None
        if backend != "mediapipe":
            raise ValueError(f"Unsupported DMS face backend: {backend}")
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            raise RuntimeError(MEDIAPIPE_REQUIRED_MESSAGE) from exc
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr: np.ndarray) -> FaceLandmarkResult:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return FaceLandmarkResult(face_found=False)
        height, width = frame_bgr.shape[:2]
        face = result.multi_face_landmarks[0]
        landmarks = {
            idx: (lm.x * width, lm.y * height)
            for idx, lm in enumerate(face.landmark)
        }
        xs = [point[0] for point in landmarks.values()]
        ys = [point[1] for point in landmarks.values()]
        x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
        x2, y2 = min(width - 1, int(max(xs))), min(height - 1, int(max(ys)))
        return FaceLandmarkResult(
            face_found=True,
            bbox=(x1, y1, x2, y2),
            landmarks_px=landmarks,
            confidence=0.85,
        )

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
