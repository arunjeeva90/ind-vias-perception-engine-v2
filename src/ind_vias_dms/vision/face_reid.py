from __future__ import annotations

import numpy as np


class FaceReIdentifier:
    """Placeholder for future ArcFace/MobileFaceNet ONNX driver re-identification."""

    def compute_embedding(
        self,
        frame: np.ndarray,
        face_box: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        return None

    def compare(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        return 0.0
