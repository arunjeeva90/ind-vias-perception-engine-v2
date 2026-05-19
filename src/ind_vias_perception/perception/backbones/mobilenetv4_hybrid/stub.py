from __future__ import annotations

import cv2
import numpy as np


class MobileNetV4HybridStub:
    """Interface-compatible placeholder for the production MobileNetV4-Hybrid backbone.

    This is deterministic and lightweight. Replace with real trained implementation.
    """

    name = "mobilenetv4_hybrid_stub"

    def forward(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        img = cv2.resize(frame, (320, 192)).astype(np.float32) / 255.0
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        edges = cv2.Canny((gray * 255).astype(np.uint8), 60, 120).astype(np.float32) / 255.0
        return {
            "p2": img,
            "p3": cv2.resize(img, (160, 96)),
            "p4": cv2.resize(img, (80, 48)),
            "geometry": np.dstack([gray, edges, gray * 0 + 1.0]),
        }
