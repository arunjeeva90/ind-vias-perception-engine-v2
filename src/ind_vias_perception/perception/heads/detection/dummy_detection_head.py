from __future__ import annotations

import numpy as np
from ind_vias_perception.common.types import BBox2D, Detection, FramePacket, ObjectClass


class DummyDetectionHead:
    name = "dummy_detection"

    def forward(self, features: dict[str, np.ndarray], packet: FramePacket) -> list[Detection]:
        h, w = packet.frame.shape[:2]
        return [
            Detection(
                bbox=BBox2D(w * 0.42, h * 0.42, w * 0.58, h * 0.72),
                label=ObjectClass.CAR,
                confidence=0.86,
            )
        ]
