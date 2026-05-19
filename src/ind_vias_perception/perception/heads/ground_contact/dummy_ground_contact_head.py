from __future__ import annotations

import numpy as np
from ind_vias_perception.common.types import Detection, FramePacket


class DummyGroundContactHead:
    name = "dummy_ground_contact"

    def forward(self, detections: list[Detection], packet: FramePacket) -> list[Detection]:
        for det in detections:
            det.ground_contact = det.bbox.bottom_center
        return detections
