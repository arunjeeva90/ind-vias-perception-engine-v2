from __future__ import annotations

import numpy as np
from ind_vias_perception.common.types import Detection, FramePacket


class DummyDepthHead:
    name = "dummy_depth"

    def forward(self, detections: list[Detection], packet: FramePacket) -> list[Detection]:
        for det in detections:
            det.metadata["relative_depth"] = 1.0 / max(det.bbox.height, 1.0)
        return detections
