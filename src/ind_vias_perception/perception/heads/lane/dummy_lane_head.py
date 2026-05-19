from __future__ import annotations

import numpy as np
from ind_vias_perception.common.types import FramePacket


class DummyLaneHead:
    name = "dummy_lane"

    def forward(self, features: dict[str, np.ndarray], packet: FramePacket) -> dict[str, object]:
        h, w = packet.frame.shape[:2]
        return {"left_boundary": [(w*0.35, h), (w*0.47, h*0.55)], "right_boundary": [(w*0.65, h), (w*0.53, h*0.55)], "confidence": 0.65}
