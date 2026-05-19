from __future__ import annotations

import numpy as np
from ind_vias_perception.common.types import FramePacket


class DummyFreeSpaceHead:
    name = "dummy_freespace"

    def forward(self, features: dict[str, np.ndarray], packet: FramePacket) -> np.ndarray:
        h, w = packet.frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(h*0.55):, int(w*0.2):int(w*0.8)] = 1
        return mask
