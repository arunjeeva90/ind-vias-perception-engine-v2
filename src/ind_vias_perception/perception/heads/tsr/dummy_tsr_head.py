from __future__ import annotations

import numpy as np
from ind_vias_perception.common.types import FramePacket


class DummyTSRHead:
    name = "dummy_tsr"

    def forward(self, features: dict[str, np.ndarray], packet: FramePacket) -> list[dict[str, object]]:
        return []
