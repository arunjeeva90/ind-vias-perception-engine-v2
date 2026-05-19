from __future__ import annotations

import numpy as np
from ind_vias_perception.common.types import FramePacket, SceneQuality


class DummySceneQualityHead:
    name = "dummy_scene_quality"

    def forward(self, features: dict[str, np.ndarray], packet: FramePacket) -> SceneQuality:
        gray_mean = float(np.mean(packet.frame)) / 255.0
        night = max(0.0, 0.35 - gray_mean)
        glare = max(0.0, gray_mean - 0.85)
        return SceneQuality(glare=glare, night=night, complexity=0.35)
