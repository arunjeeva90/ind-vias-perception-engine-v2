from __future__ import annotations

from enum import Enum
from ind_vias_perception.common.types import SceneQuality


class SentinelState(str, Enum):
    NOMINAL = "nominal"
    GLARE = "glare"
    NIGHT = "night"
    OCCLUSION = "occlusion"
    DEGRADED = "degraded"


class SentinelFSM:
    def update(self, scene: SceneQuality) -> SentinelState:
        if scene.glare > 0.6:
            return SentinelState.GLARE
        if scene.night > 0.6:
            return SentinelState.NIGHT
        if scene.occlusion > 0.6:
            return SentinelState.OCCLUSION
        if scene.degraded_score > 0.6:
            return SentinelState.DEGRADED
        return SentinelState.NOMINAL
