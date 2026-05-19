from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundPlaneEstimate:
    horizon_v_px: float
    confidence: float


class RansacGroundPlaneStub:
    def estimate(self, default_horizon_v_px: float) -> GroundPlaneEstimate:
        return GroundPlaneEstimate(horizon_v_px=default_horizon_v_px, confidence=0.70)
