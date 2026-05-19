from __future__ import annotations

from ind_vias_perception.common.types import Detection, FramePacket


class DummyUncertaintyHead:
    name = "dummy_uncertainty"

    def forward(self, detections: list[Detection], packet: FramePacket) -> list[Detection]:
        for det in detections:
            det.sigma_depth = max(0.15, 1.0 - det.confidence)
        return detections
