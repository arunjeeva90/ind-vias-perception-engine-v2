from __future__ import annotations

from ind_vias_perception.common.types import Detection, FramePacket


class DummyGroundContactHead:
    name = "dummy_ground_contact"

    def forward(self, detections: list[Detection], packet: FramePacket) -> list[Detection]:
        for det in detections:
            det.ground_contact = det.bbox.bottom_center
            det.metadata["u_gc"] = float(det.ground_contact[0])
            det.metadata["v_gc"] = float(det.ground_contact[1])
        return detections
