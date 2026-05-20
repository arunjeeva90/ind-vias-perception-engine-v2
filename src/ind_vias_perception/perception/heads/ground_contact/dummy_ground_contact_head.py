from __future__ import annotations

from ind_vias_perception.common.types import Detection, FramePacket


class DummyGroundContactHead:
    name = "dummy_ground_contact"

    def forward(self, detections: list[Detection], packet: FramePacket) -> list[Detection]:
        image_height = packet.frame.shape[0]
        for det in detections:
            u_gc, v_gc = det.bbox.bottom_center
            v_gc = min(v_gc, image_height - 1)
            det.ground_contact = (u_gc, v_gc)
            det.metadata["u_gc"] = float(det.ground_contact[0])
            det.metadata["v_gc"] = float(det.ground_contact[1])
        return detections
