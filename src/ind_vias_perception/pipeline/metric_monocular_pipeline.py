from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from ind_vias_perception.common.types import FramePacket, PerceptionOutput
from ind_vias_perception.config.settings import Settings
from ind_vias_perception.geometry.scale_fusion.inverse_variance import fuse_inverse_variance
from ind_vias_perception.ttc.depth_ttc.depth_derivative import ttc_from_depth
from ind_vias_perception.ttc.fusion.uncertainty_weighted import fuse_ttc


@dataclass
class MetricMonocularPipeline:
    settings: Settings
    backbone: Any
    neck: Any
    detection_head: Any
    lane_head: Any
    freespace_head: Any
    ground_contact_head: Any
    depth_head: Any
    uncertainty_head: Any
    scene_quality_head: Any
    tsr_head: Any
    geometric_anchor: Any
    semantic_anchor: Any
    tracker: Any
    cais: Any
    sentinel: Any
    safety_gate: Any

    def process(self, packet: FramePacket) -> PerceptionOutput:
        features = self.backbone.forward(packet.frame)
        features = self.neck.forward(features)

        detections = self.detection_head.forward(features, packet)
        _lane = self.lane_head.forward(features, packet)
        _freespace = self.freespace_head.forward(features, packet)
        _tsr = self.tsr_head.forward(features, packet)
        scene = self.scene_quality_head.forward(features, packet)

        detections = self.ground_contact_head.forward(detections, packet)
        detections = self.depth_head.forward(detections, packet)
        detections = self.uncertainty_head.forward(detections, packet)

        for det in detections:
            geo = self.geometric_anchor.estimate(det, self.settings.camera)
            sem = self.semantic_anchor.estimate(det, self.settings.camera.fy_px)
            distance, sigma = fuse_inverse_variance([geo, sem])
            det.distance_m = distance
            det.sigma_depth = max(det.sigma_depth, min(1.0, sigma))

        detections = self.tracker.update(detections, packet.timestamp_s)

        for det in detections:
            ttc_depth = ttc_from_depth(det.distance_m, det.metadata.get("relative_velocity_mps"))
            det.ttc_s = fuse_ttc([(ttc_depth, det.sigma_depth)])

        cais_decision = self.cais.decide(detections, scene)
        sentinel_state = self.sentinel.update(scene)
        payload = self.safety_gate.evaluate(detections, sentinel_state)
        payload["cais_mode"] = cais_decision.mode
        payload["target_fps"] = cais_decision.target_fps
        return PerceptionOutput(detections=detections, scene_quality=scene, mode=cais_decision.mode, safety_payload=payload)
