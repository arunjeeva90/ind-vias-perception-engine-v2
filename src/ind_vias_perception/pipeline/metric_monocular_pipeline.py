from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
from ind_vias_perception.common.types import FramePacket, PerceptionOutput
from ind_vias_perception.config.settings import Settings
from ind_vias_perception.geometry.scale_fusion.robust_distance import robust_fuse_distance_m
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
            sem = self.semantic_anchor.estimate(
                det,
                self.settings.camera.fy_px,
                self.settings.camera.fx_px,
            )
            det.metadata["distance_ground_m"] = float(geo.scale_or_distance_m)
            det.metadata["in_ego_corridor"] = point_in_ego_corridor(
                det.metadata.get("u_gc", det.bbox.bottom_center[0]),
                det.metadata.get("v_gc", det.bbox.bottom_center[1]),
                packet.frame.shape[1],
                packet.frame.shape[0],
                self.settings.raw.get("ego_corridor", {}),
            )
            distance, source = robust_fuse_distance_m(
                geo.scale_or_distance_m,
                sem.scale_or_distance_m,
                self.settings.camera,
                prefer_semantic=not bool(det.metadata["in_ego_corridor"]) or near_horizon(det, self.settings),
            )
            det.metadata["distance_fused_camera_m"] = float(distance)
            det.metadata["distance_source"] = source
            det.metadata["distance_camera_m"] = float(distance)
            det.distance_m = bumper_relative_distance_m(
                distance,
                self.settings.vehicle.camera_to_front_bumper_offset_m,
            )
            det.metadata["distance_bumper_m"] = float(det.distance_m)
            det.sigma_depth = max(det.sigma_depth, 0.35 if source == "fused" else 0.55)

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


def bumper_relative_distance_m(distance_camera_m: float, offset_m: float) -> float:
    if not math.isfinite(distance_camera_m):
        return distance_camera_m
    return max(distance_camera_m - offset_m, 0.0)


def near_horizon(det, settings: Settings) -> bool:
    v_gc = float(det.metadata.get("v_gc", det.bbox.y2))
    height = max(settings.camera.image_height, 1)
    margin = 0.08 * height
    return v_gc <= settings.camera.horizon_v_px + margin


def point_in_ego_corridor(
    u_px: float,
    v_px: float,
    image_width: int,
    image_height: int,
    cfg: dict[str, object],
) -> bool:
    if not cfg or not cfg.get("enabled", False):
        return False
    top_y = float(cfg.get("top_y_norm", 0.45)) * image_height
    bottom_y = float(cfg.get("bottom_y_norm", 1.0)) * image_height
    if v_px < top_y or v_px > bottom_y:
        return False
    t = (v_px - top_y) / max(bottom_y - top_y, 1.0)
    top_width = float(cfg.get("top_width_norm", 0.18)) * image_width
    bottom_width = float(cfg.get("bottom_width_norm", 0.45)) * image_width
    width = top_width + t * (bottom_width - top_width)
    center_x = float(cfg.get("center_x_norm", 0.5)) * image_width
    return center_x - width * 0.5 <= u_px <= center_x + width * 0.5
