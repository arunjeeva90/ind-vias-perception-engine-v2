from __future__ import annotations

from ind_vias_perception.config.settings import Settings
from ind_vias_perception.perception.backbones.mobilenetv4_hybrid.stub import MobileNetV4HybridStub
from ind_vias_perception.perception.necks.bifpn.bifpn_stub import BiFPNStub
from ind_vias_perception.perception.heads.detection.dummy_detection_head import DummyDetectionHead
from ind_vias_perception.perception.heads.detection.onnx_detection_head import ONNXDetectionHead
from ind_vias_perception.perception.heads.lane.dummy_lane_head import DummyLaneHead
from ind_vias_perception.perception.heads.freespace.dummy_freespace_head import DummyFreeSpaceHead
from ind_vias_perception.perception.heads.ground_contact.dummy_ground_contact_head import DummyGroundContactHead
from ind_vias_perception.perception.heads.depth.dummy_depth_head import DummyDepthHead
from ind_vias_perception.perception.heads.uncertainty.dummy_uncertainty_head import DummyUncertaintyHead
from ind_vias_perception.perception.heads.scene_quality.dummy_scene_quality_head import DummySceneQualityHead
from ind_vias_perception.perception.heads.tsr.dummy_tsr_head import DummyTSRHead
from ind_vias_perception.geometry.scale_anchors.geometric_anchor import GeometricGroundContactAnchor
from ind_vias_perception.geometry.scale_anchors.semantic_anchor import SemanticObjectSizeAnchor
from ind_vias_perception.temporal.trackers.simple_tracker import SimpleDistanceTracker
from ind_vias_perception.temporal.ego_motion.yaw_detector import OpticalFlowYawDetector
from ind_vias_perception.runtime.cais.controller import CAISController
from ind_vias_perception.safety.sentinel_fsm.fsm import SentinelFSM
from ind_vias_perception.safety.safety_gate.gate import SafetyGate
from ind_vias_perception.pipeline.metric_monocular_pipeline import MetricMonocularPipeline


def build_detection_head(settings: Settings):
    detection_cfg = settings.raw.get("detection", {})
    backend = detection_cfg.get("backend", "dummy")
    if backend == "dummy":
        return DummyDetectionHead()
    if backend == "onnx":
        class_names = {int(k): str(v) for k, v in detection_cfg.get("class_names", {}).items()}
        return ONNXDetectionHead(
            model_path=detection_cfg.get("onnx_model_path", "models/weights/detector.onnx"),
            input_size=tuple(detection_cfg.get("input_size", [640, 640])),
            confidence_threshold=detection_cfg.get("confidence_threshold", 0.25),
            nms_threshold=detection_cfg.get("nms_threshold", 0.45),
            class_names=class_names,
        )
    raise ValueError(f"Unsupported detection backend: {backend}")


def build_pipeline(settings: Settings) -> MetricMonocularPipeline:
    cais_cfg = settings.raw.get("runtime", {}).get("cais", {}) | settings.raw.get("cais", {})
    semantic_priors = settings.raw.get("semantic_priors", {})
    ego_motion_cfg = settings.raw.get("ego_motion", {})
    tracking_cfg = settings.raw.get("tracking", {})
    return MetricMonocularPipeline(
        settings=settings,
        backbone=MobileNetV4HybridStub(),
        neck=BiFPNStub(),
        detection_head=build_detection_head(settings),
        lane_head=DummyLaneHead(),
        freespace_head=DummyFreeSpaceHead(),
        ground_contact_head=DummyGroundContactHead(),
        depth_head=DummyDepthHead(),
        uncertainty_head=DummyUncertaintyHead(),
        scene_quality_head=DummySceneQualityHead(),
        tsr_head=DummyTSRHead(),
        geometric_anchor=GeometricGroundContactAnchor(),
        semantic_anchor=SemanticObjectSizeAnchor(semantic_priors),
        tracker=SimpleDistanceTracker(
            max_age=int(tracking_cfg.get("max_age", 10)),
            min_hits=int(tracking_cfg.get("min_hits", 2)),
            iou_weight=float(tracking_cfg.get("iou_weight", 0.50)),
            center_weight=float(tracking_cfg.get("center_weight", 0.25)),
            class_mismatch_penalty=float(tracking_cfg.get("class_mismatch_penalty", 0.25)),
            distance_weight=float(tracking_cfg.get("distance_weight", 0.20)),
            max_association_cost=float(tracking_cfg.get("max_association_cost", 1.0)),
        ),
        ego_yaw_detector=OpticalFlowYawDetector(
            min_flow_points=int(ego_motion_cfg.get("min_flow_points", 25)),
            median_dx_threshold=float(ego_motion_cfg.get("median_dx_threshold", 2.0)),
            yaw_score_threshold=float(ego_motion_cfg.get("yaw_score_threshold", 0.55)),
            smoothing_window=int(ego_motion_cfg.get("smoothing_window", 5)),
            required_turning_frames=int(ego_motion_cfg.get("required_turning_frames", 3)),
        ),
        cais=CAISController(**cais_cfg),
        sentinel=SentinelFSM(),
        safety_gate=SafetyGate(
            settings.raw.get("ego_corridor", {}),
            settings.raw.get("safety_confirmation", {}),
        ),
    )
