from __future__ import annotations

from ind_vias_perception.config.settings import Settings
from ind_vias_perception.perception.backbones.mobilenetv4_hybrid.stub import MobileNetV4HybridStub
from ind_vias_perception.perception.necks.bifpn.bifpn_stub import BiFPNStub
from ind_vias_perception.perception.heads.detection.dummy_detection_head import DummyDetectionHead
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
from ind_vias_perception.runtime.cais.controller import CAISController
from ind_vias_perception.safety.sentinel_fsm.fsm import SentinelFSM
from ind_vias_perception.safety.safety_gate.gate import SafetyGate
from ind_vias_perception.pipeline.metric_monocular_pipeline import MetricMonocularPipeline


def build_pipeline(settings: Settings) -> MetricMonocularPipeline:
    cais_cfg = settings.raw.get("runtime", {}).get("cais", {})
    return MetricMonocularPipeline(
        settings=settings,
        backbone=MobileNetV4HybridStub(),
        neck=BiFPNStub(),
        detection_head=DummyDetectionHead(),
        lane_head=DummyLaneHead(),
        freespace_head=DummyFreeSpaceHead(),
        ground_contact_head=DummyGroundContactHead(),
        depth_head=DummyDepthHead(),
        uncertainty_head=DummyUncertaintyHead(),
        scene_quality_head=DummySceneQualityHead(),
        tsr_head=DummyTSRHead(),
        geometric_anchor=GeometricGroundContactAnchor(),
        semantic_anchor=SemanticObjectSizeAnchor(),
        tracker=SimpleDistanceTracker(),
        cais=CAISController(**cais_cfg),
        sentinel=SentinelFSM(),
        safety_gate=SafetyGate(),
    )
