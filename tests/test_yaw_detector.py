from __future__ import annotations

import numpy as np
import cv2

from ind_vias_perception.common.types import FramePacket
from ind_vias_perception.config.settings import load_settings
from ind_vias_perception.pipeline.factory import build_pipeline
from ind_vias_perception.temporal.ego_motion.yaw_detector import (
    OpticalFlowYawDetector,
    YawDetectionResult,
    analyze_flow,
)


def test_analyze_flow_detects_consistent_horizontal_yaw():
    prev = np.array([[[float(i), 10.0]] for i in range(30)], dtype=np.float32)
    nxt = prev.copy()
    nxt[:, 0, 0] += 3.0
    status = np.ones((30, 1), dtype=np.uint8)

    result = analyze_flow(
        prev,
        nxt,
        status,
        min_flow_points=25,
        median_dx_threshold=2.0,
        yaw_score_threshold=0.55,
    )

    assert result.turning_detected is True
    assert result.median_dx == 3.0
    assert result.flow_points == 30
    assert result.yaw_score == 1.0


def test_analyze_flow_rejects_too_few_points():
    prev = np.array([[[float(i), 10.0]] for i in range(10)], dtype=np.float32)
    nxt = prev.copy()
    nxt[:, 0, 0] += 3.0
    status = np.ones((10, 1), dtype=np.uint8)

    result = analyze_flow(prev, nxt, status, min_flow_points=25)

    assert result.turning_detected is False
    assert result.flow_points == 10


def test_analyze_flow_rejects_low_median_dx():
    prev = np.array([[[float(i), 10.0]] for i in range(30)], dtype=np.float32)
    nxt = prev.copy()
    nxt[:, 0, 0] += 0.5
    status = np.ones((30, 1), dtype=np.uint8)

    result = analyze_flow(prev, nxt, status, min_flow_points=25, median_dx_threshold=2.0)

    assert result.turning_detected is False
    assert result.median_dx == 0.5


def test_single_frame_yaw_spike_does_not_become_turning():
    detector = OpticalFlowYawDetector(required_turning_frames=3, min_flow_points=50)

    result = detector.update_from_measurement(YawDetectionResult(True, 1.0, 4.0, 80))

    assert result.ego_motion_state == "uncertain"
    assert result.turning_detected is False
    assert result.turning_confirmation_count == 1


def test_repeated_yaw_over_required_frames_becomes_turning():
    detector = OpticalFlowYawDetector(required_turning_frames=3, min_flow_points=50)

    detector.update_from_measurement(YawDetectionResult(True, 1.0, 4.0, 80))
    detector.update_from_measurement(YawDetectionResult(True, 0.9, 3.5, 80))
    result = detector.update_from_measurement(YawDetectionResult(True, 0.8, 3.2, 80))

    assert result.ego_motion_state == "turning"
    assert result.turning_detected is True
    assert result.yaw_confidence == 1.0
    assert result.turning_confirmation_count == 3


def test_low_flow_points_gives_uncertain():
    detector = OpticalFlowYawDetector(required_turning_frames=3, min_flow_points=50)

    result = detector.update_from_measurement(YawDetectionResult(True, 1.0, 4.0, 10))

    assert result.ego_motion_state == "uncertain"
    assert result.turning_detected is False
    assert result.yaw_confidence == 0.0


def test_straight_sequence_remains_straight():
    detector = OpticalFlowYawDetector(required_turning_frames=3, min_flow_points=50)

    for _ in range(5):
        result = detector.update_from_measurement(YawDetectionResult(False, 0.1, 0.2, 80))

    assert result.ego_motion_state == "straight"
    assert result.turning_detected is False
    assert result.turning_confirmation_count == 0


def test_yaw_detector_handles_large_frame_without_crash():
    detector = OpticalFlowYawDetector(max_feature_width=640, max_feature_height=640)
    frame = np.zeros((1440, 1440, 3), dtype=np.uint8)

    first = detector.update(frame)
    second = detector.update(frame)

    assert first.ego_motion_state == "uncertain"
    assert second.ego_motion_state == "uncertain"
    assert second.yaw_confidence == 0.0
    height, width = [int(part) for part in second.roi_shape.split("x")]
    assert height <= 640
    assert width <= 640


def test_yaw_detector_handles_good_features_returning_none(monkeypatch):
    detector = OpticalFlowYawDetector()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector.update(frame)

    monkeypatch.setattr(cv2, "goodFeaturesToTrack", lambda *args, **kwargs: None)
    result = detector.update(frame)

    assert result.ego_motion_state == "uncertain"
    assert result.yaw_confidence == 0.0
    assert result.reason_codes == "ego_motion_feature_failure"


def test_yaw_detector_handles_cv2_error_and_returns_uncertain(monkeypatch):
    detector = OpticalFlowYawDetector()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector.update(frame)

    def raise_cv2_error(*_args, **_kwargs):
        raise cv2.error("forced")

    monkeypatch.setattr(cv2, "goodFeaturesToTrack", raise_cv2_error)
    result = detector.update(frame)

    assert result.ego_motion_state == "uncertain"
    assert result.yaw_confidence == 0.0
    assert result.reason_codes == "opencv_memory_error"


def test_full_pipeline_continues_when_yaw_detector_feature_extraction_fails(monkeypatch):
    settings = load_settings("configs/default.yaml")
    pipeline = build_pipeline(settings)
    pipeline.ego_yaw_enabled = True
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    pipeline.process(FramePacket(frame=frame, timestamp_s=0.0, frame_id=0))

    def raise_cv2_error(*_args, **_kwargs):
        raise cv2.error("forced")

    monkeypatch.setattr(cv2, "goodFeaturesToTrack", raise_cv2_error)
    out = pipeline.process(FramePacket(frame=frame, timestamp_s=1 / 30.0, frame_id=1))

    assert out.scene_quality.ego_motion_state == "uncertain"
    assert out.scene_quality.ego_motion_reason_codes == "opencv_memory_error"
    assert out.safety_payload["ego_motion_reason_codes"] == "opencv_memory_error"
