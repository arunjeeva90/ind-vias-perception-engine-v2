from __future__ import annotations

import numpy as np

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
