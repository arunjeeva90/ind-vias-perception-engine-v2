from __future__ import annotations

import numpy as np

from ind_vias_perception.temporal.ego_motion.yaw_detector import analyze_flow


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
