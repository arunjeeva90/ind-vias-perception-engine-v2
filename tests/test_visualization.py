from __future__ import annotations

import numpy as np

from ind_vias_perception.apps.visualization import draw_perception_output
from ind_vias_perception.common.types import BBox2D, Detection, ObjectClass, PerceptionOutput, SceneQuality


def _sample_output() -> PerceptionOutput:
    det = Detection(
        bbox=BBox2D(20, 20, 80, 90),
        label=ObjectClass.CAR,
        confidence=0.86,
        distance_m=12.5,
        ttc_s=3.2,
    )
    return PerceptionOutput(
        detections=[det],
        scene_quality=SceneQuality(),
        mode="nominal",
        safety_payload={
            "warning_level": "visual",
            "sentinel_state": "nominal",
            "cais_mode": "nominal",
        },
    )


def test_visualization_preserves_image_shape_and_dtype():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    annotated = draw_perception_output(frame, _sample_output())

    assert annotated.shape == frame.shape
    assert annotated.dtype == frame.dtype


def test_visualization_changes_some_pixels():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    annotated = draw_perception_output(frame, _sample_output())

    assert np.any(annotated != frame)
