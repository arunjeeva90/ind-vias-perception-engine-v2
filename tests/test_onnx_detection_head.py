from __future__ import annotations

import numpy as np

from ind_vias_perception.common.types import ObjectClass
from ind_vias_perception.perception.heads.detection.onnx_detection_head import (
    ONNXDetectionHead,
    letterbox,
)


class _DummyNet:
    def setInput(self, blob):
        self.blob = blob

    def forward(self):
        return np.empty((0, 13), dtype=np.float32)


def _head() -> ONNXDetectionHead:
    return ONNXDetectionHead(
        model_path="unused.onnx",
        input_size=(640, 640),
        confidence_threshold=0.25,
        nms_threshold=0.45,
        class_names={
            0: "car",
            1: "bus",
            2: "truck",
            3: "motorcycle",
            4: "auto_rickshaw",
            5: "pedestrian",
            6: "cyclist",
            7: "animal",
        },
        net=_DummyNet(),
    )


def test_letterbox_resizes_with_padding():
    frame = np.zeros((320, 640, 3), dtype=np.uint8)

    resized, meta = letterbox(frame, (640, 640))

    assert resized.shape == (640, 640, 3)
    assert meta.scale == 1.0
    assert meta.pad_x == 0.0
    assert meta.pad_y == 160.0


def test_postprocess_converts_synthetic_model_rows_to_detections():
    head = _head()
    _, meta = letterbox(np.zeros((640, 640, 3), dtype=np.uint8), (640, 640))
    row = np.array([[320, 320, 160, 160, 0.9, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    detections = head.postprocess(row, meta)

    assert len(detections) == 1
    assert detections[0].label == ObjectClass.CAR
    assert abs(detections[0].confidence - 0.72) < 1e-6
    assert detections[0].bbox.x1 == 240.0
    assert detections[0].bbox.y1 == 240.0
    assert detections[0].bbox.x2 == 400.0
    assert detections[0].bbox.y2 == 400.0


def test_postprocess_filters_low_confidence_and_maps_cyclist_alias():
    head = _head()
    _, meta = letterbox(np.zeros((640, 640, 3), dtype=np.uint8), (640, 640))
    rows = np.array(
        [
            [320, 320, 160, 160, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
            [320, 320, 160, 160, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
        ],
        dtype=np.float32,
    )

    detections = head.postprocess(rows, meta)

    assert len(detections) == 1
    assert detections[0].label == ObjectClass.BICYCLE


def test_postprocess_accepts_rows_without_objectness_score():
    head = _head()
    _, meta = letterbox(np.zeros((640, 640, 3), dtype=np.uint8), (640, 640))
    row = np.array([[320, 320, 160, 160, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    detections = head.postprocess(row, meta)

    assert len(detections) == 1
    assert detections[0].label == ObjectClass.CAR
    assert abs(detections[0].confidence - 0.8) < 1e-6
