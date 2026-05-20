from __future__ import annotations

import pytest
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


def _coco_head() -> ONNXDetectionHead:
    return ONNXDetectionHead(
        model_path="unused.onnx",
        input_size=(640, 640),
        confidence_threshold=0.25,
        nms_threshold=0.45,
        class_names={
            0: "pedestrian",
            1: "cyclist",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
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


def test_missing_model_raises_clear_error(tmp_path):
    missing_model = tmp_path / "missing_detector.onnx"

    with pytest.raises(FileNotFoundError, match="ONNX detector model not found"):
        ONNXDetectionHead(model_path=missing_model)


def test_debug_logs_raw_and_confidence_filtered_counts(capsys):
    head = _head()
    head.debug = True
    _, meta = letterbox(np.zeros((640, 640, 3), dtype=np.uint8), (640, 640))
    rows = np.array(
        [
            [320, 320, 160, 160, 0.9, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [320, 320, 160, 160, 0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    head.postprocess(rows, meta)

    captured = capsys.readouterr()
    assert "output_shapes=[(2, 13)]" in captured.out
    assert "after_confidence=1" in captured.out
    assert "after_class_filter=1" in captured.out
    assert "after_nms=1" in captured.out


def test_postprocess_accepts_yolov8_1x84x8400_output():
    head = _coco_head()
    _, meta = letterbox(np.zeros((640, 640, 3), dtype=np.uint8), (640, 640))
    output = np.zeros((1, 84, 8400), dtype=np.float32)
    output[0, 0, 123] = 320
    output[0, 1, 123] = 320
    output[0, 2, 123] = 160
    output[0, 3, 123] = 160
    output[0, 4 + 2, 123] = 0.91

    detections = head.postprocess(output, meta)

    assert len(detections) == 1
    assert detections[0].label == ObjectClass.CAR
    assert abs(detections[0].confidence - 0.91) < 1e-6
    assert detections[0].bbox.x1 == 240.0
    assert detections[0].bbox.y1 == 240.0
    assert detections[0].bbox.x2 == 400.0
    assert detections[0].bbox.y2 == 400.0


def test_yolov8_coco_subset_filters_unmapped_classes():
    head = _coco_head()
    _, meta = letterbox(np.zeros((640, 640, 3), dtype=np.uint8), (640, 640))
    output = np.zeros((1, 84, 8400), dtype=np.float32)
    output[0, 0, 123] = 320
    output[0, 1, 123] = 320
    output[0, 2, 123] = 160
    output[0, 3, 123] = 160
    output[0, 4 + 10, 123] = 0.91

    detections = head.postprocess(output, meta)

    assert detections == []


def test_bbox_exceeding_image_bottom_is_clipped():
    head = _coco_head()
    _, meta = letterbox(np.zeros((100, 100, 3), dtype=np.uint8), (640, 640))
    row = np.zeros((1, 84), dtype=np.float32)
    row[0, :4] = [320, 650, 160, 100]
    row[0, 4 + 2] = 0.9

    detections = head.postprocess(row, meta)

    assert len(detections) == 1
    assert detections[0].bbox.y2 == 99.0
    assert detections[0].metadata["bbox_clipped"] is True


def test_bbox_exceeding_left_and_right_bounds_is_clipped():
    head = _coco_head()
    _, meta = letterbox(np.zeros((100, 100, 3), dtype=np.uint8), (640, 640))
    row = np.zeros((1, 84), dtype=np.float32)
    row[0, :4] = [320, 320, 900, 160]
    row[0, 4 + 2] = 0.9

    detections = head.postprocess(row, meta)

    assert len(detections) == 1
    assert detections[0].bbox.x1 == 0.0
    assert detections[0].bbox.x2 == 99.0
    assert detections[0].metadata["bbox_clipped"] is True


def test_invalid_negative_size_box_is_rejected():
    head = _coco_head()
    _, meta = letterbox(np.zeros((100, 100, 3), dtype=np.uint8), (640, 640))
    row = np.zeros((1, 84), dtype=np.float32)
    row[0, :4] = [320, 320, -20, 160]
    row[0, 4 + 2] = 0.9

    detections = head.postprocess(row, meta)

    assert detections == []
