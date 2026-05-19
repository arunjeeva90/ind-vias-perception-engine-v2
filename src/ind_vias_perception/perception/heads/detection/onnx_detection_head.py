from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ind_vias_perception.common.types import BBox2D, Detection, FramePacket, ObjectClass


@dataclass(frozen=True)
class LetterboxMeta:
    scale: float
    pad_x: float
    pad_y: float
    input_width: int
    input_height: int
    original_width: int
    original_height: int


class ONNXDetectionHead:
    name = "onnx_detection"

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        class_names: dict[int, str] | None = None,
        net: Any | None = None,
    ):
        self.model_path = Path(model_path)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.class_names = class_names or {}
        if net is not None:
            self.net = net
        else:
            if not self.model_path.exists():
                raise FileNotFoundError(f"ONNX detector model not found: {self.model_path}")
            self.net = cv2.dnn.readNetFromONNX(str(self.model_path))

    def forward(self, features: dict[str, np.ndarray], packet: FramePacket) -> list[Detection]:
        blob, meta = self.preprocess(packet.frame)
        self.net.setInput(blob)
        outputs = self.net.forward()
        return self.postprocess(outputs, meta)

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, LetterboxMeta]:
        letterboxed, meta = letterbox(frame, self.input_size)
        blob = cv2.dnn.blobFromImage(
            letterboxed,
            scalefactor=1.0 / 255.0,
            size=self.input_size,
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
        return blob, meta

    def postprocess(
        self,
        outputs: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
        meta: LetterboxMeta,
    ) -> list[Detection]:
        expected_columns = None
        if self.class_names:
            expected_columns = (4 + len(self.class_names), 5 + len(self.class_names))
        rows = _flatten_outputs(outputs, expected_columns=expected_columns)
        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        for row in rows:
            decoded = self._decode_row(row, meta)
            if decoded is None:
                continue
            box, confidence, class_id = decoded
            boxes.append(box)
            confidences.append(confidence)
            class_ids.append(class_id)

        if not boxes:
            return []
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        detections: list[Detection] = []
        for index in np.array(indices).reshape(-1):
            x, y, w, h = boxes[int(index)]
            detections.append(
                Detection(
                    bbox=BBox2D(float(x), float(y), float(x + w), float(y + h)),
                    label=self._object_class(class_ids[int(index)]),
                    confidence=float(confidences[int(index)]),
                    metadata={"class_id": float(class_ids[int(index)])},
                )
            )
        return detections

    def _decode_row(self, row: np.ndarray, meta: LetterboxMeta) -> tuple[list[int], float, int] | None:
        if row.size < 6:
            return None

        if self.class_names and row.size == 4 + len(self.class_names):
            objectness = 1.0
            class_scores = row[4:]
        else:
            objectness = float(row[4])
            class_scores = row[5:]
        class_id = int(np.argmax(class_scores))
        confidence = float(objectness * class_scores[class_id])
        if confidence < self.confidence_threshold:
            return None

        cx, cy, width, height = map(float, row[:4])
        x1 = (cx - width * 0.5 - meta.pad_x) / meta.scale
        y1 = (cy - height * 0.5 - meta.pad_y) / meta.scale
        x2 = (cx + width * 0.5 - meta.pad_x) / meta.scale
        y2 = (cy + height * 0.5 - meta.pad_y) / meta.scale
        x1 = max(0.0, min(meta.original_width - 1.0, x1))
        y1 = max(0.0, min(meta.original_height - 1.0, y1))
        x2 = max(0.0, min(meta.original_width - 1.0, x2))
        y2 = max(0.0, min(meta.original_height - 1.0, y2))
        box_width = max(0, int(round(x2 - x1)))
        box_height = max(0, int(round(y2 - y1)))
        if box_width == 0 or box_height == 0:
            return None
        return [int(round(x1)), int(round(y1)), box_width, box_height], confidence, class_id

    def _object_class(self, class_id: int) -> ObjectClass:
        name = str(self.class_names.get(class_id, ObjectClass.UNKNOWN.value)).lower()
        aliases = {"cyclist": ObjectClass.BICYCLE.value, "bike": ObjectClass.BICYCLE.value}
        name = aliases.get(name, name)
        try:
            return ObjectClass(name)
        except ValueError:
            return ObjectClass.UNKNOWN


def letterbox(frame: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, LetterboxMeta]:
    input_width, input_height = int(input_size[0]), int(input_size[1])
    original_height, original_width = frame.shape[:2]
    scale = min(input_width / original_width, input_height / original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    pad_x = (input_width - resized_width) / 2.0
    pad_y = (input_height - resized_height) / 2.0

    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    output = np.full((input_height, input_width, 3), 114, dtype=frame.dtype)
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    output[top : top + resized_height, left : left + resized_width] = resized
    return output, LetterboxMeta(
        scale=scale,
        pad_x=float(left),
        pad_y=float(top),
        input_width=input_width,
        input_height=input_height,
        original_width=original_width,
        original_height=original_height,
    )


def _flatten_outputs(
    outputs: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
    expected_columns: tuple[int, ...] | None = None,
) -> np.ndarray:
    if isinstance(outputs, (list, tuple)):
        arrays = [_flatten_outputs(output, expected_columns) for output in outputs]
        return np.concatenate(arrays, axis=0) if arrays else np.empty((0, 0), dtype=np.float32)

    array = np.asarray(outputs)
    array = np.squeeze(array)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim != 2:
        return array.reshape(-1, array.shape[-1])
    if expected_columns is not None:
        if array.shape[1] in expected_columns:
            return array
        if array.shape[0] in expected_columns:
            return array.T
    if array.shape[0] < array.shape[1] and array.shape[0] <= 128:
        return array.T
    return array
