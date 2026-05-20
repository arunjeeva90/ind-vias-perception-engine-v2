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


@dataclass(frozen=True)
class DecodedCandidate:
    box: list[int]
    confidence: float
    class_id: int


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
        debug: bool = False,
    ):
        self.model_path = Path(model_path)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.class_names = class_names or {}
        self.debug = debug
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
        rows = _flatten_outputs(outputs)
        output_shapes = _output_shapes(outputs)
        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []
        raw_count = len(rows)
        after_confidence_count = 0
        after_class_count = 0
        candidate_scores: list[float] = []
        candidate_class_ids: list[int] = []

        for row in rows:
            decoded = self._decode_candidate(row, meta)
            if decoded is None:
                continue
            candidate_scores.append(decoded.confidence)
            candidate_class_ids.append(decoded.class_id)
            if decoded.confidence < self.confidence_threshold:
                continue
            after_confidence_count += 1
            if self.class_names and decoded.class_id not in self.class_names:
                continue
            after_class_count += 1
            boxes.append(decoded.box)
            confidences.append(decoded.confidence)
            class_ids.append(decoded.class_id)

        nms_indices = []
        if boxes:
            nms_indices = cv2.dnn.NMSBoxes(
                boxes,
                confidences,
                self.confidence_threshold,
                self.nms_threshold,
            )
        flat_indices = np.array(nms_indices).reshape(-1)
        if self.debug:
            top_scores, top_class_ids = _top_candidates(candidate_scores, candidate_class_ids)
            print(
                "ONNXDetectionHead debug: "
                f"output_shapes={output_shapes}, "
                f"raw={raw_count}, "
                f"top_scores={top_scores}, "
                f"top_class_ids={top_class_ids}, "
                f"after_confidence={after_confidence_count}, "
                f"after_class_filter={after_class_count}, "
                f"after_nms={len(flat_indices)}"
            )
        if len(flat_indices) == 0:
            return []
        detections: list[Detection] = []
        for index in flat_indices:
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

    def _decode_candidate(self, row: np.ndarray, meta: LetterboxMeta) -> DecodedCandidate | None:
        if row.size < 6:
            return None

        if _is_yolov8_row(row.size, self.class_names):
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
        return DecodedCandidate(
            box=[int(round(x1)), int(round(y1)), box_width, box_height],
            confidence=confidence,
            class_id=class_id,
        )

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


def _flatten_outputs(outputs: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
    if isinstance(outputs, (list, tuple)):
        arrays = [_flatten_outputs(output) for output in outputs]
        return np.concatenate(arrays, axis=0) if arrays else np.empty((0, 0), dtype=np.float32)

    array = np.asarray(outputs)
    array = np.squeeze(array)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim != 2:
        return array.reshape(-1, array.shape[-1])
    if array.shape[0] < array.shape[1] and array.shape[0] in {84, 85}:
        return array.T
    return array


def _is_yolov8_row(row_size: int, class_names: dict[int, str]) -> bool:
    if row_size == 84:
        return True
    if row_size == 85:
        return False
    if class_names:
        dense_class_count = max(class_names) + 1
        if row_size == 4 + dense_class_count:
            return True
        if row_size == 5 + dense_class_count:
            return False
        if row_size == 4 + len(class_names):
            return True
        if row_size == 5 + len(class_names):
            return False
    return row_size > 85


def _output_shapes(outputs: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]) -> list[tuple[int, ...]]:
    if isinstance(outputs, (list, tuple)):
        return [tuple(np.asarray(output).shape) for output in outputs]
    return [tuple(np.asarray(outputs).shape)]


def _top_candidates(scores: list[float], class_ids: list[int]) -> tuple[list[float], list[int]]:
    ranked = sorted(zip(scores, class_ids), key=lambda item: item[0], reverse=True)[:10]
    top_scores = [round(float(score), 4) for score, _ in ranked]
    top_class_ids = [int(class_id) for _, class_id in ranked]
    return top_scores, top_class_ids
