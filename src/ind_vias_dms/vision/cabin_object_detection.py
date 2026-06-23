from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import (
    CabinEvidenceObject,
    CabinEvidenceObjectType,
    CabinEvidenceRegion,
    CabinEvidenceRelation,
)


DEFAULT_CLASS_MAP = {
    "classes": {
        "0": "PHONE",
        "1": "SEATBELT",
        "2": "CIGARETTE",
        "3": "HAND",
        "4": "UNKNOWN_OBJECT",
    },
    "aliases": {
        "cell phone": "PHONE",
        "mobile": "PHONE",
        "mobile phone": "PHONE",
        "seat belt": "SEATBELT",
        "seatbelt": "SEATBELT",
        "cigarette": "CIGARETTE",
        "smoke": "CIGARETTE",
        "hand": "HAND",
    },
}


class CabinClassMap:
    def __init__(self, path: str = "") -> None:
        self.path = path
        self.status = "CLASS_MAP_DEFAULT"
        payload = DEFAULT_CLASS_MAP
        if path:
            try:
                with open(path, "r", encoding="utf-8-sig") as f:
                    payload = json.load(f)
                self.status = "CLASS_MAP_READY"
            except OSError:
                payload = DEFAULT_CLASS_MAP
                self.status = "CLASS_MAP_MISSING"
            except json.JSONDecodeError:
                payload = DEFAULT_CLASS_MAP
                self.status = "CLASS_MAP_INVALID"
        self.classes = {str(key): str(value) for key, value in payload.get("classes", {}).items()}
        aliases = dict(DEFAULT_CLASS_MAP.get("aliases", {}))
        aliases.update(payload.get("aliases", {}))
        self.aliases = {
            str(key).strip().lower(): str(value).strip().upper()
            for key, value in aliases.items()
        }

    def object_type_for(self, class_id: int | float | str) -> CabinEvidenceObjectType:
        raw = self.classes.get(str(int(class_id))) if _is_number(class_id) else str(class_id)
        canonical = self.canonical_name(raw or class_id)
        return _enum_value(CabinEvidenceObjectType, canonical, CabinEvidenceObjectType.UNKNOWN_OBJECT)

    def has_class_id(self, class_id: int | float | str) -> bool:
        if not _is_number(class_id):
            return False
        return str(int(class_id)) in self.classes

    def canonical_name(self, raw_value: Any) -> str:
        value = str(raw_value or "").strip()
        if not value:
            return CabinEvidenceObjectType.UNKNOWN_OBJECT.value
        alias = self.aliases.get(value.lower())
        return (alias or value).upper().replace("-", "_").replace(" ", "_")


class SyntheticCabinTimeline:
    def __init__(self, path: str, default_confidence: float = 0.90) -> None:
        self.path = path
        self.default_confidence = default_confidence
        self.events: list[dict[str, Any]] = []
        self.status = "SYNTHETIC_TIMELINE_NOT_CONFIGURED"
        if path:
            self.events = self._load(path)

    def active_objects(self, timestamp_ms: int) -> list[CabinEvidenceObject]:
        objects: list[CabinEvidenceObject] = []
        for event in self.events:
            if int(event.get("start_ms", 0)) <= timestamp_ms <= int(event.get("end_ms", -1)):
                objects.append(_event_to_object(event, timestamp_ms, self.default_confidence))
        return objects

    def _load(self, path: str) -> list[dict[str, Any]]:
        timeline_path = Path(path)
        if not timeline_path.exists():
            self.status = "SYNTHETIC_TIMELINE_MISSING"
            return []
        try:
            if timeline_path.suffix.lower() == ".csv":
                with open(timeline_path, newline="", encoding="utf-8") as f:
                    rows = [dict(row) for row in csv.DictReader(f)]
                self.status = "SYNTHETIC_TIMELINE_READY"
                return rows
            with open(timeline_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            events = payload.get("events", []) if isinstance(payload, dict) else payload
            self.status = "SYNTHETIC_TIMELINE_READY"
            return [event for event in events if isinstance(event, dict)]
        except (OSError, json.JSONDecodeError, csv.Error, ValueError):
            self.status = "SYNTHETIC_TIMELINE_INVALID"
            return []


class CabinObjectDetector:
    """Cabin object evidence detector facade.

    The ONNX path is a contract backend only: it uses OpenCV DNN if a local
    model is supplied, returns empty evidence safely when it is not, and never
    lets cabin detections drive final DMS warnings by itself.
    """

    def __init__(self, config: DMSConfig) -> None:
        evidence_config = config.cabin_evidence or {}
        self.enabled = bool(evidence_config.get("enabled", True))
        self.backend = str(evidence_config.get("detector_backend", "dummy"))
        self.model_path = str(evidence_config.get("model_path", ""))
        self.class_map_path = str(evidence_config.get("class_map_path", ""))
        self.input_width = int(evidence_config.get("input_width", 640))
        self.input_height = int(evidence_config.get("input_height", 640))
        self.min_confidence = float(evidence_config.get("raw_phone_min_confidence", evidence_config.get("min_confidence", 0.35)))
        self.nms_iou_threshold = float(evidence_config.get("nms_iou_threshold", 0.45))
        self.max_detections = int(evidence_config.get("max_detections", 50))
        self.normalize_bboxes = bool(evidence_config.get("normalize_bboxes", True))
        self.allow_unknown_objects = bool(evidence_config.get("allow_unknown_objects", False))
        self.roi_association_enabled = bool(evidence_config.get("roi_association_enabled", True))
        self.relation_inference_enabled = bool(evidence_config.get("relation_inference_enabled", True))
        self.near_ear_face_x_margin = float(evidence_config.get("phone_to_ear_face_x_margin", evidence_config.get("near_ear_face_x_margin", 0.35)))
        self.near_ear_face_y_margin_top = float(evidence_config.get("phone_to_ear_face_y_margin_top", evidence_config.get("near_ear_face_y_margin_top", 0.30)))
        self.near_ear_face_y_margin_bottom = float(evidence_config.get("phone_to_ear_face_y_margin_bottom", evidence_config.get("near_ear_face_y_margin_bottom", 0.65)))
        self.near_ear_max_below_face = float(evidence_config.get("phone_to_ear_max_below_face", evidence_config.get("near_ear_max_below_face", 0.45)))
        self.class_map = CabinClassMap(self.class_map_path)
        self.synthetic_timeline_path = str(evidence_config.get("synthetic_timeline_path", ""))
        self.synthetic_default_confidence = float(evidence_config.get("synthetic_default_confidence", 0.90))
        self.synthetic_timeline = SyntheticCabinTimeline(
            self.synthetic_timeline_path,
            self.synthetic_default_confidence,
        )
        self.synthetic_active = False
        self.net = None
        self.last_raw_output_shapes: list[list[int]] = []
        self.last_parser_format = "NOT_RUN"
        self.last_yolo_debug = self._empty_yolo_debug("NOT_RUN")
        self.backend_status = "DISABLED" if not self.enabled else "DUMMY_READY"
        if self.enabled and self.backend == "onnx":
            self._load_onnx_model()

    def detect(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
        context: dict[str, Any] | None = None,
    ) -> list[CabinEvidenceObject]:
        if not self.enabled:
            self.backend_status = "DISABLED"
            self.synthetic_active = False
            return []
        if self.backend == "synthetic":
            objects = [
                obj
                for obj in self.synthetic_timeline.active_objects(timestamp_ms)
                if obj.confidence >= self.min_confidence
            ]
            self.synthetic_active = bool(objects)
            self.backend_status = "SYNTHETIC_ACTIVE" if objects else self.synthetic_timeline.status
            return objects
        if self.backend == "onnx":
            self.synthetic_active = False
            return self._detect_onnx(frame, timestamp_ms, context or {})
        if self.backend != "dummy":
            self.backend_status = "MODEL_NOT_CONFIGURED"
            self.synthetic_active = False
            return []
        self.backend_status = "DUMMY_READY"
        self.synthetic_active = False
        return []

    def parse_outputs(
        self,
        outputs: Any,
        frame_shape: tuple[int, int] | tuple[int, int, int],
        timestamp_ms: int = 0,
        context: dict[str, Any] | None = None,
    ) -> list[CabinEvidenceObject]:
        output = _first_array(outputs)
        self.last_raw_output_shapes = _output_shapes(outputs)
        self.last_parser_format = "NOT_RUN"
        self.last_yolo_debug = self._empty_yolo_debug("NOT_RUN")
        if output is None:
            self.backend_status = "UNSUPPORTED_OUTPUT_SHAPE"
            self.last_parser_format = "UNSUPPORTED"
            return []
        output = np.asarray(output, dtype=np.float32)
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        parser_format = self._parser_format_for(output)
        if parser_format is None:
            self.backend_status = "UNSUPPORTED_OUTPUT_SHAPE"
            self.last_parser_format = "UNSUPPORTED"
            return []
        if parser_format == "YOLOV8_CXCYWH_CLASS_SCORES":
            output = output.T
            self.last_yolo_debug = self._initial_yolo_debug(output, "channel_first_transposed")
        self.last_parser_format = parser_format
        height, width = int(frame_shape[0]), int(frame_shape[1])
        context = dict(context or {})
        context["_frame_shape"] = frame_shape
        detections = []
        for row in output:
            parsed = self._parse_row(row, width, height, parser_format)
            if parsed is None:
                continue
            bbox, confidence, class_id = parsed
            is_yolo = parser_format == "YOLOV8_CXCYWH_CLASS_SCORES"
            if is_yolo and confidence >= self.min_confidence:
                self.last_yolo_debug["yolo_debug_candidates_above_conf"] += 1
            if confidence < self.min_confidence:
                continue
            if not self.allow_unknown_objects and not self.class_map.has_class_id(class_id):
                continue
            if is_yolo:
                self.last_yolo_debug["yolo_debug_candidates_after_class_map_filter"] += 1
            bbox = _clamp_bbox(bbox)
            if not _valid_bbox(bbox):
                continue
            if is_yolo:
                self.last_yolo_debug["yolo_debug_candidates_after_bbox_validation"] += 1
            detections.append((bbox, confidence, class_id))
        detections = _nms(detections, self.nms_iou_threshold, self.max_detections)
        if parser_format == "YOLOV8_CXCYWH_CLASS_SCORES":
            self.last_yolo_debug["yolo_debug_candidates_after_nms"] = len(detections)
        evidence = [
            self._to_evidence(bbox, confidence, class_id, timestamp_ms, context or {})
            for bbox, confidence, class_id in detections
        ]
        self.backend_status = "OK"
        return evidence

    def _empty_yolo_debug(self, parser_axis_used: str = "NOT_RUN") -> dict[str, Any]:
        return {
            "yolo_debug_total_candidates": 0,
            "yolo_debug_max_score": 0.0,
            "yolo_debug_max_class_id": None,
            "yolo_debug_top_classes_before_filter": [],
            "yolo_debug_candidates_above_conf": 0,
            "yolo_debug_candidates_after_class_map_filter": 0,
            "yolo_debug_candidates_after_bbox_validation": 0,
            "yolo_debug_candidates_after_nms": 0,
            "yolo_debug_class_map_keys": sorted(self.class_map.classes.keys(), key=lambda key: (0, int(key)) if key.isdigit() else (1, key)),
            "yolo_debug_parser_axis_used": parser_axis_used,
        }

    def _initial_yolo_debug(self, rows: np.ndarray, parser_axis_used: str) -> dict[str, Any]:
        debug = self._empty_yolo_debug(parser_axis_used)
        debug["yolo_debug_total_candidates"] = int(rows.shape[0])
        top: list[dict[str, Any]] = []
        max_score = -1.0
        max_class_id: int | None = None
        for row in rows:
            if row.shape[0] < 5:
                continue
            scores = row[4:]
            if scores.size == 0:
                continue
            class_id = int(np.argmax(scores))
            score = float(scores[class_id])
            if score > max_score:
                max_score = score
                max_class_id = class_id
            top.append({
                "class_id": class_id,
                "score": score,
                "bbox_cxcywh": [float(value) for value in row[:4]],
            })
        top.sort(key=lambda item: item["score"], reverse=True)
        debug["yolo_debug_max_score"] = max(0.0, max_score)
        debug["yolo_debug_max_class_id"] = max_class_id
        debug["yolo_debug_top_classes_before_filter"] = top[:10]
        return debug

    def _parser_format_for(self, output: np.ndarray) -> str | None:
        if output.ndim != 2:
            return None
        rows, cols = int(output.shape[0]), int(output.shape[1])
        if cols == 6:
            return "N6_XYXY_CONF_CLASS"
        if rows >= 5 and cols > rows:
            return "YOLOV8_CXCYWH_CLASS_SCORES"
        if cols > 6:
            return "N_5C_OBJECTNESS_CLASS_SCORES"
        return None

    def _load_onnx_model(self) -> None:
        if not self.model_path or not Path(self.model_path).exists():
            self.backend_status = "MODEL_MISSING"
            self.net = None
            return
        try:
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            self.backend_status = "OK"
        except cv2.error:
            self.net = None
            self.backend_status = "MODEL_LOAD_FAILED"

    def _detect_onnx(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
        context: dict[str, Any],
    ) -> list[CabinEvidenceObject]:
        if self.net is None:
            if self.backend_status not in {"MODEL_MISSING", "MODEL_LOAD_FAILED"}:
                self._load_onnx_model()
            return []
        try:
            blob = cv2.dnn.blobFromImage(
                frame,
                scalefactor=1.0 / 255.0,
                size=(self.input_width, self.input_height),
                swapRB=True,
                crop=False,
            )
            self.net.setInput(blob)
            outputs = self.net.forward()
            return self.parse_outputs(outputs, frame.shape, timestamp_ms, context)
        except cv2.error:
            self.backend_status = "MODEL_ERROR"
            return []

    def _parse_row(
        self,
        row: np.ndarray,
        width: int,
        height: int,
        parser_format: str,
    ) -> tuple[list[float], float, int] | None:
        if parser_format == "YOLOV8_CXCYWH_CLASS_SCORES":
            if row.shape[0] < 5:
                return None
            cx, cy, bw, bh = [float(value) for value in row[:4]]
            scores = row[4:]
            if scores.size == 0:
                return None
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            bbox = self._normalize_yolo_xywh([cx, cy, bw, bh])
            return bbox, confidence, class_id
        if parser_format == "N6_XYXY_CONF_CLASS":
            x1, y1, x2, y2, confidence, class_id = [float(value) for value in row[:6]]
            bbox = self._normalize_xyxy([x1, y1, x2, y2], width, height)
            return bbox, confidence, int(class_id)
        x1, y1, x2, y2 = [float(value) for value in row[:4]]
        objectness = float(row[4])
        scores = row[5:]
        if scores.size == 0:
            return None
        class_id = int(np.argmax(scores))
        confidence = objectness * float(scores[class_id])
        if x2 > x1 and y2 > y1:
            bbox = self._normalize_xyxy([x1, y1, x2, y2], width, height)
        else:
            bbox = self._normalize_xywh([x1, y1, x2, y2], width, height)
        return bbox, confidence, class_id

    def _normalize_xyxy(self, bbox: list[float], width: int, height: int) -> list[float]:
        if self.normalize_bboxes:
            return bbox
        x1, y1, x2, y2 = bbox
        return [x1 / width, y1 / height, x2 / width, y2 / height]

    def _normalize_xywh(self, bbox: list[float], width: int, height: int) -> list[float]:
        cx, cy, bw, bh = bbox
        if not self.normalize_bboxes:
            cx, bw = cx / width, bw / width
            cy, bh = cy / height, bh / height
        return [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]

    def _normalize_yolo_xywh(self, bbox: list[float]) -> list[float]:
        cx, cy, bw, bh = bbox
        cx, bw = cx / max(1.0, float(self.input_width)), bw / max(1.0, float(self.input_width))
        cy, bh = cy / max(1.0, float(self.input_height)), bh / max(1.0, float(self.input_height))
        return [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]

    def _to_evidence(
        self,
        bbox: list[float],
        confidence: float,
        class_id: int,
        timestamp_ms: int,
        context: dict[str, Any],
    ) -> CabinEvidenceObject:
        object_type = self.class_map.object_type_for(class_id)
        region = self._associate_region(bbox, context)
        relation = self._infer_relation(object_type, bbox, region, context)
        return CabinEvidenceObject(
            object_type=object_type,
            bbox=bbox,
            confidence=float(confidence),
            source="onnx",
            region=region,
            relation_to_driver=relation,
            first_seen_ms=timestamp_ms,
            last_seen_ms=timestamp_ms,
        )

    def _associate_region(self, bbox: list[float], context: dict[str, Any]) -> CabinEvidenceRegion:
        if not self.roi_association_enabled:
            return CabinEvidenceRegion.UNKNOWN
        cx, cy = _bbox_center(bbox)
        driver_roi = _context_roi(context)
        if driver_roi and _point_in_bbox(cx, cy, driver_roi):
            return CabinEvidenceRegion.DRIVER
        if cx >= 0.55:
            return CabinEvidenceRegion.PASSENGER
        if cy <= 0.55:
            return CabinEvidenceRegion.REAR
        return CabinEvidenceRegion.UNKNOWN

    def _infer_relation(
        self,
        object_type: CabinEvidenceObjectType,
        bbox: list[float],
        region: CabinEvidenceRegion,
        context: dict[str, Any],
    ) -> CabinEvidenceRelation:
        if not self.relation_inference_enabled:
            return CabinEvidenceRelation.UNKNOWN
        _, cy = _bbox_center(bbox)
        driver_roi = _context_roi(context) or [0.0, 0.0, 0.5, 1.0]
        _, ry1, _, ry2 = driver_roi
        rh = max(0.01, ry2 - ry1)
        lower_driver = ry1 + 0.68 * rh
        if object_type == CabinEvidenceObjectType.PHONE:
            if region == CabinEvidenceRegion.DRIVER and self._is_near_driver_ear(bbox, context):
                return CabinEvidenceRelation.NEAR_EAR
            if region == CabinEvidenceRegion.DRIVER and cy >= lower_driver:
                return CabinEvidenceRelation.NEAR_LAP
            if region == CabinEvidenceRegion.DRIVER:
                return CabinEvidenceRelation.NEAR_HAND
        if object_type == CabinEvidenceObjectType.CIGARETTE and region == CabinEvidenceRegion.DRIVER:
            return CabinEvidenceRelation.NEAR_MOUTH
        if object_type == CabinEvidenceObjectType.SEATBELT and region == CabinEvidenceRegion.DRIVER:
            return CabinEvidenceRelation.ACROSS_TORSO
        if object_type == CabinEvidenceObjectType.HAND and region == CabinEvidenceRegion.DRIVER:
            return CabinEvidenceRelation.NEAR_HAND
        return CabinEvidenceRelation.UNKNOWN

    def _is_near_driver_ear(self, bbox: list[float], context: dict[str, Any]) -> bool:
        face_bbox = _context_face_bbox_norm(context)
        if face_bbox is None:
            return False
        cx, cy = _bbox_center(bbox)
        fx1, fy1, fx2, fy2 = face_bbox
        face_width = max(0.01, fx2 - fx1)
        face_height = max(0.01, fy2 - fy1)
        x_min = max(0.0, fx1 - self.near_ear_face_x_margin * face_width)
        x_max = min(1.0, fx2 + self.near_ear_face_x_margin * face_width)
        y_min = max(0.0, fy1 - self.near_ear_face_y_margin_top * face_height)
        y_max = min(1.0, fy2 + self.near_ear_face_y_margin_bottom * face_height)
        if not (x_min <= cx <= x_max and y_min <= cy <= y_max):
            return False
        if cy > fy2 + self.near_ear_max_below_face * face_height:
            return False
        near_left_edge = abs(cx - fx1) <= self.near_ear_face_x_margin * face_width
        near_right_edge = abs(cx - fx2) <= self.near_ear_face_x_margin * face_width
        horizontally_adjacent = cx < fx1 or cx > fx2 or near_left_edge or near_right_edge
        return horizontally_adjacent


def _event_to_object(
    event: dict[str, Any],
    timestamp_ms: int,
    default_confidence: float,
) -> CabinEvidenceObject:
    object_type = _enum_value(CabinEvidenceObjectType, event.get("object_type"), CabinEvidenceObjectType.UNKNOWN_OBJECT)
    region = _enum_value(CabinEvidenceRegion, event.get("region"), CabinEvidenceRegion.UNKNOWN)
    relation = _enum_value(
        CabinEvidenceRelation,
        event.get("relation_to_driver"),
        CabinEvidenceRelation.UNKNOWN,
    )
    bbox = event.get("bbox", [])
    if not isinstance(bbox, list):
        bbox = []
    confidence = float(event.get("confidence", default_confidence) or default_confidence)
    return CabinEvidenceObject(
        object_type=object_type,
        bbox=[float(value) for value in bbox[:4]],
        confidence=confidence,
        source="synthetic",
        region=region,
        relation_to_driver=relation,
        first_seen_ms=timestamp_ms,
        last_seen_ms=timestamp_ms,
    )


def _enum_value(enum_cls: type, raw_value: Any, default: Any) -> Any:
    value = str(raw_value or default.value).upper()
    for item in enum_cls:
        if item.value == value:
            return item
    return default


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _first_array(outputs: Any) -> np.ndarray | None:
    if isinstance(outputs, (list, tuple)):
        if not outputs:
            return None
        return np.asarray(outputs[0])
    return np.asarray(outputs) if outputs is not None else None


def _output_shapes(outputs: Any) -> list[list[int]]:
    if isinstance(outputs, (list, tuple)):
        return [list(np.asarray(output).shape) for output in outputs]
    if outputs is None:
        return []
    return [list(np.asarray(outputs).shape)]


def _clamp_bbox(bbox: list[float]) -> list[float]:
    return [max(0.0, min(1.0, float(value))) for value in bbox[:4]]


def _valid_bbox(bbox: list[float]) -> bool:
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    return x2 > x1 and y2 > y1 and (x2 - x1) * (y2 - y1) > 0.0001


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _point_in_bbox(x: float, y: float, bbox: list[float]) -> bool:
    x1, y1, x2, y2 = bbox[:4]
    return x1 <= x <= x2 and y1 <= y <= y2



def _context_face_bbox_norm(context: dict[str, Any]) -> list[float] | None:
    face = context.get("driver_face")
    bbox = getattr(face, "bbox", None)
    if bbox is None:
        return None
    values = [float(value) for value in bbox[:4]]
    if len(values) != 4:
        return None
    if max(abs(value) for value in values) <= 1.5:
        return values
    shape = context.get("_frame_shape")
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        height, width = float(shape[0]), float(shape[1])
        if width > 0.0 and height > 0.0:
            x1, y1, x2, y2 = values
            return [x1 / width, y1 / height, x2 / width, y2 / height]
    return None

def _context_roi(context: dict[str, Any]) -> list[float] | None:
    roi = context.get("driver_roi_norm")
    if isinstance(roi, (list, tuple)) and len(roi) >= 4:
        return [float(value) for value in roi[:4]]
    return None


def _nms(
    detections: list[tuple[list[float], float, int]],
    iou_threshold: float,
    max_detections: int,
) -> list[tuple[list[float], float, int]]:
    selected: list[tuple[list[float], float, int]] = []
    for candidate in sorted(detections, key=lambda item: item[1], reverse=True):
        bbox, _, class_id = candidate
        if any(class_id == kept[2] and _iou(bbox, kept[0]) > iou_threshold for kept in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max_detections:
            break
    return selected


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


