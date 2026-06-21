from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import (
    CabinEvidenceObject,
    CabinEvidenceObjectType,
    CabinEvidenceRegion,
    CabinEvidenceRelation,
)


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

    v0.2.5 intentionally ships with a disabled dummy backend. This keeps the
    DMS contract ready for future ONNX/public/internal detectors without
    allowing object detections to drive final DMS warnings yet.
    """

    def __init__(self, config: DMSConfig) -> None:
        evidence_config = config.cabin_evidence or {}
        self.enabled = bool(evidence_config.get("enabled", True))
        self.backend = str(evidence_config.get("detector_backend", "dummy"))
        self.model_path = str(evidence_config.get("model_path", ""))
        self.min_confidence = float(evidence_config.get("min_confidence", 0.35))
        self.synthetic_timeline_path = str(evidence_config.get("synthetic_timeline_path", ""))
        self.synthetic_default_confidence = float(evidence_config.get("synthetic_default_confidence", 0.90))
        self.synthetic_timeline = SyntheticCabinTimeline(
            self.synthetic_timeline_path,
            self.synthetic_default_confidence,
        )
        self.synthetic_active = False
        self.backend_status = "DISABLED" if not self.enabled else "DUMMY_READY"

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
            self.backend_status = (
                "SYNTHETIC_ACTIVE"
                if objects
                else self.synthetic_timeline.status
            )
            return objects
        if self.backend != "dummy":
            self.backend_status = "MODEL_NOT_CONFIGURED"
            self.synthetic_active = False
            return []
        self.backend_status = "DUMMY_READY"
        self.synthetic_active = False
        return []


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
