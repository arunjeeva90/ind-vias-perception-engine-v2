from __future__ import annotations

import csv
import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ind_vias_dms.core.types import DMSState
from ind_vias_dms.utils.jsonl_writer import JSONLWriter


class DebugTraceRecorder:
    def __init__(
        self,
        trace_path: str | None = None,
        event_log_path: str | None = None,
        event_json_path: str | None = None,
        review_bundle_dir: str | None = None,
        save_event_keyframes: bool = False,
        save_event_crops: bool = False,
        keyframe_before_ms: int = 500,
        keyframe_after_ms: int = 500,
    ) -> None:
        self.review_bundle_dir = Path(review_bundle_dir) if review_bundle_dir else None
        if self.review_bundle_dir is not None:
            self.review_bundle_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_path or str(self.review_bundle_dir / "frame_trace.jsonl")
            event_log_path = event_log_path or str(self.review_bundle_dir / "event_timeline.csv")
            event_json_path = event_json_path or str(self.review_bundle_dir / "event_timeline.json")
        self.trace = JSONLWriter(trace_path)
        self.event_log_path = Path(event_log_path) if event_log_path else None
        self.event_json_path = Path(event_json_path) if event_json_path else None
        self.save_event_keyframes = save_event_keyframes
        self.save_event_crops = save_event_crops
        self.keyframe_before_ms = keyframe_before_ms
        self.keyframe_after_ms = keyframe_after_ms
        self.events: list[dict[str, Any]] = []
        self._last_event_key: tuple[str, str, str] | None = None
        self._last_cabin_key: tuple[str, str, str] | None = None

    @property
    def enabled(self) -> bool:
        return (
            self.trace.file is not None
            or self.event_log_path is not None
            or self.event_json_path is not None
            or self.review_bundle_dir is not None
        )

    def write_frame(self, state: DMSState, context: dict[str, object], frame: np.ndarray) -> None:
        if not self.enabled:
            return
        record = build_debug_record(state, context, frame)
        self.trace.write(record)
        self._capture_event_if_changed(record, state, frame)

    def close(self) -> None:
        self.trace.close()
        if self.event_log_path is not None:
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.event_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp_ms",
                        "frame_id",
                        "banner",
                        "attention",
                        "substate",
                        "availability",
                        "decision_path",
                        "reason_codes",
                        "raw_observation_codes",
                        "classification_reason_codes",
                        "cabin_phone_state",
                        "cabin_seatbelt_state",
                        "cabin_smoking_state",
                        "cabin_event_type",
                    ],
                )
                writer.writeheader()
                writer.writerows(self.events)
        if self.event_json_path is not None:
            self.event_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.event_json_path, "w", encoding="utf-8") as f:
                json.dump(self.events, f, indent=2)
        if self.review_bundle_dir is not None:
            summary = {
                "event_count": len(self.events),
                "keyframe_before_ms": self.keyframe_before_ms,
                "keyframe_after_ms": self.keyframe_after_ms,
                "outputs": {
                    "frame_trace": str(self.review_bundle_dir / "frame_trace.jsonl"),
                    "event_timeline_csv": str(self.review_bundle_dir / "event_timeline.csv"),
                    "event_timeline_json": str(self.review_bundle_dir / "event_timeline.json"),
                },
            }
            with open(self.review_bundle_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    def _capture_event_if_changed(
        self,
        record: dict[str, Any],
        state: DMSState,
        frame: np.ndarray,
    ) -> None:
        cabin_key = (
            state.cabin_evidence.phone_state.value,
            state.cabin_evidence.seatbelt_state.value,
            state.cabin_evidence.smoking_state.value,
        )
        cabin_event_type = _cabin_event_type(self._last_cabin_key, cabin_key)
        key = (
            state.dms_v02.final_banner,
            state.attention.attention_state.value,
            state.attention.attention_substate.value,
            cabin_event_type,
        )
        self._last_cabin_key = cabin_key
        if key == self._last_event_key:
            return
        self._last_event_key = key
        event = {
            "timestamp_ms": state.timestamp_ms,
            "frame_id": state.frame_id,
            "banner": state.dms_v02.final_banner,
            "attention": state.attention.attention_state.value,
            "substate": state.attention.attention_substate.value,
            "availability": state.driver_availability.state.value,
            "decision_path": state.dms_v02.final_decision_path or state.attention.final_decision_path,
            "reason_codes": ",".join(record.get("reason_codes", [])),
            "raw_observation_codes": ",".join(record.get("raw_observation_codes", [])),
            "classification_reason_codes": ",".join(record.get("classification_reason_codes", [])),
            "cabin_phone_state": state.cabin_evidence.phone_state.value,
            "cabin_seatbelt_state": state.cabin_evidence.seatbelt_state.value,
            "cabin_smoking_state": state.cabin_evidence.smoking_state.value,
            "cabin_event_type": cabin_event_type,
        }
        self.events.append(event)
        if self.review_bundle_dir is not None and self.save_event_keyframes:
            keyframe_dir = self.review_bundle_dir / "keyframes"
            keyframe_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(keyframe_dir / f"frame_{state.frame_id:06d}.jpg"), frame)
        if self.review_bundle_dir is not None and self.save_event_crops:
            face = record.get("driver_face_box_px")
            if isinstance(face, list) and len(face) == 4:
                x1, y1, x2, y2 = [int(v) for v in face]
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size:
                    crop_dir = self.review_bundle_dir / "driver_crops"
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(crop_dir / f"frame_{state.frame_id:06d}.jpg"), crop)


def build_debug_record(state: DMSState, context: dict[str, object], frame: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size else 0.0
    face = context.get("face")
    face_box = getattr(face, "bbox", None)
    contradiction_flags: list[str] = []
    if (
        state.dms_v02.final_banner == "NORMAL"
        and {
            "HEAD_DOWN",
            "GAZE_OFF_ROAD",
            "PHONE_DOWN_SUSPECTED",
            "POSSIBLE_PHONE_POSTURE",
        }
        & set(state.attention.attention_reason_codes + state.phone_use.reason_codes)
    ):
        contradiction_flags.append("NORMAL_WITH_ACTIVE_DISTRACTION_EVIDENCE")
    if (
        state.dms_v02.final_banner == "NORMAL"
        and state.attention.attention_state.value == "DEGRADED"
        and state.driver_identity.driver_track_hold_state != "PROPOSAL_VISIBLE_HELD"
    ):
        contradiction_flags.append("NORMAL_WHILE_ATTENTION_DEGRADED")
    if any(reason in state.drowsiness.perclos_validity_reason_codes for reason in state.phone_use.reason_codes):
        contradiction_flags.append("PERCLOS_REASON_CONTAINS_PHONE_REASON")
    if state.driver_availability.state.value == "UNAVAILABLE" and state.driver_identity.driver_proposal_visible:
        contradiction_flags.append("DRIVER_UNAVAILABLE_WITH_PROPOSAL")
    if state.driver_identity.driver_proposal_visible:
        contradiction_flags.append("PROPOSAL_ONLY_DRIVER_FRAME")
    if state.dms_v02.final_banner == "DMS DEGRADED" and abs(state.gaze.relative_yaw_deg) <= 30.0:
        contradiction_flags.append("DMS_DEGRADED_VISIBLE_NEAR_ROAD")
    if state.dms_v02.final_banner == "NORMAL":
        final_codes = set(state.dms_v02.classification_reason_codes or state.dms_v02.reason_codes)
        if final_codes & {"PHONE_TO_EAR_SUSPECTED", "PHONE_TO_EAR_CONFIRMED", "PHONE_TO_EAR_CANDIDATE"}:
            contradiction_flags.append("NORMAL_WITH_STALE_PHONE_REASON")
        if final_codes & {"FACE_LOST", "DRIVER_FACE_LOST_TEMP", "SIDE_PROFILE_FACE_LOST"}:
            contradiction_flags.append("NORMAL_WITH_STALE_FACE_LOST_REASON")
        if "HEAD_POSE_UNRELIABLE" in final_codes:
            contradiction_flags.append("NORMAL_WITH_STALE_POSE_UNRELIABLE_REASON")
        if final_codes & {
            "PHONE_TO_EAR_SUSPECTED",
            "HEAD_DOWN",
            "FACE_LOST",
            "LOW_EYE_VISIBILITY",
            "SIDE_PROFILE_FACE_LOST",
            "GAZE_OFF_ROAD_SUSTAINED",
            "HEAD_POSE_UNRELIABLE",
        }:
            contradiction_flags.append("NORMAL_WITH_STALE_CLASSIFICATION_REASON")

    classification_reason_codes = list(
        state.dms_v02.classification_reason_codes or state.dms_v02.reason_codes
    )
    raw_observation_codes = list(state.dms_v02.raw_observation_codes)

    return {
        "timestamp_ms": state.timestamp_ms,
        "frame_id": state.frame_id,
        "frame_quality": {
            "brightness": float(np.mean(gray)) if gray.size else 0.0,
            "contrast": float(np.std(gray)) if gray.size else 0.0,
            "blur_laplacian_var": blur,
        },
        "camera_status": state.dms_health.camera_status.value,
        "face_detection_status": state.dms_health.face_detection_status.value,
        "face_backend": state.dms_health.face_backend,
        "nir_mode": state.dms_health.nir_mode,
        "face_proposals": state.dms_health.face_proposals,
        "driver_face_box_px": list(face_box) if face_box else [],
        "driver_identity": _jsonable(asdict(state.driver_identity)),
        "occupants": _jsonable(asdict(state.occupants)),
        "occupancy": _jsonable(asdict(state.occupancy)),
        "gaze": _jsonable(asdict(state.gaze)),
        "drowsiness": _jsonable(asdict(state.drowsiness)),
        "phone_use": _jsonable(asdict(state.phone_use)),
        "cabin_evidence": _jsonable(asdict(state.cabin_evidence)),
        "phone_state": state.cabin_evidence.phone_state.value,
        "seatbelt_state": state.cabin_evidence.seatbelt_state.value,
        "smoking_state": state.cabin_evidence.smoking_state.value,
        "cabin_evidence_count": state.cabin_evidence.cabin_evidence_count,
        "evidence_objects": _jsonable(asdict(state.cabin_evidence).get("evidence_objects", [])),
        "affect_final_dms_state": state.cabin_evidence.affect_final_dms_state,
        "distraction": _jsonable(asdict(state.distraction)),
        "attention": _jsonable(asdict(state.attention)),
        "dms_v02": _jsonable(asdict(state.dms_v02)),
        "availability": _jsonable(asdict(state.driver_availability)),
        "raw_observation_codes": raw_observation_codes,
        "classification_reason_codes": classification_reason_codes,
        "reason_codes": classification_reason_codes,
        "contradiction_flags": contradiction_flags,
    }


def _cabin_event_type(
    previous: tuple[str, str, str] | None,
    current: tuple[str, str, str],
) -> str:
    phone, belt, smoking = current
    prev_phone, prev_belt, prev_smoking = previous or ("NO_PHONE", "SEATBELT_UNKNOWN", "NO_SMOKING")
    if prev_phone == phone and prev_belt == belt and prev_smoking == smoking:
        return ""
    if prev_phone == "NO_PHONE" and phone == "PHONE_OBJECT_CANDIDATE":
        return "CABIN_PHONE_CANDIDATE_STARTED"
    if prev_phone == "PHONE_OBJECT_CANDIDATE" and phone in {
        "PHONE_IN_HAND_SUSPECTED",
        "PHONE_TO_EAR_SUSPECTED",
        "PHONE_DOWN_TEXTING_SUSPECTED",
    }:
        return "CABIN_PHONE_SUSPECTED"
    if prev_phone != "NO_PHONE" and phone == "NO_PHONE":
        return "CABIN_PHONE_CLEARED"
    if prev_belt == "SEATBELT_UNKNOWN" and belt in {"SEATBELT_NOT_VISIBLE", "SEATBELT_NOT_WORN_SUSPECTED"}:
        return "CABIN_SEATBELT_UNKNOWN"
    if prev_belt != "SEATBELT_WORN_CONFIRMED" and belt == "SEATBELT_WORN_CONFIRMED":
        return "CABIN_SEATBELT_WORN_CONFIRMED"
    if prev_smoking == "NO_SMOKING" and smoking == "HAND_TO_MOUTH_CANDIDATE":
        return "CABIN_SMOKING_CANDIDATE_STARTED"
    if prev_smoking == "HAND_TO_MOUTH_CANDIDATE" and smoking == "SMOKING_SUSPECTED":
        return "CABIN_SMOKING_SUSPECTED"
    if prev_smoking != "NO_SMOKING" and smoking == "NO_SMOKING":
        return "CABIN_SMOKING_CLEARED"
    return ""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
