from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import DMSState


class LearningMemoryWriter:
    def __init__(
        self,
        path: str | None,
        config: DMSConfig,
        save_keyframes: bool = False,
        save_crops: bool = False,
    ) -> None:
        self.path = Path(path) if path else None
        self.config = config
        self.save_keyframes = save_keyframes
        self.save_crops = save_crops
        self._file = None
        self._asset_dir: Path | None = None
        self._last_cabin_key: tuple[str, str, str] | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "a", encoding="utf-8")
            self._asset_dir = self.path.with_suffix("")
            if save_keyframes or save_crops:
                self._asset_dir.mkdir(parents=True, exist_ok=True)

    def write_frame(self, state: DMSState, context: dict[str, object], frame: np.ndarray) -> None:
        cabin_event_type = self._cabin_transition_event(state)
        if self._file is None:
            return
        should_record = self._should_record(state, cabin_event_type)
        self._last_cabin_key = self._cabin_key(state)
        if not should_record:
            return
        keyframe = ""
        if self.save_keyframes and self._asset_dir is not None:
            keyframe_path = self._asset_dir / f"frame_{state.frame_id:06d}.jpg"
            cv2.imwrite(str(keyframe_path), frame)
            keyframe = str(keyframe_path)
        record = {
            "schema_version": "dms_learning_event_v1",
            "learning_memory_status": "EVENT_RECORDED",
            "learning_mode": "DEVELOPMENT_OFFLINE_REVIEW_ONLY",
            "online_model_update_enabled": False,
            "session_id": state.driver_identity.driver_session_id or "UNKNOWN",
            "frame_id": state.frame_id,
            "timestamp_ms": state.timestamp_ms,
            "event_type": self._event_type(state, cabin_event_type),
            "human_label": None,
            "model_decision": state.dms_v02.final_banner,
            "expected_decision": None,
            "confidence": state.attention.attention_confidence,
            "raw_observation_codes": state.dms_v02.raw_observation_codes,
            "classification_reason_codes": state.dms_v02.classification_reason_codes,
            "active_thresholds": self._threshold_snapshot(),
            "calibration": {
                "road_axis_yaw_ref_deg": state.gaze.road_axis_yaw_ref_deg,
                "road_axis_pitch_ref_deg": state.gaze.road_axis_pitch_ref_deg,
                "road_axis_roll_ref_deg": state.gaze.road_axis_roll_ref_deg,
                "source": state.gaze.road_axis_calibration_source,
            },
            "keyframe": keyframe,
            "driver_crop": "",
            "phone_crop": "",
            "cabin_evidence": {
                "phone_state": state.cabin_evidence.phone_state.value,
                "phone_relation": state.cabin_evidence.phone_relation,
                "phone_source": state.cabin_evidence.phone_source,
                "phone_confidence": state.cabin_evidence.phone_confidence,
                "cabin_phone_observed": state.cabin_evidence.cabin_phone_observed,
                "cabin_phone_observed_count": state.cabin_evidence.cabin_phone_observed_count,
                "cabin_phone_observed_regions": state.cabin_evidence.cabin_phone_observed_regions,
                "driver_phone_state": state.cabin_evidence.driver_phone_state.value,
                "driver_phone_relation": state.cabin_evidence.driver_phone_relation,
                "driver_phone_confidence": state.cabin_evidence.driver_phone_confidence,
                "driver_phone_source": state.cabin_evidence.driver_phone_source,
                "driver_phone_relevant_count": state.cabin_evidence.driver_phone_relevant_count,
                "driver_phone_pre_candidate": state.cabin_evidence.driver_phone_pre_candidate,
                "driver_phone_track_age_ms": state.cabin_evidence.driver_phone_track_age_ms,
                "driver_phone_consecutive_frames": state.cabin_evidence.driver_phone_consecutive_frames,
                "driver_phone_last_seen_ms": state.cabin_evidence.driver_phone_last_seen_ms,
                "driver_phone_stale_hold_active": state.cabin_evidence.driver_phone_stale_hold_active,
                "driver_phone_relation_geometry_valid": state.cabin_evidence.driver_phone_relation_geometry_valid,
                "driver_phone_relation_geometry_reason": state.cabin_evidence.driver_phone_relation_geometry_reason,
                "driver_phone_relation_threshold_used": state.cabin_evidence.driver_phone_relation_threshold_used,
                "overlay_phone_semantic_level": state.cabin_evidence.overlay_phone_semantic_level,
                "overlay_phone_hidden_duplicate_count": state.cabin_evidence.overlay_phone_hidden_duplicate_count,
                "current_driver_phone_relevant_count": state.cabin_evidence.current_driver_phone_relevant_count,
                "current_ignored_phone_count": state.cabin_evidence.current_ignored_phone_count,
                "ignored_phone_count": state.cabin_evidence.ignored_phone_count,
                "ignored_phone_reasons": state.cabin_evidence.ignored_phone_reasons,
                "seatbelt_state": state.cabin_evidence.seatbelt_state.value,
                "smoking_state": state.cabin_evidence.smoking_state.value,
                "evidence_count": state.cabin_evidence.cabin_evidence_count,
                "affect_final_dms_state": state.cabin_evidence.affect_final_dms_state,
                "sources": sorted({obj.source for obj in state.cabin_evidence.evidence_objects}),
                "synthetic_active": state.cabin_evidence.synthetic_active,
            },
            "review_outputs_supported": [
                "false_positive_library",
                "false_negative_library",
                "regression_scenarios",
                "threshold_tuning_notes",
                "future_training_dataset_candidates",
                "reviewed_human_labels",
            ],
            "notes": "Development-phase learning memory only; no live in-vehicle model-weight or threshold updates are performed.",
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @staticmethod
    def _should_record(state: DMSState, cabin_event_type: str) -> bool:
        return (
            state.dms_v02.final_banner != "NORMAL"
            or state.phone_use.driver_state not in {"NO_PHONE", "UNKNOWN"}
            or state.vehicle.sanctioned_task_state != "NONE"
            or state.vehicle.dms_alert_suppression_reason
            in {"STANDBY", "STARTUP_INITIALIZING"}
            or bool(cabin_event_type)
            or state.attention.attention_substate.value
            in {"FACE_LOST", "SIDE_PROFILE_TRACKED", "SIDE_PROFILE_ATTENTION_LOSS"}
        )

    def _event_type(self, state: DMSState, cabin_event_type: str = "") -> str:
        reasons = set(
            state.dms_v02.classification_reason_codes
            + state.dms_v02.reason_codes
            + state.attention.attention_reason_codes
            + state.phone_use.reason_codes
            + state.vehicle.vehicle_speed_reason_codes
            + state.vehicle.sanctioned_task_reason_codes
        )
        if "MIRROR_CHECK_ALLOWED" in reasons:
            return "INDICATOR_SANCTIONED_MIRROR_CHECK"
        if cabin_event_type:
            return cabin_event_type
        if state.vehicle.dms_alert_suppression_reason in {"STANDBY", "STARTUP_INITIALIZING"}:
            return "DMS_ALERT_SUPPRESSED_SPEED_GATE"
        if state.dms_v02.final_banner == "DMS DEGRADED" and state.attention.attention_substate.value.startswith("SIDE"):
            return "SIDE_PROFILE_FALSE_DEGRADED"
        if state.dms_v02.final_banner == "DMS DEGRADED" and state.attention.yaw_classifiable:
            return "DEGRADED_WHILE_CLASSIFIABLE"
        if state.dms_v02.final_banner == "NORMAL" and (
            state.attention.attention_state.value != "NORMAL"
            or bool({"HEAD_DOWN", "GAZE_OFF_ROAD", "POSSIBLE_PHONE_POSTURE"} & reasons)
        ):
            return "NORMAL_WHILE_DISTRACTED"
        if state.phone_use.driver_state in {"PHONE_DOWN_SUSPECTED", "PHONE_TEXTING_SCROLLING_SUSPECTED"}:
            return "PHONE_POSTURE_ONLY"
        if "POSSIBLE_PHONE_POSTURE" in reasons and state.phone_use.driver_state in {"NO_PHONE", "UNKNOWN"}:
            return "PHONE_MISSED"
        if state.phone_use.driver_state not in {"NO_PHONE", "UNKNOWN"}:
            return "PHONE_POSTURE_ONLY"
        if state.dms_v02.final_banner == "DROWSINESS WARNING":
            return "DROWSINESS_FALSE_POSITIVE"
        if "OCCUPANT_FALSE_POSITIVE" in reasons:
            return "OCCUPANT_FALSE_POSITIVE"
        if "ROAD_AXIS_CALIBRATION_UPDATE" in reasons:
            return "ROAD_AXIS_CALIBRATION_UPDATE"
        return state.dms_v02.final_banner.replace(" ", "_")

    def _cabin_transition_event(self, state: DMSState) -> str:
        current = self._cabin_key(state)
        previous = self._last_cabin_key or ("NO_PHONE", "SEATBELT_UNKNOWN", "NO_SMOKING")
        phone, belt, smoking = current
        prev_phone, prev_belt, prev_smoking = previous
        if previous == current:
            return ""
        if prev_phone == "NO_PHONE" and phone == "PHONE_OBJECT_CANDIDATE":
            return "PHONE_IN_DRIVER_ROI"
        if prev_phone == "PHONE_OBJECT_CANDIDATE" and phone == "PHONE_IN_HAND_SUSPECTED":
            return "PHONE_DISTRACTION_STARTED"
        if prev_phone == "PHONE_OBJECT_CANDIDATE" and phone == "PHONE_TO_EAR_SUSPECTED":
            return "PHONE_TO_EAR_STARTED"
        if prev_phone == "PHONE_OBJECT_CANDIDATE" and phone == "PHONE_DOWN_TEXTING_SUSPECTED":
            return "PHONE_DISTRACTION_STARTED"
        if phone == "PHONE_CONFIRMED" and prev_phone != "PHONE_CONFIRMED":
            return "PHONE_DISTRACTION_STARTED"
        if prev_phone != "NO_PHONE" and phone == "NO_PHONE":
            return "PHONE_CLEARED"
        if prev_belt == "SEATBELT_UNKNOWN" and belt in {"SEATBELT_NOT_VISIBLE", "SEATBELT_NOT_WORN_SUSPECTED"}:
            return "CABIN_SEATBELT_UNKNOWN"
        if prev_belt != "SEATBELT_WORN_CONFIRMED" and belt == "SEATBELT_WORN_CONFIRMED":
            return "CABIN_SEATBELT_WORN_CONFIRMED"
        if prev_smoking == "NO_SMOKING" and smoking == "HAND_TO_MOUTH_CANDIDATE":
            return "CABIN_SMOKING_CANDIDATE"
        if smoking == "SMOKING_SUSPECTED" and prev_smoking != "SMOKING_SUSPECTED":
            return "CABIN_SMOKING_SUSPECTED"
        if prev_smoking != "NO_SMOKING" and smoking == "NO_SMOKING":
            return "CABIN_SMOKING_CLEARED"
        return ""

    @staticmethod
    def _cabin_key(state: DMSState) -> tuple[str, str, str]:
        return (
            state.cabin_evidence.phone_state.value,
            state.cabin_evidence.seatbelt_state.value,
            state.cabin_evidence.smoking_state.value,
        )

    def _threshold_snapshot(self) -> dict[str, Any]:
        return {
            "side_glance_monitor_deg": self.config.side_glance_monitor_deg,
            "side_glance_warning_deg": self.config.side_glance_warning_deg,
            "side_glance_monitor_ms": self.config.side_glance_monitor_ms,
            "side_glance_warning_ms": self.config.side_glance_warning_ms,
            "side_glance_recovery_ms": self.config.side_glance_recovery_ms,
            "phone_down_warning_ms": self.config.phone_down_warning_ms,
            "eye_closure_microsleep_ms": self.config.eye_closure_microsleep_ms,
        }


