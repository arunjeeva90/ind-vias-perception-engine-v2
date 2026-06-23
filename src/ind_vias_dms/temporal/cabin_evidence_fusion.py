from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import (
    CabinEvidenceLifecycleState,
    CabinEvidenceObject,
    CabinEvidenceObjectType,
    CabinEvidenceRegion,
    CabinEvidenceRelation,
    CabinEvidenceState,
    CabinPhoneState,
    CabinSeatbeltState,
    CabinSmokingState,
)


_DRIVER_PHONE_RELATIONS = {
    CabinEvidenceRelation.NEAR_HAND,
    CabinEvidenceRelation.NEAR_LAP,
    CabinEvidenceRelation.NEAR_EAR,
}
_COMPATIBLE_DRIVER_PHONE_RELATIONS = {
    frozenset({CabinEvidenceRelation.NEAR_HAND, CabinEvidenceRelation.NEAR_LAP}),
}


class CabinEvidenceFusion:
    """Temporal semantic fusion for cabin object evidence.

    This class only emits semantic evidence states. It does not alter final DMS
    decisions unless a future config explicitly wires that behavior.
    """

    def __init__(self, config: DMSConfig) -> None:
        self.config = config.cabin_evidence or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.backend = str(self.config.get("detector_backend", "dummy"))
        self.affect_final_dms_state = bool(self.config.get("affect_final_dms_state", False))
        self.temporal_confirm_ms = int(self.config.get("temporal_confirm_ms", 1200))
        self.temporal_clear_ms = int(self.config.get("temporal_clear_ms", 700))
        self.phone_confirm_ms = int(self.config.get("phone_confirm_ms", 1500))
        self.phone_to_ear_confirm_ms = int(self.config.get("phone_to_ear_confirm_ms", 1200))
        self.phone_down_texting_confirm_ms = int(self.config.get("phone_down_texting_confirm_ms", 1500))
        self.seatbelt_confirm_ms = int(self.config.get("seatbelt_confirm_ms", 3000))
        self.smoking_confirm_ms = int(self.config.get("smoking_confirm_ms", 2500))
        self.driver_phone_only = bool(self.config.get("phone_driver_roi_only", self.config.get("driver_phone_only", True)))
        self.phone_ignore_outside_driver_roi = bool(self.config.get("phone_ignore_outside_driver_roi", True))
        self.allow_unknown_region_phone = bool(self.config.get("allow_unknown_region_phone", False))
        self.raw_phone_min_confidence = float(self.config.get("raw_phone_min_confidence", self.config.get("min_confidence", 0.25)))
        self.driver_phone_min_confidence = float(self.config.get("phone_driver_roi_min_confidence", self.config.get("driver_phone_min_confidence", 0.25)))
        self.driver_phone_near_ear_min_confidence = float(self.config.get("phone_to_ear_min_confidence", self.config.get("driver_phone_near_ear_min_confidence", 0.25)))
        self.driver_phone_near_hand_min_confidence = float(self.config.get("phone_distraction_min_confidence", self.config.get("driver_phone_near_hand_min_confidence", self.driver_phone_min_confidence)))
        self.driver_phone_near_lap_min_confidence = float(self.config.get("phone_distraction_min_confidence", self.config.get("driver_phone_near_lap_min_confidence", self.driver_phone_min_confidence)))
        self.phone_pending_min_frames = int(self.config.get("phone_pending_min_frames", self.config.get("driver_phone_min_stable_frames", 3)))
        self.phone_distraction_min_frames = int(self.config.get("phone_distraction_min_frames", self.config.get("driver_phone_min_stable_frames", 8)))
        self.phone_distraction_min_duration_ms = int(self.config.get("phone_distraction_min_duration_ms", self.config.get("phone_distraction_without_head_down_ms", 700)))
        self.phone_to_ear_min_frames = int(self.config.get("phone_to_ear_min_frames", self.config.get("driver_phone_min_stable_frames", 6)))
        self.phone_to_ear_min_duration_ms = int(self.config.get("phone_to_ear_min_duration_ms", self.config.get("phone_to_ear_confirm_ms", 700)))
        self.driver_phone_min_stable_frames = self.phone_pending_min_frames
        self.driver_phone_min_duration_ms = 0
        self.driver_phone_clear_ms = int(self.config.get("phone_clear_ms", self.config.get("driver_phone_clear_ms", 700)))
        self.driver_phone_track_hold_ms = int(self.config.get("phone_track_hold_ms", self.config.get("driver_phone_track_hold_ms", 700)))
        self.driver_phone_max_raw_gap_ms = int(
            self.config.get("phone_max_gap_ms", self.config.get("driver_phone_max_raw_gap_ms", max(700, self.driver_phone_clear_ms)))
        )
        self.driver_phone_allow_gap_if_iou_stable = bool(
            self.config.get("driver_phone_allow_gap_if_iou_stable", True)
        )
        self.driver_phone_hold_does_not_increment_consecutive = bool(
            self.config.get("driver_phone_hold_does_not_increment_consecutive", True)
        )
        self.driver_phone_clear_immediately_on_ignored_only = bool(
            self.config.get("driver_phone_clear_immediately_on_ignored_only", True)
        )
        self.driver_phone_min_bbox_area = float(self.config.get("driver_phone_min_bbox_area", 0.003))
        self.driver_phone_max_bbox_area = float(self.config.get("driver_phone_max_bbox_area", 0.080))
        self.driver_phone_min_aspect_ratio = float(self.config.get("driver_phone_min_aspect_ratio", 0.35))
        self.driver_phone_max_aspect_ratio = float(self.config.get("driver_phone_max_aspect_ratio", 2.80))
        self.driver_phone_hard_min_bbox_area = float(
            self.config.get("driver_phone_hard_min_bbox_area", self.driver_phone_min_bbox_area)
        )
        self.driver_phone_hard_max_bbox_area = float(self.config.get("driver_phone_hard_max_bbox_area", 0.250))
        self.driver_phone_soft_max_bbox_area = float(
            self.config.get("driver_phone_soft_max_bbox_area", self.driver_phone_max_bbox_area)
        )
        self.driver_phone_hard_min_aspect_ratio = float(
            self.config.get("driver_phone_hard_min_aspect_ratio", 0.20)
        )
        self.driver_phone_hard_max_aspect_ratio = float(
            self.config.get("driver_phone_hard_max_aspect_ratio", 5.00)
        )
        self.driver_phone_reject_large_square_area = float(
            self.config.get("driver_phone_reject_large_square_area", 0.045)
        )
        self.driver_phone_large_square_aspect_min = float(
            self.config.get("driver_phone_large_square_aspect_min", 0.75)
        )
        self.driver_phone_large_square_aspect_max = float(
            self.config.get("driver_phone_large_square_aspect_max", 1.35)
        )
        self.driver_phone_large_square_min_confidence = float(
            self.config.get("driver_phone_large_square_min_confidence", 0.70)
        )
        self.driver_phone_soft_large_area_min_confidence = float(
            self.config.get("driver_phone_soft_large_area_min_confidence", 0.55)
        )
        self.driver_phone_low_plausibility_extra_frames = int(
            self.config.get("driver_phone_low_plausibility_extra_frames", 5)
        )
        self.driver_phone_low_plausibility_extra_ms = int(
            self.config.get("driver_phone_low_plausibility_extra_ms", 300)
        )
        self.driver_phone_max_center_jump = float(self.config.get("phone_max_center_jump", self.config.get("driver_phone_max_center_jump", 0.25)))
        self.driver_phone_min_iou_for_same_track = float(
            self.config.get("phone_min_iou_for_same_track", self.config.get("driver_phone_min_iou_for_same_track", 0.08))
        )
        self.driver_phone_event_debounce_ms = int(self.config.get("driver_phone_event_debounce_ms", 300))
        self.driver_phone_candidate_on_debounce_ms = int(
            self.config.get("driver_phone_candidate_on_debounce_ms", 300)
        )
        self.driver_phone_clear_debounce_ms = int(self.config.get("driver_phone_clear_debounce_ms", 500))
        self.driver_phone_event_cooldown_ms = int(self.config.get("driver_phone_event_cooldown_ms", 500))
        self.driver_interaction_roi_expand_x = float(self.config.get("driver_interaction_roi_expand_x", 0.35))
        self.driver_interaction_roi_expand_y_top = float(self.config.get("driver_interaction_roi_expand_y_top", 0.25))
        self.driver_interaction_roi_expand_y_bottom = float(self.config.get("driver_interaction_roi_expand_y_bottom", 1.20))
        self.passenger_phone_observed_only = bool(self.config.get("passenger_phone_observed_only", True))
        self._tracks: dict[CabinEvidenceObjectType, CabinEvidenceObject] = {}
        self._last_state = CabinEvidenceState(
            enabled=self.enabled,
            detector_backend=self.backend,
            affect_final_dms_state=self.affect_final_dms_state,
            backend_status="DUMMY_READY" if self.enabled else "DISABLED",
        )

    def update(
        self,
        raw_objects: Iterable[CabinEvidenceObject],
        timestamp_ms: int,
        backend_status: str = "DUMMY_READY",
    ) -> CabinEvidenceState:
        if not self.enabled:
            return CabinEvidenceState(
                enabled=False,
                detector_backend=self.backend,
                backend_status="DISABLED",
                affect_final_dms_state=False,
                reason_codes=["CABIN_EVIDENCE_DISABLED"],
            )

        objects = [obj for obj in raw_objects if obj.confidence > 0.0]
        phone_objects = [obj for obj in objects if obj.object_type == CabinEvidenceObjectType.PHONE and obj.confidence >= self.raw_phone_min_confidence]
        phone_observed_regions = sorted({obj.region.value for obj in phone_objects})
        driver_phone_objects, ignored_phone_objects, ignored_phone_reasons = self._driver_phone_candidates(phone_objects)
        selected_phone = max(driver_phone_objects, key=lambda obj: obj.confidence) if driver_phone_objects else None
        previous_phone = self._tracks.get(CabinEvidenceObjectType.PHONE)
        phone_track_reset_reason = ""

        raw_by_type: dict[CabinEvidenceObjectType, CabinEvidenceObject] = {}
        for obj in objects:
            if obj.object_type == CabinEvidenceObjectType.PHONE:
                continue
            previous = raw_by_type.get(obj.object_type)
            if previous is None or obj.confidence > previous.confidence:
                raw_by_type[obj.object_type] = obj

        if selected_phone is not None:
            if previous_phone is not None and not self._same_driver_phone_track(previous_phone, selected_phone, timestamp_ms):
                phone_track_reset_reason = self._driver_phone_track_reset_reason(previous_phone, selected_phone, timestamp_ms) or "PHONE_UNSTABLE_TRACK"
                del self._tracks[CabinEvidenceObjectType.PHONE]
                previous_phone = None
                ignored_phone_reasons.append(phone_track_reset_reason)
            raw_by_type[CabinEvidenceObjectType.PHONE] = selected_phone
        elif phone_objects and self.driver_phone_clear_immediately_on_ignored_only:
            if previous_phone is not None:
                del self._tracks[CabinEvidenceObjectType.PHONE]
                phone_track_reset_reason = "DRIVER_PHONE_STALE_CLEARED_IGNORED_ONLY"
                ignored_phone_reasons.append(phone_track_reset_reason)

        fused_objects: list[CabinEvidenceObject] = []
        for object_type, obj in raw_by_type.items():
            previous = self._tracks.get(object_type)
            clear_ms = self.driver_phone_track_hold_ms if object_type == CabinEvidenceObjectType.PHONE else self.temporal_clear_ms
            same_track = (
                self._same_driver_phone_track(previous, obj, timestamp_ms)
                if object_type == CabinEvidenceObjectType.PHONE
                else (
                    previous is not None
                    and timestamp_ms - previous.last_seen_ms <= clear_ms
                    and previous.relation_to_driver == obj.relation_to_driver
                )
            )
            first_seen = previous.first_seen_ms if same_track and previous is not None else timestamp_ms
            stable_count = (previous.stable_count + 1) if same_track and previous is not None else 1
            duration_ms = max(0, timestamp_ms - first_seen)
            fused = replace(
                obj,
                first_seen_ms=first_seen,
                last_seen_ms=timestamp_ms,
                duration_ms=duration_ms,
                stable_count=stable_count,
                state=self._lifecycle_for(object_type, obj.relation_to_driver, duration_ms),
            )
            self._tracks[object_type] = fused
            fused_objects.append(fused)

        for object_type in list(self._tracks):
            if object_type in raw_by_type:
                continue
            clear_ms = self.driver_phone_track_hold_ms if object_type == CabinEvidenceObjectType.PHONE else self.temporal_clear_ms
            if timestamp_ms - self._tracks[object_type].last_seen_ms > clear_ms:
                if object_type == CabinEvidenceObjectType.PHONE:
                    phone_track_reset_reason = "PHONE_TRACK_HOLD_EXPIRED"
                del self._tracks[object_type]
            else:
                fused_objects.append(self._tracks[object_type])

        fused_objects.extend(ignored_phone_objects)

        phone_track = self._tracks.get(CabinEvidenceObjectType.PHONE)
        phone_raw_detected_this_frame = bool(phone_objects)
        driver_phone_fresh_this_frame = selected_phone is not None and phone_track is not None and phone_track.last_seen_ms == timestamp_ms
        phone_gap_ms = timestamp_ms - phone_track.last_seen_ms if phone_track is not None else 0
        phone_track_held = bool(phone_track is not None and not driver_phone_fresh_this_frame and phone_gap_ms <= self.driver_phone_track_hold_ms)
        phone_object = self._active_phone_object(timestamp_ms) if driver_phone_fresh_this_frame else None
        phone_state = self._phone_state(timestamp_ms) if driver_phone_fresh_this_frame else CabinPhoneState.NO_PHONE
        phone_pre_candidate = bool(driver_phone_fresh_this_frame and phone_track is not None and phone_object is None)
        phone_last_seen_ms = phone_track.last_seen_ms if phone_track is not None else 0
        phone_track_age_ms = max(0, timestamp_ms - phone_track.first_seen_ms) if phone_track is not None else 0
        phone_consecutive_frames = phone_track.stable_count if phone_track is not None else 0
        phone_stale_hold_active = phone_track_held
        phone_hold_remaining_ms = max(0, self.driver_phone_track_hold_ms - phone_gap_ms) if phone_track is not None else 0
        phone_plausibility_score, phone_plausibility_reason = self._phone_visual_plausibility(phone_track) if phone_track is not None else (1.0, "NO_DRIVER_PHONE_TRACK")
        unique_ignored_reasons = list(dict.fromkeys(ignored_phone_reasons))
        state = CabinEvidenceState(
            enabled=True,
            detector_backend=self.backend,
            backend_status=backend_status,
            model_path=str(self.config.get("model_path", "")),
            class_map_path=str(self.config.get("class_map_path", "")),
            synthetic_active=any(obj.source == "synthetic" for obj in fused_objects),
            affect_final_dms_state=self.affect_final_dms_state,
            cabin_phone_observed=bool(phone_objects),
            cabin_phone_observed_count=len(phone_objects),
            cabin_phone_observed_regions=phone_observed_regions,
            phone_state=phone_state,
            phone_relation=phone_object.relation_to_driver.value if phone_object is not None else "NONE",
            phone_source=phone_object.source if phone_object is not None else "NONE",
            phone_confidence=phone_object.confidence if phone_object is not None else 0.0,
            driver_phone_state=phone_state,
            phone_scenario=self._phone_scenario(phone_state),
            driver_roi_phone=driver_phone_fresh_this_frame,
            phone_driver_roi_hit=driver_phone_fresh_this_frame,
            phone_inside_driver_roi=driver_phone_fresh_this_frame,
            phone_track_confidence_smoothed=phone_object.confidence if phone_object is not None else 0.0,
            phone_track_fresh_this_frame=driver_phone_fresh_this_frame,
            phone_track_held=phone_track_held,
            phone_track_gap_ms=phone_gap_ms,
            phone_to_ear_geometry_valid=bool(phone_track is not None and phone_track.relation_to_driver == CabinEvidenceRelation.NEAR_EAR),
            phone_to_ear_geometry_reason=("FACE_ADJACENT" if phone_track is not None and phone_track.relation_to_driver == CabinEvidenceRelation.NEAR_EAR else "NOT_NEAR_EAR"),
            phone_behavior_support=False,
            phone_behavior_support_reason="NOT_EVALUATED",
            phone_to_ear_active=phone_state == CabinPhoneState.PHONE_TO_EAR_SUSPECTED,
            phone_distraction_active=phone_state in {CabinPhoneState.PHONE_DISTRACTION, CabinPhoneState.PHONE_CONFIRMED},
            phone_box_source=phone_object.source if phone_object is not None else "NONE",
            phone_outside_driver_roi_detected=any(reason in {"PHONE_OUTSIDE_DRIVER_INTERACTION_ROI", "PASSENGER_PHONE_OBSERVED_ONLY", "REAR_PHONE_OBSERVED_ONLY", "UNKNOWN_REGION_PHONE_IGNORED"} for reason in unique_ignored_reasons),
            phone_overlay_label=self._phone_overlay_label(phone_state, phone_track_held),
            phone_overlay_drawn=phone_state != CabinPhoneState.NO_PHONE,
            status_page_index=1,
            driver_phone_relation=phone_object.relation_to_driver.value if phone_object is not None else "NONE",
            driver_phone_source=phone_object.source if phone_object is not None else "NONE",
            driver_phone_confidence=phone_object.confidence if phone_object is not None else 0.0,
            driver_phone_relevant_count=len(driver_phone_objects),
            driver_phone_pre_candidate=phone_pre_candidate,
            driver_phone_track_age_ms=phone_track_age_ms,
            driver_phone_consecutive_frames=phone_consecutive_frames,
            driver_phone_last_seen_ms=phone_last_seen_ms,
            driver_phone_stale_hold_active=phone_stale_hold_active,
            phone_raw_detected_this_frame=phone_raw_detected_this_frame,
            driver_phone_fresh_this_frame=driver_phone_fresh_this_frame,
            driver_phone_track_held=phone_track_held,
            driver_phone_last_raw_seen_ms=phone_last_seen_ms,
            driver_phone_track_hold_remaining_ms=phone_hold_remaining_ms,
            driver_phone_track_gap_ms=phone_gap_ms,
            driver_phone_track_signature=self._phone_track_signature(phone_track),
            driver_phone_track_reset_reason=phone_track_reset_reason,
            driver_phone_visual_plausibility_score=phone_plausibility_score,
            driver_phone_visual_plausibility_reason=phone_plausibility_reason,
            driver_phone_relation_geometry_valid=phone_track is not None,
            driver_phone_relation_geometry_reason=("RELATION_GEOMETRY_ACCEPTED" if phone_track is not None else "NO_DRIVER_PHONE_TRACK"),
            driver_phone_relation_threshold_used=self._driver_phone_threshold(phone_track) if phone_track is not None else 0.0,
            overlay_phone_semantic_level=self._overlay_semantic_level(phone_state, phone_pre_candidate, phone_track_held),
            overlay_phone_hidden_duplicate_count=0,
            current_driver_phone_relevant_count=len(driver_phone_objects),
            current_ignored_phone_count=max(0, len(phone_objects) - len(driver_phone_objects)),
            ignored_phone_count=max(0, len(phone_objects) - len(driver_phone_objects)),
            ignored_phone_reasons=unique_ignored_reasons,
            seatbelt_state=self._seatbelt_state(timestamp_ms),
            smoking_state=self._smoking_state(timestamp_ms),
            cabin_evidence_count=len(fused_objects),
            evidence_objects=fused_objects,
        )
        state.phone_reason_codes = self._phone_reasons(state.phone_state)
        state.seatbelt_reason_codes = self._seatbelt_reasons(state.seatbelt_state)
        state.smoking_reason_codes = self._smoking_reasons(state.smoking_state)
        state.reason_codes = (
            state.phone_reason_codes + state.seatbelt_reason_codes + state.smoking_reason_codes
        )
        self._last_state = state
        return state

    def _driver_phone_candidates(
        self,
        phone_objects: list[CabinEvidenceObject],
    ) -> tuple[list[CabinEvidenceObject], list[CabinEvidenceObject], list[str]]:
        accepted: list[CabinEvidenceObject] = []
        ignored: list[CabinEvidenceObject] = []
        ignored_reasons: list[str] = []
        for obj in phone_objects:
            reason = self._driver_phone_reject_reason(obj)
            if reason:
                ignored_reasons.append(reason)
                ignored.append(replace(obj, state=CabinEvidenceLifecycleState.REJECTED))
            else:
                accepted.append(obj)
        return accepted, ignored, list(dict.fromkeys(ignored_reasons))

    def _driver_phone_reject_reason(self, obj: CabinEvidenceObject) -> str:
        if not self.driver_phone_only:
            return ""
        threshold = self._driver_phone_threshold(obj)
        if obj.confidence < threshold:
            return "DRIVER_PHONE_LOW_CONFIDENCE"
        if obj.region == CabinEvidenceRegion.PASSENGER:
            return "PASSENGER_PHONE_OBSERVED_ONLY"
        if obj.region == CabinEvidenceRegion.REAR:
            return "REAR_PHONE_OBSERVED_ONLY"
        if obj.region == CabinEvidenceRegion.UNKNOWN and not self.allow_unknown_region_phone:
            return "UNKNOWN_REGION_PHONE_IGNORED"
        if obj.relation_to_driver not in _DRIVER_PHONE_RELATIONS:
            return "PHONE_RELATION_NOT_DRIVER_RELEVANT"
        bbox_reason = self._phone_bbox_reject_reason(obj)
        if bbox_reason:
            return bbox_reason
        if not _bbox_center_inside(obj.bbox, self._driver_interaction_roi()):
            return "PHONE_OUTSIDE_DRIVER_INTERACTION_ROI"
        return ""

    def _driver_phone_threshold(self, obj: CabinEvidenceObject | None) -> float:
        if obj is None:
            return self.driver_phone_min_confidence
        if obj.relation_to_driver == CabinEvidenceRelation.NEAR_EAR:
            return self.driver_phone_near_ear_min_confidence
        if obj.relation_to_driver == CabinEvidenceRelation.NEAR_HAND:
            return self.driver_phone_near_hand_min_confidence
        if obj.relation_to_driver == CabinEvidenceRelation.NEAR_LAP:
            return self.driver_phone_near_lap_min_confidence
        return self.driver_phone_min_confidence

    @staticmethod
    def _overlay_semantic_level(phone_state: CabinPhoneState, pre_candidate: bool, track_held: bool = False) -> str:
        if phone_state == CabinPhoneState.PHONE_CONFIRMED:
            return "CONFIRMED"
        if phone_state in {
            CabinPhoneState.PHONE_TO_EAR_SUSPECTED,
            CabinPhoneState.PHONE_IN_HAND_SUSPECTED,
            CabinPhoneState.PHONE_DOWN_TEXTING_SUSPECTED,
        }:
            return "SUSPECTED"
        if phone_state != CabinPhoneState.NO_PHONE:
            return "HELD" if track_held else "CANDIDATE"
        if pre_candidate:
            return "HELD" if track_held else "PENDING"
        return "NONE"

    def _phone_bbox_reject_reason(self, obj: CabinEvidenceObject) -> str:
        area, aspect = _bbox_area_aspect(obj.bbox)
        if area < self.driver_phone_hard_min_bbox_area:
            return "PHONE_BBOX_TOO_SMALL"
        if area > self.driver_phone_hard_max_bbox_area:
            return "PHONE_BBOX_TOO_LARGE"
        if aspect < self.driver_phone_hard_min_aspect_ratio or aspect > self.driver_phone_hard_max_aspect_ratio:
            return "PHONE_BBOX_ASPECT_IMPLAUSIBLE"
        if (
            area >= self.driver_phone_reject_large_square_area
            and self.driver_phone_large_square_aspect_min <= aspect <= self.driver_phone_large_square_aspect_max
            and obj.confidence < self.driver_phone_large_square_min_confidence
        ):
            score, reason = self._phone_visual_plausibility(obj)
            if score <= 0.15:
                return reason
        return ""

    def _phone_visual_plausibility(self, obj: CabinEvidenceObject | None) -> tuple[float, str]:
        if obj is None:
            return 1.0, "NO_DRIVER_PHONE_TRACK"
        area, aspect = _bbox_area_aspect(obj.bbox)
        if area < self.driver_phone_hard_min_bbox_area:
            return 0.0, "PHONE_BBOX_TOO_SMALL"
        if area > self.driver_phone_hard_max_bbox_area:
            return 0.0, "PHONE_BBOX_TOO_LARGE"
        if aspect < self.driver_phone_hard_min_aspect_ratio or aspect > self.driver_phone_hard_max_aspect_ratio:
            return 0.0, "PHONE_BBOX_ASPECT_IMPLAUSIBLE"
        if area > self.driver_phone_soft_max_bbox_area:
            return 0.45, "LOW_PLAUSIBILITY_LARGE_BBOX"
        if (
            area >= self.driver_phone_reject_large_square_area
            and self.driver_phone_large_square_aspect_min <= aspect <= self.driver_phone_large_square_aspect_max
            and obj.confidence < self.driver_phone_large_square_min_confidence
        ):
            return 0.45, "LOW_PLAUSIBILITY_LARGE_SQUARE"
        return 1.0, "OK"

    def _same_driver_phone_track(
        self,
        previous: CabinEvidenceObject | None,
        current: CabinEvidenceObject,
        timestamp_ms: int,
    ) -> bool:
        if previous is None:
            return False
        gap_ms = timestamp_ms - previous.last_seen_ms
        if gap_ms > self.driver_phone_max_raw_gap_ms:
            return False
        if not _compatible_phone_relation(previous.relation_to_driver, current.relation_to_driver):
            return False
        stable_geometry = (
            _bbox_center_distance(previous.bbox, current.bbox) <= self.driver_phone_max_center_jump
            or _bbox_iou(previous.bbox, current.bbox) >= self.driver_phone_min_iou_for_same_track
        )
        if gap_ms <= self.driver_phone_clear_ms:
            return stable_geometry
        return bool(self.driver_phone_allow_gap_if_iou_stable and stable_geometry)

    def _driver_phone_track_reset_reason(
        self,
        previous: CabinEvidenceObject | None,
        current: CabinEvidenceObject,
        timestamp_ms: int,
    ) -> str:
        if previous is None:
            return ""
        gap_ms = timestamp_ms - previous.last_seen_ms
        if gap_ms > self.driver_phone_max_raw_gap_ms:
            return "PHONE_RAW_GAP_EXCEEDED"
        if not _compatible_phone_relation(previous.relation_to_driver, current.relation_to_driver):
            return "PHONE_RELATION_CHANGED"
        return "PHONE_UNSTABLE_TRACK"

    @staticmethod
    def _phone_track_signature(phone: CabinEvidenceObject | None) -> str:
        if phone is None:
            return ""
        return f"{phone.relation_to_driver.value}:{phone.first_seen_ms}:{round(phone.bbox[0], 2)}:{round(phone.bbox[1], 2)}"

    def _driver_interaction_roi(self) -> list[float]:
        base = [0.0, 0.0, 0.5, 1.0]
        x1, y1, x2, y2 = base
        width = x2 - x1
        height = y2 - y1
        return [
            max(0.0, x1 - width * self.driver_interaction_roi_expand_x),
            max(0.0, y1 - height * self.driver_interaction_roi_expand_y_top),
            min(1.0, x2 + width * self.driver_interaction_roi_expand_x),
            min(1.0, y2 + height * self.driver_interaction_roi_expand_y_bottom),
        ]

    def _lifecycle_for(
        self,
        object_type: CabinEvidenceObjectType,
        relation: CabinEvidenceRelation,
        duration_ms: int,
    ) -> CabinEvidenceLifecycleState:
        if object_type == CabinEvidenceObjectType.PHONE:
            if duration_ms >= self.phone_confirm_ms:
                return CabinEvidenceLifecycleState.CONFIRMED
            if duration_ms >= min(
                self.phone_to_ear_confirm_ms if relation == CabinEvidenceRelation.NEAR_EAR else self.phone_confirm_ms,
                self.phone_down_texting_confirm_ms
                if relation in {CabinEvidenceRelation.NEAR_LAP, CabinEvidenceRelation.NEAR_HAND}
                else self.phone_confirm_ms,
            ):
                return CabinEvidenceLifecycleState.SUSPECTED
        if object_type == CabinEvidenceObjectType.SEATBELT and duration_ms >= self.seatbelt_confirm_ms:
            return CabinEvidenceLifecycleState.CONFIRMED
        if object_type == CabinEvidenceObjectType.CIGARETTE and duration_ms >= self.smoking_confirm_ms:
            return CabinEvidenceLifecycleState.CONFIRMED
        if duration_ms >= self.temporal_confirm_ms:
            return CabinEvidenceLifecycleState.SUSPECTED
        return CabinEvidenceLifecycleState.CANDIDATE

    def _phone_state(self, timestamp_ms: int) -> CabinPhoneState:
        phone = self._active_phone_object(timestamp_ms)
        if phone is None:
            return CabinPhoneState.NO_PHONE
        age_ms = max(0, timestamp_ms - phone.first_seen_ms)
        if phone.duration_ms >= self.phone_confirm_ms:
            return CabinPhoneState.PHONE_CONFIRMED
        if (
            phone.relation_to_driver == CabinEvidenceRelation.NEAR_EAR
            and phone.stable_count >= self.phone_to_ear_min_frames
            and age_ms >= self.phone_to_ear_min_duration_ms
        ):
            return CabinPhoneState.PHONE_TO_EAR_SUSPECTED
        if phone.stable_count >= self.phone_distraction_min_frames and age_ms >= self.phone_distraction_min_duration_ms:
            return CabinPhoneState.PHONE_DISTRACTION
        return CabinPhoneState.PHONE_OBJECT_CANDIDATE

    def _active_phone_object(self, timestamp_ms: int) -> CabinEvidenceObject | None:
        phone = self._tracks.get(CabinEvidenceObjectType.PHONE)
        if phone is None:
            return None
        if timestamp_ms != phone.last_seen_ms:
            return None
        if timestamp_ms - phone.last_seen_ms > self.driver_phone_track_hold_ms:
            return None
        required_frames = self.phone_pending_min_frames
        required_ms = 0
        score, _ = self._phone_visual_plausibility(phone)
        if score < 1.0:
            required_frames += self.driver_phone_low_plausibility_extra_frames
            required_ms += self.driver_phone_low_plausibility_extra_ms
        if phone.stable_count < required_frames:
            return None
        if max(0, timestamp_ms - phone.first_seen_ms) < required_ms:
            return None
        return phone

    @staticmethod
    def _phone_scenario(phone_state: CabinPhoneState) -> str:
        if phone_state == CabinPhoneState.NO_PHONE:
            return "NONE"
        if phone_state == CabinPhoneState.PHONE_OBJECT_CANDIDATE:
            return "PENDING"
        if phone_state == CabinPhoneState.PHONE_TO_EAR_SUSPECTED:
            return "PHONE_TO_EAR"
        if phone_state == CabinPhoneState.PHONE_CONFIRMED:
            return "PHONE_CONFIRMED"
        return "PHONE_DISTRACTION"

    def _phone_overlay_label(self, phone_state: CabinPhoneState, held: bool) -> str:
        if held:
            return "PHONE TRACK / HELD"
        scenario = self._phone_scenario(phone_state)
        if scenario == "PENDING":
            return "PHONE IN DRIVER ROI / PENDING"
        if scenario == "PHONE_TO_EAR":
            return "PHONE TO EAR / SUSPECTED"
        if scenario in {"PHONE_DISTRACTION", "PHONE_CONFIRMED"}:
            return "PHONE DISTRACTION"
        return ""

    def _seatbelt_state(self, timestamp_ms: int) -> CabinSeatbeltState:
        belt = self._tracks.get(CabinEvidenceObjectType.SEATBELT)
        if belt is None:
            return CabinSeatbeltState.SEATBELT_UNKNOWN
        if timestamp_ms - belt.last_seen_ms > self.temporal_clear_ms:
            return CabinSeatbeltState.SEATBELT_NOT_VISIBLE
        if belt.duration_ms >= self.seatbelt_confirm_ms:
            return CabinSeatbeltState.SEATBELT_WORN_CONFIRMED
        return CabinSeatbeltState.SEATBELT_CONFIDENCE_LOW

    def _smoking_state(self, timestamp_ms: int) -> CabinSmokingState:
        cigarette = self._tracks.get(CabinEvidenceObjectType.CIGARETTE)
        hand = self._tracks.get(CabinEvidenceObjectType.HAND)
        smoking_evidence = cigarette or (
            hand if hand is not None and hand.relation_to_driver == CabinEvidenceRelation.NEAR_MOUTH else None
        )
        if smoking_evidence is None:
            return CabinSmokingState.NO_SMOKING
        if timestamp_ms - smoking_evidence.last_seen_ms > self.temporal_clear_ms:
            return CabinSmokingState.NO_SMOKING
        if smoking_evidence.object_type == CabinEvidenceObjectType.CIGARETTE and smoking_evidence.duration_ms >= self.smoking_confirm_ms:
            return CabinSmokingState.SMOKING_CONFIRMED
        if smoking_evidence.duration_ms >= self.temporal_confirm_ms:
            return CabinSmokingState.SMOKING_SUSPECTED
        return CabinSmokingState.HAND_TO_MOUTH_CANDIDATE

    @staticmethod
    def _phone_reasons(state: CabinPhoneState) -> list[str]:
        return [] if state == CabinPhoneState.NO_PHONE else [state.value]

    @staticmethod
    def _seatbelt_reasons(state: CabinSeatbeltState) -> list[str]:
        return [] if state == CabinSeatbeltState.SEATBELT_UNKNOWN else [state.value]

    @staticmethod
    def _smoking_reasons(state: CabinSmokingState) -> list[str]:
        return [] if state == CabinSmokingState.NO_SMOKING else [state.value]


def _bbox_center_inside(bbox: list[float], roi: list[float]) -> bool:
    if len(bbox) != 4 or len(roi) != 4:
        return False
    cx, cy = _bbox_center(bbox)
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    if len(bbox) != 4:
        return 0.0, 0.0
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_area_aspect(bbox: list[float]) -> tuple[float, float]:
    if len(bbox) != 4:
        return 0.0, 0.0
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    aspect = width / height if height > 0.0 else 0.0
    return width * height, aspect


def _bbox_center_distance(a: list[float], b: list[float]) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _bbox_iou(a: list[float], b: list[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(value) for value in a[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _compatible_phone_relation(a: CabinEvidenceRelation, b: CabinEvidenceRelation) -> bool:
    if a == b:
        return True
    return frozenset({a, b}) in _COMPATIBLE_DRIVER_PHONE_RELATIONS

