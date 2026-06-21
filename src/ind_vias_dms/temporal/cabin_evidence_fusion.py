from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import (
    CabinEvidenceLifecycleState,
    CabinEvidenceObject,
    CabinEvidenceObjectType,
    CabinEvidenceRelation,
    CabinEvidenceState,
    CabinPhoneState,
    CabinSeatbeltState,
    CabinSmokingState,
)


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

        raw_by_type: dict[CabinEvidenceObjectType, CabinEvidenceObject] = {}
        for obj in raw_objects:
            if obj.confidence <= 0.0:
                continue
            previous = raw_by_type.get(obj.object_type)
            if previous is None or obj.confidence > previous.confidence:
                raw_by_type[obj.object_type] = obj

        fused_objects: list[CabinEvidenceObject] = []
        for object_type, obj in raw_by_type.items():
            previous = self._tracks.get(object_type)
            same_track = (
                previous is not None
                and timestamp_ms - previous.last_seen_ms <= self.temporal_clear_ms
                and previous.relation_to_driver == obj.relation_to_driver
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
            if timestamp_ms - self._tracks[object_type].last_seen_ms > self.temporal_clear_ms:
                del self._tracks[object_type]
            else:
                fused_objects.append(self._tracks[object_type])

        phone_object = self._active_phone_object(timestamp_ms)
        state = CabinEvidenceState(
            enabled=True,
            detector_backend=self.backend,
            backend_status=backend_status,
            synthetic_active=any(obj.source == "synthetic" for obj in fused_objects),
            affect_final_dms_state=self.affect_final_dms_state,
            phone_state=self._phone_state(timestamp_ms),
            phone_relation=phone_object.relation_to_driver.value if phone_object is not None else "NONE",
            phone_source=phone_object.source if phone_object is not None else "NONE",
            phone_confidence=phone_object.confidence if phone_object is not None else 0.0,
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
        if phone.duration_ms >= self.phone_confirm_ms:
            return CabinPhoneState.PHONE_CONFIRMED
        if phone.relation_to_driver == CabinEvidenceRelation.NEAR_EAR and phone.duration_ms >= self.phone_to_ear_confirm_ms:
            return CabinPhoneState.PHONE_TO_EAR_SUSPECTED
        if phone.relation_to_driver == CabinEvidenceRelation.NEAR_HAND and phone.duration_ms >= self.temporal_confirm_ms:
            return CabinPhoneState.PHONE_IN_HAND_SUSPECTED
        if phone.relation_to_driver == CabinEvidenceRelation.NEAR_LAP and phone.duration_ms >= self.phone_down_texting_confirm_ms:
            return CabinPhoneState.PHONE_DOWN_TEXTING_SUSPECTED
        return CabinPhoneState.PHONE_OBJECT_CANDIDATE

    def _active_phone_object(self, timestamp_ms: int) -> CabinEvidenceObject | None:
        phone = self._tracks.get(CabinEvidenceObjectType.PHONE)
        if phone is None:
            return None
        if timestamp_ms - phone.last_seen_ms > self.temporal_clear_ms:
            return None
        return phone

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
