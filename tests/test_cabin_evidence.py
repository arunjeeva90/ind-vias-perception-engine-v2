from __future__ import annotations

import json

import numpy as np

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import (
    CabinEvidenceObject,
    CabinEvidenceObjectType,
    CabinEvidenceRegion,
    CabinEvidenceRelation,
    CabinPhoneState,
    CabinSeatbeltState,
    CabinSmokingState,
    DMSState,
)
from ind_vias_dms.temporal.cabin_evidence_fusion import CabinEvidenceFusion
from ind_vias_dms.utils.debug_trace import DebugTraceRecorder
from ind_vias_dms.vision.cabin_object_detection import CabinObjectDetector
from ind_vias_dms.visualization.overlay import banner_decision, status_dashboard_lines


def _phone(timestamp_ms: int = 0) -> CabinEvidenceObject:
    return CabinEvidenceObject(
        object_type=CabinEvidenceObjectType.PHONE,
        bbox=[0.25, 0.55, 0.35, 0.70],
        confidence=0.82,
        source="unit_test",
        region=CabinEvidenceRegion.DRIVER,
        relation_to_driver=CabinEvidenceRelation.NEAR_LAP,
        first_seen_ms=timestamp_ms,
        last_seen_ms=timestamp_ms,
    )


def test_cabin_evidence_types_serialize():
    state = DMSState()
    state.cabin_evidence.evidence_objects.append(_phone())
    state.cabin_evidence.cabin_evidence_count = 1
    payload = state.to_dict()

    assert payload["cabin_evidence"]["evidence_objects"][0]["object_type"] == "PHONE"
    assert payload["cabin_evidence"]["affect_final_dms_state"] is False


def test_dummy_detector_returns_empty_evidence_safely():
    detector = CabinObjectDetector(DMSConfig(cabin_evidence={"enabled": True, "detector_backend": "dummy"}))

    evidence = detector.detect(np.zeros((32, 32, 3), dtype=np.uint8), 100)

    assert evidence == []
    assert detector.backend_status == "DUMMY_READY"


def test_temporal_fusion_does_not_confirm_one_frame_phone():
    fusion = CabinEvidenceFusion(
        DMSConfig(cabin_evidence={"temporal_confirm_ms": 1200, "phone_confirm_ms": 1500})
    )

    state = fusion.update([_phone()], 1000)

    assert state.phone_state == CabinPhoneState.PHONE_OBJECT_CANDIDATE
    assert state.evidence_objects[0].state.value == "CANDIDATE"


def test_phone_candidate_becomes_suspected_after_duration():
    fusion = CabinEvidenceFusion(
        DMSConfig(
            cabin_evidence={
                "temporal_confirm_ms": 1200,
                "phone_down_texting_confirm_ms": 1500,
                "phone_confirm_ms": 2500,
            }
        )
    )

    fusion.update([_phone()], 1000)
    state = fusion.update([_phone()], 2600)

    assert state.phone_state == CabinPhoneState.PHONE_DOWN_TEXTING_SUSPECTED


def test_evidence_clears_after_clear_time():
    fusion = CabinEvidenceFusion(DMSConfig(cabin_evidence={"temporal_clear_ms": 700}))

    fusion.update([_phone()], 1000)
    held = fusion.update([], 1500)
    cleared = fusion.update([], 1801)

    assert held.phone_state == CabinPhoneState.PHONE_OBJECT_CANDIDATE
    assert cleared.phone_state == CabinPhoneState.NO_PHONE


def test_cabin_evidence_does_not_change_final_banner_when_disabled_for_decisions():
    state = DMSState()
    state.cabin_evidence.affect_final_dms_state = False
    state.cabin_evidence.phone_state = CabinPhoneState.PHONE_CONFIRMED

    label, _ = banner_decision(state)

    assert label == "NORMAL"


def test_seatbelt_unknown_does_not_create_warning():
    state = DMSState()
    state.cabin_evidence.seatbelt_state = CabinSeatbeltState.SEATBELT_UNKNOWN

    label, _ = banner_decision(state)

    assert label == "NORMAL"


def test_smoking_candidate_does_not_create_warning():
    state = DMSState()
    state.cabin_evidence.smoking_state = CabinSmokingState.HAND_TO_MOUTH_CANDIDATE

    label, _ = banner_decision(state)

    assert label == "NORMAL"


def test_dummy_cabin_evidence_does_not_emit_repeated_phone_cleared_events(tmp_path):
    event_path = tmp_path / "events.json"
    recorder = DebugTraceRecorder(event_json_path=str(event_path))
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    for frame_id in range(100):
        state = DMSState(frame_id=frame_id, timestamp_ms=frame_id * 33)
        recorder.write_frame(state, {}, frame)
    recorder.close()

    events = json.loads(event_path.read_text(encoding="utf-8"))
    cabin_events = [event for event in events if event.get("cabin_event_type")]

    assert cabin_events == []


def test_phone_candidate_cleared_event_emits_once(tmp_path):
    event_path = tmp_path / "events.json"
    recorder = DebugTraceRecorder(event_json_path=str(event_path))
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    candidate = DMSState(frame_id=1, timestamp_ms=100)
    candidate.cabin_evidence.phone_state = CabinPhoneState.PHONE_OBJECT_CANDIDATE
    cleared = DMSState(frame_id=2, timestamp_ms=200)

    recorder.write_frame(candidate, {}, frame)
    recorder.write_frame(cleared, {}, frame)
    recorder.write_frame(DMSState(frame_id=3, timestamp_ms=300), {}, frame)
    recorder.close()

    events = json.loads(event_path.read_text(encoding="utf-8"))
    cabin_events = [event.get("cabin_event_type") for event in events if event.get("cabin_event_type")]

    assert cabin_events == ["CABIN_PHONE_CANDIDATE_STARTED", "CABIN_PHONE_CLEARED"]


def test_status_dashboard_shows_compact_cabin_evidence_near_vehicle_context():
    labels = [label for label, _ in status_dashboard_lines(DMSState(), fps=30.0)]

    assert labels.index("Cabin phone") < labels.index("HMI banner")
    assert labels.index("Cabin belt") < labels.index("HMI banner")
    assert labels.index("Cabin smoking") < labels.index("HMI banner")
    assert labels.index("Cabin affect") < labels.index("HMI banner")
