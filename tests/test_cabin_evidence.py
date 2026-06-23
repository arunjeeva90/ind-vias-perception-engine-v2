from __future__ import annotations

import json
from argparse import Namespace

import cv2
import numpy as np

import apps.inspect_cabin_onnx as inspect_app
from apps.inspect_cabin_onnx import console_summary, inspect_model, write_report
from apps.sample_dms_frame import sample_frame
from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import (
    CabinEvidenceLifecycleState,
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
from ind_vias_dms.utils.learning_memory import LearningMemoryWriter
from ind_vias_dms.vision.cabin_object_detection import CabinClassMap, CabinObjectDetector, SyntheticCabinTimeline
from ind_vias_dms.visualization.overlay import (
    _cabin_evidence_label,
    _visible_cabin_evidence_objects,
    banner_decision,
    status_dashboard_lines,
)


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
        DMSConfig(cabin_evidence={"temporal_confirm_ms": 1200, "phone_confirm_ms": 1500, "driver_phone_min_stable_frames": 1, "driver_phone_min_duration_ms": 0, "driver_phone_clear_ms": 5000})
    )

    state = fusion.update([_phone()], 1000)

    assert state.phone_state == CabinPhoneState.PHONE_OBJECT_CANDIDATE
    assert state.evidence_objects[0].state.value == "CANDIDATE"


def test_phone_candidate_becomes_suspected_after_duration():
    fusion = CabinEvidenceFusion(
        DMSConfig(
            cabin_evidence={
                    "temporal_confirm_ms": 1200,
                    "temporal_clear_ms": 5000,
                    "phone_down_texting_confirm_ms": 1500,
                    "phone_confirm_ms": 2500,
                    "driver_phone_min_stable_frames": 1,
                    "driver_phone_clear_ms": 5000,
            }
        )
    )

    fusion.update([_phone()], 1000)
    state = fusion.update([_phone()], 2600)

    assert state.phone_state == CabinPhoneState.PHONE_DISTRACTION


def test_evidence_clears_after_clear_time():
    fusion = CabinEvidenceFusion(DMSConfig(cabin_evidence={"temporal_clear_ms": 700, "driver_phone_min_stable_frames": 1, "driver_phone_min_duration_ms": 0, "driver_phone_clear_ms": 700}))

    fusion.update([_phone()], 1000)
    held = fusion.update([], 1500)
    cleared = fusion.update([], 1801)

    assert held.phone_state == CabinPhoneState.NO_PHONE
    assert held.driver_phone_track_held is True
    assert held.phone_scenario == "NONE"
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

    assert cabin_events == ["PHONE_IN_DRIVER_ROI_STARTED", "PHONE_CLEARED"]


def test_status_dashboard_shows_compact_cabin_evidence_near_vehicle_context():
    labels = [label for label, _ in status_dashboard_lines(DMSState(), fps=30.0)]

    assert labels.index("Cabin backend") < labels.index("HMI banner")
    assert labels.index("Cabin objects") < labels.index("HMI banner")
    assert labels.index("Phone scenario") < labels.index("HMI banner")
    assert labels.index("Driver ROI phone") < labels.index("HMI banner")
    assert labels.index("Cabin belt") < labels.index("HMI banner")
    assert labels.index("Cabin smoking") < labels.index("HMI banner")
    assert labels.index("Cabin affect") < labels.index("HMI banner")


def test_synthetic_timeline_parser_loads_valid_json(tmp_path):
    path = tmp_path / "timeline.json"
    path.write_text(
        '{"version":"test","events":[{"start_ms":10,"end_ms":20,"object_type":"PHONE","bbox":[0,0,1,1]}]}',
        encoding="utf-8",
    )

    timeline = SyntheticCabinTimeline(str(path))

    assert timeline.status == "SYNTHETIC_TIMELINE_READY"
    assert len(timeline.events) == 1


def test_missing_synthetic_timeline_is_safe(tmp_path):
    detector = CabinObjectDetector(
        DMSConfig(
            cabin_evidence={
                "enabled": True,
                "detector_backend": "synthetic",
                "synthetic_timeline_path": str(tmp_path / "missing.json"),
            }
        )
    )

    evidence = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8), 100)

    assert evidence == []
    assert detector.backend_status == "SYNTHETIC_TIMELINE_MISSING"


def test_active_synthetic_phone_near_hand_produces_evidence(tmp_path):
    path = tmp_path / "timeline.json"
    path.write_text(
        '{"events":[{"start_ms":0,"end_ms":1000,"object_type":"PHONE","confidence":0.9,'
        '"bbox":[0.1,0.2,0.3,0.4],"region":"DRIVER","relation_to_driver":"NEAR_HAND"}]}',
        encoding="utf-8",
    )
    detector = CabinObjectDetector(
        DMSConfig(cabin_evidence={"detector_backend": "synthetic", "synthetic_timeline_path": str(path)})
    )

    evidence = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8), 500)

    assert len(evidence) == 1
    assert evidence[0].object_type == CabinEvidenceObjectType.PHONE
    assert evidence[0].relation_to_driver == CabinEvidenceRelation.NEAR_HAND
    assert evidence[0].source == "synthetic"


def test_active_synthetic_phone_near_ear_produces_evidence(tmp_path):
    path = tmp_path / "timeline.json"
    path.write_text(
        '{"events":[{"start_ms":0,"end_ms":1000,"object_type":"PHONE",'
        '"bbox":[0.1,0.2,0.3,0.4],"region":"DRIVER","relation_to_driver":"NEAR_EAR"}]}',
        encoding="utf-8",
    )
    detector = CabinObjectDetector(
        DMSConfig(cabin_evidence={"detector_backend": "synthetic", "synthetic_timeline_path": str(path)})
    )

    evidence = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8), 500)

    assert len(evidence) == 1
    assert evidence[0].relation_to_driver == CabinEvidenceRelation.NEAR_EAR


def test_phone_near_hand_sustained_becomes_in_hand_suspected():
    fusion = CabinEvidenceFusion(
        DMSConfig(cabin_evidence={"temporal_confirm_ms": 1200, "temporal_clear_ms": 5000, "phone_confirm_ms": 2500, "driver_phone_min_stable_frames": 1, "driver_phone_clear_ms": 5000})
    )
    phone = _phone()
    phone.relation_to_driver = CabinEvidenceRelation.NEAR_HAND

    fusion.update([phone], 1000)
    state = fusion.update([phone], 2300)

    assert state.phone_state == CabinPhoneState.PHONE_DISTRACTION
    assert state.phone_relation == "NEAR_HAND"
    assert state.phone_source == "unit_test"
    assert state.phone_confidence == 0.82


def test_phone_near_ear_sustained_becomes_to_ear_suspected():
    fusion = CabinEvidenceFusion(
        DMSConfig(
            cabin_evidence={
                "temporal_clear_ms": 5000,
                "phone_to_ear_confirm_ms": 1200,
                "phone_confirm_ms": 2500,
                "driver_phone_min_stable_frames": 1,
                "driver_phone_clear_ms": 5000,
            }
        )
    )
    phone = _phone()
    phone.relation_to_driver = CabinEvidenceRelation.NEAR_EAR

    fusion.update([phone], 1000)
    state = fusion.update([phone], 2300)

    assert state.phone_state == CabinPhoneState.PHONE_TO_EAR_SUSPECTED


def test_seatbelt_across_torso_sustained_becomes_worn_confirmed():
    fusion = CabinEvidenceFusion(DMSConfig(cabin_evidence={"temporal_clear_ms": 5000, "seatbelt_confirm_ms": 3000}))
    belt = CabinEvidenceObject(
        object_type=CabinEvidenceObjectType.SEATBELT,
        confidence=0.9,
        relation_to_driver=CabinEvidenceRelation.ACROSS_TORSO,
        source="synthetic",
    )

    fusion.update([belt], 1000)
    state = fusion.update([belt], 4100)

    assert state.seatbelt_state == CabinSeatbeltState.SEATBELT_WORN_CONFIRMED


def test_cigarette_near_mouth_sustained_becomes_smoking_suspected():
    fusion = CabinEvidenceFusion(
        DMSConfig(cabin_evidence={"temporal_confirm_ms": 1200, "temporal_clear_ms": 5000, "smoking_confirm_ms": 5000})
    )
    cigarette = CabinEvidenceObject(
        object_type=CabinEvidenceObjectType.CIGARETTE,
        confidence=0.9,
        relation_to_driver=CabinEvidenceRelation.NEAR_MOUTH,
        source="synthetic",
    )

    fusion.update([cigarette], 1000)
    state = fusion.update([cigarette], 2400)

    assert state.smoking_state == CabinSmokingState.SMOKING_SUSPECTED


def test_synthetic_evidence_clear_transition_once(tmp_path):
    event_path = tmp_path / "events.json"
    recorder = DebugTraceRecorder(event_json_path=str(event_path))
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    candidate = DMSState(frame_id=1, timestamp_ms=100)
    candidate.cabin_evidence.phone_state = CabinPhoneState.PHONE_OBJECT_CANDIDATE
    candidate.cabin_evidence.synthetic_active = True
    cleared = DMSState(frame_id=2, timestamp_ms=200)

    recorder.write_frame(candidate, {}, frame)
    recorder.write_frame(cleared, {}, frame)
    recorder.write_frame(DMSState(frame_id=3, timestamp_ms=300), {}, frame)
    recorder.close()

    events = json.loads(event_path.read_text(encoding="utf-8"))
    cabin_events = [event.get("cabin_event_type") for event in events if event.get("cabin_event_type")]

    assert cabin_events == ["PHONE_IN_DRIVER_ROI_STARTED", "PHONE_CLEARED"]


def test_synthetic_overlay_label_includes_syn_prefix():
    label = _cabin_evidence_label("PHONE", "CANDIDATE", "synthetic", "NEAR_HAND")

    assert label == "PHONE IN DRIVER ROI / PENDING"


def test_learning_memory_does_not_repeat_unchanged_cabin_state(tmp_path):
    path = tmp_path / "learning.jsonl"
    writer = LearningMemoryWriter(str(path), DMSConfig())
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for frame_id in range(3):
        state = DMSState(frame_id=frame_id, timestamp_ms=frame_id * 100)
        state.dms_v02.final_banner = "NORMAL"
        state.cabin_evidence.phone_state = CabinPhoneState.PHONE_OBJECT_CANDIDATE
        writer.write_frame(state, {}, frame)
    writer.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert [record["event_type"] for record in records] == ["PHONE_IN_DRIVER_ROI"]


def test_learning_memory_records_cabin_phone_transitions(tmp_path):
    path = tmp_path / "learning.jsonl"
    writer = LearningMemoryWriter(str(path), DMSConfig())
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    sequence = [
        CabinPhoneState.PHONE_OBJECT_CANDIDATE,
        CabinPhoneState.PHONE_DISTRACTION,
        CabinPhoneState.PHONE_CONFIRMED,
        CabinPhoneState.NO_PHONE,
    ]
    for frame_id, phone_state in enumerate(sequence):
        state = DMSState(frame_id=frame_id, timestamp_ms=frame_id * 1000)
        state.dms_v02.final_banner = "NORMAL"
        state.cabin_evidence.phone_state = phone_state
        state.cabin_evidence.phone_relation = "NEAR_HAND" if phone_state != CabinPhoneState.NO_PHONE else "NONE"
        writer.write_frame(state, {}, frame)
    writer.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert [record["event_type"] for record in records] == [
        "PHONE_IN_DRIVER_ROI",
        "PHONE_DISTRACTION_STARTED",
        "PHONE_CLEARED",
    ]
    assert records[1]["cabin_evidence"]["phone_relation"] == "NEAR_HAND"


def test_phone_relation_is_retained_when_phone_confirmed():
    fusion = CabinEvidenceFusion(
        DMSConfig(cabin_evidence={"temporal_clear_ms": 5000, "phone_confirm_ms": 1500, "driver_phone_min_stable_frames": 1, "driver_phone_min_duration_ms": 0, "driver_phone_clear_ms": 5000})
    )
    phone = _phone()
    phone.source = "synthetic"
    phone.relation_to_driver = CabinEvidenceRelation.NEAR_EAR

    fusion.update([phone], 1000)
    state = fusion.update([phone], 2600)

    assert state.phone_state == CabinPhoneState.PHONE_CONFIRMED
    assert state.phone_relation == "NEAR_EAR"
    assert state.phone_source == "synthetic"
    assert state.phone_confidence == 0.82


def _phone_obj(region=CabinEvidenceRegion.DRIVER, relation=CabinEvidenceRelation.NEAR_LAP, bbox=None, confidence=0.82):
    return CabinEvidenceObject(
        object_type=CabinEvidenceObjectType.PHONE,
        bbox=bbox or [0.25, 0.55, 0.35, 0.70],
        confidence=confidence,
        source="unit_test",
        region=region,
        relation_to_driver=relation,
    )


def _driver_phone_fusion(**overrides):
    config = {
        "temporal_clear_ms": 5000,
        "phone_confirm_ms": 2500,
        "driver_phone_min_stable_frames": 1,
        "driver_phone_min_duration_ms": 0,
    }
    config.update(overrides)
    return CabinEvidenceFusion(DMSConfig(cabin_evidence=config))


def test_driver_phone_near_lap_becomes_candidate():
    fusion = _driver_phone_fusion()

    state = fusion.update([_phone_obj(relation=CabinEvidenceRelation.NEAR_LAP)], 1000)

    assert state.cabin_phone_observed is True
    assert state.cabin_phone_observed_count == 1
    assert state.cabin_phone_observed_regions == ["DRIVER"]
    assert state.driver_phone_state == CabinPhoneState.PHONE_OBJECT_CANDIDATE
    assert state.phone_state == CabinPhoneState.PHONE_OBJECT_CANDIDATE
    assert state.driver_phone_relation == "NEAR_LAP"
    assert state.driver_phone_relevant_count == 1
    assert state.ignored_phone_count == 0


def test_driver_phone_near_ear_becomes_candidate():
    fusion = _driver_phone_fusion()

    state = fusion.update([_phone_obj(relation=CabinEvidenceRelation.NEAR_EAR, bbox=[0.20, 0.10, 0.30, 0.24])], 1000)

    assert state.driver_phone_state == CabinPhoneState.PHONE_OBJECT_CANDIDATE
    assert state.driver_phone_relation == "NEAR_EAR"


def test_passenger_phone_observed_does_not_change_driver_phone_state():
    fusion = _driver_phone_fusion()
    passenger_phone = _phone_obj(
        region=CabinEvidenceRegion.PASSENGER,
        relation=CabinEvidenceRelation.UNKNOWN,
        bbox=[0.65, 0.45, 0.75, 0.65],
    )

    state = fusion.update([passenger_phone], 1000)

    assert state.cabin_phone_observed is True
    assert state.cabin_phone_observed_count == 1
    assert state.cabin_phone_observed_regions == ["PASSENGER"]
    assert state.driver_phone_state == CabinPhoneState.NO_PHONE
    assert state.phone_state == CabinPhoneState.NO_PHONE
    assert state.ignored_phone_count == 1
    assert "PASSENGER_PHONE_OBSERVED_ONLY" in state.ignored_phone_reasons
    assert state.evidence_objects[0].state.value == "REJECTED"


def test_unknown_region_phone_ignored_when_not_allowed():
    fusion = _driver_phone_fusion(allow_unknown_region_phone=False)
    unknown_phone = _phone_obj(region=CabinEvidenceRegion.UNKNOWN, relation=CabinEvidenceRelation.NEAR_HAND)

    state = fusion.update([unknown_phone], 1000)

    assert state.driver_phone_state == CabinPhoneState.NO_PHONE
    assert state.cabin_phone_observed_regions == ["UNKNOWN"]
    assert "UNKNOWN_REGION_PHONE_IGNORED" in state.ignored_phone_reasons


def test_phone_outside_expanded_driver_roi_does_not_change_driver_phone_state():
    fusion = _driver_phone_fusion()
    outside_phone = _phone_obj(region=CabinEvidenceRegion.DRIVER, relation=CabinEvidenceRelation.NEAR_HAND, bbox=[0.92, 0.55, 0.98, 0.68])

    state = fusion.update([outside_phone], 1000)

    assert state.cabin_phone_observed is True
    assert state.driver_phone_state == CabinPhoneState.NO_PHONE
    assert "PHONE_OUTSIDE_DRIVER_INTERACTION_ROI" in state.ignored_phone_reasons


def test_driver_phone_requires_stable_frames_and_duration_by_default():
    fusion = CabinEvidenceFusion(DMSConfig(cabin_evidence={"temporal_clear_ms": 5000}))
    phone = _phone_obj(confidence=0.82)
    state = None
    for index in range(7):
        state = fusion.update([phone], 1000 + index * 50)

    assert state is not None
    assert state.driver_phone_state == CabinPhoneState.PHONE_OBJECT_CANDIDATE
    assert state.phone_scenario == "PENDING"
    state = None
    for index in range(8, 16):
        state = fusion.update([phone], 1000 + index * 50)

    assert state is not None
    assert state.driver_phone_state == CabinPhoneState.PHONE_DISTRACTION
    assert state.phone_scenario == "PHONE_DISTRACTION"
    assert state.driver_phone_consecutive_frames >= 8
    assert state.driver_phone_track_age_ms >= 700

def test_driver_phone_status_lines_are_visible():
    state = DMSState()
    state.cabin_evidence.cabin_phone_observed = True
    state.cabin_evidence.cabin_phone_observed_regions = ["DRIVER"]
    state.cabin_evidence.driver_phone_state = CabinPhoneState.PHONE_OBJECT_CANDIDATE
    labels = [label for label, _ in status_dashboard_lines(state, fps=30.0)]

    assert "Cabin phone obs" in labels
    assert "Cabin phone regs" in labels
    assert "Phone scenario" in labels
    assert "Driver ROI phone" in labels
    assert "Phone track age" in labels
    assert "Ignored phone" in labels


def test_debug_trace_includes_driver_phone_gating_fields(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    recorder = DebugTraceRecorder(trace_path=str(trace_path))
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    state = DMSState(frame_id=1, timestamp_ms=100)
    state.cabin_evidence.cabin_phone_observed = True
    state.cabin_evidence.cabin_phone_observed_count = 1
    state.cabin_evidence.cabin_phone_observed_regions = ["PASSENGER"]
    state.cabin_evidence.ignored_phone_count = 1
    state.cabin_evidence.ignored_phone_reasons = ["PASSENGER_PHONE_OBSERVED_ONLY"]

    recorder.write_frame(state, {}, frame)
    recorder.close()
    record = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])

    assert record["cabin_phone_observed"] is True
    assert record["cabin_phone_observed_count"] == 1
    assert record["cabin_phone_observed_regions"] == ["PASSENGER"]
    assert record["driver_phone_state"] == "NO_PHONE"
    assert record["ignored_phone_count"] == 1
    assert record["current_ignored_phone_count"] == 0
    assert record["driver_phone_consecutive_frames"] == 0
    assert record["ignored_phone_reasons"] == ["PASSENGER_PHONE_OBSERVED_ONLY"]


def test_passenger_phone_does_not_emit_driver_phone_event(tmp_path):
    event_path = tmp_path / "events.json"
    recorder = DebugTraceRecorder(event_json_path=str(event_path))
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    state = DMSState(frame_id=1, timestamp_ms=100)
    state.cabin_evidence.cabin_phone_observed = True
    state.cabin_evidence.cabin_phone_observed_regions = ["PASSENGER"]
    state.cabin_evidence.ignored_phone_count = 1
    state.cabin_evidence.ignored_phone_reasons = ["PASSENGER_PHONE_OBSERVED_ONLY"]

    recorder.write_frame(state, {}, frame)
    recorder.close()

    events = json.loads(event_path.read_text(encoding="utf-8"))
    assert [event.get("cabin_event_type") for event in events if event.get("cabin_event_type")] == []


def _onnx_detector(**overrides):
    config = {
        "enabled": True,
        "detector_backend": "onnx",
        "model_path": "",
        "min_confidence": 0.35,
        "nms_iou_threshold": 0.45,
        "normalize_bboxes": True,
    }
    config.update(overrides)
    return CabinObjectDetector(DMSConfig(cabin_evidence=config))


def test_onnx_missing_model_path_returns_empty_safely():
    detector = _onnx_detector(model_path="")

    evidence = detector.detect(np.zeros((32, 32, 3), dtype=np.uint8), 0)

    assert evidence == []
    assert detector.backend_status == "MODEL_MISSING"


def test_onnx_nonexistent_model_path_returns_empty_safely(tmp_path):
    detector = _onnx_detector(model_path=str(tmp_path / "missing.onnx"))

    evidence = detector.detect(np.zeros((32, 32, 3), dtype=np.uint8), 0)

    assert evidence == []
    assert detector.backend_status == "MODEL_MISSING"


def test_cabin_class_map_loads_json_and_aliases(tmp_path):
    path = tmp_path / "class_map.json"
    path.write_text(
        '{"classes":{"0":"cell phone","1":"seat belt"},"aliases":{"cell phone":"PHONE","seat belt":"SEATBELT"}}',
        encoding="utf-8",
    )

    class_map = CabinClassMap(str(path))

    assert class_map.status == "CLASS_MAP_READY"
    assert class_map.object_type_for(0) == CabinEvidenceObjectType.PHONE
    assert class_map.object_type_for(1) == CabinEvidenceObjectType.SEATBELT
    assert class_map.canonical_name("mobile") == "PHONE"


def test_onnx_parser_handles_n6_output():
    detector = _onnx_detector()
    output = np.array([[0.10, 0.20, 0.30, 0.40, 0.90, 0]], dtype=np.float32)

    evidence = detector.parse_outputs(output, (100, 100, 3), 100, {"driver_roi_norm": [0.0, 0.0, 0.5, 1.0]})

    assert len(evidence) == 1
    assert evidence[0].object_type == CabinEvidenceObjectType.PHONE
    assert evidence[0].source == "onnx"


def test_onnx_parser_handles_1_n_6_output():
    detector = _onnx_detector()
    output = np.array([[[0.10, 0.20, 0.30, 0.40, 0.90, 1]]], dtype=np.float32)

    evidence = detector.parse_outputs(output, (100, 100, 3), 100, {"driver_roi_norm": [0.0, 0.0, 0.5, 1.0]})

    assert len(evidence) == 1
    assert evidence[0].object_type == CabinEvidenceObjectType.SEATBELT


def test_onnx_parser_handles_n_5_plus_c_output():
    detector = _onnx_detector()
    output = np.array([[0.10, 0.20, 0.30, 0.40, 0.90, 0.10, 0.80, 0.05]], dtype=np.float32)

    evidence = detector.parse_outputs(output, (100, 100, 3), 100, {"driver_roi_norm": [0.0, 0.0, 0.5, 1.0]})

    assert len(evidence) == 1
    assert evidence[0].object_type == CabinEvidenceObjectType.SEATBELT


def test_onnx_parser_unknown_shape_is_safe():
    detector = _onnx_detector()

    evidence = detector.parse_outputs(np.zeros((2, 2, 2, 2), dtype=np.float32), (100, 100, 3))

    assert evidence == []
    assert detector.backend_status == "UNSUPPORTED_OUTPUT_SHAPE"


def test_onnx_bboxes_are_clamped_and_low_confidence_filtered():
    detector = _onnx_detector(min_confidence=0.50)
    output = np.array([
        [-0.10, 0.20, 1.20, 0.40, 0.90, 0],
        [0.20, 0.20, 0.30, 0.30, 0.20, 0],
    ], dtype=np.float32)

    evidence = detector.parse_outputs(output, (100, 100, 3), 100, {"driver_roi_norm": [0.0, 0.0, 1.0, 1.0]})

    assert len(evidence) == 1
    assert evidence[0].bbox == [0.0, 0.20000000298023224, 1.0, 0.4000000059604645]


def test_onnx_nms_removes_duplicate_detections():
    detector = _onnx_detector(nms_iou_threshold=0.30)
    output = np.array([
        [0.10, 0.20, 0.40, 0.60, 0.90, 0],
        [0.12, 0.22, 0.42, 0.62, 0.80, 0],
    ], dtype=np.float32)

    evidence = detector.parse_outputs(output, (100, 100, 3), 100, {"driver_roi_norm": [0.0, 0.0, 1.0, 1.0]})

    assert len(evidence) == 1


def test_onnx_roi_association_maps_driver_region():
    detector = _onnx_detector()
    output = np.array([[0.10, 0.20, 0.30, 0.40, 0.90, 0]], dtype=np.float32)

    evidence = detector.parse_outputs(output, (100, 100, 3), 100, {"driver_roi_norm": [0.0, 0.0, 0.5, 1.0]})

    assert evidence[0].region == CabinEvidenceRegion.DRIVER


def test_onnx_relation_infers_phone_near_ear_and_lap():
    detector = _onnx_detector()
    context = {"driver_roi_norm": [0.0, 0.0, 0.5, 1.0], "driver_face": Namespace(bbox=[0.20, 0.10, 0.40, 0.35])}
    output = np.array([
        [0.10, 0.10, 0.30, 0.25, 0.90, 0],
        [0.10, 0.75, 0.30, 0.90, 0.80, 0],
    ], dtype=np.float32)

    evidence = detector.parse_outputs(output, (100, 100, 3), 100, context)
    relations = {obj.relation_to_driver for obj in evidence}

    assert CabinEvidenceRelation.NEAR_EAR in relations
    assert CabinEvidenceRelation.NEAR_LAP in relations


def test_onnx_relation_infers_seatbelt_across_torso():
    detector = _onnx_detector()
    output = np.array([[0.10, 0.25, 0.35, 0.90, 0.90, 1]], dtype=np.float32)

    evidence = detector.parse_outputs(output, (100, 100, 3), 100, {"driver_roi_norm": [0.0, 0.0, 0.5, 1.0]})

    assert evidence[0].relation_to_driver == CabinEvidenceRelation.ACROSS_TORSO


def test_onnx_evidence_flows_through_temporal_fusion():
    detector = _onnx_detector()
    fusion = CabinEvidenceFusion(DMSConfig(cabin_evidence={"temporal_confirm_ms": 1200, "temporal_clear_ms": 5000, "phone_confirm_ms": 2500, "driver_phone_min_stable_frames": 1, "driver_phone_clear_ms": 5000}))
    output = np.array([[0.10, 0.50, 0.30, 0.65, 0.90, 0]], dtype=np.float32)
    context = {"driver_roi_norm": [0.0, 0.0, 0.5, 1.0], "driver_face": Namespace(bbox=[0.20, 0.10, 0.40, 0.35])}

    first = detector.parse_outputs(output, (100, 100, 3), 1000, context)
    second = detector.parse_outputs(output, (100, 100, 3), 2300, context)
    fusion.update(first, 1000, backend_status=detector.backend_status)
    state = fusion.update(second, 2300, backend_status=detector.backend_status)

    assert state.phone_state == CabinPhoneState.PHONE_DISTRACTION
    assert state.affect_final_dms_state is False


def test_status_lines_include_cabin_backend_status():
    labels = [label for label, _ in status_dashboard_lines(DMSState(), fps=30.0)]

    assert "Cabin status" in labels


def test_onnx_overlay_label_uses_det_prefix():
    label = _cabin_evidence_label("PHONE", "CANDIDATE", "onnx", "NEAR_HAND")

    assert label == "PHONE IN DRIVER ROI / PENDING"


def test_face_adjacent_low_confidence_phone_infers_near_ear_and_is_pending():
    detector = _onnx_detector()
    output = np.array([[0.37, 0.30, 0.47, 0.45, 0.36, 0]], dtype=np.float32)
    evidence = detector.parse_outputs(
        output,
        (1080, 1920, 3),
        timestamp_ms=1000,
        context={
            "driver_roi_norm": [0.0, 0.0, 0.55, 1.0],
            "driver_face": Namespace(bbox=[0.20, 0.20, 0.40, 0.50]),
        },
    )

    assert len(evidence) == 1
    assert evidence[0].relation_to_driver == CabinEvidenceRelation.NEAR_EAR

    fusion = CabinEvidenceFusion(DMSConfig(cabin_evidence={"temporal_clear_ms": 5000}))
    state = fusion.update(evidence, 1000)

    assert state.driver_phone_state == CabinPhoneState.NO_PHONE
    assert state.driver_phone_pre_candidate is True
    assert state.ignored_phone_count == 0
    assert state.driver_phone_relation_threshold_used == 0.25


def test_chest_level_phone_does_not_infer_near_ear():
    detector = _onnx_detector()
    output = np.array([[0.28, 0.62, 0.38, 0.76, 0.86, 0]], dtype=np.float32)
    evidence = detector.parse_outputs(
        output,
        (1080, 1920, 3),
        timestamp_ms=1000,
        context={
            "driver_roi_norm": [0.0, 0.0, 0.55, 1.0],
            "driver_face": Namespace(bbox=[0.20, 0.20, 0.40, 0.50]),
        },
    )

    assert len(evidence) == 1
    assert evidence[0].relation_to_driver in {
        CabinEvidenceRelation.NEAR_HAND,
        CabinEvidenceRelation.NEAR_LAP,
        CabinEvidenceRelation.UNKNOWN,
    }
    assert evidence[0].relation_to_driver != CabinEvidenceRelation.NEAR_EAR

def test_phone_overlay_label_marks_pending_and_promoted_states():
    pending = _cabin_evidence_label("PHONE", "CANDIDATE", "onnx", "NEAR_EAR", "PENDING")
    promoted = _cabin_evidence_label("PHONE", "CANDIDATE", "onnx", "NEAR_EAR", "CANDIDATE")
    ignored = _cabin_evidence_label(
        "PHONE",
        "REJECTED",
        "onnx",
        "NEAR_EAR",
        "IGNORED",
        ["DRIVER_PHONE_LOW_CONFIDENCE"],
    )

    assert pending == "PHONE IN DRIVER ROI / PENDING"
    assert promoted == "PHONE DISTRACTION"
    assert ignored == "PHONE / IGNORED"

def test_duplicate_accepted_and_ignored_phone_shows_one_overlay_box():
    state = DMSState()
    accepted = _phone_obj(relation=CabinEvidenceRelation.NEAR_EAR, bbox=[0.20, 0.20, 0.34, 0.42], confidence=0.72)
    ignored = _phone_obj(relation=CabinEvidenceRelation.NEAR_EAR, bbox=[0.21, 0.21, 0.35, 0.43], confidence=0.38)
    ignored.state = CabinEvidenceLifecycleState.REJECTED
    state.cabin_evidence.evidence_objects = [ignored, accepted]
    state.cabin_evidence.driver_phone_state = CabinPhoneState.PHONE_OBJECT_CANDIDATE
    state.cabin_evidence.driver_phone_relation = "NEAR_EAR"

    visible, hidden = _visible_cabin_evidence_objects(state)

    assert len(visible) == 1
    assert hidden == 1
    assert visible[0].state != CabinEvidenceLifecycleState.REJECTED

def _coco_phone_class_map(tmp_path):
    path = tmp_path / "coco_phone_map.json"
    path.write_text(
        json.dumps({"version": "test", "classes": {"67": "PHONE"}, "aliases": {"cell phone": "PHONE"}}),
        encoding="utf-8",
    )
    return path


def _yolov8_output(class_id=67, confidence=0.90, cx=160.0, cy=320.0, bw=128.0, bh=128.0, columns=8400):
    output = np.zeros((1, 84, columns), dtype=np.float32)
    output[0, 0, 0] = cx
    output[0, 1, 0] = cy
    output[0, 2, 0] = bw
    output[0, 3, 0] = bh
    output[0, 4 + class_id, 0] = confidence
    return output


def test_onnx_parser_handles_yolov8_1_84_8400_phone_output(tmp_path):
    detector = _onnx_detector(class_map_path=str(_coco_phone_class_map(tmp_path)), min_confidence=0.10)
    output = _yolov8_output(confidence=0.37)

    evidence = detector.parse_outputs(output, (100, 100, 3), 100, {"driver_roi_norm": [0.0, 0.0, 0.5, 1.0]})

    assert len(evidence) == 1
    assert evidence[0].object_type == CabinEvidenceObjectType.PHONE
    assert np.allclose(evidence[0].bbox, [0.15, 0.40, 0.35, 0.60])
    assert detector.backend_status == "OK"
    assert detector.last_parser_format == "YOLOV8_CXCYWH_CLASS_SCORES"
    assert detector.last_yolo_debug["yolo_debug_max_class_id"] == 67
    assert detector.last_yolo_debug["yolo_debug_candidates_above_conf"] == 1
    assert detector.last_yolo_debug["yolo_debug_candidates_after_class_map_filter"] == 1
    assert detector.last_yolo_debug["yolo_debug_candidates_after_bbox_validation"] == 1
    assert detector.last_yolo_debug["yolo_debug_candidates_after_nms"] == 1


def test_onnx_parser_handles_yolov8_84_8400_phone_output(tmp_path):
    detector = _onnx_detector(class_map_path=str(_coco_phone_class_map(tmp_path)), min_confidence=0.10)
    output = _yolov8_output()[0]

    evidence = detector.parse_outputs(output, (100, 100, 3), 100, {"driver_roi_norm": [0.0, 0.0, 0.5, 1.0]})

    assert len(evidence) == 1
    assert evidence[0].object_type == CabinEvidenceObjectType.PHONE
    assert detector.last_parser_format == "YOLOV8_CXCYWH_CLASS_SCORES"


def test_onnx_yolov8_unmapped_high_confidence_class_is_ignored(tmp_path):
    detector = _onnx_detector(class_map_path=str(_coco_phone_class_map(tmp_path)))
    output = _yolov8_output(class_id=10, confidence=0.95)

    evidence = detector.parse_outputs(output, (100, 100, 3), 100, {"driver_roi_norm": [0.0, 0.0, 0.5, 1.0]})

    assert evidence == []
    assert detector.backend_status == "OK"


def test_onnx_yolov8_score_037_is_filtered_at_confidence_050(tmp_path):
    detector = _onnx_detector(class_map_path=str(_coco_phone_class_map(tmp_path)), min_confidence=0.50)
    output = _yolov8_output(confidence=0.37)

    evidence = detector.parse_outputs(output, (100, 100, 3), 100, {"driver_roi_norm": [0.0, 0.0, 0.5, 1.0]})

    assert evidence == []
    assert detector.backend_status == "OK"


def test_onnx_yolov8_argmax_candidate_must_be_mapped_class(tmp_path):
    detector = _onnx_detector(class_map_path=str(_coco_phone_class_map(tmp_path)), min_confidence=0.10)
    output = _yolov8_output(class_id=0, confidence=0.95)
    output[0, 0, 1] = 448.0
    output[0, 1, 1] = 320.0
    output[0, 2, 1] = 128.0
    output[0, 3, 1] = 128.0
    output[0, 4 + 67, 1] = 0.37

    evidence = detector.parse_outputs(output, (100, 100, 3), 100, {"driver_roi_norm": [0.0, 0.0, 1.0, 1.0]})

    assert len(evidence) == 1
    assert evidence[0].object_type == CabinEvidenceObjectType.PHONE
    assert np.allclose(evidence[0].bbox, [0.60, 0.40, 0.80, 0.60])
    assert detector.last_yolo_debug["yolo_debug_candidates_above_conf"] == 2
    assert detector.last_yolo_debug["yolo_debug_candidates_after_class_map_filter"] == 1


def test_coco_phone_class_map_key_67_maps_without_off_by_one(tmp_path):
    class_map = CabinClassMap(str(_coco_phone_class_map(tmp_path)))

    assert class_map.has_class_id(67) is True
    assert class_map.object_type_for(67) == CabinEvidenceObjectType.PHONE
    assert class_map.has_class_id(66) is False
    assert class_map.object_type_for(66) == CabinEvidenceObjectType.UNKNOWN_OBJECT


def test_onnx_yolov8_duplicate_phone_boxes_are_nms_filtered(tmp_path):
    detector = _onnx_detector(class_map_path=str(_coco_phone_class_map(tmp_path)), nms_iou_threshold=0.30)
    output = _yolov8_output()
    output[0, 0, 1] = 166.0
    output[0, 1, 1] = 320.0
    output[0, 2, 1] = 128.0
    output[0, 3, 1] = 128.0
    output[0, 4 + 67, 1] = 0.80

    evidence = detector.parse_outputs(output, (100, 100, 3), 100, {"driver_roi_norm": [0.0, 0.0, 0.5, 1.0]})

    assert len(evidence) == 1



def test_inspect_cabin_onnx_missing_model_writes_report(tmp_path):
    report_path = tmp_path / "inspection.json"
    args = Namespace(
        model=str(tmp_path / "missing.onnx"),
        class_map="configs/dms/cabin_object_class_map.json",
        image=None,
        video=None,
        frame_index=0,
        output_json=str(report_path),
        output_image=None,
        input_width=640,
        input_height=640,
        conf=0.35,
        nms=0.45,
        max_detections=50,
    )

    report = inspect_model(args)
    write_report(report, args.output_json)
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["backend_status"] == "MODEL_MISSING"
    assert payload["class_map_loaded"] is True
    assert payload["class_map_status"] == "CLASS_MAP_READY"
    assert payload["model_exists"] is False
    assert payload["parsed_detection_count"] == 0
    assert "MODEL_MISSING" in payload["warnings"]


def test_inspect_cabin_onnx_invalid_class_map_is_safe(tmp_path):
    bad_map = tmp_path / "bad.json"
    bad_map.write_text("{not-json", encoding="utf-8")
    args = Namespace(
        model=str(tmp_path / "missing.onnx"),
        class_map=str(bad_map),
        image=None,
        video=None,
        frame_index=0,
        output_json=None,
        output_image=None,
        input_width=640,
        input_height=640,
        conf=0.35,
        nms=0.45,
        max_detections=50,
    )

    report = inspect_model(args)

    assert report["class_map_loaded"] is False
    assert report["class_map_status"] == "CLASS_MAP_INVALID"
    assert "CLASS_MAP_INVALID" in report["warnings"]
    assert report["backend_status"] == "MODEL_MISSING"


def test_inspect_cabin_onnx_missing_class_map_is_safe(tmp_path):
    args = Namespace(
        model=str(tmp_path / "missing.onnx"),
        class_map=str(tmp_path / "missing_class_map.json"),
        image=None,
        video=None,
        frame_index=0,
        output_json=None,
        output_image=None,
        input_width=640,
        input_height=640,
        conf=0.35,
        nms=0.45,
        max_detections=50,
    )

    report = inspect_model(args)

    assert report["backend_status"] == "MODEL_MISSING"
    assert report["class_map_loaded"] is False
    assert report["class_map_status"] == "CLASS_MAP_MISSING"
    assert "CLASS_MAP_MISSING" in report["warnings"]


def test_inspect_report_schema_fields_are_stable(tmp_path):
    args = Namespace(
        model=str(tmp_path / "missing.onnx"),
        class_map="configs/dms/cabin_object_class_map.json",
        image=None,
        video=None,
        frame_index=0,
        output_json=None,
        output_image=None,
        input_width=320,
        input_height=320,
        conf=0.4,
        nms=0.5,
        max_detections=10,
    )

    report = inspect_model(args)

    for key in {
        "model_path",
        "model_exists",
        "class_map_path",
        "class_map_loaded",
        "class_map_status",
        "backend_status",
        "input_width",
        "input_height",
        "raw_output_shapes",
        "parsed_detection_count",
        "detections",
        "warnings",
        "errors",
        "parser_status",
        "parser_format",
        "yolo_debug_total_candidates",
        "yolo_debug_max_score",
        "yolo_debug_max_class_id",
        "yolo_debug_top_classes_before_filter",
        "yolo_debug_candidates_above_conf",
        "yolo_debug_candidates_after_class_map_filter",
        "yolo_debug_candidates_after_bbox_validation",
        "yolo_debug_candidates_after_nms",
        "yolo_debug_class_map_keys",
        "yolo_debug_parser_axis_used",
    }:
        assert key in report


def test_onnx_parser_records_raw_output_shapes():
    detector = _onnx_detector()
    output = np.array([[[0.10, 0.20, 0.30, 0.40, 0.90, 0]]], dtype=np.float32)

    evidence = detector.parse_outputs(output, (100, 100, 3))

    assert len(evidence) == 1
    assert detector.last_raw_output_shapes == [[1, 1, 6]]



def test_sample_dms_frame_missing_video_fails_safely(tmp_path):
    args = Namespace(
        video=str(tmp_path / "missing.mp4"),
        camera=None,
        frame_index=0,
        time_ms=None,
        output=str(tmp_path / "frame.jpg"),
    )

    ok, message, shape = sample_frame(args)

    assert ok is False
    assert message == "SOURCE_OPEN_FAILED"
    assert shape is None


def test_sample_dms_frame_extracts_tiny_generated_video(tmp_path):
    video_path = tmp_path / "tiny.mp4"
    output_path = tmp_path / "frame.jpg"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (16, 12))
    assert writer.isOpened()
    writer.write(np.full((12, 16, 3), 127, dtype=np.uint8))
    writer.release()
    args = Namespace(video=str(video_path), camera=None, frame_index=0, time_ms=None, output=str(output_path))

    ok, message, shape = sample_frame(args)

    assert ok is True
    assert message == str(output_path)
    assert shape == (12, 16, 3)
    assert output_path.exists()


def test_inspect_report_contains_expected_shapes_for_unsupported_parser(tmp_path, monkeypatch):
    model_path = tmp_path / "candidate.onnx"
    image_path = tmp_path / "frame.jpg"
    model_path.write_bytes(b"not-a-real-model")
    cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))

    class FakeDetector:
        def __init__(self, config):
            self.backend_status = "OK"
            self.last_raw_output_shapes = []

        def detect(self, frame, timestamp_ms, context=None):
            self.backend_status = "UNSUPPORTED_OUTPUT_SHAPE"
            self.last_raw_output_shapes = [[1, 3, 7, 9]]
            return []

    monkeypatch.setattr(inspect_app, "CabinObjectDetector", FakeDetector)
    args = Namespace(
        model=str(model_path),
        class_map="configs/dms/cabin_object_class_map.json",
        image=str(image_path),
        video=None,
        frame_index=0,
        output_json=None,
        output_image=None,
        input_width=640,
        input_height=640,
        conf=0.35,
        nms=0.45,
        max_detections=50,
        save_raw_shapes_only=False,
    )

    report = inspect_model(args)

    assert report["parser_status"] == "UNSUPPORTED_OUTPUT_SHAPE"
    assert report["raw_output_shapes"] == [[1, 3, 7, 9]]
    assert "[N,6] as x1,y1,x2,y2,conf,class_id" in report["expected_supported_shapes"]
    assert "[1,4+C,N] YOLOv8-style output, e.g. [1,84,8400]" in report["expected_supported_shapes"]


def test_inspect_no_detections_is_distinct_from_model_missing(tmp_path, monkeypatch):
    model_path = tmp_path / "candidate.onnx"
    image_path = tmp_path / "frame.jpg"
    model_path.write_bytes(b"not-a-real-model")
    cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))

    class FakeDetector:
        def __init__(self, config):
            self.backend_status = "OK"
            self.last_raw_output_shapes = []

        def detect(self, frame, timestamp_ms, context=None):
            self.backend_status = "OK"
            self.last_raw_output_shapes = [[1, 0, 6]]
            return []

    monkeypatch.setattr(inspect_app, "CabinObjectDetector", FakeDetector)
    args = Namespace(
        model=str(model_path),
        class_map="configs/dms/cabin_object_class_map.json",
        image=str(image_path),
        video=None,
        frame_index=0,
        output_json=None,
        output_image=None,
        input_width=640,
        input_height=640,
        conf=0.35,
        nms=0.45,
        max_detections=50,
        save_raw_shapes_only=False,
    )

    report = inspect_model(args)

    assert report["backend_status"] == "OK"
    assert report["parser_status"] == "NO_DETECTIONS"
    assert "MODEL_MISSING" not in report["warnings"]


def test_console_summary_includes_missing_model_status(tmp_path):
    args = Namespace(
        model=str(tmp_path / "missing.onnx"),
        class_map="configs/dms/cabin_object_class_map.json",
        image=None,
        video=None,
        frame_index=0,
        output_json=None,
        output_image=None,
        input_width=640,
        input_height=640,
        conf=0.35,
        nms=0.45,
        max_detections=50,
        save_raw_shapes_only=False,
    )

    summary = console_summary(inspect_model(args))

    assert "Cabin ONNX Inspection" in summary
    assert "Model exists: NO" in summary
    assert "Backend status: MODEL_MISSING" in summary
    assert "Parser format:" in summary





def test_passenger_phone_clears_stale_driver_phone_immediately():
    fusion = _driver_phone_fusion(driver_phone_clear_immediately_on_ignored_only=True)
    driver_state = fusion.update([_phone_obj()], 1000)
    assert driver_state.driver_phone_state == CabinPhoneState.PHONE_OBJECT_CANDIDATE

    passenger_phone = _phone_obj(
        region=CabinEvidenceRegion.PASSENGER,
        relation=CabinEvidenceRelation.UNKNOWN,
        bbox=[0.65, 0.45, 0.75, 0.65],
    )
    state = fusion.update([passenger_phone], 1033)

    assert state.cabin_phone_observed is True
    assert state.driver_phone_state == CabinPhoneState.NO_PHONE
    assert state.driver_phone_pre_candidate is False
    assert "PASSENGER_PHONE_OBSERVED_ONLY" in state.ignored_phone_reasons
    assert "DRIVER_PHONE_STALE_CLEARED_IGNORED_ONLY" in state.ignored_phone_reasons


def test_unknown_region_phone_clears_stale_driver_phone_when_not_allowed():
    fusion = _driver_phone_fusion(allow_unknown_region_phone=False)
    fusion.update([_phone_obj()], 1000)

    unknown_phone = _phone_obj(region=CabinEvidenceRegion.UNKNOWN, relation=CabinEvidenceRelation.NEAR_HAND)
    state = fusion.update([unknown_phone], 1033)

    assert state.cabin_phone_observed is True
    assert state.driver_phone_state == CabinPhoneState.NO_PHONE
    assert "UNKNOWN_REGION_PHONE_IGNORED" in state.ignored_phone_reasons


def test_outside_driver_roi_phone_clears_stale_driver_phone():
    fusion = _driver_phone_fusion()
    fusion.update([_phone_obj()], 1000)

    outside_phone = _phone_obj(
        region=CabinEvidenceRegion.DRIVER,
        relation=CabinEvidenceRelation.NEAR_HAND,
        bbox=[0.92, 0.55, 0.98, 0.68],
    )
    state = fusion.update([outside_phone], 1033)

    assert state.cabin_phone_observed is True
    assert state.driver_phone_state == CabinPhoneState.NO_PHONE
    assert "PHONE_OUTSIDE_DRIVER_INTERACTION_ROI" in state.ignored_phone_reasons


def test_weak_driver_phone_below_driver_confidence_does_not_become_candidate():
    fusion = CabinEvidenceFusion(DMSConfig(cabin_evidence={"driver_phone_min_confidence": 0.45}))

    state = fusion.update([_phone_obj(confidence=0.40)], 1000)

    assert state.cabin_phone_observed is True
    assert state.driver_phone_state == CabinPhoneState.NO_PHONE
    assert "DRIVER_PHONE_LOW_CONFIDENCE" in state.ignored_phone_reasons


def test_stable_driver_near_lap_phone_becomes_candidate_after_gate():
    fusion = CabinEvidenceFusion(DMSConfig(cabin_evidence={"temporal_clear_ms": 5000}))
    state = None
    for index in range(8):
        state = fusion.update([_phone_obj(relation=CabinEvidenceRelation.NEAR_LAP)], 1000 + index * 50)

    assert state is not None
    assert state.driver_phone_state == CabinPhoneState.PHONE_OBJECT_CANDIDATE
    assert state.driver_phone_relation == "NEAR_LAP"
    assert state.driver_phone_track_age_ms >= 300
    assert state.driver_phone_consecutive_frames >= 8


def test_stable_driver_near_ear_phone_becomes_candidate_after_gate():
    fusion = CabinEvidenceFusion(DMSConfig(cabin_evidence={"temporal_clear_ms": 5000}))
    phone = _phone_obj(relation=CabinEvidenceRelation.NEAR_EAR, bbox=[0.20, 0.10, 0.30, 0.24])
    state = None
    for index in range(8):
        state = fusion.update([phone], 1000 + index * 50)

    assert state is not None
    assert state.driver_phone_state == CabinPhoneState.PHONE_OBJECT_CANDIDATE
    assert state.driver_phone_relation == "NEAR_EAR"


def test_driver_phone_with_unstable_jumping_bbox_does_not_become_candidate():
    fusion = CabinEvidenceFusion(
        DMSConfig(
            cabin_evidence={
                "temporal_clear_ms": 5000,
                "driver_phone_max_center_jump": 0.05,
                "driver_phone_min_iou_for_same_track": 0.60,
            }
        )
    )
    boxes = ([0.10, 0.55, 0.20, 0.70], [0.50, 0.55, 0.60, 0.70])
    state = None
    for index in range(10):
        state = fusion.update([_phone_obj(bbox=list(boxes[index % 2]))], 1000 + index * 50)

    assert state is not None
    assert state.driver_phone_state == CabinPhoneState.NO_PHONE
    assert "PHONE_UNSTABLE_TRACK" in state.ignored_phone_reasons


def test_large_torso_sized_phone_like_bbox_requires_low_plausibility_extra_stability():
    fusion = _driver_phone_fusion()
    state = fusion.update([_phone_obj(bbox=[0.20, 0.45, 0.55, 0.90], confidence=0.90)], 1000)

    assert state.cabin_phone_observed is True
    assert state.driver_phone_state == CabinPhoneState.NO_PHONE
    assert state.driver_phone_pre_candidate is True
    assert state.driver_phone_visual_plausibility_reason == "LOW_PLAUSIBILITY_LARGE_BBOX"


def test_very_small_phone_like_bbox_is_ignored():
    fusion = _driver_phone_fusion()
    state = fusion.update([_phone_obj(bbox=[0.20, 0.45, 0.22, 0.47], confidence=0.90)], 1000)

    assert state.cabin_phone_observed is True
    assert state.driver_phone_state == CabinPhoneState.NO_PHONE
    assert "PHONE_BBOX_TOO_SMALL" in state.ignored_phone_reasons


def test_implausible_phone_aspect_ratio_is_ignored():
    fusion = _driver_phone_fusion()
    state = fusion.update([_phone_obj(bbox=[0.15, 0.55, 0.55, 0.60], confidence=0.90)], 1000)

    assert state.cabin_phone_observed is True
    assert state.driver_phone_state == CabinPhoneState.NO_PHONE
    assert "PHONE_BBOX_ASPECT_IMPLAUSIBLE" in state.ignored_phone_reasons


def test_large_square_low_confidence_phone_like_bbox_requires_low_plausibility_extra_stability():
    fusion = _driver_phone_fusion()
    state = fusion.update([_phone_obj(bbox=[0.25, 0.50, 0.48, 0.73], confidence=0.60)], 1000)

    assert state.cabin_phone_observed is True
    assert state.driver_phone_state == CabinPhoneState.NO_PHONE
    assert state.driver_phone_pre_candidate is True
    assert state.driver_phone_visual_plausibility_reason == "LOW_PLAUSIBILITY_LARGE_SQUARE"


def test_one_frame_phone_flicker_does_not_emit_driver_phone_event(tmp_path):
    fusion = CabinEvidenceFusion(DMSConfig(cabin_evidence={"temporal_clear_ms": 5000}))
    event_path = tmp_path / "events.json"
    recorder = DebugTraceRecorder(event_json_path=str(event_path))
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    recorder.write_frame(DMSState(frame_id=0, timestamp_ms=900), {}, frame)
    dms_state = DMSState(frame_id=1, timestamp_ms=1000)
    dms_state.cabin_evidence = fusion.update([_phone_obj()], 1000)
    recorder.write_frame(dms_state, {}, frame)
    dms_state = DMSState(frame_id=2, timestamp_ms=1033)
    dms_state.cabin_evidence = fusion.update([], 1033)
    recorder.write_frame(dms_state, {}, frame)
    recorder.close()

    events = json.loads(event_path.read_text(encoding="utf-8"))
    assert [event.get("cabin_event_type") for event in events if event.get("cabin_event_type")] == []


def test_near_ear_track_survives_short_raw_gap_with_stable_geometry():
    fusion = CabinEvidenceFusion(
        DMSConfig(
            cabin_evidence={
                "driver_phone_min_stable_frames": 2,
                "driver_phone_min_duration_ms": 0,
                "driver_phone_clear_ms": 200,
                "driver_phone_track_hold_ms": 600,
                "driver_phone_max_raw_gap_ms": 600,
                "driver_phone_max_center_jump": 0.18,
                "driver_phone_min_iou_for_same_track": 0.10,
            }
        )
    )
    first = _phone_obj(relation=CabinEvidenceRelation.NEAR_EAR, bbox=[0.20, 0.10, 0.30, 0.24], confidence=0.38)
    second = _phone_obj(relation=CabinEvidenceRelation.NEAR_EAR, bbox=[0.205, 0.105, 0.305, 0.245], confidence=0.39)

    fusion.update([first], 1000)
    state = fusion.update([second], 1233)

    assert state.driver_phone_track_reset_reason == ""
    assert state.driver_phone_track_age_ms >= 233
    assert state.driver_phone_consecutive_frames >= 2


def test_near_ear_track_promotes_to_suspected_despite_short_raw_gap():
    fusion = CabinEvidenceFusion(
        DMSConfig(
            cabin_evidence={
                "phone_to_ear_confirm_ms": 1200,
                "phone_confirm_ms": 2500,
                "driver_phone_min_stable_frames": 2,
                "driver_phone_min_duration_ms": 0,
                "driver_phone_clear_ms": 200,
                "driver_phone_track_hold_ms": 600,
                "driver_phone_max_raw_gap_ms": 600,
            }
        )
    )
    phone = _phone_obj(relation=CabinEvidenceRelation.NEAR_EAR, bbox=[0.20, 0.10, 0.30, 0.24], confidence=0.39)

    fusion.update([phone], 0)
    fusion.update([phone], 100)
    held = fusion.update([], 333)
    fusion.update([phone], 566)
    fusion.update([phone], 900)
    state = fusion.update([phone], 1233)

    assert held.driver_phone_track_held is True
    assert state.driver_phone_state == CabinPhoneState.PHONE_TO_EAR_SUSPECTED
    assert state.driver_phone_relation == "NEAR_EAR"


def test_driver_phone_candidate_does_not_clear_on_one_missed_frame():
    fusion = CabinEvidenceFusion(
        DMSConfig(
            cabin_evidence={
                "driver_phone_min_stable_frames": 1,
                "driver_phone_min_duration_ms": 0,
                "driver_phone_track_hold_ms": 600,
            }
        )
    )
    fusion.update([_phone_obj()], 1000)
    state = fusion.update([], 1033)

    assert state.driver_phone_state == CabinPhoneState.NO_PHONE
    assert state.phone_scenario == "NONE"
    assert state.driver_roi_phone is False
    assert state.phone_distraction_active is False
    assert state.driver_phone_track_held is True
    assert state.driver_phone_fresh_this_frame is False


def test_driver_phone_clears_after_track_hold_expires():
    fusion = CabinEvidenceFusion(
        DMSConfig(
            cabin_evidence={
                "driver_phone_min_stable_frames": 1,
                "driver_phone_min_duration_ms": 0,
                "driver_phone_track_hold_ms": 600,
            }
        )
    )
    fusion.update([_phone_obj()], 1000)
    state = fusion.update([], 1701)

    assert state.driver_phone_state == CabinPhoneState.NO_PHONE
    assert state.driver_phone_track_reset_reason == "PHONE_TRACK_HOLD_EXPIRED"


def test_held_phone_track_label_is_not_fresh_det_phone():
    label = _cabin_evidence_label("PHONE", "CANDIDATE", "onnx", "NEAR_EAR", "HELD")

    assert label == "PHONE TRACK / HELD"
    assert not label.startswith("DET PHONE")


def test_phone_overlay_diagnostics_fields_are_serialized():
    state = DMSState()
    state.cabin_evidence.overlay_phone_drawn_count = 1
    state.cabin_evidence.overlay_phone_drawn_labels = ["PHONE TRACK / NEAR_EAR / HELD"]
    state.cabin_evidence.overlay_phone_drawn_boxes = [[0.2, 0.1, 0.3, 0.24]]
    state.cabin_evidence.overlay_phone_track_label = "PHONE TRACK / NEAR_EAR / HELD"
    state.cabin_evidence.overlay_phone_track_is_held = True

    payload = state.to_dict()["cabin_evidence"]

    assert payload["overlay_phone_drawn_count"] == 1
    assert payload["overlay_phone_drawn_labels"] == ["PHONE TRACK / NEAR_EAR / HELD"]
    assert payload["overlay_phone_track_is_held"] is True


def test_held_track_with_no_fresh_driver_roi_phone_has_no_semantic_state():
    fusion = CabinEvidenceFusion(
        DMSConfig(
            cabin_evidence={
                "phone_pending_min_frames": 1,
                "phone_distraction_min_frames": 1,
                "phone_distraction_min_duration_ms": 0,
                "phone_track_hold_ms": 700,
            }
        )
    )
    fresh = fusion.update([_phone_obj()], 1000)
    held = fusion.update([], 1100)

    assert fresh.phone_scenario == "PHONE_DISTRACTION"
    assert held.driver_phone_track_held is True
    assert held.driver_roi_phone is False
    assert held.phone_inside_driver_roi is False
    assert held.phone_scenario == "NONE"
    assert held.phone_distraction_active is False
    assert held.phone_to_ear_active is False


def test_status_lines_keep_head_angle_visible_near_top():
    labels = [label for label, _ in status_dashboard_lines(DMSState(), fps=30.0)]

    assert "Status page" in labels
    assert "Head angle" in labels
    assert "Head raw/rel" in labels
    assert labels.index("Head angle") < labels.index("Cabin backend")


def test_overlay_phone_label_uses_simplified_public_text():
    assert _cabin_evidence_label("PHONE", "CANDIDATE", "onnx", "NEAR_LAP", "PENDING") == "PHONE IN DRIVER ROI / PENDING"
    assert _cabin_evidence_label("PHONE", "CANDIDATE", "onnx", "NEAR_HAND", "SUSPECTED") == "PHONE DISTRACTION"
    assert _cabin_evidence_label("PHONE", "CANDIDATE", "onnx", "NEAR_EAR", "SUSPECTED") == "PHONE TO EAR / SUSPECTED"
    assert _cabin_evidence_label("PHONE", "REJECTED", "onnx", "UNKNOWN", "IGNORED", ["PHONE_OUTSIDE_DRIVER_INTERACTION_ROI"]) == "PHONE OUTSIDE DRIVER ROI / IGNORED"


def test_phone_state_serializes_simplified_debug_fields():
    state = DMSState()
    state.cabin_evidence.driver_roi_phone = True
    state.cabin_evidence.phone_scenario = "PHONE_DISTRACTION"
    state.cabin_evidence.phone_overlay_label = "PHONE DISTRACTION"
    state.cabin_evidence.phone_overlay_drawn = True
    state.cabin_evidence.status_page_index = 1

    payload = state.to_dict()["cabin_evidence"]

    assert payload["driver_roi_phone"] is True
    assert payload["phone_scenario"] == "PHONE_DISTRACTION"
    assert payload["phone_overlay_drawn"] is True
    assert payload["status_page_index"] == 1

