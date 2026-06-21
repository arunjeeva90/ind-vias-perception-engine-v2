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
from ind_vias_dms.utils.learning_memory import LearningMemoryWriter
from ind_vias_dms.vision.cabin_object_detection import CabinClassMap, CabinObjectDetector, SyntheticCabinTimeline
from ind_vias_dms.visualization.overlay import _cabin_evidence_label, banner_decision, status_dashboard_lines


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
                    "temporal_clear_ms": 5000,
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

    assert labels.index("Cabin backend") < labels.index("HMI banner")
    assert labels.index("Cabin objects") < labels.index("HMI banner")
    assert labels.index("Cabin phone") < labels.index("HMI banner")
    assert labels.index("Cabin phone rel") < labels.index("HMI banner")
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
        DMSConfig(cabin_evidence={"temporal_confirm_ms": 1200, "temporal_clear_ms": 5000, "phone_confirm_ms": 2500})
    )
    phone = _phone()
    phone.relation_to_driver = CabinEvidenceRelation.NEAR_HAND

    fusion.update([phone], 1000)
    state = fusion.update([phone], 2300)

    assert state.phone_state == CabinPhoneState.PHONE_IN_HAND_SUSPECTED
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

    assert cabin_events == ["CABIN_PHONE_CANDIDATE_STARTED", "CABIN_PHONE_CLEARED"]


def test_synthetic_overlay_label_includes_syn_prefix():
    label = _cabin_evidence_label("PHONE", "CANDIDATE", "synthetic", "NEAR_HAND")

    assert label.startswith("SYNTH ")
    assert "NEAR_HAND" in label


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

    assert [record["event_type"] for record in records] == ["CABIN_PHONE_CANDIDATE"]


def test_learning_memory_records_cabin_phone_transitions(tmp_path):
    path = tmp_path / "learning.jsonl"
    writer = LearningMemoryWriter(str(path), DMSConfig())
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    sequence = [
        CabinPhoneState.PHONE_OBJECT_CANDIDATE,
        CabinPhoneState.PHONE_IN_HAND_SUSPECTED,
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
        "CABIN_PHONE_CANDIDATE",
        "CABIN_PHONE_IN_HAND_SUSPECTED",
        "CABIN_PHONE_CONFIRMED",
        "CABIN_PHONE_CLEARED",
    ]
    assert records[1]["cabin_evidence"]["phone_relation"] == "NEAR_HAND"


def test_phone_relation_is_retained_when_phone_confirmed():
    fusion = CabinEvidenceFusion(
        DMSConfig(cabin_evidence={"temporal_clear_ms": 5000, "phone_confirm_ms": 1500})
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
    context = {"driver_roi_norm": [0.0, 0.0, 0.5, 1.0]}
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
    fusion = CabinEvidenceFusion(DMSConfig(cabin_evidence={"temporal_confirm_ms": 1200, "temporal_clear_ms": 5000, "phone_confirm_ms": 2500}))
    output = np.array([[0.10, 0.50, 0.30, 0.65, 0.90, 0]], dtype=np.float32)
    context = {"driver_roi_norm": [0.0, 0.0, 0.5, 1.0]}

    first = detector.parse_outputs(output, (100, 100, 3), 1000, context)
    second = detector.parse_outputs(output, (100, 100, 3), 2300, context)
    fusion.update(first, 1000, backend_status=detector.backend_status)
    state = fusion.update(second, 2300, backend_status=detector.backend_status)

    assert state.phone_state == CabinPhoneState.PHONE_IN_HAND_SUSPECTED
    assert state.affect_final_dms_state is False


def test_status_lines_include_cabin_backend_status():
    labels = [label for label, _ in status_dashboard_lines(DMSState(), fps=30.0)]

    assert "Cabin status" in labels


def test_onnx_overlay_label_uses_det_prefix():
    label = _cabin_evidence_label("PHONE", "CANDIDATE", "onnx", "NEAR_HAND")

    assert label.startswith("DET ")
    assert "NEAR_HAND" in label
