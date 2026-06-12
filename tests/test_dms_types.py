from __future__ import annotations

import json

from ind_vias_dms.core.types import (
    AttentionOutput,
    AttentionState,
    AttentionSubstate,
    AvailabilityState,
    DMSV02DecisionState,
    DMSV02Level,
    DMSState,
    DriverIdentityState,
    DriverObservability,
    DriverObservabilityState,
    DistractionLevel,
    DistractionState,
    DistractionType,
    DrowsinessLevel,
    DrowsinessState,
    GazeZone,
    OccupantFace,
    OccupancyState,
    OccupantsState,
    PhoneUseState,
)
from ind_vias_dms.interface.dms_packet import dumps_dms_state


def test_dms_state_serializes_to_dict_and_json():
    state = DMSState(
        timestamp_ms=123,
        frame_id=7,
        occupants=OccupantsState(
            count=1,
            driver_track_id=4,
            faces=[OccupantFace(4, "DRIVER", [0.5, 0.2, 0.8, 0.8], True)],
        ),
        phone_use=PhoneUseState(state="NO_PHONE", driver_state="NO_PHONE"),
        dms_v02=DMSV02DecisionState(final_level=DMSV02Level.MONITOR, final_banner="DMS MONITOR"),
        occupancy=OccupancyState(cabin_occupant_count=1, driver_present=True),
        attention=AttentionOutput(
            attention_state=AttentionState.ATTENTION_LOST,
            attention_substate=AttentionSubstate.PHONE_SUSPECTED,
            attention_confidence=0.7,
            attention_reason_codes=["POSSIBLE_PHONE_POSTURE"],
        ),
        drowsiness=DrowsinessState(
            raw_eye_state="CLOSED",
            effective_eye_state="UNKNOWN",
            perclos_valid=False,
            perclos_validity_reason_codes=["EYE_CLOSURE_SUPPRESSED_BY_DOWNWARD_GAZE"],
        ),
        distraction=DistractionState(
            type=DistractionType.PHONE_SUSPECTED,
            reason_codes=["POSSIBLE_PHONE_POSTURE"],
        ),
        driver_identity=DriverIdentityState(
            driver_session_id="D1",
            driver_track_id=4,
            session_state="ACTIVE",
        ),
        driver_observability=DriverObservability(
            state=DriverObservabilityState.PARTIALLY_OBSERVABLE,
            reason_codes=["OCCLUSION_GLASSES"],
        ),
    )
    payload = state.to_dict()

    assert payload["timestamp_ms"] == 123
    assert payload["frame_id"] == 7
    assert payload["gaze"]["zone"] == "UNKNOWN"
    assert payload["occupants"]["driver_track_id"] == 4
    assert payload["phone_use"]["driver_state"] == "NO_PHONE"
    assert payload["attention"]["attention_state"] == "ATTENTION_LOST"
    assert payload["attention"]["attention_substate"] == "PHONE_SUSPECTED"
    assert payload["attention"]["attention_reason_codes"] == ["POSSIBLE_PHONE_POSTURE"]
    assert payload["attention_state"] == "ATTENTION_LOST"
    assert payload["attention_substate"] == "PHONE_SUSPECTED"
    assert payload["dms_v02"]["final_level"] == "MONITOR"
    assert payload["occupancy"]["cabin_occupant_count"] == 1
    assert payload["drowsiness"]["raw_eye_state"] == "CLOSED"
    assert payload["drowsiness"]["effective_eye_state"] == "UNKNOWN"
    assert payload["drowsiness"]["perclos_valid"] is False
    assert payload["drowsiness"]["perclos_validity_reason_codes"] == [
        "EYE_CLOSURE_SUPPRESSED_BY_DOWNWARD_GAZE"
    ]
    assert payload["distraction"]["reason_codes"] == ["POSSIBLE_PHONE_POSTURE"]
    assert payload["driver_identity"]["driver_session_id"] == "D1"
    assert payload["driver_observability"]["state"] == "PARTIALLY_OBSERVABLE"
    assert json.loads(dumps_dms_state(state))["driver_presence"]["state"] == "UNKNOWN"


def test_dms_enum_values_are_stable():
    assert GazeZone.ROAD.value == "ROAD"
    assert DrowsinessLevel.MICROSLEEP.value == "MICROSLEEP"
    assert DistractionLevel.HIGH.value == "HIGH"
    assert AvailabilityState.UNAVAILABLE.value == "UNAVAILABLE"
    assert AttentionState.ATTENTION_LOST.value == "ATTENTION_LOST"
    assert AttentionSubstate.AMBIGUOUS.value == "AMBIGUOUS"
    assert AttentionSubstate.PHONE_DOWN_SUSPECTED.value == "PHONE_DOWN_SUSPECTED"
    assert DriverObservabilityState.UNOBSERVABLE_TEMP.value == "UNOBSERVABLE_TEMP"
