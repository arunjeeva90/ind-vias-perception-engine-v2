from __future__ import annotations

import json

from ind_vias_dms.core.types import (
    AvailabilityState,
    DMSState,
    DistractionLevel,
    DrowsinessLevel,
    GazeZone,
)
from ind_vias_dms.interface.dms_packet import dumps_dms_state


def test_dms_state_serializes_to_dict_and_json():
    state = DMSState(timestamp_ms=123, frame_id=7)
    payload = state.to_dict()

    assert payload["timestamp_ms"] == 123
    assert payload["frame_id"] == 7
    assert payload["gaze"]["zone"] == "UNKNOWN"
    assert json.loads(dumps_dms_state(state))["driver_presence"]["state"] == "UNKNOWN"


def test_dms_enum_values_are_stable():
    assert GazeZone.ROAD.value == "ROAD"
    assert DrowsinessLevel.MICROSLEEP.value == "MICROSLEEP"
    assert DistractionLevel.HIGH.value == "HIGH"
    assert AvailabilityState.UNAVAILABLE.value == "UNAVAILABLE"
