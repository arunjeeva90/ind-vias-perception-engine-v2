from __future__ import annotations

from ind_vias_perception.common.types import BBox2D, Detection, ObjectClass
from ind_vias_perception.safety.safety_gate.gate import SafetyGate
from ind_vias_perception.safety.sentinel_fsm.fsm import SentinelState


def test_safety_gate_uses_bumper_distance_when_available():
    det = Detection(BBox2D(0, 0, 10, 10), ObjectClass.CAR, 0.9, distance_m=20.0)
    det.track_id = 1
    det.metadata["distance_bumper_m"] = 18.55

    payload = SafetyGate().evaluate([det], SentinelState.NOMINAL)

    assert payload["target_distance_m"] == 18.55
