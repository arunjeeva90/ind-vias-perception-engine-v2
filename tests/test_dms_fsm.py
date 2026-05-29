from __future__ import annotations

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import DistractionLevel, DistractionType, DrowsinessLevel, GazeZone
from ind_vias_dms.temporal.distraction_fsm import DistractionFSM
from ind_vias_dms.temporal.drowsiness_fsm import DrowsinessFSM


def test_drowsiness_fsm_enters_microsleep_after_threshold():
    config = DMSConfig(microsleep_duration_ms=1500)
    fsm = DrowsinessFSM(config)

    level = fsm.update(
        perclos_short=0.0,
        perclos_long=0.0,
        eye_closure_duration_ms=1600,
        face_present=True,
    )

    assert level == DrowsinessLevel.MICROSLEEP


def test_distraction_fsm_enters_high_after_eyes_off_road_threshold():
    config = DMSConfig(eyes_off_road_warning_ms=2000)
    fsm = DistractionFSM(config)

    level, kind = fsm.update(GazeZone.DOWN, eyes_off_road_duration_ms=2200)

    assert level == DistractionLevel.HIGH
    assert kind == DistractionType.PHONE_SUSPECTED


def test_distraction_fsm_no_face_handling_does_not_crash():
    config = DMSConfig(no_face_timeout_ms=1000)
    fsm = DistractionFSM(config)

    level, kind = fsm.update(
        GazeZone.UNKNOWN,
        eyes_off_road_duration_ms=0,
        no_face_duration_ms=1200,
    )

    assert level == DistractionLevel.UNKNOWN
    assert kind == DistractionType.UNKNOWN
