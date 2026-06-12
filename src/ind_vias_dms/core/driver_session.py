from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.occupant_manager import CabinOccupantManager, TrackedFace


class DriverSessionState(str, Enum):
    ACTIVE = "ACTIVE"
    LOST_TEMP = "LOST_TEMP"
    LOST_LONG = "LOST_LONG"
    SWAPPED = "SWAPPED"
    UNKNOWN = "UNKNOWN"


@dataclass
class DriverSessionUpdate:
    driver_session_id: str | None = None
    driver_track_id: int | None = None
    session_state: DriverSessionState = DriverSessionState.UNKNOWN
    reassociated: bool = False
    time_since_seen_ms: int = 0
    reason_codes: list[str] | None = None


class DriverSessionManager:
    def __init__(self, config: DMSConfig, occupant_manager: CabinOccupantManager) -> None:
        self.config = config
        self.occupant_manager = occupant_manager
        self.current_driver_session_id: str | None = None
        self.current_driver_track_id: int | None = None
        self.last_driver_box_norm: tuple[float, float, float, float] | None = None
        self.last_driver_center_norm: tuple[float, float] | None = None
        self.last_driver_area_norm: float | None = None
        self.last_seen_timestamp_ms: int | None = None
        self.lost_since_timestamp_ms: int | None = None
        self.session_state = DriverSessionState.UNKNOWN
        self._next_session_index = 1
        self.reassociated_until_ms: int | None = None

    def update(
        self,
        driver: TrackedFace | None,
        timestamp_ms: int,
    ) -> DriverSessionUpdate:
        reasons: list[str] = []
        if driver is None:
            return self._mark_lost(timestamp_ms)

        box = driver.observation.box_norm
        reassociated = False
        swapped = False
        if self.current_driver_session_id is None:
            self._start_new_session(driver)
            reasons.append("DRIVER_SESSION_RESET")
        elif self._time_since_seen(timestamp_ms) > self.config.driver_swap_absence_ms:
            self._start_new_session(driver)
            swapped = True
            reasons.append("POSSIBLE_DRIVER_SWAP")
        elif self.current_driver_track_id != driver.track_id:
            if self._can_reassociate(driver, timestamp_ms):
                reassociated = True
                reasons.append("DRIVER_REASSOCIATED")
                self.current_driver_track_id = driver.track_id
            elif self._time_since_seen(timestamp_ms) > self.config.driver_swap_absence_ms:
                self._start_new_session(driver)
                swapped = True
                reasons.append("POSSIBLE_DRIVER_SWAP")
            else:
                reassociated = True
                reasons.extend(["DRIVER_REASSOCIATED", "DRIVER_SESSION_HELD"])
                self.current_driver_track_id = driver.track_id
        else:
            self.current_driver_track_id = driver.track_id

        self._remember_driver_geometry(box, timestamp_ms)
        self.session_state = DriverSessionState.SWAPPED if swapped else DriverSessionState.ACTIVE
        if reassociated:
            self.reassociated_until_ms = timestamp_ms + self.config.driver_reassociated_display_ms
        display_reassociated = reassociated or (
            self.reassociated_until_ms is not None and timestamp_ms <= self.reassociated_until_ms
        )
        self.lost_since_timestamp_ms = None
        return DriverSessionUpdate(
            driver_session_id=self.current_driver_session_id,
            driver_track_id=self.current_driver_track_id,
            session_state=self.session_state,
            reassociated=display_reassociated,
            time_since_seen_ms=0,
            reason_codes=reasons,
        )

    def reset(self) -> None:
        self.current_driver_session_id = None
        self.current_driver_track_id = None
        self.last_driver_box_norm = None
        self.last_driver_center_norm = None
        self.last_driver_area_norm = None
        self.last_seen_timestamp_ms = None
        self.lost_since_timestamp_ms = None
        self.session_state = DriverSessionState.UNKNOWN
        self.reassociated_until_ms = None

    def _mark_lost(self, timestamp_ms: int) -> DriverSessionUpdate:
        if self.lost_since_timestamp_ms is None:
            self.lost_since_timestamp_ms = timestamp_ms
        time_since_seen = self._time_since_seen(timestamp_ms)
        if self.current_driver_session_id is None:
            self.session_state = DriverSessionState.UNKNOWN
            reasons = ["DRIVER_FACE_NOT_VISIBLE"]
        elif time_since_seen <= self.config.driver_session_hold_ms:
            self.session_state = DriverSessionState.LOST_TEMP
            reasons = ["DRIVER_FACE_LOST_TEMP", "DRIVER_SESSION_HELD"]
        else:
            self.session_state = DriverSessionState.LOST_LONG
            reasons = ["DRIVER_FACE_NOT_VISIBLE"]
        return DriverSessionUpdate(
            driver_session_id=self.current_driver_session_id,
            driver_track_id=None,
            session_state=self.session_state,
            reassociated=False,
            time_since_seen_ms=time_since_seen,
            reason_codes=reasons,
        )

    def _start_new_session(self, driver: TrackedFace) -> None:
        self.current_driver_session_id = f"D{self._next_session_index}"
        self._next_session_index += 1
        self.current_driver_track_id = driver.track_id

    def _can_reassociate(self, driver: TrackedFace, timestamp_ms: int) -> bool:
        box = driver.observation.box_norm
        if (
            box is None
            or self.last_driver_center_norm is None
            or self.last_driver_area_norm is None
            or self._time_since_seen(timestamp_ms) > self.config.driver_swap_absence_ms
        ):
            return False
        center = _center(box)
        area = _area(box)
        area_ratio = area / max(1e-6, self.last_driver_area_norm)
        roi_overlap = CabinOccupantManager._overlap(box, self.occupant_manager._roi("driver_roi_norm"))
        return (
            _distance(center, self.last_driver_center_norm)
            <= self.config.driver_reid_center_distance_threshold
            and self.config.driver_reid_area_ratio_min
            <= area_ratio
            <= self.config.driver_reid_area_ratio_max
            and roi_overlap >= self.config.driver_reid_roi_overlap_min
        )

    def _remember_driver_geometry(
        self,
        box: tuple[float, float, float, float] | None,
        timestamp_ms: int,
    ) -> None:
        if box is None:
            return
        self.last_driver_box_norm = box
        self.last_driver_center_norm = _center(box)
        self.last_driver_area_norm = _area(box)
        self.last_seen_timestamp_ms = timestamp_ms

    def _time_since_seen(self, timestamp_ms: int) -> int:
        if self.last_seen_timestamp_ms is None:
            return 0
        return max(0, timestamp_ms - self.last_seen_timestamp_ms)


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
