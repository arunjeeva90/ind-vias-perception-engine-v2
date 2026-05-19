from __future__ import annotations

from dataclasses import dataclass
from ind_vias_perception.common.types import Detection


@dataclass
class TrackState:
    track_id: int
    last_distance_m: float | None
    last_timestamp_s: float


class SimpleDistanceTracker:
    def __init__(self):
        self._next_id = 1
        self._track: TrackState | None = None

    def update(self, detections: list[Detection], timestamp_s: float) -> list[Detection]:
        if not detections:
            return []
        # MVP: assign nearest object as ego target. Replace with DIoU/Hungarian/IMM later.
        det = min(detections, key=lambda d: d.distance_m if d.distance_m is not None else 1e9)
        if self._track is None:
            self._track = TrackState(self._next_id, det.distance_m, timestamp_s)
            self._next_id += 1
        else:
            dt = max(1e-3, timestamp_s - self._track.last_timestamp_s)
            if det.distance_m is not None and self._track.last_distance_m is not None:
                rel_v = (det.distance_m - self._track.last_distance_m) / dt
                det.metadata["relative_velocity_mps"] = rel_v
            self._track.last_distance_m = det.distance_m
            self._track.last_timestamp_s = timestamp_s
        det.track_id = self._track.track_id
        return detections
