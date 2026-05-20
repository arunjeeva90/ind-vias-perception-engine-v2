from __future__ import annotations

from dataclasses import dataclass
import math

from ind_vias_perception.common.types import BBox2D, Detection


@dataclass
class TrackState:
    track_id: int
    detection: Detection
    last_timestamp_s: float
    hits: int = 1
    missing_frames: int = 0


class SimpleDistanceTracker:
    def __init__(
        self,
        max_age: int = 10,
        min_hits: int = 2,
        iou_weight: float = 0.50,
        center_weight: float = 0.25,
        class_mismatch_penalty: float = 0.25,
        distance_weight: float = 0.20,
        max_association_cost: float = 1.0,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_weight = iou_weight
        self.center_weight = center_weight
        self.class_mismatch_penalty = class_mismatch_penalty
        self.distance_weight = distance_weight
        self.max_association_cost = max_association_cost
        self._next_id = 1
        self._tracks: list[TrackState] = []

    def update(self, detections: list[Detection], timestamp_s: float) -> list[Detection]:
        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        active_track_ids: set[int] = set()
        results: list[Detection] = []

        pairs = sorted(
            (
                (self._association_cost(track, det), track_idx, det_idx)
                for track_idx, track in enumerate(self._tracks)
                for det_idx, det in enumerate(detections)
            ),
            key=lambda item: item[0],
        )
        for cost, track_idx, det_idx in pairs:
            if cost > self.max_association_cost:
                continue
            if track_idx in assigned_tracks or det_idx in assigned_detections:
                continue
            track = self._tracks[track_idx]
            det = detections[det_idx]
            self._update_track(track, det, timestamp_s)
            assigned_tracks.add(track_idx)
            assigned_detections.add(det_idx)
            active_track_ids.add(track.track_id)
            results.append(det)

        for det_idx, det in enumerate(detections):
            if det_idx in assigned_detections:
                continue
            track = self._start_track(det, timestamp_s)
            active_track_ids.add(track.track_id)
            results.append(track.detection)

        predicted: list[Detection] = []
        for track_idx, track in enumerate(self._tracks):
            if track_idx in assigned_tracks or track.track_id in active_track_ids:
                continue
            track.missing_frames += 1
            if track.missing_frames <= self.max_age:
                predicted.append(self._predicted_detection(track))

        self._tracks = [track for track in self._tracks if track.missing_frames <= self.max_age]
        results.extend(predicted)
        return results

    def _start_track(self, det: Detection, timestamp_s: float) -> TrackState:
        det.track_id = self._next_id
        det.metadata["track_predicted"] = False
        det.metadata["missing_frames"] = 0.0
        det.metadata["track_hits"] = 1.0
        track = TrackState(self._next_id, det, timestamp_s)
        self._next_id += 1
        self._tracks.append(track)
        return track

    def _update_track(self, track: TrackState, det: Detection, timestamp_s: float) -> None:
        dt = max(1e-3, timestamp_s - track.last_timestamp_s)
        last_distance = _track_distance(track.detection)
        current_distance = _track_distance(det)
        if current_distance is not None and last_distance is not None:
            det.metadata["relative_velocity_mps"] = (current_distance - last_distance) / dt
        track.detection = det
        track.last_timestamp_s = timestamp_s
        track.hits += 1
        track.missing_frames = 0
        det.track_id = track.track_id
        det.metadata["track_predicted"] = False
        det.metadata["missing_frames"] = 0.0
        det.metadata["track_hits"] = float(track.hits)

    def _predicted_detection(self, track: TrackState) -> Detection:
        det = _copy_detection(track.detection)
        det.track_id = track.track_id
        det.metadata["track_predicted"] = True
        det.metadata["missing_frames"] = float(track.missing_frames)
        det.metadata["track_hits"] = float(track.hits)
        return det

    def _association_cost(self, track: TrackState, det: Detection) -> float:
        last = track.detection
        iou_cost = 1.0 - _iou(last.bbox, det.bbox)
        center_cost = min(1.0, _center_distance(last.bbox, det.bbox) / max(_bbox_diag(last.bbox), 1.0))
        class_cost = self.class_mismatch_penalty if last.label != det.label else 0.0
        distance_cost = _distance_cost(_track_distance(last), _track_distance(det))
        relevance_bonus = 0.1 if last.metadata.get("in_ego_corridor") == det.metadata.get("in_ego_corridor") else 0.0
        return (
            self.iou_weight * iou_cost
            + self.center_weight * center_cost
            + class_cost
            + self.distance_weight * distance_cost
            - relevance_bonus
        )


def _track_distance(det: Detection) -> float | None:
    distance = det.metadata.get("distance_bumper_m", det.distance_m)
    if distance is None:
        return None
    return float(distance)


def _distance_cost(a: float | None, b: float | None) -> float:
    if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
        return 0.0
    return min(1.0, abs(a - b) / max(max(a, b), 1.0))


def _iou(a: BBox2D, b: BBox2D) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.width * a.height + b.width * b.height - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _center_distance(a: BBox2D, b: BBox2D) -> float:
    ax = (a.x1 + a.x2) * 0.5
    ay = (a.y1 + a.y2) * 0.5
    bx = (b.x1 + b.x2) * 0.5
    by = (b.y1 + b.y2) * 0.5
    return math.hypot(ax - bx, ay - by)


def _bbox_diag(bbox: BBox2D) -> float:
    return math.hypot(bbox.width, bbox.height)


def _copy_detection(det: Detection) -> Detection:
    return Detection(
        bbox=det.bbox,
        label=det.label,
        confidence=det.confidence,
        ground_contact=det.ground_contact,
        distance_m=det.distance_m,
        sigma_depth=det.sigma_depth,
        track_id=det.track_id,
        ttc_s=det.ttc_s,
        metadata=dict(det.metadata),
    )
