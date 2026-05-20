from __future__ import annotations

from ind_vias_perception.common.types import BBox2D, Detection, ObjectClass
from ind_vias_perception.temporal.trackers.simple_tracker import SimpleDistanceTracker


def _det(
    x1: float = 100,
    distance_m: float = 20.0,
    label: ObjectClass = ObjectClass.CAR,
) -> Detection:
    det = Detection(BBox2D(x1, 100, x1 + 100, 220), label, 0.9, distance_m=distance_m)
    det.metadata["distance_bumper_m"] = distance_m
    det.metadata["in_ego_corridor"] = True
    return det


def test_same_object_keeps_id_across_frames():
    tracker = SimpleDistanceTracker()

    first = tracker.update([_det(100, 20.0)], 0.0)[0]
    second = tracker.update([_det(105, 19.5)], 0.1)[0]

    assert first.track_id == second.track_id
    assert second.metadata["track_predicted"] is False


def test_brief_miss_preserves_track_as_predicted():
    tracker = SimpleDistanceTracker(max_age=2)

    first = tracker.update([_det(100, 20.0)], 0.0)[0]
    predicted = tracker.update([], 0.1)[0]

    assert predicted.track_id == first.track_id
    assert predicted.metadata["track_predicted"] is True
    assert predicted.metadata["missing_frames"] == 1.0


def test_distance_mismatch_discourages_wrong_association():
    tracker = SimpleDistanceTracker(max_association_cost=0.4, distance_weight=1.0)

    first = tracker.update([_det(100, 20.0)], 0.0)[0]
    second = tracker.update([_det(102, 80.0)], 0.1)[0]

    assert second.track_id != first.track_id
