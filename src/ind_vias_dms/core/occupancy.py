from __future__ import annotations

from dataclasses import dataclass

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.occupant_manager import OccupantSelection, TrackedFace
from ind_vias_dms.core.types import OccupancySeatState, OccupancyState


SEAT_KEYS = ("driver", "front_passenger", "rear_left", "rear_center", "rear_right")


@dataclass
class _SeatTrack:
    stable_frames: int = 0
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = 0.0


class CabinOccupancyManager:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.partial_tracks: dict[str, _SeatTrack] = {}

    def update(self, selection: OccupantSelection, timestamp_ms: int) -> OccupancyState:
        seats = self._empty_seats()
        reasons: list[str] = []
        for face in selection.faces:
            key = _zone_to_key(face.zone)
            if key not in seats:
                key = "rear_center" if face.zone.startswith("REAR") else "front_passenger"
            seats[key] = self._seat_from_face(face)

        for rejected in selection.rejected_proposals or []:
            zone = str(rejected.get("zone", "UNKNOWN"))
            key = _zone_to_key(zone)
            if key not in {"rear_left", "rear_center", "rear_right", "front_passenger"}:
                continue
            if key == "front_passenger" and not self.config.enable_front_passenger_detection:
                continue
            if key.startswith("rear") and not self.config.enable_rear_occupant_detection:
                continue
            reason_codes = set(rejected.get("reason_codes", []))
            if self._reject_as_static_object(reason_codes):
                reasons.append("STATIC_OBJECT_SUPPRESSED")
                continue
            bbox = tuple(float(v) for v in rejected.get("box_norm", (0.0, 0.0, 0.0, 0.0)))
            if not self._plausible_partial_box(bbox, key):
                reasons.append("OCCUPANT_BOX_REJECTED")
                continue
            track = self.partial_tracks.setdefault(key, _SeatTrack())
            if track.bbox is not None and _center_jump(track.bbox, bbox) > self.config.rear_occupant_max_jump_ratio:
                track.stable_frames = 0
                reasons.append("OCCUPANT_TRACK_JUMP")
            track.stable_frames += 1
            track.bbox = bbox
            track.confidence = max(track.confidence, self.config.min_partial_occupant_confidence)
            needed = self.config.rear_occupant_confirm_frames if key.startswith("rear") else self.config.front_passenger_confirm_frames
            if track.stable_frames >= needed and seats[key].occupied != "true":
                seats[key] = OccupancySeatState(
                    occupied="partial",
                    occupant_type=key.upper(),
                    detection_source="PARTIAL_FACE",
                    confidence=track.confidence,
                    track_id=None,
                    stable_frames=track.stable_frames,
                    occlusion_state="PARTIAL_FACE_OCCLUDED" if key.startswith("rear") else "UNKNOWN",
                    face_visible=True,
                    body_visible=False,
                    bbox=list(bbox),
                    depth_layer="REAR_ROW" if key.startswith("rear") else "UNKNOWN",
                    slot_reason="REAR_OCCUPANT_CONFIRMED" if key.startswith("rear") else "OCCUPANT_CONFIRMED",
                )
            elif seats[key].occupied == "unknown":
                seats[key] = OccupancySeatState(
                    occupied="possible",
                    occupant_type=key.upper(),
                    detection_source="PARTIAL_FACE",
                    confidence=track.confidence,
                    stable_frames=track.stable_frames,
                    occlusion_state="UNKNOWN",
                    face_visible=False,
                    body_visible=False,
                    bbox=list(bbox),
                    depth_layer="REAR_ROW" if key.startswith("rear") else "UNKNOWN",
                    slot_reason="OCCUPANT_PENDING_TEMPORAL_CONFIRMATION",
                )

        cabin_count = sum(1 for seat in seats.values() if seat.occupied in {"true", "partial"})
        face_count = sum(1 for seat in seats.values() if seat.face_visible and seat.occupied in {"true", "partial"})
        body_count = sum(1 for seat in seats.values() if seat.body_visible)
        confidence_values = [seat.confidence for seat in seats.values() if seat.occupied in {"true", "partial"}]
        confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        return OccupancyState(
            cabin_occupant_count=cabin_count,
            face_count=face_count,
            body_count=body_count,
            driver_present=seats["driver"].occupied == "true",
            front_passenger_present=seats["front_passenger"].occupied in {"true", "partial"},
            rear_left_present=seats["rear_left"].occupied,
            rear_center_present=seats["rear_center"].occupied,
            rear_right_present=seats["rear_right"].occupied,
            unknown_occupant_count=0,
            occupancy_confidence=confidence,
            occupancy_reason_codes=list(dict.fromkeys(reasons)),
            seats=seats,
        )

    def _empty_seats(self) -> dict[str, OccupancySeatState]:
        return {
            key: OccupancySeatState(
                occupied="false" if key in {"driver", "front_passenger"} else "unknown",
                occupant_type=key.upper(),
                detection_source="UNKNOWN",
            )
            for key in SEAT_KEYS
        }

    @staticmethod
    def _seat_from_face(face: TrackedFace) -> OccupancySeatState:
        key = _zone_to_key(face.zone)
        return OccupancySeatState(
            occupied="true",
            occupant_type=face.zone,
            detection_source="FACE",
            confidence=face.observation.confidence,
            track_id=face.track_id,
            stable_frames=1,
            occlusion_state="NONE",
            face_visible=True,
            body_visible=False,
            bbox=list(face.observation.box_norm or (0.0, 0.0, 0.0, 0.0)),
            depth_layer=face.depth_layer,
            slot_reason=face.slot_reason,
        )

    def _reject_as_static_object(self, reason_codes: set[str]) -> bool:
        return self.config.rear_static_headrest_suppression and bool(
            reason_codes & {"HEADREST_LIKE_STATIC_BOX", "STATIC_OBJECT_SUPPRESSED"}
        )

    def _plausible_partial_box(self, bbox: tuple[float, float, float, float], key: str) -> bool:
        x1, y1, x2, y2 = bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height
        min_area = self.config.min_rear_face_box_area_norm if key.startswith("rear") else self.config.min_occupant_box_area_norm
        if area < min_area or width <= 0.0 or height <= 0.0:
            return False
        aspect = width / max(height, 1e-6)
        return 0.25 <= aspect <= 1.5


def _zone_to_key(zone: str) -> str:
    return {
        "DRIVER": "driver",
        "FRONT_PASSENGER": "front_passenger",
        "REAR_LEFT": "rear_left",
        "REAR_CENTER": "rear_center",
        "REAR_RIGHT": "rear_right",
    }.get(zone, "unknown")


def _center_jump(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ac = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    bc = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    return ((ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2) ** 0.5
