from __future__ import annotations

from dataclasses import dataclass

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.vision.face_landmarks import FaceLandmarkResult


@dataclass
class TrackedFace:
    observation: FaceLandmarkResult
    track_id: int
    zone: str
    selected_as_driver: bool = False
    missed_frames: int = 0
    driver_candidate_score: float = 0.0
    front_layer_score: float = 0.0
    rear_layer_penalty: float = 0.0
    seat_slot: str = "UNKNOWN"
    depth_layer: str = "UNKNOWN"
    slot_reason: str = ""


@dataclass
class OccupantSelection:
    faces: list[TrackedFace]
    driver: TrackedFace | None
    driver_track_changed: bool = False
    proposal_count: int = 0
    unconfirmed_proposal_count: int = 0
    rejected_proposals: list[dict[str, object]] | None = None


class CabinOccupantManager:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.next_track_id = 1
        self.previous: list[TrackedFace] = []
        self.confirmation: dict[int, tuple[int, int]] = {}
        self.driver_track_id: int | None = None
        self.driver_last_seen_ms: int | None = None

    def update(
        self,
        observations: list[FaceLandmarkResult],
        frame_shape: tuple[int, int, int],
        timestamp_ms: int,
    ) -> OccupantSelection:
        tracked_all = self._assign_tracks(observations, frame_shape)
        proposal_count = len(tracked_all)
        rejected: list[dict[str, object]] = []
        confirmed: list[TrackedFace] = []
        unconfirmed_count = 0
        for face in tracked_all:
            face.zone = self.assign_zone(face.observation.box_norm)
        driver_candidate = self._select_driver(tracked_all, timestamp_ms)
        driver_track_id = driver_candidate.track_id if driver_candidate is not None else None
        for face in tracked_all:
            ok, reasons = self._confirm_face(face, timestamp_ms, frame_shape, face.track_id == driver_track_id)
            if ok:
                confirmed.append(face)
            else:
                unconfirmed_count += 1
                rejected.append(
                    {
                        "track_id": face.track_id,
                        "zone": face.zone,
                        "box_norm": list(face.observation.box_norm or (0.0, 0.0, 0.0, 0.0)),
                        "reason_codes": reasons,
                    }
                )
        tracked = confirmed
        for face in tracked:
            face.zone = self.assign_zone(face.observation.box_norm)
        driver = self._select_driver(tracked, timestamp_ms)
        changed = False
        if driver is not None:
            driver.selected_as_driver = True
            changed = self.driver_track_id is not None and driver.track_id != self.driver_track_id
            self.driver_track_id = driver.track_id
            self.driver_last_seen_ms = timestamp_ms
        if not self.config.retain_non_driver_landmarks:
            for face in tracked_all:
                if not face.selected_as_driver:
                    # Multi-face FaceMesh landmarks are useful to validate a
                    # human face, but passenger landmarks must not enter the
                    # driver eye/head/gaze decision path.
                    face.observation.landmarks_px = {}
        self.previous = tracked_all
        return OccupantSelection(
            tracked,
            driver,
            changed,
            proposal_count=proposal_count,
            unconfirmed_proposal_count=unconfirmed_count,
            rejected_proposals=rejected,
        )

    def assign_zone(self, box_norm: tuple[float, float, float, float] | None) -> str:
        if box_norm is None:
            return "UNKNOWN"
        if self._overlap(box_norm, self._roi("driver_roi_norm")) > 0.05:
            return "DRIVER"
        if self._overlap(box_norm, self._roi("front_passenger_roi_norm")) > 0.05:
            return "FRONT_PASSENGER"
        center_x = (box_norm[0] + box_norm[2]) / 2.0
        if center_x < 0.33:
            return "REAR_LEFT"
        if center_x > 0.66:
            return "REAR_RIGHT"
        return "REAR_CENTER"

    def assign_depth_layer(self, box_norm: tuple[float, float, float, float] | None, area_norm: float = 0.0) -> str:
        if box_norm is None:
            return "UNKNOWN"
        center_y = (box_norm[1] + box_norm[3]) / 2.0
        if center_y >= 0.45:
            return "FRONT_ROW" if area_norm >= self.config.min_face_box_area_norm else "UNKNOWN"
        if area_norm >= self.config.front_row_face_min_area_norm and center_y >= 0.22:
            return "FRONT_ROW"
        if area_norm <= self.config.rear_row_face_max_area_norm or center_y < 0.45:
            return "REAR_ROW"
        return "UNKNOWN"

    def _assign_tracks(
        self,
        observations: list[FaceLandmarkResult],
        frame_shape: tuple[int, int, int],
    ) -> list[TrackedFace]:
        assigned: set[int] = set()
        tracked = []
        height, width = frame_shape[:2]
        for obs in observations:
            best_idx = None
            best_score = 0.0
            for idx, prev in enumerate(self.previous):
                if idx in assigned or obs.bbox is None or prev.observation.bbox is None:
                    continue
                iou_score = _iou(obs.bbox, prev.observation.bbox)
                center_score = 1.0 - min(
                    1.0,
                    _center_distance_norm(obs.bbox, prev.observation.bbox, width, height),
                )
                score = max(iou_score, center_score)
                if (
                    iou_score >= self.config.face_track_iou_threshold
                    or center_score >= 1.0 - self.config.face_track_center_distance_threshold
                ) and score > best_score:
                    best_idx = idx
                    best_score = score
            if best_idx is None:
                track_id = self.next_track_id
                self.next_track_id += 1
            else:
                assigned.add(best_idx)
                track_id = self.previous[best_idx].track_id
            tracked.append(TrackedFace(obs, track_id, "UNKNOWN"))
        return tracked

    def _select_driver(self, faces: list[TrackedFace], timestamp_ms: int) -> TrackedFace | None:
        driver_roi = self._roi("driver_roi_norm")
        candidates: list[TrackedFace] = []
        for face in faces:
            score, front_score, rear_penalty, reason = self._driver_candidate_score(face, driver_roi, timestamp_ms)
            face.driver_candidate_score = score
            face.front_layer_score = front_score
            face.rear_layer_penalty = rear_penalty
            face.seat_slot = "DRIVER" if score >= self.config.driver_min_candidate_score else face.zone
            face.depth_layer = self.assign_depth_layer(
                face.observation.box_norm,
                _area_norm(face.observation.box_norm),
            )
            face.slot_reason = reason
            if score >= self.config.driver_min_candidate_score:
                candidates.append(face)
        if not candidates:
            if (
                self.driver_last_seen_ms is not None
                and timestamp_ms - self.driver_last_seen_ms > self.config.driver_track_hold_ms
            ):
                self.driver_track_id = None
            return None
        roi_center = ((driver_roi[0] + driver_roi[2]) / 2.0, (driver_roi[1] + driver_roi[3]) / 2.0)
        if self.config.driver_largest_face_in_roi_priority:
            # Mounted-camera contract: among validated faces inside the driver
            # ROI, the largest image-space face is the nearest/front driver.
            # Candidate quality remains the tie-breaker; all non-selected faces
            # are passengers even if their boxes also overlap the driver ROI.
            candidates.sort(
                key=lambda face: (
                    _area_norm(face.observation.box_norm),
                    face.driver_candidate_score,
                    -_norm_center_distance(face.observation.box_norm, roi_center),
                ),
                reverse=True,
            )
        else:
            candidates.sort(
                key=lambda face: (
                    face.driver_candidate_score,
                    -_norm_center_distance(face.observation.box_norm, roi_center),
                ),
                reverse=True,
            )
        return candidates[0]

    def _driver_candidate_score(
        self,
        face: TrackedFace,
        driver_roi: tuple[float, float, float, float],
        timestamp_ms: int,
    ) -> tuple[float, float, float, str]:
        quality = face.observation.quality
        box = face.observation.box_norm
        roi_overlap = self._overlap(box, driver_roi)
        area_norm = _area_norm(box)
        depth_layer = self.assign_depth_layer(box, area_norm)
        front_score = 1.0 if depth_layer == "FRONT_ROW" else (0.55 if depth_layer == "UNKNOWN" else 0.0)
        rear_penalty = 0.45 if depth_layer == "REAR_ROW" else 0.0
        continuity = 0.25 if face.track_id == self.driver_track_id else 0.0
        if (
            self.driver_track_id is not None
            and self.driver_last_seen_ms is not None
            and timestamp_ms - self.driver_last_seen_ms <= self.config.prefer_previous_driver_track_ms
            and face.track_id == self.driver_track_id
        ):
            continuity = 0.35
        if quality is None:
            return 0.0, front_score, rear_penalty, "FACE_PROPOSAL_NOT_VALIDATED"
        if self.config.enable_strict_driver_face_validation and not quality.is_valid_driver_face:
            return 0.0, front_score, rear_penalty, ",".join(quality.rejection_reason_codes or ["FACE_PROPOSAL_NOT_VALIDATED"])
        if roi_overlap < 0.05:
            return 0.0, front_score, rear_penalty, "OUTSIDE_DRIVER_ROI"
        quality_score = (
            quality.face_completeness_score * 0.35
            + quality.landmark_coverage_score * 0.25
            + min(1.0, quality.proposal_confidence) * 0.15
        )
        size_score = min(1.0, area_norm / max(self.config.front_row_face_min_area_norm, 1e-6)) * 0.15
        score = min(1.0, roi_overlap * 0.25 + front_score * 0.20 + quality_score + size_score + continuity - rear_penalty)
        reason = "DRIVER_SLOT_CONFIRMED_FRONT_ROW" if score >= self.config.driver_min_candidate_score else "NO_FRONT_ROW_SUPPORT"
        if depth_layer == "REAR_ROW" and roi_overlap >= self.config.rear_overlap_driver_reject_threshold:
            score = 0.0
            reason = "REAR_LAYER_REJECTED_AS_DRIVER"
        return score, front_score, rear_penalty, reason

    def _roi(self, name: str) -> tuple[float, float, float, float]:
        if self.config.auto_generate_rois_from_layout:
            return self._generated_roi(name)
        value = getattr(self.config, name)
        if value:
            return (value["x_min"], value["y_min"], value["x_max"], value["y_max"])
        return self._generated_roi(name)

    def _generated_roi(self, name: str) -> tuple[float, float, float, float]:
        driver_right = self.config.driver_image_side.upper() == "RIGHT"
        if self.config.mirror_input:
            driver_right = not driver_right
        if name == "driver_roi_norm":
            return (0.45, 0.10, 1.00, 0.95) if driver_right else (0.00, 0.10, 0.55, 0.95)
        if name == "front_passenger_roi_norm":
            return (0.00, 0.10, 0.55, 0.95) if driver_right else (0.45, 0.10, 1.00, 0.95)
        if name == "rear_left_roi_norm":
            return (0.00, 0.00, 0.36, 0.70)
        if name == "rear_center_roi_norm":
            return (0.32, 0.00, 0.68, 0.70)
        if name == "rear_right_roi_norm":
            return (0.64, 0.00, 1.00, 0.70)
        return (0.00, 0.00, 1.00, 0.70)

    def driver_roi_state(self) -> str:
        source = "AUTO" if self.config.auto_generate_rois_from_layout else "EXPLICIT"
        roi = self._roi("driver_roi_norm")
        side = "RIGHT" if (roi[0] + roi[2]) / 2.0 > 0.5 else "LEFT"
        return f"{source}_{side}"

    @staticmethod
    def _overlap(
        box: tuple[float, float, float, float] | None,
        roi: tuple[float, float, float, float],
    ) -> float:
        if box is None:
            return 0.0
        x1, y1 = max(box[0], roi[0]), max(box[1], roi[1])
        x2, y2 = min(box[2], roi[2]), min(box[3], roi[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
        return inter / area

    def _confirm_face(
        self,
        face: TrackedFace,
        timestamp_ms: int,
        frame_shape: tuple[int, int, int],
        is_driver_candidate: bool,
    ) -> tuple[bool, list[str]]:
        if is_driver_candidate:
            quality = face.observation.quality
            if quality is None or not quality.is_valid_driver_face:
                return False, (quality.rejection_reason_codes if quality else ["FACE_PROPOSAL_NOT_VALIDATED"])
            return True, []
        if (
            face.slot_reason == "REAR_LAYER_REJECTED_AS_DRIVER"
            and not self.config.retain_non_driver_faces_in_driver_roi
        ):
            return False, ["REAR_LAYER_REJECTED_AS_DRIVER"]
        if face.observation.quality is None:
            return False, ["FACE_PROPOSAL_NOT_VALIDATED"]
        if not self.config.non_driver_false_positive_suppression_enabled:
            return True, []
        reasons: list[str] = []
        obs = face.observation
        height, width = frame_shape[:2]
        frame_area = float(max(1, width * height))
        area_norm = obs.area / frame_area
        if obs.confidence < self.config.non_driver_face_min_confidence:
            reasons.append("LOW_FACE_CONFIDENCE")
        if area_norm < self.config.non_driver_min_face_area_norm:
            reasons.append("FACE_BOX_TOO_SMALL")
        landmark_count = len(obs.landmarks_px or {})
        if self.config.non_driver_require_landmarks and landmark_count == 0:
            reasons.append("LANDMARK_VALIDATION_FAILED")
        elif landmark_count < self.config.non_driver_reject_if_landmark_count_below:
            reasons.append("LANDMARK_VALIDATION_FAILED")
        if not self._valid_occupant_roi(face):
            reasons.append("OUTSIDE_VALID_OCCUPANT_ROI")
        if self.config.non_driver_reject_static_headrest_like_boxes and self._headrest_like(obs):
            reasons.append("HEADREST_LIKE_STATIC_BOX")
        frames, first_seen_ms = self.confirmation.get(face.track_id, (0, timestamp_ms))
        frames += 1
        self.confirmation[face.track_id] = (frames, first_seen_ms)
        if (
            frames < self.config.non_driver_confirm_frames
            or timestamp_ms - first_seen_ms < self.config.non_driver_confirm_time_ms
        ):
            reasons.append("TEMPORAL_CONFIRMATION_PENDING")
        return not reasons, list(dict.fromkeys(reasons))

    def _valid_occupant_roi(self, face: TrackedFace) -> bool:
        if face.zone == "FRONT_PASSENGER":
            return self._overlap(face.observation.box_norm, self._roi("front_passenger_roi_norm")) >= 0.25
        if face.zone in {"REAR_LEFT", "REAR_CENTER", "REAR_RIGHT"}:
            return self._overlap(face.observation.box_norm, self._roi(f"{face.zone.lower()}_roi_norm")) >= 0.10
        return face.zone in {"DRIVER", "UNKNOWN"}

    @staticmethod
    def _headrest_like(face: FaceLandmarkResult) -> bool:
        if face.box_norm is None:
            return False
        x1, y1, x2, y2 = face.box_norm
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height
        center_y = (y1 + y2) / 2.0
        return area < 0.012 and width < 0.09 and center_y < 0.58


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)


def suppress_duplicate_faces(
    faces: list[FaceLandmarkResult],
    iou_threshold: float,
    center_distance_threshold: float,
) -> list[FaceLandmarkResult]:
    kept: list[FaceLandmarkResult] = []
    for face in sorted(faces, key=lambda item: (item.confidence, item.area), reverse=True):
        duplicate = False
        for existing in kept:
            if face.bbox is None or existing.bbox is None:
                continue
            if _iou(face.bbox, existing.bbox) > iou_threshold:
                duplicate = True
                break
            if _norm_box_center_distance(face.box_norm, existing.box_norm) < center_distance_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(face)
    return kept


def _center_distance_norm(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    width: int,
    height: int,
) -> float:
    ac = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    bc = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    return (((ac[0] - bc[0]) / width) ** 2 + ((ac[1] - bc[1]) / height) ** 2) ** 0.5


def _norm_center_distance(
    box: tuple[float, float, float, float] | None,
    point: tuple[float, float],
) -> float:
    if box is None:
        return 1.0
    center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    return ((center[0] - point[0]) ** 2 + (center[1] - point[1]) ** 2) ** 0.5


def _norm_box_center_distance(
    a: tuple[float, float, float, float] | None,
    b: tuple[float, float, float, float] | None,
) -> float:
    if a is None or b is None:
        return 1.0
    ac = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    bc = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    return ((ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2) ** 0.5


def _area_norm(box: tuple[float, float, float, float] | None) -> float:
    if box is None:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
