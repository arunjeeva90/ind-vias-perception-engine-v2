from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
from typing import Deque

from ind_vias_perception.common.types import Detection, ObjectClass, SceneQuality


@dataclass(frozen=True)
class CorridorBounds:
    left: float
    right: float


@dataclass(frozen=True)
class CrossingClassification:
    state: str
    confidence: float
    valid_for_safety: bool
    reason_codes: list[str]
    lateral_displacement_px: float
    corridor_approach: bool
    boundary_suppressed: bool
    tiny_object_suppressed: bool


class LateralCutInDetector:
    def __init__(
        self,
        enabled: bool = False,
        history_size: int = 10,
        min_history: int = 5,
        lateral_velocity_threshold_px_s: float = 25.0,
        max_relevant_distance_m: float = 22.0,
        lateral_ttc_threshold_s: float = 2.8,
        min_confidence_for_warning: float = 0.75,
        min_relevance_for_warning: float = 0.45,
        min_corridor_overlap_for_warning: float = 0.15,
        require_valid_distance_for_warning: bool = True,
        suppress_near_image_boundary: bool = True,
        boundary_margin_px: float = 20.0,
        min_corridor_overlap_delta: float = 0.08,
        required_corridor_entry_frames: int = 3,
        min_lateral_history_count: int = 4,
        min_lateral_ttc_s: float = 0.4,
        max_lateral_ttc_s: float = 4.0,
        crossing_cfg: dict[str, object] | None = None,
        ego_corridor: dict[str, object] | None = None,
    ):
        self.enabled = enabled
        self.history_size = history_size
        self.min_history = min_history
        self.lateral_velocity_threshold_px_s = lateral_velocity_threshold_px_s
        self.max_relevant_distance_m = max_relevant_distance_m
        self.lateral_ttc_threshold_s = lateral_ttc_threshold_s
        self.min_confidence_for_warning = min_confidence_for_warning
        self.min_relevance_for_warning = min_relevance_for_warning
        self.min_corridor_overlap_for_warning = min_corridor_overlap_for_warning
        self.require_valid_distance_for_warning = require_valid_distance_for_warning
        self.suppress_near_image_boundary = suppress_near_image_boundary
        self.boundary_margin_px = boundary_margin_px
        self.min_corridor_overlap_delta = min_corridor_overlap_delta
        self.required_corridor_entry_frames = required_corridor_entry_frames
        self.min_lateral_history_count = min_lateral_history_count
        self.min_lateral_ttc_s = min_lateral_ttc_s
        self.max_lateral_ttc_s = max_lateral_ttc_s
        crossing_cfg = crossing_cfg or {}
        self.crossing_enabled = bool(crossing_cfg.get("enabled", True))
        self.crossing_min_history_count = int(crossing_cfg.get("min_history_count", 8))
        self.crossing_min_lateral_displacement_px = float(
            crossing_cfg.get("min_lateral_displacement_px", 60.0)
        )
        self.crossing_max_distance_m = float(crossing_cfg.get("max_distance_m", 25.0))
        self.crossing_min_confidence = float(crossing_cfg.get("min_confidence", 0.7))
        self.crossing_min_bbox_area_ratio = float(
            crossing_cfg.get("min_bbox_area_ratio", 0.0008)
        )
        self.ego_corridor = ego_corridor or {}
        self._history: dict[int, Deque[tuple[float, float, float]]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )
        self._overlap_history: dict[int, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )

    def update(
        self,
        detections: list[Detection],
        timestamp_s: float,
        image_width: int,
        image_height: int,
        scene: SceneQuality,
    ) -> list[Detection]:
        if not self.enabled:
            for det in detections:
                _set_default_metadata(det)
            return detections

        active_track_ids: set[int] = set()
        for det in detections:
            u_gc, v_gc = _ground_contact(det)
            track_id = det.track_id
            if track_id is not None:
                active_track_ids.add(track_id)
                self._history[track_id].append((timestamp_s, u_gc, v_gc))
            history_count = len(self._history[track_id]) if track_id is not None else 0

            bounds = corridor_bounds_at_y(v_gc, image_width, image_height, self.ego_corridor)
            side_state = side_state_for_x(u_gc, bounds)
            corridor_overlap = corridor_overlap_ratio(det, bounds)
            if track_id is not None:
                self._overlap_history[track_id].append(corridor_overlap)
            overlap_delta = self._overlap_delta(track_id)
            overlap_increasing = self._overlap_increasing(track_id)
            corridor_entry_confirmed = (
                overlap_delta > self.min_corridor_overlap_delta and overlap_increasing
            )
            velocity = self._velocity_px_s(track_id)
            smoothed_velocity = self._smoothed_velocity_px_s(track_id)
            lateral_motion_stable = self._lateral_motion_stable(track_id)
            crossing_trend, entry_side = self._crossing_trend(
                track_id,
                side_state,
                smoothed_velocity,
                image_width,
                image_height,
                corridor_overlap,
                corridor_entry_confirmed,
            )
            distance = _target_distance_m(det)
            far = distance is None or not math.isfinite(distance) or distance > self.max_relevant_distance_m
            ttc_lateral = lateral_ttc_s(u_gc, smoothed_velocity, bounds, side_state)
            confidence = cutin_confidence(
                side_state,
                smoothed_velocity,
                ttc_lateral,
                history_count >= self.min_history,
                far,
                scene.ego_motion_state,
                scene.yaw_confidence,
                self.lateral_velocity_threshold_px_s,
                self.lateral_ttc_threshold_s,
            )
            crossing = self._crossing_state(
                det,
                track_id,
                distance,
                smoothed_velocity,
                image_width,
                image_height,
            )
            valid, reasons = self._valid_for_warning(
                det=det,
                side_state=side_state,
                velocity=smoothed_velocity,
                history_count=history_count,
                distance=distance,
                ttc_lateral=ttc_lateral,
                confidence=confidence,
                corridor_overlap=corridor_overlap,
                corridor_entry_confirmed=corridor_entry_confirmed,
                lateral_motion_stable=lateral_motion_stable,
                crossing_trend=crossing_trend,
                image_width=image_width,
                image_height=image_height,
                scene=scene,
            )
            cutin_state = cutin_state_from_motion(
                side_state,
                smoothed_velocity,
                ttc_lateral,
                confidence,
                self.lateral_velocity_threshold_px_s,
                self.lateral_ttc_threshold_s,
            )
            det.metadata["side_state"] = side_state
            det.metadata["lateral_velocity_px_s"] = velocity
            det.metadata["lateral_velocity_px_s_smoothed"] = smoothed_velocity
            det.metadata["lateral_center_history_count"] = float(history_count)
            det.metadata["lateral_motion_stable"] = lateral_motion_stable
            det.metadata["lateral_history_count"] = float(history_count)
            det.metadata["corridor_overlap_ratio"] = corridor_overlap
            det.metadata["corridor_overlap_delta"] = overlap_delta
            det.metadata["corridor_overlap_increasing"] = overlap_increasing
            det.metadata["corridor_entry_confirmed"] = corridor_entry_confirmed
            det.metadata["ttc_lateral_s"] = ttc_lateral
            det.metadata["cutin_state"] = cutin_state
            det.metadata["cutin_confidence"] = confidence
            det.metadata["cutin_valid_for_safety"] = valid
            det.metadata["cutin_reason_codes"] = ",".join(reasons)
            det.metadata["cutin_warning_candidate"] = "cut_in_risk" if valid else "none"
            det.metadata["cutin_crossing_trend"] = crossing_trend
            det.metadata["cutin_entry_side"] = entry_side
            det.metadata["cutin_warning_eligible"] = valid
            det.metadata["cut_in_risk"] = valid
            det.metadata["cutin_warning_suppressed"] = (
                scene.ego_motion_state != "straight"
            )
            det.metadata["crossing_state"] = crossing.state
            det.metadata["crossing_confidence"] = crossing.confidence
            det.metadata["crossing_history_count"] = float(history_count)
            det.metadata["crossing_valid_for_safety"] = crossing.valid_for_safety
            det.metadata["crossing_reason_codes"] = ",".join(crossing.reason_codes)
            det.metadata["crossing_lateral_displacement_px"] = crossing.lateral_displacement_px
            det.metadata["crossing_corridor_approach"] = crossing.corridor_approach
            det.metadata["crossing_boundary_suppressed"] = crossing.boundary_suppressed
            det.metadata["crossing_tiny_object_suppressed"] = crossing.tiny_object_suppressed

        stale = set(self._history) - active_track_ids
        for track_id in stale:
            if len(self._history[track_id]) == 0:
                self._history.pop(track_id, None)
        return detections

    def _velocity_px_s(self, track_id: int | None) -> float:
        if track_id is None:
            return 0.0
        history = self._history[track_id]
        if len(history) < self.min_history:
            return 0.0
        t0, x0, _ = history[0]
        t1, x1, _ = history[-1]
        dt = max(t1 - t0, 1e-3)
        return (x1 - x0) / dt

    def _smoothed_velocity_px_s(self, track_id: int | None) -> float:
        if track_id is None:
            return 0.0
        history = self._history[track_id]
        if len(history) < self.min_lateral_history_count:
            return 0.0
        half = max(1, len(history) // 2)
        early = list(history)[:half]
        late = list(history)[-half:]
        early_x = sum(x for _, x, _ in early) / len(early)
        late_x = sum(x for _, x, _ in late) / len(late)
        early_t = sum(t for t, _, _ in early) / len(early)
        late_t = sum(t for t, _, _ in late) / len(late)
        return (late_x - early_x) / max(late_t - early_t, 1e-3)

    def _lateral_motion_stable(self, track_id: int | None) -> bool:
        if track_id is None:
            return False
        history = self._history[track_id]
        if len(history) < self.min_lateral_history_count:
            return False
        xs = [x for _, x, _ in history]
        deltas = [b - a for a, b in zip(xs, xs[1:])]
        meaningful = [delta for delta in deltas if abs(delta) >= 2.0]
        if len(meaningful) < self.min_lateral_history_count - 1:
            return False
        positives = sum(delta > 0.0 for delta in meaningful)
        negatives = sum(delta < 0.0 for delta in meaningful)
        return max(positives, negatives) / len(meaningful) >= 0.8

    def _overlap_delta(self, track_id: int | None) -> float:
        if track_id is None or len(self._overlap_history[track_id]) < 2:
            return 0.0
        history = self._overlap_history[track_id]
        return history[-1] - history[0]

    def _overlap_increasing(self, track_id: int | None) -> bool:
        if track_id is None:
            return False
        history = self._overlap_history[track_id]
        if len(history) < self.required_corridor_entry_frames:
            return False
        recent = list(history)[-self.required_corridor_entry_frames:]
        return all(b > a for a, b in zip(recent, recent[1:]))

    def _crossing_trend(
        self,
        track_id: int | None,
        current_side: str,
        velocity: float,
        image_width: int,
        image_height: int,
        corridor_overlap: float,
        corridor_entry_confirmed: bool,
    ) -> tuple[bool, str]:
        if track_id is None:
            return False, "UNKNOWN"
        history = self._history[track_id]
        if len(history) < self.min_history:
            return False, "UNKNOWN"
        _, first_x, first_y = history[0]
        _, last_x, last_y = history[-1]
        first_bounds = corridor_bounds_at_y(first_y, image_width, image_height, self.ego_corridor)
        last_bounds = corridor_bounds_at_y(last_y, image_width, image_height, self.ego_corridor)
        entry_side = side_state_for_x(first_x, first_bounds)
        if entry_side not in {"LEFT", "RIGHT"} or current_side == "IN":
            return False, entry_side
        first_gap = _distance_to_corridor(first_x, first_bounds, entry_side)
        last_gap = _distance_to_corridor(last_x, last_bounds, entry_side)
        moving_toward = (
            entry_side == "LEFT" and velocity >= self.lateral_velocity_threshold_px_s
        ) or (
            entry_side == "RIGHT" and velocity <= -self.lateral_velocity_threshold_px_s
        )
        overlap_ok = corridor_overlap >= self.min_corridor_overlap_for_warning
        return (
            moving_toward and last_gap < first_gap and overlap_ok and corridor_entry_confirmed,
            entry_side,
        )

    def _crossing_state(
        self,
        det: Detection,
        track_id: int | None,
        distance: float | None,
        smoothed_velocity: float,
        image_width: int,
        image_height: int,
    ) -> CrossingClassification:
        crossing_classes = {
            ObjectClass.PEDESTRIAN,
            ObjectClass.BICYCLE,
            ObjectClass.MOTORCYCLE,
            ObjectClass.TWO_WHEELER_AGENT,
        }
        reasons: list[str] = []
        if not self.crossing_enabled:
            reasons.append("disabled")
        if det.label not in crossing_classes:
            reasons.append("non_vru_class")
        if track_id is None:
            reasons.append("insufficient_history")
        if reasons:
            return CrossingClassification("none", 0.0, False, reasons, 0.0, False, False, False)

        history = self._history[track_id]
        history_count = len(history)
        if len(history) < self.crossing_min_history_count:
            return CrossingClassification(
                "uncertain",
                0.0,
                False,
                ["insufficient_history"],
                0.0,
                False,
                False,
                False,
            )
        if distance is None or not math.isfinite(distance) or distance > self.crossing_max_distance_m:
            reasons.append("too_far")
        boundary_suppressed = near_image_boundary(
            det,
            image_width,
            image_height,
            self.boundary_margin_px,
        )
        if boundary_suppressed:
            reasons.append("near_boundary")
        area_ratio = det.bbox.width * det.bbox.height / max(float(image_width * image_height), 1.0)
        tiny_suppressed = area_ratio < self.crossing_min_bbox_area_ratio
        if tiny_suppressed:
            reasons.append("tiny_bbox")

        displacement = history[-1][1] - history[0][1]
        if abs(displacement) < self.crossing_min_lateral_displacement_px:
            return CrossingClassification(
                "none",
                0.0,
                False,
                reasons + ["low_lateral_displacement"],
                displacement,
                False,
                boundary_suppressed,
                tiny_suppressed,
            )

        first_x = history[0][1]
        first_y = history[0][2]
        last_x = history[-1][1]
        last_y = history[-1][2]
        first_bounds = corridor_bounds_at_y(first_y, image_width, image_height, self.ego_corridor)
        last_bounds = corridor_bounds_at_y(last_y, image_width, image_height, self.ego_corridor)
        first_side = side_state_for_x(first_x, first_bounds)
        last_side = side_state_for_x(last_x, last_bounds)
        first_gap = _gap_to_corridor_any_side(first_x, first_bounds)
        last_gap = _gap_to_corridor_any_side(last_x, last_bounds)
        crossed_corridor = first_side != last_side and (first_side == "IN" or last_side == "IN")
        corridor_approach = crossed_corridor or last_gap < first_gap
        stable = self._lateral_motion_stable(track_id)
        confidence = min(
            1.0,
            abs(displacement) / max(self.crossing_min_lateral_displacement_px * 2.0, 1.0),
        )
        if not stable:
            return CrossingClassification(
                "uncertain",
                confidence * 0.5,
                False,
                reasons + ["parallel_motion"],
                displacement,
                corridor_approach,
                boundary_suppressed,
                tiny_suppressed,
            )
        if not corridor_approach:
            return CrossingClassification(
                "parallel",
                min(confidence, 0.5),
                False,
                reasons + ["not_approaching_corridor", "parallel_motion"],
                displacement,
                False,
                boundary_suppressed,
                tiny_suppressed,
            )
        state = "left_to_right" if smoothed_velocity > 0.0 else "right_to_left"
        if confidence < self.crossing_min_confidence:
            reasons.append("low_lateral_displacement")
        valid = not reasons and history_count >= self.crossing_min_history_count
        return CrossingClassification(
            state,
            confidence,
            valid,
            reasons or ["valid_crossing"],
            displacement,
            corridor_approach,
            boundary_suppressed,
            tiny_suppressed,
        )

    def _valid_for_warning(
        self,
        det: Detection,
        side_state: str,
        velocity: float,
        history_count: int,
        distance: float | None,
        ttc_lateral: float | None,
        confidence: float,
        corridor_overlap: float,
        corridor_entry_confirmed: bool,
        lateral_motion_stable: bool,
        crossing_trend: bool,
        image_width: int,
        image_height: int,
        scene: SceneQuality,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if det.label == ObjectClass.PEDESTRIAN:
            reasons.append("pedestrian_crossing_not_vehicle_cutin")
        if side_state == "IN":
            reasons.append("in_path_longitudinal_only")
        if scene.ego_motion_state != "straight":
            reasons.append("ego_not_straight")
        if self.require_valid_distance_for_warning and not det.metadata.get(
            "distance_valid_for_safety", True
        ):
            reasons.append("invalid_distance_for_safety")
        if distance is None or not math.isfinite(distance):
            reasons.append("distance_missing")
        elif distance > self.max_relevant_distance_m:
            reasons.append("distance_too_far")
        if history_count < self.min_history:
            reasons.append("insufficient_lateral_history")
        if history_count < self.min_lateral_history_count:
            reasons.append("insufficient_lateral_center_history")
        if not lateral_motion_stable:
            reasons.append("lateral_motion_unstable")
        if abs(velocity) < self.lateral_velocity_threshold_px_s:
            reasons.append("lateral_velocity_too_low")
        if ttc_lateral is None or not math.isfinite(ttc_lateral):
            reasons.append("lateral_ttc_missing")
        elif ttc_lateral < self.min_lateral_ttc_s:
            reasons.append("lateral_ttc_too_low")
        elif ttc_lateral > min(self.lateral_ttc_threshold_s, self.max_lateral_ttc_s):
            reasons.append("lateral_ttc_too_high")
        if confidence < self.min_confidence_for_warning:
            reasons.append("cutin_confidence_too_low")
        if not corridor_entry_confirmed:
            reasons.append("corridor_entry_not_confirmed")
        relevance = float(det.metadata.get("target_relevance", 0.0))
        if relevance < self.min_relevance_for_warning:
            if corridor_overlap < self.min_corridor_overlap_for_warning or not corridor_entry_confirmed:
                reasons.append("insufficient_corridor_entry")
            if not crossing_trend:
                reasons.append("low_relevance_no_crossing_trend")
            if not corridor_entry_confirmed or not lateral_motion_stable:
                reasons.append("low_relevance_without_confirmed_entry")
        if self.suppress_near_image_boundary and near_image_boundary(
            det,
            image_width,
            image_height,
            self.boundary_margin_px,
        ):
            reasons.append("near_image_boundary")
        if side_state == "LEFT" and velocity <= 0.0:
            reasons.append("not_moving_toward_corridor")
        if side_state == "RIGHT" and velocity >= 0.0:
            reasons.append("not_moving_toward_corridor")
        if scene.degraded_score >= 0.5 or scene.night >= 0.5 or scene.glare >= 0.5 or det.confidence < 0.5:
            if not (
                det.metadata.get("distance_valid_for_safety", True)
                and lateral_motion_stable
                and corridor_entry_confirmed
            ):
                reasons.append("scene_quality_cut_in_suppressed")
        return not reasons, reasons or ["eligible_cut_in"]


def corridor_bounds_at_y(
    y_px: float,
    image_width: int,
    image_height: int,
    cfg: dict[str, object] | None,
) -> CorridorBounds:
    cfg = cfg or {}
    center_x = float(cfg.get("center_x_norm", 0.5)) * image_width
    top_y = float(cfg.get("top_y_norm", 0.45)) * image_height
    bottom_y = float(cfg.get("bottom_y_norm", 1.0)) * image_height
    top_width = float(cfg.get("top_width_norm", 0.18)) * image_width
    bottom_width = float(cfg.get("bottom_width_norm", 0.45)) * image_width
    if bottom_y <= top_y:
        width = bottom_width
    else:
        t = min(1.0, max(0.0, (y_px - top_y) / (bottom_y - top_y)))
        width = top_width + t * (bottom_width - top_width)
    return CorridorBounds(center_x - width * 0.5, center_x + width * 0.5)


def side_state_for_x(x_px: float, bounds: CorridorBounds) -> str:
    if x_px < bounds.left:
        return "LEFT"
    if x_px > bounds.right:
        return "RIGHT"
    return "IN"


def lateral_ttc_s(
    x_px: float,
    velocity_px_s: float,
    bounds: CorridorBounds,
    side_state: str,
) -> float | None:
    if side_state == "LEFT" and velocity_px_s > 0.0:
        return max((bounds.left - x_px) / velocity_px_s, 0.0)
    if side_state == "RIGHT" and velocity_px_s < 0.0:
        return max((x_px - bounds.right) / abs(velocity_px_s), 0.0)
    return None


def corridor_overlap_ratio(det: Detection, bounds: CorridorBounds) -> float:
    width = max(det.bbox.width, 1.0)
    overlap = max(0.0, min(det.bbox.x2, bounds.right) - max(det.bbox.x1, bounds.left))
    return min(1.0, overlap / width)


def _distance_to_corridor(x_px: float, bounds: CorridorBounds, side_state: str) -> float:
    if side_state == "LEFT":
        return max(bounds.left - x_px, 0.0)
    if side_state == "RIGHT":
        return max(x_px - bounds.right, 0.0)
    return 0.0


def _gap_to_corridor_any_side(x_px: float, bounds: CorridorBounds) -> float:
    if x_px < bounds.left:
        return bounds.left - x_px
    if x_px > bounds.right:
        return x_px - bounds.right
    return 0.0


def near_image_boundary(
    det: Detection,
    image_width: int,
    image_height: int,
    margin_px: float,
) -> bool:
    return (
        det.bbox.x1 <= margin_px
        or det.bbox.y1 <= margin_px
        or det.bbox.x2 >= image_width - margin_px
        or det.bbox.y2 >= image_height - margin_px
    )


def cutin_confidence(
    side_state: str,
    velocity_px_s: float,
    ttc_lateral: float | None,
    enough_history: bool,
    far: bool,
    ego_motion_state: str,
    yaw_confidence: float,
    velocity_threshold: float,
    ttc_threshold: float,
) -> float:
    if side_state == "IN":
        return 0.8
    if far or not enough_history or ttc_lateral is None:
        return 0.0
    if abs(velocity_px_s) < velocity_threshold or ttc_lateral > ttc_threshold:
        return 0.0
    confidence = min(1.0, abs(velocity_px_s) / max(velocity_threshold * 2.0, 1.0))
    confidence *= max(0.0, 1.0 - ttc_lateral / max(ttc_threshold * 2.0, 1.0))
    if ego_motion_state == "turning":
        confidence *= 0.25 if yaw_confidence >= 0.8 else 0.5
    return confidence


def cutin_state_from_motion(
    side_state: str,
    velocity_px_s: float,
    ttc_lateral: float | None,
    confidence: float,
    velocity_threshold: float,
    ttc_threshold: float,
) -> str:
    if side_state == "IN":
        return "IN_PATH"
    if ttc_lateral is None or confidence < 0.5 or ttc_lateral > ttc_threshold:
        return "NONE"
    if side_state == "LEFT" and velocity_px_s >= velocity_threshold:
        return "LEFT_CUT_IN"
    if side_state == "RIGHT" and velocity_px_s <= -velocity_threshold:
        return "RIGHT_CUT_IN"
    return "NONE"


def _ground_contact(det: Detection) -> tuple[float, float]:
    u_gc = float(det.metadata.get("u_gc", det.bbox.bottom_center[0]))
    v_gc = float(det.metadata.get("v_gc", det.bbox.bottom_center[1]))
    return u_gc, v_gc


def _target_distance_m(det: Detection) -> float | None:
    distance = det.metadata.get("distance_bumper_m", det.distance_m)
    if distance is None:
        return None
    return float(distance)


def _set_default_metadata(det: Detection) -> None:
    det.metadata.setdefault("side_state", "IN" if det.metadata.get("in_ego_corridor") else "NONE")
    det.metadata.setdefault("lateral_velocity_px_s", 0.0)
    det.metadata.setdefault("lateral_velocity_px_s_smoothed", 0.0)
    det.metadata.setdefault("lateral_center_history_count", 0.0)
    det.metadata.setdefault("lateral_motion_stable", False)
    det.metadata.setdefault("ttc_lateral_s", None)
    det.metadata.setdefault("lateral_history_count", 0.0)
    det.metadata.setdefault("corridor_overlap_ratio", 0.0)
    det.metadata.setdefault("corridor_overlap_delta", 0.0)
    det.metadata.setdefault("corridor_overlap_increasing", False)
    det.metadata.setdefault("corridor_entry_confirmed", False)
    det.metadata.setdefault("cutin_state", "NONE")
    det.metadata.setdefault("cutin_confidence", 0.0)
    det.metadata.setdefault("cutin_valid_for_safety", False)
    det.metadata.setdefault("cutin_reason_codes", "disabled")
    det.metadata.setdefault("cutin_warning_candidate", "none")
    det.metadata.setdefault("cutin_crossing_trend", False)
    det.metadata.setdefault("cutin_entry_side", "UNKNOWN")
    det.metadata.setdefault("cutin_warning_eligible", False)
    det.metadata.setdefault("cut_in_risk", False)
    det.metadata.setdefault("crossing_state", "none")
    det.metadata.setdefault("crossing_confidence", 0.0)
    det.metadata.setdefault("crossing_history_count", 0.0)
    det.metadata.setdefault("crossing_valid_for_safety", False)
    det.metadata.setdefault("crossing_reason_codes", "disabled")
    det.metadata.setdefault("crossing_lateral_displacement_px", 0.0)
    det.metadata.setdefault("crossing_corridor_approach", False)
    det.metadata.setdefault("crossing_boundary_suppressed", False)
    det.metadata.setdefault("crossing_tiny_object_suppressed", False)
