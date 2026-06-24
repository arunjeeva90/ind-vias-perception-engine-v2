
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import (
    CabinEvidenceObjectType,
    CabinEvidenceRelation,
    CabinEvidenceState,
    CabinSeatbeltState,
    SeatbeltAuthenticity,
)


class SeatbeltDetectionPlaceholder:
    """Legacy placeholder kept for backward compatibility."""

    def process(self, frame: object) -> SeatbeltAuthenticity:
        # TODO(v0.3): add visual belt-path and buckle-signal fusion.
        return SeatbeltAuthenticity()


@dataclass
class _HeuristicCandidate:
    found: bool = False
    line_norm: list[float] | None = None
    score: float = 0.0
    contrast: float = 0.0
    width_px: float = 0.0
    expected_width_px: float = 0.0
    angle_deg: float = 0.0
    center_norm: tuple[float, float] = (0.0, 0.0)
    corridor_score: float = 0.0
    width_score: float = 0.0
    edge_pair_score: float = 0.0
    anchor_score: float = 0.0
    reason_codes: list[str] | None = None
    rejected_reason: str = ""


@dataclass
class _TrackState:
    active: bool = False
    start_ms: int = 0
    last_seen_ms: int = 0
    stable_frames: int = 0
    center_norm: tuple[float, float] = (0.0, 0.0)
    angle_deg: float = 0.0
    width_px: float = 0.0
    line_norm: list[float] | None = None
    score: float = 0.0
    confirmed: bool = False


class SeatbeltDetectionModule:
    """Prototype seatbelt authenticity evidence module.

    The v0.3.0 path keeps final DMS decisions untouched and adds a conservative
    visual heuristic for a dark diagonal belt path across the driver torso. COCO
    ONNX cabin models do not contain a seatbelt class, so this module does not
    claim model-backed belt detection unless explicit SEATBELT cabin evidence is
    present from a future custom model/synthetic source.
    """

    def __init__(self, config: DMSConfig) -> None:
        cabin_cfg = config.cabin_evidence or {}
        belt_cfg = config.seatbelt_detection or {}
        self.backend: str = str(belt_cfg.get("backend", "disabled" if not belt_cfg else "heuristic"))
        self.enabled: bool = bool(belt_cfg.get("enabled", bool(belt_cfg)))
        self.affect_final_dms_state: bool = bool(belt_cfg.get("affect_final_dms_state", False))
        self.min_confirm_ms: int = int(belt_cfg.get("min_confirm_ms", 1500))
        self.clear_ms: int = int(belt_cfg.get("clear_ms", 1000))
        self.driver_torso_roi_from_face: bool = bool(belt_cfg.get("driver_torso_roi_from_face", True))
        self.torso_roi_top_offset_face_h: float = float(belt_cfg.get("torso_roi_top_offset_face_h", 0.8))
        self.torso_roi_bottom_offset_face_h: float = float(belt_cfg.get("torso_roi_bottom_offset_face_h", 3.8))
        self.torso_roi_width_expand_face_w: float = float(belt_cfg.get("torso_roi_width_expand_face_w", 1.8))
        self.belt_dark_line_enabled: bool = bool(belt_cfg.get("belt_dark_line_enabled", True))
        self.belt_edge_detection_enabled: bool = bool(belt_cfg.get("belt_edge_detection_enabled", True))
        self.belt_min_diagonal_length_ratio: float = float(belt_cfg.get("belt_min_diagonal_length_ratio", 0.35))
        self.belt_min_contrast: float = float(belt_cfg.get("belt_min_contrast", 25.0))
        self.belt_min_width_px: int = int(belt_cfg.get("belt_min_width_px", 8))
        self.belt_max_width_px: int = int(belt_cfg.get("belt_max_width_px", 80))
        self.belt_temporal_stability_required: bool = bool(belt_cfg.get("belt_temporal_stability_required", True))
        self.driver_image_side: str = str(getattr(config, "driver_image_side", "LEFT")).upper()
        self.belt_corridor_enabled: bool = bool(belt_cfg.get("belt_corridor_enabled", True))
        self.belt_corridor_width_ratio: float = float(belt_cfg.get("belt_corridor_width_ratio", 0.28))
        self.belt_min_corridor_overlap: float = float(belt_cfg.get("belt_min_corridor_overlap", 0.60))
        self.belt_require_upper_anchor_hit: bool = bool(belt_cfg.get("belt_require_upper_anchor_hit", True))
        self.belt_require_lower_anchor_hit: bool = bool(belt_cfg.get("belt_require_lower_anchor_hit", True))
        self.belt_require_chest_crossing: bool = bool(belt_cfg.get("belt_require_chest_crossing", True))
        self.width_ratio_min: float = float(belt_cfg.get("seatbelt_width_ratio_to_face_min", 0.16))
        self.width_ratio_max: float = float(belt_cfg.get("seatbelt_width_ratio_to_face_max", 0.45))
        self.width_ratio_nominal: float = float(belt_cfg.get("seatbelt_nominal_width_ratio_to_face", 0.30))
        self.lanyard_width_ratio_max: float = float(belt_cfg.get("seatbelt_lanyard_max_width_ratio_to_face", 0.12))
        self.require_parallel_edges: bool = bool(belt_cfg.get("seatbelt_require_parallel_edges", True))
        self.edge_width_tolerance_ratio: float = float(belt_cfg.get("seatbelt_edge_pair_width_tolerance_ratio", 0.45))
        self.dark_fill_min_contrast: float = float(belt_cfg.get("seatbelt_dark_fill_min_contrast", 18.0))
        self.min_band_length_ratio: float = float(belt_cfg.get("seatbelt_min_band_length_ratio", 0.45))
        self.reject_horizontal_angle_deg: float = float(belt_cfg.get("seatbelt_reject_horizontal_angle_deg", 25.0))
        self.same_track_required: bool = bool(belt_cfg.get("seatbelt_same_track_required", True))
        self.track_max_center_jump_norm: float = float(belt_cfg.get("seatbelt_track_max_center_jump_norm", 0.06))
        self.track_max_angle_delta_deg: float = float(belt_cfg.get("seatbelt_track_max_angle_delta_deg", 12.0))
        self.track_max_width_delta_ratio: float = float(belt_cfg.get("seatbelt_track_max_width_delta_ratio", 0.35))
        self.track_min_stable_frames: int = int(belt_cfg.get("seatbelt_track_min_stable_frames", 8))
        self.track_min_stable_ms: int = int(belt_cfg.get("seatbelt_track_min_stable_ms", self.min_confirm_ms))
        self.candidate_visible_min_ms: int = int(belt_cfg.get("seatbelt_candidate_visible_min_ms", 300))
        self.short_occlusion_hold_ms: int = int(belt_cfg.get("seatbelt_short_occlusion_hold_ms", 700))

        self.seatbelt_not_worn_absence_ms: int = int(
            cabin_cfg.get("seatbelt_not_worn_absence_ms", 10000)
        )
        self.seatbelt_misuse_min_confidence: float = float(
            cabin_cfg.get("seatbelt_misuse_min_confidence", 0.5)
        )
        self.seatbelt_worn_min_confidence: float = float(
            cabin_cfg.get("seatbelt_worn_min_confidence", 0.6)
        )
        self.seatbelt_not_worn_requires_driver_present: bool = bool(
            cabin_cfg.get("seatbelt_not_worn_requires_driver_present", True)
        )

        self._first_frame_ms: int | None = None
        self._last_seatbelt_seen_ms: int | None = None
        self._seatbelt_seen_count: int = 0
        self._belt_candidate_since_ms: int | None = None
        self._no_belt_since_ms: int | None = None
        self._last_torso_roi_px: tuple[int, int, int, int] | None = None
        self._last_torso_roi_ms: int | None = None
        self._last_candidate_line_norm: list[float] = []
        self._last_candidate_score: float = 0.0
        self._last_face_width_px: float = 0.0
        self._track = _TrackState()
        self._confirmed_frames: int = 0
        self._candidate_frames: int = 0
        self._unstable_track_reset_count: int = 0

    def reset(self) -> None:
        """Reset internal state for a new driver session or scene change."""
        self._first_frame_ms = None
        self._last_seatbelt_seen_ms = None
        self._seatbelt_seen_count = 0
        self._belt_candidate_since_ms = None
        self._no_belt_since_ms = None
        self._last_torso_roi_px = None
        self._last_torso_roi_ms = None
        self._last_candidate_line_norm = []
        self._last_candidate_score = 0.0
        self._last_face_width_px = 0.0
        self._track = _TrackState()
        self._confirmed_frames = 0
        self._candidate_frames = 0
        self._unstable_track_reset_count = 0

    def process(
        self,
        frame: np.ndarray | object,
        cabin_evidence_state: CabinEvidenceState | None = None,
        timestamp_ms: int = 0,
        driver_present: bool = True,
        driver_face_bbox: tuple[int, int, int, int] | None = None,
        driver_proposal_bbox_norm: list[float] | tuple[float, float, float, float] | None = None,
    ) -> SeatbeltAuthenticity:
        """Analyze prototype visual seatbelt evidence plus existing cabin state."""
        if self._first_frame_ms is None:
            self._first_frame_ms = timestamp_ms

        if cabin_evidence_state is None:
            return SeatbeltAuthenticity(
                buckle_switch="UNKNOWN",
                visual_belt_path="NOT_AVAILABLE",
                final_state="UNKNOWN",
                confidence=0.0,
            )

        self._prepare_cabin_fields(cabin_evidence_state)
        explicit_model_seatbelt = self._has_explicit_seatbelt_object(cabin_evidence_state)
        if not explicit_model_seatbelt and cabin_evidence_state.detector_backend == "onnx":
            self._add_reason(cabin_evidence_state, "SEATBELT_MODEL_CLASS_NOT_AVAILABLE")

        if self.enabled and self.backend == "heuristic" and isinstance(frame, np.ndarray) and driver_present:
            self._update_heuristic(frame, cabin_evidence_state, timestamp_ms, driver_face_bbox, driver_proposal_bbox_norm)
        elif self.enabled and self.backend == "heuristic" and not driver_present:
            cabin_evidence_state.seatbelt_backend = "heuristic"
            cabin_evidence_state.seatbelt_state = CabinSeatbeltState.SEATBELT_UNKNOWN
            cabin_evidence_state.seatbelt_confidence = 0.0
            self._add_reason(cabin_evidence_state, "SEATBELT_TORSO_NOT_VISIBLE")

        seatbelt_state = cabin_evidence_state.seatbelt_state
        if seatbelt_state in (
            CabinSeatbeltState.SEATBELT_CANDIDATE,
            CabinSeatbeltState.SEATBELT_WORN_CONFIRMED,
            CabinSeatbeltState.SEATBELT_CONFIDENCE_LOW,
            CabinSeatbeltState.SEATBELT_MISUSE_SUSPECTED,
        ):
            self._seatbelt_seen_count += 1
            self._last_seatbelt_seen_ms = timestamp_ms

        visual_belt_path = self._determine_belt_path(cabin_evidence_state)
        final_state, confidence = self._classify_state(
            seatbelt_state,
            cabin_evidence_state,
            timestamp_ms,
            visual_belt_path,
            driver_present=driver_present,
        )

        return SeatbeltAuthenticity(
            buckle_switch="UNKNOWN",
            visual_belt_path=visual_belt_path,
            final_state=final_state,
            confidence=confidence,
        )

    def _prepare_cabin_fields(self, cabin_evidence: CabinEvidenceState) -> None:
        cabin_evidence.seatbelt_backend = self.backend if self.enabled else "disabled"
        cabin_evidence.seatbelt_affect_final_dms_state = self.affect_final_dms_state
        if not cabin_evidence.seatbelt_reason_codes:
            cabin_evidence.seatbelt_reason_codes = []

    @staticmethod
    def _add_reason(cabin_evidence: CabinEvidenceState, reason: str) -> None:
        if reason not in cabin_evidence.seatbelt_reason_codes:
            cabin_evidence.seatbelt_reason_codes.append(reason)

    @staticmethod
    def _has_explicit_seatbelt_object(cabin_evidence: CabinEvidenceState) -> bool:
        return any(obj.object_type == CabinEvidenceObjectType.SEATBELT for obj in cabin_evidence.evidence_objects)

    def _update_heuristic(
        self,
        frame: np.ndarray,
        cabin_evidence: CabinEvidenceState,
        timestamp_ms: int,
        driver_face_bbox: tuple[int, int, int, int] | None,
        driver_proposal_bbox_norm: list[float] | tuple[float, float, float, float] | None = None,
    ) -> None:
        roi_px = self._torso_roi(frame.shape, driver_face_bbox, timestamp_ms, driver_proposal_bbox_norm)
        cabin_evidence.seatbelt_backend = "heuristic"
        self._reset_frame_debug(cabin_evidence)
        if roi_px is None:
            cabin_evidence.seatbelt_state = CabinSeatbeltState.SEATBELT_UNKNOWN
            self._reset_track("SEATBELT_TORSO_NOT_VISIBLE", cabin_evidence)
            self._add_reason(cabin_evidence, "SEATBELT_TORSO_NOT_VISIBLE")
            return

        cabin_evidence.seatbelt_torso_roi = self._norm_box(roi_px, frame.shape)
        candidate = self._detect_belt_candidate(frame, roi_px, self._last_face_width_px)
        self._copy_candidate_debug(cabin_evidence, candidate)
        if candidate.found:
            self._update_track(candidate, timestamp_ms, cabin_evidence)
            age_ms = timestamp_ms - self._track.start_ms if self._track.active else 0
            temporal_score = min(
                age_ms / max(1, self.track_min_stable_ms),
                self._track.stable_frames / max(1, self.track_min_stable_frames),
                1.0,
            )
            cabin_evidence.seatbelt_temporal_score = round(temporal_score, 3)
            cabin_evidence.seatbelt_track_age_ms = age_ms
            cabin_evidence.seatbelt_track_stable_frames = self._track.stable_frames
            cabin_evidence.seatbelt_confidence = round(min(1.0, 0.25 + candidate.score * 0.55 + temporal_score * 0.20), 3)
            self._add_reason(cabin_evidence, "SEATBELT_TRACK_STARTED" if self._track.stable_frames == 1 else "SEATBELT_TRACK_STABLE")
            for reason in candidate.reason_codes or []:
                self._add_reason(cabin_evidence, reason)
            confirmed = self._track.stable_frames >= self.track_min_stable_frames and age_ms >= self.track_min_stable_ms
            visible_candidate = age_ms >= self.candidate_visible_min_ms
            if confirmed:
                cabin_evidence.seatbelt_state = CabinSeatbeltState.SEATBELT_WORN_CONFIRMED
                cabin_evidence.seatbelt_candidate_line = list(candidate.line_norm or [])
                cabin_evidence.seatbelt_confirmed_ms = age_ms
                self._confirmed_frames += 1
                self._track.confirmed = True
                self._add_reason(cabin_evidence, "SEATBELT_WORN_CONFIRMED")
            elif visible_candidate:
                cabin_evidence.seatbelt_state = CabinSeatbeltState.SEATBELT_CANDIDATE
                cabin_evidence.seatbelt_candidate_line = list(candidate.line_norm or [])
                cabin_evidence.seatbelt_confirmed_ms = age_ms
                self._candidate_frames += 1
                self._add_reason(cabin_evidence, "SEATBELT_HEURISTIC_CANDIDATE")
                self._add_reason(cabin_evidence, "SEATBELT_CONFIRMATION_BLOCKED_UNSTABLE_TRACK")
            else:
                cabin_evidence.seatbelt_state = CabinSeatbeltState.SEATBELT_NOT_VISIBLE
                cabin_evidence.seatbelt_candidate_line = []
                cabin_evidence.seatbelt_confirmed_ms = 0
                self._add_reason(cabin_evidence, "SEATBELT_CONFIRMATION_BLOCKED_UNSTABLE_TRACK")
            self._no_belt_since_ms = None
        else:
            self._handle_missing_candidate(cabin_evidence, timestamp_ms, candidate)
        cabin_evidence.seatbelt_confirmed_frames = self._confirmed_frames
        cabin_evidence.seatbelt_candidate_frames = self._candidate_frames
        cabin_evidence.seatbelt_unstable_track_reset_count = self._unstable_track_reset_count

    def _reset_frame_debug(self, cabin_evidence: CabinEvidenceState) -> None:
        cabin_evidence.seatbelt_confidence = 0.0
        cabin_evidence.seatbelt_candidate_line = []
        cabin_evidence.seatbelt_candidate_score = 0.0
        cabin_evidence.seatbelt_corridor_score = 0.0
        cabin_evidence.seatbelt_width_score = 0.0
        cabin_evidence.seatbelt_edge_pair_score = 0.0
        cabin_evidence.seatbelt_anchor_score = 0.0
        cabin_evidence.seatbelt_temporal_score = 0.0
        cabin_evidence.seatbelt_candidate_width_px = 0.0
        cabin_evidence.seatbelt_expected_width_px = 0.0
        cabin_evidence.seatbelt_confirmed_ms = 0
        cabin_evidence.seatbelt_track_age_ms = 0
        cabin_evidence.seatbelt_track_stable_frames = 0
        cabin_evidence.seatbelt_track_reset_reason = ""
        cabin_evidence.seatbelt_rejected_reason = ""
        cabin_evidence.seatbelt_overlay_drawn = False
        cabin_evidence.seatbelt_total_raw_candidates = 0
        cabin_evidence.seatbelt_rejected_lanyard_count = 0
        cabin_evidence.seatbelt_rejected_arm_shadow_count = 0
        cabin_evidence.seatbelt_rejected_outside_corridor_count = 0

    def _copy_candidate_debug(self, cabin_evidence: CabinEvidenceState, candidate: _HeuristicCandidate) -> None:
        cabin_evidence.seatbelt_candidate_score = candidate.score
        cabin_evidence.seatbelt_corridor_score = candidate.corridor_score
        cabin_evidence.seatbelt_width_score = candidate.width_score
        cabin_evidence.seatbelt_edge_pair_score = candidate.edge_pair_score
        cabin_evidence.seatbelt_anchor_score = candidate.anchor_score
        cabin_evidence.seatbelt_candidate_width_px = round(candidate.width_px, 2)
        cabin_evidence.seatbelt_expected_width_px = round(candidate.expected_width_px, 2)
        if candidate.rejected_reason:
            cabin_evidence.seatbelt_rejected_reason = candidate.rejected_reason
            if "LANYARD" in candidate.rejected_reason:
                cabin_evidence.seatbelt_rejected_lanyard_count = 1
            if candidate.rejected_reason in {"SEATBELT_ARM_SHADOW_REJECTED", "SEATBELT_WIDTH_TOO_WIDE_SHADOW", "SEATBELT_HORIZONTAL_SHADOW_REJECTED"}:
                cabin_evidence.seatbelt_rejected_arm_shadow_count = 1
            if candidate.rejected_reason == "SEATBELT_OUTSIDE_BELT_CORRIDOR_REJECTED":
                cabin_evidence.seatbelt_rejected_outside_corridor_count = 1

    def _handle_missing_candidate(self, cabin_evidence: CabinEvidenceState, timestamp_ms: int, candidate: _HeuristicCandidate) -> None:
        for reason in candidate.reason_codes or []:
            self._add_reason(cabin_evidence, reason)
        if self._track.confirmed and timestamp_ms - self._track.last_seen_ms <= self.short_occlusion_hold_ms:
            cabin_evidence.seatbelt_state = CabinSeatbeltState.SEATBELT_HELD_OCCLUDED
            cabin_evidence.seatbelt_candidate_line = list(self._track.line_norm or [])
            cabin_evidence.seatbelt_track_age_ms = timestamp_ms - self._track.start_ms
            cabin_evidence.seatbelt_track_stable_frames = self._track.stable_frames
            cabin_evidence.seatbelt_confidence = max(0.45, min(0.85, self._track.score))
            self._add_reason(cabin_evidence, "SEATBELT_HELD_OCCLUDED")
            return
        if self._track.active and timestamp_ms - self._track.last_seen_ms > self.short_occlusion_hold_ms:
            self._reset_track("SEATBELT_TRACK_RESET_LINE_JUMP", cabin_evidence)
        if self._no_belt_since_ms is None:
            self._no_belt_since_ms = timestamp_ms
        self._add_reason(cabin_evidence, "SEATBELT_NO_STABLE_EVIDENCE")
        if timestamp_ms - self._no_belt_since_ms >= self.seatbelt_not_worn_absence_ms:
            cabin_evidence.seatbelt_state = CabinSeatbeltState.SEATBELT_NOT_WORN_SUSPECTED
        else:
            cabin_evidence.seatbelt_state = CabinSeatbeltState.SEATBELT_NOT_VISIBLE

    def _update_track(self, candidate: _HeuristicCandidate, timestamp_ms: int, cabin_evidence: CabinEvidenceState) -> None:
        if not self._track.active:
            self._track = _TrackState(True, timestamp_ms, timestamp_ms, 1, candidate.center_norm, candidate.angle_deg, candidate.width_px, list(candidate.line_norm or []), candidate.score, False)
            return
        reset_reason = self._track_reset_reason(candidate)
        if reset_reason:
            self._reset_track(reset_reason, cabin_evidence)
            self._track = _TrackState(True, timestamp_ms, timestamp_ms, 1, candidate.center_norm, candidate.angle_deg, candidate.width_px, list(candidate.line_norm or []), candidate.score, False)
            return
        self._track.last_seen_ms = timestamp_ms
        self._track.stable_frames += 1
        self._track.center_norm = candidate.center_norm
        self._track.angle_deg = candidate.angle_deg
        self._track.width_px = candidate.width_px
        self._track.line_norm = list(candidate.line_norm or [])
        self._track.score = candidate.score

    def _track_reset_reason(self, candidate: _HeuristicCandidate) -> str:
        if not self.same_track_required:
            return ""
        center_jump = math.hypot(candidate.center_norm[0] - self._track.center_norm[0], candidate.center_norm[1] - self._track.center_norm[1])
        if center_jump > self.track_max_center_jump_norm:
            return "SEATBELT_TRACK_RESET_LINE_JUMP"
        if abs(candidate.angle_deg - self._track.angle_deg) > self.track_max_angle_delta_deg:
            return "SEATBELT_TRACK_RESET_ANGLE_CHANGE"
        width_delta = abs(candidate.width_px - self._track.width_px) / max(1.0, self._track.width_px)
        if width_delta > self.track_max_width_delta_ratio:
            return "SEATBELT_TRACK_RESET_ANCHOR_CHANGE"
        return ""

    def _reset_track(self, reason: str, cabin_evidence: CabinEvidenceState | None = None) -> None:
        if self._track.active:
            self._unstable_track_reset_count += 1
        self._track = _TrackState()
        if cabin_evidence is not None:
            cabin_evidence.seatbelt_track_reset_reason = reason
            self._add_reason(cabin_evidence, reason)

    def _torso_roi(
        self,
        frame_shape: tuple[int, ...],
        face_bbox: tuple[int, int, int, int] | None,
        timestamp_ms: int,
        proposal_bbox_norm: list[float] | tuple[float, float, float, float] | None = None,
    ) -> tuple[int, int, int, int] | None:
        height, width = frame_shape[:2]
        if face_bbox is None and proposal_bbox_norm and len(proposal_bbox_norm) >= 4:
            px1, py1, px2, py2 = proposal_bbox_norm[:4]
            if px2 > px1 and py2 > py1:
                face_bbox = (int(px1 * width), int(py1 * height), int(px2 * width), int(py2 * height))
        if self.driver_torso_roi_from_face and face_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in face_bbox]
            face_w = max(1, x2 - x1)
            face_h = max(1, y2 - y1)
            self._last_face_width_px = float(face_w)
            cx = (x1 + x2) / 2.0
            roi_x1 = int(max(0, cx - face_w * self.torso_roi_width_expand_face_w))
            roi_x2 = int(min(width - 1, cx + face_w * self.torso_roi_width_expand_face_w))
            roi_y1 = int(max(0, y1 + face_h * self.torso_roi_top_offset_face_h))
            roi_y2 = int(min(height - 1, y1 + face_h * self.torso_roi_bottom_offset_face_h))
            if roi_x2 - roi_x1 >= 20 and roi_y2 - roi_y1 >= 30:
                self._last_torso_roi_px = (roi_x1, roi_y1, roi_x2, roi_y2)
                self._last_torso_roi_ms = timestamp_ms
                return self._last_torso_roi_px
        if (
            self._last_torso_roi_px is not None
            and self._last_torso_roi_ms is not None
            and timestamp_ms - self._last_torso_roi_ms <= self.clear_ms
        ):
            return self._last_torso_roi_px
        return None

    def _detect_belt_candidate(
        self,
        frame: np.ndarray,
        roi_px: tuple[int, int, int, int],
        face_width_px: float | None = None,
    ) -> _HeuristicCandidate:
        x1, y1, x2, y2 = roi_px
        roi = frame[y1:y2, x1:x2]
        face_width_px = float(face_width_px or self._last_face_width_px)
        if roi.size == 0 or face_width_px <= 0:
            return _HeuristicCandidate(False, reason_codes=["SEATBELT_TORSO_NOT_VISIBLE"], rejected_reason="SEATBELT_TORSO_NOT_VISIBLE")
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        corridor = self._corridor_mask(enhanced.shape[:2])
        threshold = min(float(np.mean(enhanced) - self.belt_min_contrast * 0.25), float(np.percentile(enhanced, 38)))
        dark_mask = (enhanced < max(0, threshold)).astype(np.uint8) * 255
        dark_mask = cv2.bitwise_and(dark_mask, corridor)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[_HeuristicCandidate] = []
        rejected: list[_HeuristicCandidate] = []
        for contour in contours:
            if cv2.contourArea(contour) < 40:
                continue
            cand = self._candidate_from_contour(enhanced, contour, roi_px, frame.shape, face_width_px)
            if cand.found:
                candidates.append(cand)
            else:
                rejected.append(cand)
        if candidates:
            best = max(candidates, key=lambda c: c.score)
            best.reason_codes = list(dict.fromkeys((best.reason_codes or []) + ["SEATBELT_BAND_VALID"]))
            return best
        if rejected:
            priority = [
                "SEATBELT_WIDTH_TOO_NARROW_LANYARD",
                "SEATBELT_LANYARD_REJECTED",
                "SEATBELT_WIDTH_TOO_WIDE_SHADOW",
                "SEATBELT_HORIZONTAL_SHADOW_REJECTED",
                "SEATBELT_OUTSIDE_BELT_CORRIDOR_REJECTED",
                "SEATBELT_UPPER_ANCHOR_MISSING",
                "SEATBELT_LOWER_ANCHOR_MISSING",
            ]
            for reason in priority:
                for cand in rejected:
                    if cand.rejected_reason == reason or reason in (cand.reason_codes or []):
                        return cand
            return rejected[0]
        if self._has_lanyard_like_dark_shape(enhanced):
            return _HeuristicCandidate(False, reason_codes=["SEATBELT_LANYARD_REJECTED"], rejected_reason="SEATBELT_LANYARD_REJECTED")
        return _HeuristicCandidate(False, reason_codes=["SEATBELT_LOW_CONTRAST", "SEATBELT_NO_STABLE_EVIDENCE"], rejected_reason="SEATBELT_LOW_CONTRAST")


    def _has_lanyard_like_dark_shape(self, gray: np.ndarray) -> bool:
        threshold = min(float(np.mean(gray) - self.belt_min_contrast * 0.25), float(np.percentile(gray, 30)))
        mask = (gray < max(0, threshold)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_h, roi_w = gray.shape[:2]
        for contour in contours:
            if cv2.contourArea(contour) < 20:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            x_span = w / max(1, roi_w)
            y_span = h / max(1, roi_h)
            if y < roi_h * 0.45 and (x_span < 0.18 or y_span < 0.35 or y + h < roi_h * 0.62):
                return True
        return False

    def _candidate_from_contour(
        self,
        gray: np.ndarray,
        contour: np.ndarray,
        roi_px: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
        face_width_px: float,
    ) -> _HeuristicCandidate:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
        i, j = max(pairs, key=lambda ij: float(np.linalg.norm(box[ij[0]] - box[ij[1]])))
        p1, p2 = box[i], box[j]
        roi_h, roi_w = gray.shape[:2]
        length = float(np.linalg.norm(p2 - p1))
        short_side = max(1.0, float(min(rect[1])))
        angle = abs(math.degrees(math.atan2(float(p2[1] - p1[1]), float(p2[0] - p1[0]))))
        if angle > 90:
            angle = 180 - angle
        length_ratio = length / max(1.0, math.hypot(roi_w, roi_h))
        width_ratio = short_side / max(1.0, face_width_px)
        expected_width = max(1.0, face_width_px * self.width_ratio_nominal)
        line_local = (float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]))
        metrics = self._corridor_metrics(line_local, gray.shape[:2])
        if angle < self.reject_horizontal_angle_deg or angle > 78:
            return self._reject("SEATBELT_HORIZONTAL_SHADOW_REJECTED")
        if length_ratio < self.min_band_length_ratio:
            if max(float(p1[1]), float(p2[1])) < roi_h * 0.58:
                return self._reject("SEATBELT_SHORT_STRAP_REJECTED", ["SEATBELT_LANYARD_REJECTED", "SEATBELT_NECK_CENTER_STRAP_REJECTED"])
            if metrics["overlap"] < self.belt_min_corridor_overlap or not metrics["upper"] or not metrics["lower"]:
                return self._reject("SEATBELT_OUTSIDE_BELT_CORRIDOR_REJECTED")
            return self._reject("SEATBELT_SHORT_STRAP_REJECTED")
        if width_ratio <= self.lanyard_width_ratio_max or width_ratio < self.width_ratio_min:
            return self._reject("SEATBELT_WIDTH_TOO_NARROW_LANYARD", ["SEATBELT_LANYARD_REJECTED"])
        if width_ratio > self.width_ratio_max:
            return self._reject("SEATBELT_WIDTH_TOO_WIDE_SHADOW", ["SEATBELT_ARM_SHADOW_REJECTED"])
        if metrics["overlap"] < self.belt_min_corridor_overlap:
            return self._reject("SEATBELT_OUTSIDE_BELT_CORRIDOR_REJECTED")
        if self.belt_require_upper_anchor_hit and not metrics["upper"]:
            return self._reject("SEATBELT_UPPER_ANCHOR_MISSING")
        if self.belt_require_lower_anchor_hit and not metrics["lower"]:
            return self._reject("SEATBELT_LOWER_ANCHOR_MISSING")
        if self.belt_require_chest_crossing and not metrics["chest"]:
            return self._reject("SEATBELT_CHEST_CROSSING_MISSING")
        fill_contrast = self._dark_fill_contrast(gray, contour)
        if fill_contrast < self.dark_fill_min_contrast:
            return self._reject("SEATBELT_DARK_FILL_LOW")
        width_score = max(0.0, 1.0 - abs(short_side - expected_width) / max(1.0, expected_width * (1.0 + self.edge_width_tolerance_ratio)))
        corridor_score = float(metrics["overlap"])
        anchor_score = (0.5 if metrics["upper"] else 0.0) + (0.5 if metrics["lower"] else 0.0)
        edge_pair_score = width_score if self.require_parallel_edges else 1.0
        dark_score = min(1.0, fill_contrast / max(1.0, self.dark_fill_min_contrast * 2.0))
        score = round(0.25 * corridor_score + 0.22 * width_score + 0.18 * edge_pair_score + 0.20 * anchor_score + 0.15 * dark_score, 3)
        rx1, ry1, _, _ = roi_px
        line_norm = self._norm_line((rx1 + p1[0], ry1 + p1[1], rx1 + p2[0], ry1 + p2[1]), frame_shape)
        center = ((float(p1[0] + p2[0]) * 0.5) / max(1, roi_w), (float(p1[1] + p2[1]) * 0.5) / max(1, roi_h))
        return _HeuristicCandidate(
            True,
            line_norm=line_norm,
            score=score,
            contrast=fill_contrast,
            width_px=short_side,
            expected_width_px=expected_width,
            angle_deg=angle,
            center_norm=center,
            corridor_score=corridor_score,
            width_score=round(width_score, 3),
            edge_pair_score=round(edge_pair_score, 3),
            anchor_score=round(anchor_score, 3),
            reason_codes=["SEATBELT_PARALLEL_EDGES_FOUND", "SEATBELT_WIDTH_PLAUSIBLE", "SEATBELT_DARK_FILL_VALID"],
        )

    @staticmethod
    def _reject(reason: str, extra_reasons: list[str] | None = None) -> _HeuristicCandidate:
        return _HeuristicCandidate(False, reason_codes=list(dict.fromkeys((extra_reasons or []) + [reason])), rejected_reason=reason)

    def _corridor_mask(self, roi_shape: tuple[int, int]) -> np.ndarray:
        h, w = roi_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        upper, lower = self._corridor_points(w, h)
        thickness = max(12, int(min(w, h) * self.belt_corridor_width_ratio))
        cv2.line(mask, upper, lower, 255, thickness)
        return mask

    def _corridor_points(self, w: int, h: int) -> tuple[tuple[int, int], tuple[int, int]]:
        if self.driver_image_side == "RIGHT":
            return (int(w * 0.82), int(h * 0.08)), (int(w * 0.22), int(h * 0.90))
        return (int(w * 0.18), int(h * 0.08)), (int(w * 0.78), int(h * 0.90))

    def _corridor_metrics(self, line: tuple[float, float, float, float], roi_shape: tuple[int, int]) -> dict[str, object]:
        h, w = roi_shape
        upper, lower = self._corridor_points(w, h)
        corridor_width = max(1.0, min(w, h) * self.belt_corridor_width_ratio * 0.5)
        samples = []
        for t in np.linspace(0.0, 1.0, 25):
            samples.append((line[0] + (line[2] - line[0]) * float(t), line[1] + (line[3] - line[1]) * float(t)))
        overlap = sum(1 for p in samples if self._point_line_distance(p, upper, lower) <= corridor_width) / len(samples)
        upper_hit = any(abs(x - upper[0]) <= w * 0.22 and abs(y - upper[1]) <= h * 0.24 for x, y in samples)
        lower_hit = any(abs(x - lower[0]) <= w * 0.26 and abs(y - lower[1]) <= h * 0.28 for x, y in samples)
        chest = any(h * 0.30 <= y <= h * 0.78 and w * 0.20 <= x <= w * 0.86 for x, y in samples)
        return {"overlap": overlap, "upper": upper_hit, "lower": lower_hit, "chest": chest}

    @staticmethod
    def _point_line_distance(p: tuple[float, float], a: tuple[int, int], b: tuple[int, int]) -> float:
        px, py = p
        ax, ay = a
        bx, by = b
        den = math.hypot(bx - ax, by - ay)
        if den <= 1e-6:
            return math.hypot(px - ax, py - ay)
        return abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax) / den

    @staticmethod
    def _dark_fill_contrast(gray: np.ndarray, contour: np.ndarray) -> float:
        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
        bg = cv2.subtract(dilated, mask)
        if not np.any(mask) or not np.any(bg):
            return 0.0
        return max(0.0, float(np.mean(gray[bg > 0])) - float(np.mean(gray[mask > 0])))

    def _line_dark_contrast(self, gray: np.ndarray, line: tuple[int, int, int, int]) -> float:
        line_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        bg_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        thickness = max(3, min(self.belt_max_width_px, self.belt_min_width_px) // 2)
        cv2.line(line_mask, line[:2], line[2:], 255, thickness)
        cv2.line(bg_mask, line[:2], line[2:], 255, max(thickness * 4, 12))
        bg_mask = cv2.subtract(bg_mask, line_mask)
        if not np.any(line_mask) or not np.any(bg_mask):
            return 0.0
        line_mean = float(np.mean(gray[line_mask > 0]))
        bg_mean = float(np.mean(gray[bg_mask > 0]))
        return max(0.0, bg_mean - line_mean)

    @staticmethod
    def _norm_box(box: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> list[float]:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        return [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]

    @staticmethod
    def _norm_line(line: tuple[float, float, float, float], frame_shape: tuple[int, ...]) -> list[float]:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = line
        return [round(float(x1) / w, 4), round(float(y1) / h, 4), round(float(x2) / w, 4), round(float(y2) / h, 4)]

    def _determine_belt_path(self, cabin_evidence: CabinEvidenceState) -> str:
        for obj in cabin_evidence.evidence_objects:
            if obj.object_type == CabinEvidenceObjectType.SEATBELT:
                if obj.relation_to_driver == CabinEvidenceRelation.ACROSS_TORSO:
                    return "ACROSS_TORSO"
                if obj.relation_to_driver == CabinEvidenceRelation.UNKNOWN:
                    return "DETECTED_POSITION_UNKNOWN"
                return f"DETECTED_{obj.relation_to_driver.value}"
        if cabin_evidence.seatbelt_candidate_line:
            return "ACROSS_TORSO"
        if cabin_evidence.seatbelt_state == CabinSeatbeltState.SEATBELT_WORN_CONFIRMED:
            return "ACROSS_TORSO"
        return "NOT_VISIBLE"

    def _classify_state(
        self,
        seatbelt_state: CabinSeatbeltState,
        cabin_evidence: CabinEvidenceState,
        timestamp_ms: int,
        visual_belt_path: str,
        driver_present: bool = True,
    ) -> tuple[str, float]:
        if seatbelt_state == CabinSeatbeltState.SEATBELT_CANDIDATE:
            return "CANDIDATE", max(0.3, cabin_evidence.seatbelt_confidence)
        if seatbelt_state == CabinSeatbeltState.SEATBELT_WORN_CONFIRMED:
            confidence = max(self._compute_confidence(timestamp_ms), cabin_evidence.seatbelt_confidence)
            if visual_belt_path not in ("ACROSS_TORSO", "NOT_VISIBLE"):
                if confidence >= self.seatbelt_misuse_min_confidence:
                    return "MISUSE_SUSPECTED", confidence
            return "WORN_CONFIRMED", max(confidence, self.seatbelt_worn_min_confidence)
        if seatbelt_state == CabinSeatbeltState.SEATBELT_MISUSE_SUSPECTED:
            confidence = self._compute_confidence(timestamp_ms)
            return "MISUSE_SUSPECTED", max(confidence, self.seatbelt_misuse_min_confidence)
        if seatbelt_state == CabinSeatbeltState.SEATBELT_CONFLICTING_EVIDENCE:
            return "CONFLICTING_EVIDENCE", max(0.2, cabin_evidence.seatbelt_confidence)
        if seatbelt_state == CabinSeatbeltState.SEATBELT_CONFIDENCE_LOW:
            confidence = self._compute_confidence(timestamp_ms)
            return "CONFIDENCE_LOW", min(confidence, 0.4)
        if seatbelt_state == CabinSeatbeltState.SEATBELT_NOT_WORN_SUSPECTED:
            confidence = self._compute_confidence(timestamp_ms)
            return "NOT_WORN_SUSPECTED", confidence
        if seatbelt_state == CabinSeatbeltState.SEATBELT_NOT_VISIBLE:
            return self._check_absence_based_not_worn(
                cabin_evidence, timestamp_ms, driver_present=driver_present,
            )
        return self._check_absence_based_not_worn(
            cabin_evidence, timestamp_ms, driver_present=driver_present,
        )

    def _check_absence_based_not_worn(
        self,
        cabin_evidence: CabinEvidenceState,
        timestamp_ms: int,
        driver_present: bool = True,
    ) -> tuple[str, float]:
        if self._first_frame_ms is None:
            return "UNKNOWN", 0.0
        observation_duration_ms = timestamp_ms - self._first_frame_ms
        if self.seatbelt_not_worn_requires_driver_present and not driver_present:
            return "UNKNOWN", 0.0
        if observation_duration_ms >= self.seatbelt_not_worn_absence_ms:
            if self._last_seatbelt_seen_ms is None:
                confidence = min(
                    0.7,
                    0.3 + 0.4 * (observation_duration_ms / max(1, self.seatbelt_not_worn_absence_ms * 2)),
                )
                return "NOT_WORN_SUSPECTED", confidence
            elapsed_since_last_seen_ms = timestamp_ms - self._last_seatbelt_seen_ms
            if elapsed_since_last_seen_ms >= self.seatbelt_not_worn_absence_ms:
                confidence = min(
                    0.7,
                    0.3 + 0.4 * (elapsed_since_last_seen_ms / max(1, self.seatbelt_not_worn_absence_ms * 2)),
                )
                return "NOT_WORN_SUSPECTED", confidence
        if cabin_evidence.seatbelt_state == CabinSeatbeltState.SEATBELT_UNKNOWN:
            return "UNKNOWN", 0.0
        return "NOT_VISIBLE", 0.0

    def _compute_confidence(self, timestamp_ms: int) -> float:
        if self._first_frame_ms is None or self._seatbelt_seen_count == 0:
            return 0.0
        count_score = min(1.0, self._seatbelt_seen_count / 30.0)
        observation_ms = timestamp_ms - self._first_frame_ms
        duration_score = min(1.0, observation_ms / 5000.0)
        confidence = 0.6 * count_score + 0.4 * duration_score
        return round(min(1.0, max(0.0, confidence)), 3)
