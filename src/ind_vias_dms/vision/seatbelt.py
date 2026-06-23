from __future__ import annotations

import numpy as np

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import (
    CabinEvidenceState,
    CabinSeatbeltState,
    SeatbeltAuthenticity,
)


class SeatbeltDetectionPlaceholder:
    """Legacy placeholder kept for backward compatibility."""

    def process(self, frame: object) -> SeatbeltAuthenticity:
        # TODO(v0.3): add visual belt-path and buckle-signal fusion.
        return SeatbeltAuthenticity()


class SeatbeltDetectionModule:
    """Seatbelt authenticity detection module.

    Fuses cabin evidence seatbelt state from the temporal tracker with
    absence-based not-worn detection and misuse detection. The module
    does NOT affect the DMS final banner (affect_final_dms_state is False).

    Configuration is read from the cabin_evidence dict in DMSConfig,
    following the same pattern as CabinEvidenceFusion.
    """

    def __init__(self, config: DMSConfig) -> None:
        cfg = config.cabin_evidence or {}
        self.seatbelt_not_worn_absence_ms: int = int(
            cfg.get("seatbelt_not_worn_absence_ms", 10000)
        )
        self.seatbelt_misuse_min_confidence: float = float(
            cfg.get("seatbelt_misuse_min_confidence", 0.5)
        )
        self.seatbelt_worn_min_confidence: float = float(
            cfg.get("seatbelt_worn_min_confidence", 0.6)
        )
        self.seatbelt_not_worn_requires_driver_present: bool = bool(
            cfg.get("seatbelt_not_worn_requires_driver_present", True)
        )
        self.affect_final_dms_state: bool = False

        # Internal tracking state
        self._first_frame_ms: int | None = None
        self._last_seatbelt_seen_ms: int | None = None
        self._seatbelt_seen_count: int = 0

    def reset(self) -> None:
        """Reset internal state for a new driver session or scene change.

        Should be called when the driver changes, leaves the seat, or a new
        session begins. This prevents stale history from contaminating the
        confidence score and absence timer.
        """
        self._first_frame_ms = None
        self._last_seatbelt_seen_ms = None
        self._seatbelt_seen_count = 0

    def process(
        self,
        frame: np.ndarray | object,
        cabin_evidence_state: CabinEvidenceState | None = None,
        timestamp_ms: int = 0,
        driver_present: bool = True,
    ) -> SeatbeltAuthenticity:
        """Analyze seatbelt state based on cabin evidence fusion output.

        Args:
            frame: Current camera frame (reserved for future visual analysis).
            cabin_evidence_state: The current CabinEvidenceState from temporal
                fusion, containing the tracked seatbelt_state.
            timestamp_ms: Current frame timestamp in milliseconds.
            driver_present: Whether the driver is currently detected as present
                (face found or body present). Used for absence-based not-worn
                detection. Defaults to True for backward compatibility.

        Returns:
            SeatbeltAuthenticity with buckle_switch, visual_belt_path,
            final_state, and confidence fields populated.
        """
        # Track observation window
        if self._first_frame_ms is None:
            self._first_frame_ms = timestamp_ms

        # If no cabin evidence available, return unknown
        if cabin_evidence_state is None:
            return SeatbeltAuthenticity(
                buckle_switch="UNKNOWN",
                visual_belt_path="NOT_AVAILABLE",
                final_state="UNKNOWN",
                confidence=0.0,
            )

        seatbelt_state = cabin_evidence_state.seatbelt_state

        # Track seatbelt observations for confidence scoring
        if seatbelt_state in (
            CabinSeatbeltState.SEATBELT_WORN_CONFIRMED,
            CabinSeatbeltState.SEATBELT_CONFIDENCE_LOW,
            CabinSeatbeltState.SEATBELT_MISUSE_SUSPECTED,
        ):
            self._seatbelt_seen_count += 1
            self._last_seatbelt_seen_ms = timestamp_ms

        # Determine visual belt path from evidence
        visual_belt_path = self._determine_belt_path(cabin_evidence_state)

        # Determine final state and confidence
        final_state, confidence = self._classify_state(
            seatbelt_state, cabin_evidence_state, timestamp_ms, visual_belt_path,
            driver_present=driver_present,
        )

        return SeatbeltAuthenticity(
            buckle_switch="UNKNOWN",  # Placeholder for CAN bus integration
            visual_belt_path=visual_belt_path,
            final_state=final_state,
            confidence=confidence,
        )

    def _determine_belt_path(self, cabin_evidence: CabinEvidenceState) -> str:
        """Determine the visual belt path from cabin evidence objects.

        Checks the evidence objects for seatbelt detections and their
        spatial relation to the driver to infer belt path.
        """
        from ind_vias_dms.core.types import (
            CabinEvidenceObjectType,
            CabinEvidenceRelation,
        )

        for obj in cabin_evidence.evidence_objects:
            if obj.object_type == CabinEvidenceObjectType.SEATBELT:
                if obj.relation_to_driver == CabinEvidenceRelation.ACROSS_TORSO:
                    return "ACROSS_TORSO"
                elif obj.relation_to_driver == CabinEvidenceRelation.UNKNOWN:
                    return "DETECTED_POSITION_UNKNOWN"
                else:
                    return f"DETECTED_{obj.relation_to_driver.value}"

        # No seatbelt object currently in evidence
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
        """Map the cabin evidence seatbelt state to an authenticity classification.

        Returns:
            Tuple of (final_state, confidence).
        """
        # Direct mapping from confirmed cabin evidence states
        if seatbelt_state == CabinSeatbeltState.SEATBELT_WORN_CONFIRMED:
            confidence = self._compute_confidence(timestamp_ms)
            # Check for misuse: belt detected but not across torso
            if visual_belt_path not in ("ACROSS_TORSO", "NOT_VISIBLE"):
                if confidence >= self.seatbelt_misuse_min_confidence:
                    return "MISUSE_SUSPECTED", confidence
            return "WORN_CONFIRMED", max(confidence, self.seatbelt_worn_min_confidence)

        if seatbelt_state == CabinSeatbeltState.SEATBELT_MISUSE_SUSPECTED:
            confidence = self._compute_confidence(timestamp_ms)
            return "MISUSE_SUSPECTED", max(confidence, self.seatbelt_misuse_min_confidence)

        if seatbelt_state == CabinSeatbeltState.SEATBELT_CONFIDENCE_LOW:
            confidence = self._compute_confidence(timestamp_ms)
            return "CONFIDENCE_LOW", min(confidence, 0.4)

        if seatbelt_state == CabinSeatbeltState.SEATBELT_NOT_WORN_SUSPECTED:
            confidence = self._compute_confidence(timestamp_ms)
            return "NOT_WORN_SUSPECTED", confidence

        if seatbelt_state == CabinSeatbeltState.SEATBELT_NOT_VISIBLE:
            # Check for absence-based not-worn detection
            return self._check_absence_based_not_worn(
                cabin_evidence, timestamp_ms, driver_present=driver_present,
            )

        # SEATBELT_UNKNOWN - check if we have enough observation time for absence detection
        return self._check_absence_based_not_worn(
            cabin_evidence, timestamp_ms, driver_present=driver_present,
        )

    def _check_absence_based_not_worn(
        self,
        cabin_evidence: CabinEvidenceState,
        timestamp_ms: int,
        driver_present: bool = True,
    ) -> tuple[str, float]:
        """Detect seatbelt-not-worn based on absence of seatbelt evidence.

        If the driver is present and enough time has passed without any
        seatbelt detection, suspect that the seatbelt is not worn. This
        handles both the case where a seatbelt was never seen, and the case
        where it was previously seen but has not been seen for a long time
        (e.g., driver unbuckled mid-drive).
        """
        if self._first_frame_ms is None:
            return "UNKNOWN", 0.0

        observation_duration_ms = timestamp_ms - self._first_frame_ms

        # Check if driver presence is required and available
        if self.seatbelt_not_worn_requires_driver_present:
            if not driver_present:
                return "UNKNOWN", 0.0

        # If enough time has passed without seatbelt being seen
        if observation_duration_ms >= self.seatbelt_not_worn_absence_ms:
            if self._last_seatbelt_seen_ms is None:
                # Never seen a seatbelt since start
                confidence = min(
                    0.7,
                    0.3 + 0.4 * (observation_duration_ms / max(1, self.seatbelt_not_worn_absence_ms * 2)),
                )
                return "NOT_WORN_SUSPECTED", confidence

            # Seatbelt was previously seen but has not been seen for a while
            elapsed_since_last_seen_ms = timestamp_ms - self._last_seatbelt_seen_ms
            if elapsed_since_last_seen_ms >= self.seatbelt_not_worn_absence_ms:
                confidence = min(
                    0.7,
                    0.3 + 0.4 * (elapsed_since_last_seen_ms / max(1, self.seatbelt_not_worn_absence_ms * 2)),
                )
                return "NOT_WORN_SUSPECTED", confidence

        return "NOT_VISIBLE", 0.0

    def _compute_confidence(self, timestamp_ms: int) -> float:
        """Compute confidence score based on evidence duration and detection count.

        Confidence increases with:
        - Number of frames where seatbelt was detected
        - Duration of continuous observation
        """
        if self._first_frame_ms is None or self._seatbelt_seen_count == 0:
            return 0.0

        # Count-based component (saturates at ~30 frames)
        count_score = min(1.0, self._seatbelt_seen_count / 30.0)

        # Duration-based component (saturates at 5 seconds of observations)
        observation_ms = timestamp_ms - self._first_frame_ms
        duration_score = min(1.0, observation_ms / 5000.0)

        # Weighted combination
        confidence = 0.6 * count_score + 0.4 * duration_score
        return round(min(1.0, max(0.0, confidence)), 3)
