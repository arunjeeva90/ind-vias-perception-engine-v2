"""Tests for the SeatbeltDetectionModule.

Validates seatbelt authenticity detection logic including:
- Basic instantiation and return types
- State mapping from cabin evidence
- Absence-based not-worn detection
- Misuse detection
- Confidence scoring
- Backward compatibility with the legacy placeholder
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import (
    CabinEvidenceObject,
    CabinEvidenceObjectType,
    CabinEvidenceRelation,
    CabinEvidenceState,
    CabinSeatbeltState,
    SeatbeltAuthenticity,
)
from ind_vias_dms.vision.seatbelt import SeatbeltDetectionModule, SeatbeltDetectionPlaceholder


def _make_config(**cabin_evidence_overrides) -> DMSConfig:
    """Create a DMSConfig with cabin_evidence overrides for seatbelt testing."""
    cabin_cfg = {
        "enabled": True,
        "seatbelt_not_worn_absence_ms": 10000,
        "seatbelt_misuse_min_confidence": 0.5,
        "seatbelt_worn_min_confidence": 0.6,
        "seatbelt_not_worn_requires_driver_present": True,
    }
    cabin_cfg.update(cabin_evidence_overrides)
    return DMSConfig(cabin_evidence=cabin_cfg)


def _make_frame() -> np.ndarray:
    """Create a dummy frame for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _make_cabin_evidence(
    seatbelt_state: CabinSeatbeltState = CabinSeatbeltState.SEATBELT_UNKNOWN,
    enabled: bool = True,
    evidence_objects: list[CabinEvidenceObject] | None = None,
) -> CabinEvidenceState:
    """Create a CabinEvidenceState with specified seatbelt state."""
    return CabinEvidenceState(
        enabled=enabled,
        seatbelt_state=seatbelt_state,
        evidence_objects=evidence_objects or [],
    )


class TestSeatbeltModuleBasic:
    """Basic instantiation and return type tests."""

    def test_seatbelt_module_returns_authenticity_dataclass(self):
        """SeatbeltDetectionModule.process() returns a SeatbeltAuthenticity."""
        config = _make_config()
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()
        cabin_evidence = _make_cabin_evidence()

        result = module.process(frame, cabin_evidence_state=cabin_evidence, timestamp_ms=1000)

        assert isinstance(result, SeatbeltAuthenticity)
        assert isinstance(result.buckle_switch, str)
        assert isinstance(result.visual_belt_path, str)
        assert isinstance(result.final_state, str)
        assert isinstance(result.confidence, float)

    def test_seatbelt_module_no_cabin_evidence_returns_unknown(self):
        """When no cabin_evidence_state is provided, return UNKNOWN."""
        config = _make_config()
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()

        result = module.process(frame, cabin_evidence_state=None, timestamp_ms=1000)

        assert result.final_state == "UNKNOWN"
        assert result.confidence == 0.0
        assert result.buckle_switch == "UNKNOWN"

    def test_seatbelt_module_does_not_affect_dms_state(self):
        """The module has affect_final_dms_state set to False."""
        config = _make_config()
        module = SeatbeltDetectionModule(config)

        assert module.affect_final_dms_state is False


class TestSeatbeltWornConfirmed:
    """Tests for SEATBELT_WORN_CONFIRMED detection."""

    def test_seatbelt_worn_confirmed_from_cabin_evidence(self):
        """When cabin evidence has SEATBELT_WORN_CONFIRMED, module returns WORN_CONFIRMED."""
        config = _make_config()
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()
        cabin_evidence = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_WORN_CONFIRMED,
            evidence_objects=[
                CabinEvidenceObject(
                    object_type=CabinEvidenceObjectType.SEATBELT,
                    relation_to_driver=CabinEvidenceRelation.ACROSS_TORSO,
                    confidence=0.8,
                )
            ],
        )

        # Feed multiple frames to build confidence
        for t in range(0, 3000, 100):
            result = module.process(frame, cabin_evidence_state=cabin_evidence, timestamp_ms=t)

        assert result.final_state == "WORN_CONFIRMED"
        assert result.confidence >= 0.6
        assert result.visual_belt_path == "ACROSS_TORSO"

    def test_seatbelt_worn_confirmed_has_minimum_confidence(self):
        """WORN_CONFIRMED always has at least the configured minimum confidence."""
        config = _make_config(seatbelt_worn_min_confidence=0.7)
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()
        cabin_evidence = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_WORN_CONFIRMED,
        )

        # Even with just one frame, confidence should be at least the minimum
        result = module.process(frame, cabin_evidence_state=cabin_evidence, timestamp_ms=100)

        assert result.final_state == "WORN_CONFIRMED"
        assert result.confidence >= 0.7


class TestSeatbeltNotVisible:
    """Tests for seatbelt not visible / unknown states."""

    def test_seatbelt_not_visible_returns_not_visible(self):
        """When cabin evidence shows NOT_VISIBLE and not enough time, return NOT_VISIBLE."""
        config = _make_config(seatbelt_not_worn_absence_ms=10000)
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()
        cabin_evidence = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_NOT_VISIBLE,
        )

        # Not enough time for absence detection
        result = module.process(frame, cabin_evidence_state=cabin_evidence, timestamp_ms=5000)

        assert result.final_state == "NOT_VISIBLE"

    def test_seatbelt_unknown_returns_not_visible_early(self):
        """When cabin evidence has UNKNOWN and observation time is short, return NOT_VISIBLE."""
        config = _make_config(seatbelt_not_worn_absence_ms=10000)
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()
        cabin_evidence = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_UNKNOWN,
        )

        result = module.process(frame, cabin_evidence_state=cabin_evidence, timestamp_ms=2000)

        # Should not jump to NOT_WORN_SUSPECTED yet
        assert result.final_state in ("NOT_VISIBLE", "UNKNOWN")


class TestSeatbeltNotWornSuspected:
    """Tests for absence-based not-worn detection."""

    def test_seatbelt_not_worn_suspected_after_absence(self):
        """After enough time with driver present but no seatbelt seen, return NOT_WORN_SUSPECTED."""
        config = _make_config(seatbelt_not_worn_absence_ms=5000)
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()
        cabin_evidence = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_UNKNOWN,
            enabled=True,
        )

        # Simulate many frames without seatbelt being detected
        result = None
        for t in range(0, 8000, 100):
            result = module.process(frame, cabin_evidence_state=cabin_evidence, timestamp_ms=t)

        assert result is not None
        assert result.final_state == "NOT_WORN_SUSPECTED"
        assert result.confidence > 0.0

    def test_seatbelt_not_worn_requires_driver_present(self):
        """When driver presence is required and driver_present is False, return UNKNOWN."""
        config = _make_config(
            seatbelt_not_worn_absence_ms=5000,
            seatbelt_not_worn_requires_driver_present=True,
        )
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()
        cabin_evidence = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_UNKNOWN,
            enabled=True,
        )

        # Even after long time, should remain UNKNOWN if driver not present
        for t in range(0, 20000, 100):
            result = module.process(
                frame, cabin_evidence_state=cabin_evidence,
                timestamp_ms=t, driver_present=False,
            )

        assert result.final_state == "UNKNOWN"

    def test_seatbelt_not_worn_from_cabin_evidence_state(self):
        """When cabin evidence directly reports NOT_WORN_SUSPECTED, module passes it through."""
        config = _make_config()
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()
        cabin_evidence = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_NOT_WORN_SUSPECTED,
        )

        result = module.process(frame, cabin_evidence_state=cabin_evidence, timestamp_ms=1000)

        assert result.final_state == "NOT_WORN_SUSPECTED"


class TestSeatbeltMisuse:
    """Tests for seatbelt misuse detection."""

    def test_seatbelt_misuse_suspected(self):
        """When seatbelt detected but not across torso, output MISUSE_SUSPECTED."""
        config = _make_config()
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()
        cabin_evidence = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_MISUSE_SUSPECTED,
            evidence_objects=[
                CabinEvidenceObject(
                    object_type=CabinEvidenceObjectType.SEATBELT,
                    relation_to_driver=CabinEvidenceRelation.NEAR_LAP,
                    confidence=0.7,
                )
            ],
        )

        # Build up some detection history for confidence
        for t in range(0, 3000, 100):
            result = module.process(frame, cabin_evidence_state=cabin_evidence, timestamp_ms=t)

        assert result.final_state == "MISUSE_SUSPECTED"
        assert result.confidence >= 0.5

    def test_seatbelt_misuse_detected_via_belt_path(self):
        """When worn confirmed but belt path is not ACROSS_TORSO, detect misuse."""
        config = _make_config()
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()
        # Belt is confirmed by temporal tracker but evidence shows unexpected position
        cabin_evidence = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_WORN_CONFIRMED,
            evidence_objects=[
                CabinEvidenceObject(
                    object_type=CabinEvidenceObjectType.SEATBELT,
                    relation_to_driver=CabinEvidenceRelation.NEAR_HAND,
                    confidence=0.7,
                )
            ],
        )

        # Feed enough frames for confidence to build
        for t in range(0, 5000, 100):
            result = module.process(frame, cabin_evidence_state=cabin_evidence, timestamp_ms=t)

        assert result.final_state == "MISUSE_SUSPECTED"
        assert result.confidence >= 0.5


class TestSeatbeltConfidenceLow:
    """Tests for confidence-low state mapping."""

    def test_seatbelt_confidence_low_maps_correctly(self):
        """When cabin evidence has CONFIDENCE_LOW, module reflects it."""
        config = _make_config()
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()
        cabin_evidence = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_CONFIDENCE_LOW,
            evidence_objects=[
                CabinEvidenceObject(
                    object_type=CabinEvidenceObjectType.SEATBELT,
                    relation_to_driver=CabinEvidenceRelation.ACROSS_TORSO,
                    confidence=0.3,
                )
            ],
        )

        result = module.process(frame, cabin_evidence_state=cabin_evidence, timestamp_ms=1000)

        assert result.final_state == "CONFIDENCE_LOW"
        # Confidence should be capped low for this state
        assert result.confidence <= 0.4


class TestSeatbeltPipelineIntegration:
    """Tests verifying the module does not affect the DMS final banner."""

    def test_seatbelt_module_does_not_affect_pipeline_banner(self):
        """The seatbelt detection module does not change the DMS v02 final banner.

        This is verified by checking that the module's affect_final_dms_state
        is False and that it only produces SeatbeltAuthenticity output without
        modifying any other pipeline state.
        """
        config = _make_config()
        module = SeatbeltDetectionModule(config)

        # The module has no mechanism to affect the DMS decision
        assert module.affect_final_dms_state is False

        # Process returns only a SeatbeltAuthenticity, not modifying any external state
        frame = _make_frame()
        cabin_evidence = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_WORN_CONFIRMED
        )
        result = module.process(frame, cabin_evidence_state=cabin_evidence, timestamp_ms=1000)
        assert isinstance(result, SeatbeltAuthenticity)


class TestSeatbeltBackwardCompatibility:
    """Tests ensuring backward compatibility with the legacy placeholder."""

    def test_seatbelt_backward_compat_placeholder_still_works(self):
        """SeatbeltDetectionPlaceholder still exists and produces valid output."""
        placeholder = SeatbeltDetectionPlaceholder()
        frame = _make_frame()

        result = placeholder.process(frame)

        assert isinstance(result, SeatbeltAuthenticity)
        assert result.buckle_switch == "UNKNOWN"
        assert result.visual_belt_path == "UNKNOWN"
        assert result.final_state == "UNKNOWN"
        assert result.confidence == 0.0

    def test_seatbelt_placeholder_signature_unchanged(self):
        """The placeholder accepts only frame as argument (original signature)."""
        placeholder = SeatbeltDetectionPlaceholder()
        frame = _make_frame()

        # Should work with just a frame argument (original interface)
        result = placeholder.process(frame)
        assert result is not None


class TestSeatbeltConfidenceScoring:
    """Tests for confidence computation logic."""

    def test_confidence_increases_with_observations(self):
        """Confidence grows as more frames with seatbelt evidence are processed."""
        config = _make_config()
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()
        cabin_evidence = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_WORN_CONFIRMED,
            evidence_objects=[
                CabinEvidenceObject(
                    object_type=CabinEvidenceObjectType.SEATBELT,
                    relation_to_driver=CabinEvidenceRelation.ACROSS_TORSO,
                    confidence=0.8,
                )
            ],
        )

        confidences = []
        for t in range(0, 5000, 100):
            result = module.process(frame, cabin_evidence_state=cabin_evidence, timestamp_ms=t)
            confidences.append(result.confidence)

        # Confidence should generally increase (non-decreasing trend)
        assert confidences[-1] >= confidences[0]
        # Final confidence should be reasonably high after 5 seconds
        assert confidences[-1] >= 0.7

    def test_confidence_capped_at_one(self):
        """Confidence never exceeds 1.0."""
        config = _make_config()
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()
        cabin_evidence = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_WORN_CONFIRMED,
            evidence_objects=[
                CabinEvidenceObject(
                    object_type=CabinEvidenceObjectType.SEATBELT,
                    relation_to_driver=CabinEvidenceRelation.ACROSS_TORSO,
                    confidence=0.9,
                )
            ],
        )

        # Process many frames
        for t in range(0, 30000, 50):
            result = module.process(frame, cabin_evidence_state=cabin_evidence, timestamp_ms=t)

        assert result.confidence <= 1.0


class TestSeatbeltBuckleSwitch:
    """Tests for buckle switch placeholder."""

    def test_buckle_switch_always_unknown(self):
        """Buckle switch is always UNKNOWN (placeholder for CAN integration)."""
        config = _make_config()
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()

        for state in CabinSeatbeltState:
            cabin_evidence = _make_cabin_evidence(seatbelt_state=state)
            result = module.process(frame, cabin_evidence_state=cabin_evidence, timestamp_ms=1000)
            assert result.buckle_switch == "UNKNOWN"


class TestSeatbeltStateTransitions:
    """Tests for state transitions (worn -> removed -> not-worn-suspected)."""

    def test_seatbelt_worn_then_removed_becomes_not_worn_suspected(self):
        """After seatbelt is confirmed worn and then removed, module transitions to NOT_WORN_SUSPECTED.

        This exercises the critical path: WORN_CONFIRMED -> seatbelt removed ->
        enough time passes -> NOT_WORN_SUSPECTED.
        """
        config = _make_config(seatbelt_not_worn_absence_ms=5000)
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()

        # Phase 1: Seatbelt is worn for several seconds
        cabin_evidence_worn = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_WORN_CONFIRMED,
            evidence_objects=[
                CabinEvidenceObject(
                    object_type=CabinEvidenceObjectType.SEATBELT,
                    relation_to_driver=CabinEvidenceRelation.ACROSS_TORSO,
                    confidence=0.8,
                )
            ],
        )
        for t in range(0, 5000, 100):
            result = module.process(frame, cabin_evidence_state=cabin_evidence_worn, timestamp_ms=t)

        assert result.final_state == "WORN_CONFIRMED"
        assert result.confidence >= 0.6

        # Phase 2: Driver unbuckles - seatbelt is no longer visible
        cabin_evidence_removed = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_NOT_VISIBLE,
            enabled=True,
        )

        # Initially after removal, state should be NOT_VISIBLE (not enough time)
        result = module.process(
            frame, cabin_evidence_state=cabin_evidence_removed,
            timestamp_ms=5100, driver_present=True,
        )
        assert result.final_state == "NOT_VISIBLE"

        # Phase 3: Enough time passes without seatbelt being seen
        for t in range(5200, 12000, 100):
            result = module.process(
                frame, cabin_evidence_state=cabin_evidence_removed,
                timestamp_ms=t, driver_present=True,
            )

        # After absence_ms elapsed since last seatbelt seen, should escalate
        assert result.final_state == "NOT_WORN_SUSPECTED"
        assert result.confidence > 0.0

    def test_seatbelt_worn_then_removed_unknown_state(self):
        """Seatbelt worn then cabin evidence goes to UNKNOWN triggers absence detection."""
        config = _make_config(seatbelt_not_worn_absence_ms=3000)
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()

        # Phase 1: Seatbelt is worn
        cabin_evidence_worn = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_WORN_CONFIRMED,
            evidence_objects=[
                CabinEvidenceObject(
                    object_type=CabinEvidenceObjectType.SEATBELT,
                    relation_to_driver=CabinEvidenceRelation.ACROSS_TORSO,
                    confidence=0.8,
                )
            ],
        )
        for t in range(0, 2000, 100):
            module.process(frame, cabin_evidence_state=cabin_evidence_worn, timestamp_ms=t)

        # Phase 2: Cabin evidence goes to UNKNOWN (belt removed)
        cabin_evidence_unknown = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_UNKNOWN,
            enabled=True,
        )
        for t in range(2000, 8000, 100):
            result = module.process(
                frame, cabin_evidence_state=cabin_evidence_unknown,
                timestamp_ms=t, driver_present=True,
            )

        # Should eventually detect not-worn
        assert result.final_state == "NOT_WORN_SUSPECTED"


class TestSeatbeltReset:
    """Tests for the reset() method."""

    def test_reset_clears_state(self):
        """reset() clears internal state so the module behaves as fresh."""
        config = _make_config(seatbelt_not_worn_absence_ms=5000)
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()

        # Build up state
        cabin_evidence_worn = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_WORN_CONFIRMED,
            evidence_objects=[
                CabinEvidenceObject(
                    object_type=CabinEvidenceObjectType.SEATBELT,
                    relation_to_driver=CabinEvidenceRelation.ACROSS_TORSO,
                    confidence=0.8,
                )
            ],
        )
        for t in range(0, 3000, 100):
            module.process(frame, cabin_evidence_state=cabin_evidence_worn, timestamp_ms=t)

        assert module._seatbelt_seen_count > 0
        assert module._last_seatbelt_seen_ms is not None

        # Reset the module
        module.reset()

        assert module._first_frame_ms is None
        assert module._last_seatbelt_seen_ms is None
        assert module._seatbelt_seen_count == 0

    def test_reset_allows_fresh_absence_detection(self):
        """After reset, absence detection works from scratch again."""
        config = _make_config(seatbelt_not_worn_absence_ms=5000)
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()

        # Build up worn state
        cabin_evidence_worn = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_WORN_CONFIRMED,
            evidence_objects=[
                CabinEvidenceObject(
                    object_type=CabinEvidenceObjectType.SEATBELT,
                    relation_to_driver=CabinEvidenceRelation.ACROSS_TORSO,
                    confidence=0.8,
                )
            ],
        )
        for t in range(0, 3000, 100):
            module.process(frame, cabin_evidence_state=cabin_evidence_worn, timestamp_ms=t)

        # Reset (simulating driver change)
        module.reset()

        # Now feed unknown evidence - should eventually trigger not-worn
        cabin_evidence_unknown = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_UNKNOWN,
            enabled=True,
        )
        for t in range(10000, 20000, 100):
            result = module.process(
                frame, cabin_evidence_state=cabin_evidence_unknown,
                timestamp_ms=t, driver_present=True,
            )

        assert result.final_state == "NOT_WORN_SUSPECTED"

    def test_driver_present_false_prevents_absence_detection(self):
        """When driver_present is False, absence detection does not fire."""
        config = _make_config(
            seatbelt_not_worn_absence_ms=5000,
            seatbelt_not_worn_requires_driver_present=True,
        )
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()
        cabin_evidence = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_UNKNOWN,
            enabled=True,
        )

        # Even after long time, with driver_present=False, stays UNKNOWN
        for t in range(0, 20000, 100):
            result = module.process(
                frame, cabin_evidence_state=cabin_evidence,
                timestamp_ms=t, driver_present=False,
            )

        assert result.final_state == "UNKNOWN"

    def test_driver_present_true_enables_absence_detection(self):
        """When driver_present is True (default), absence detection works."""
        config = _make_config(seatbelt_not_worn_absence_ms=5000)
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()
        cabin_evidence = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_UNKNOWN,
            enabled=True,
        )

        for t in range(0, 10000, 100):
            result = module.process(
                frame, cabin_evidence_state=cabin_evidence,
                timestamp_ms=t, driver_present=True,
            )

        assert result.final_state == "NOT_WORN_SUSPECTED"



def _make_heuristic_config(**seatbelt_overrides) -> DMSConfig:
    cabin_cfg = {
        "enabled": True,
        "seatbelt_not_worn_absence_ms": 2500,
        "seatbelt_misuse_min_confidence": 0.5,
        "seatbelt_worn_min_confidence": 0.6,
        "seatbelt_not_worn_requires_driver_present": True,
    }
    seatbelt_cfg = {
        "enabled": True,
        "backend": "heuristic",
        "affect_final_dms_state": False,
        "min_confirm_ms": 1500,
        "clear_ms": 1000,
        "torso_roi_top_offset_face_h": 0.8,
        "torso_roi_bottom_offset_face_h": 3.8,
        "torso_roi_width_expand_face_w": 1.8,
        "belt_min_diagonal_length_ratio": 0.35,
        "belt_min_contrast": 20,
        "belt_min_width_px": 8,
        "belt_max_width_px": 80,
        "seatbelt_track_min_stable_frames": 3,
        "seatbelt_track_min_stable_ms": 1500,
    }
    seatbelt_cfg.update(seatbelt_overrides)
    return DMSConfig(cabin_evidence=cabin_cfg, seatbelt_detection=seatbelt_cfg)


def _make_torso_frame_with_belt() -> np.ndarray:
    frame = np.full((480, 640, 3), 210, dtype=np.uint8)
    cv2.line(frame, (70, 160), (290, 420), (25, 25, 25), 22)
    return frame


class TestSeatbeltVisualHeuristic:
    def test_torso_roi_generated_from_driver_face_box(self):
        module = SeatbeltDetectionModule(_make_heuristic_config())
        frame = _make_frame()
        cabin = _make_cabin_evidence()

        module.process(frame, cabin_evidence_state=cabin, timestamp_ms=0, driver_face_bbox=(100, 50, 200, 150))

        assert cabin.seatbelt_torso_roi
        x1, y1, x2, y2 = cabin.seatbelt_torso_roi
        assert 0.0 <= x1 < x2 <= 1.0
        assert 0.0 <= y1 < y2 <= 1.0

    def test_seatbelt_heuristic_unknown_without_face_or_torso(self):
        module = SeatbeltDetectionModule(_make_heuristic_config())
        cabin = _make_cabin_evidence()

        result = module.process(_make_frame(), cabin_evidence_state=cabin, timestamp_ms=0, driver_face_bbox=None)

        assert cabin.seatbelt_state == CabinSeatbeltState.SEATBELT_UNKNOWN
        assert result.final_state == "UNKNOWN"
        assert "SEATBELT_TORSO_NOT_VISIBLE" in cabin.seatbelt_reason_codes

    def test_stable_diagonal_belt_promotes_to_worn_confirmed(self):
        module = SeatbeltDetectionModule(_make_heuristic_config())
        cabin = _make_cabin_evidence()
        frame = _make_torso_frame_with_belt()
        face = (100, 50, 200, 150)

        module.process(frame, cabin_evidence_state=cabin, timestamp_ms=0, driver_face_bbox=face)
        module.process(frame, cabin_evidence_state=cabin, timestamp_ms=800, driver_face_bbox=face)
        result = module.process(frame, cabin_evidence_state=cabin, timestamp_ms=1600, driver_face_bbox=face)

        assert cabin.seatbelt_state == CabinSeatbeltState.SEATBELT_WORN_CONFIRMED
        assert result.final_state == "WORN_CONFIRMED"
        assert cabin.seatbelt_candidate_line
        assert "SEATBELT_WORN_CONFIRMED" in cabin.seatbelt_reason_codes

    def test_short_neck_lanyard_like_line_is_rejected(self):
        module = SeatbeltDetectionModule(_make_heuristic_config())
        frame = np.full((480, 640, 3), 210, dtype=np.uint8)
        cv2.line(frame, (145, 150), (190, 250), (25, 25, 25), 10)
        cabin = _make_cabin_evidence()

        module.process(frame, cabin_evidence_state=cabin, timestamp_ms=0, driver_face_bbox=(100, 50, 200, 150))

        assert cabin.seatbelt_state != CabinSeatbeltState.SEATBELT_WORN_CONFIRMED
        assert "SEATBELT_LANYARD_REJECTED" in cabin.seatbelt_reason_codes

    def test_no_belt_visible_waits_before_not_worn_suspected(self):
        module = SeatbeltDetectionModule(_make_heuristic_config())
        frame = np.full((480, 640, 3), 210, dtype=np.uint8)
        cabin = _make_cabin_evidence()
        face = (100, 50, 200, 150)

        early = module.process(frame, cabin_evidence_state=cabin, timestamp_ms=1000, driver_face_bbox=face)
        late = module.process(frame, cabin_evidence_state=cabin, timestamp_ms=4000, driver_face_bbox=face)

        assert early.final_state in {"NOT_VISIBLE", "UNKNOWN"}
        assert late.final_state == "NOT_WORN_SUSPECTED"
        assert cabin.seatbelt_state == CabinSeatbeltState.SEATBELT_NOT_WORN_SUSPECTED

    def test_seatbelt_heuristic_does_not_affect_final_dms_state_by_default(self):
        module = SeatbeltDetectionModule(_make_heuristic_config())
        assert module.affect_final_dms_state is False

    def test_seatbelt_status_and_debug_fields_are_populated(self):
        module = SeatbeltDetectionModule(_make_heuristic_config())
        cabin = _make_cabin_evidence()
        module.process(_make_torso_frame_with_belt(), cabin_evidence_state=cabin, timestamp_ms=0, driver_face_bbox=(100, 50, 200, 150))

        assert cabin.seatbelt_backend == "heuristic"
        assert cabin.seatbelt_confidence > 0.0
        assert cabin.seatbelt_candidate_score > 0.0
        assert cabin.seatbelt_torso_roi
        assert cabin.seatbelt_affect_final_dms_state is False


    def test_dark_line_outside_belt_corridor_is_rejected(self):
        module = SeatbeltDetectionModule(_make_heuristic_config())
        frame = np.full((480, 640, 3), 210, dtype=np.uint8)
        cv2.line(frame, (290, 160), (70, 420), (25, 25, 25), 22)
        cabin = _make_cabin_evidence()

        module.process(frame, cabin_evidence_state=cabin, timestamp_ms=0, driver_face_bbox=(100, 50, 200, 150))

        assert cabin.seatbelt_state != CabinSeatbeltState.SEATBELT_CANDIDATE
        assert cabin.seatbelt_state != CabinSeatbeltState.SEATBELT_WORN_CONFIRMED
        assert set(cabin.seatbelt_reason_codes) & {
            "SEATBELT_OUTSIDE_BELT_CORRIDOR_REJECTED",
            "SEATBELT_UPPER_ANCHOR_MISSING",
            "SEATBELT_LOWER_ANCHOR_MISSING",
        }

    def test_horizontal_arm_shadow_like_candidate_is_rejected(self):
        module = SeatbeltDetectionModule(_make_heuristic_config())
        frame = np.full((480, 640, 3), 210, dtype=np.uint8)
        cv2.line(frame, (40, 270), (310, 285), (25, 25, 25), 26)
        cabin = _make_cabin_evidence()

        module.process(frame, cabin_evidence_state=cabin, timestamp_ms=0, driver_face_bbox=(100, 50, 200, 150))

        assert cabin.seatbelt_state != CabinSeatbeltState.SEATBELT_CANDIDATE
        assert "SEATBELT_HORIZONTAL_SHADOW_REJECTED" in cabin.seatbelt_reason_codes or "SEATBELT_OUTSIDE_BELT_CORRIDOR_REJECTED" in cabin.seatbelt_reason_codes

    def test_different_candidate_lines_do_not_accumulate_confirmation(self):
        module = SeatbeltDetectionModule(_make_heuristic_config(seatbelt_track_max_center_jump_norm=0.03))
        frame_a = np.full((480, 640, 3), 210, dtype=np.uint8)
        frame_b = np.full((480, 640, 3), 210, dtype=np.uint8)
        cv2.line(frame_a, (70, 160), (290, 420), (25, 25, 25), 22)
        cv2.line(frame_b, (30, 165), (250, 420), (25, 25, 25), 22)
        cabin = _make_cabin_evidence()
        face = (100, 50, 200, 150)

        module.process(frame_a, cabin_evidence_state=cabin, timestamp_ms=0, driver_face_bbox=face)
        module.process(frame_b, cabin_evidence_state=cabin, timestamp_ms=800, driver_face_bbox=face)
        module.process(frame_a, cabin_evidence_state=cabin, timestamp_ms=1600, driver_face_bbox=face)

        assert cabin.seatbelt_state != CabinSeatbeltState.SEATBELT_WORN_CONFIRMED
        assert cabin.seatbelt_unstable_track_reset_count > 0

