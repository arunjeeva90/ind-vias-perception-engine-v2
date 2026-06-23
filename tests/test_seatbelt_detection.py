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
        """When driver presence is required and cabin evidence is disabled, return UNKNOWN."""
        config = _make_config(
            seatbelt_not_worn_absence_ms=5000,
            seatbelt_not_worn_requires_driver_present=True,
        )
        module = SeatbeltDetectionModule(config)
        frame = _make_frame()
        cabin_evidence = _make_cabin_evidence(
            seatbelt_state=CabinSeatbeltState.SEATBELT_UNKNOWN,
            enabled=False,
        )

        # Even after long time, should remain UNKNOWN if driver presence not confirmed
        for t in range(0, 20000, 100):
            result = module.process(frame, cabin_evidence_state=cabin_evidence, timestamp_ms=t)

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
