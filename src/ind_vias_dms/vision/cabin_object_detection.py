from __future__ import annotations

from typing import Any

import numpy as np

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.core.types import CabinEvidenceObject


class CabinObjectDetector:
    """Cabin object evidence detector facade.

    v0.2.5 intentionally ships with a disabled dummy backend. This keeps the
    DMS contract ready for future ONNX/public/internal detectors without
    allowing object detections to drive final DMS warnings yet.
    """

    def __init__(self, config: DMSConfig) -> None:
        evidence_config = config.cabin_evidence or {}
        self.enabled = bool(evidence_config.get("enabled", True))
        self.backend = str(evidence_config.get("detector_backend", "dummy"))
        self.model_path = str(evidence_config.get("model_path", ""))
        self.min_confidence = float(evidence_config.get("min_confidence", 0.35))
        self.backend_status = "DISABLED" if not self.enabled else "DUMMY_READY"

    def detect(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
        context: dict[str, Any] | None = None,
    ) -> list[CabinEvidenceObject]:
        if not self.enabled:
            self.backend_status = "DISABLED"
            return []
        if self.backend != "dummy":
            self.backend_status = "MODEL_NOT_CONFIGURED"
            return []
        self.backend_status = "DUMMY_READY"
        return []
