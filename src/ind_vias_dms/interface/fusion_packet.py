from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FusionPacket:
    timestamp_ms: int = 0
    fused_risk_level: str = "UNKNOWN"
    dms_reason_codes: list[str] = field(default_factory=list)
    adas_reason_codes: list[str] = field(default_factory=list)
