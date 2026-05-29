from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ADASInputPacket:
    timestamp_ms: int = 0
    forward_risk_level: str = "UNKNOWN"
    reason_codes: list[str] = field(default_factory=list)
