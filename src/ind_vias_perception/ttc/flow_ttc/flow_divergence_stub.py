from __future__ import annotations

def ttc_from_flow_divergence(divergence: float | None) -> float | None:
    if divergence is None or divergence <= 1e-6:
        return None
    return 1.0 / divergence
