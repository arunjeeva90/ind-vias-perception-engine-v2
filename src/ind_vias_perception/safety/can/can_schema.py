from __future__ import annotations

CAN_SIGNALS = {
    "INDVIAS_FCW_WarningLevel": {"type": "uint8", "values": {"none": 0, "advisory": 1, "visual": 2, "strong": 3}},
    "INDVIAS_TargetDistance_cm": {"type": "uint16", "scale": 100},
    "INDVIAS_TargetTTC_10ms": {"type": "uint16", "scale": 100},
    "INDVIAS_AEBReady": {"type": "bool"},
}
