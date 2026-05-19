from __future__ import annotations

import numpy as np


class FPNStub:
    name = "fpn_stub"

    def forward(self, features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return dict(features)
