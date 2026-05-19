from __future__ import annotations

import numpy as np


class IdentityBackbone:
    name = "identity_backbone"

    def forward(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        if frame.ndim != 3:
            raise ValueError("Expected HxWxC image")
        small = frame.astype(np.float32) / 255.0
        return {"p2": small, "p3": small[::2, ::2], "p4": small[::4, ::4]}
