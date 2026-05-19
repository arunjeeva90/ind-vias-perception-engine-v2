from __future__ import annotations

import numpy as np


class OnnxBackboneAdapter:
    name = "onnx_backbone_adapter"

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None

    def load(self) -> None:
        import onnxruntime as ort
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])

    def forward(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        if self.session is None:
            raise RuntimeError("Call load() before forward()")
        raise NotImplementedError("Map ONNX output names to p2/p3/p4 for your backbone")
