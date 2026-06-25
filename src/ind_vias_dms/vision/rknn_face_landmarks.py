from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


RKNNLITE_REQUIRED_MESSAGE = (
    "rknnlite.api is required for RKNN face landmark inference. "
    "Activate .venv-rknn or install the Rockchip RKNNLite runtime package."
)


class RKNNFaceLandmarkBackend:
    def __init__(self, model_path: str | Path, input_width: int, input_height: int) -> None:
        self.model_path = Path(model_path)
        self.input_width = input_width
        self.input_height = input_height
        self._rknn: Any | None = None
        self._loaded = False

    def load(self) -> None:
        try:
            from rknnlite.api import RKNNLite  # type: ignore
        except ImportError as exc:
            raise RuntimeError(RKNNLITE_REQUIRED_MESSAGE) from exc

        if not self.model_path.exists():
            raise FileNotFoundError(f"RKNN face landmark model not found: {self.model_path}")

        rknn = RKNNLite()
        ret = rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(f"RKNNLite failed to load model {self.model_path}: ret={ret}")

        ret = rknn.init_runtime()
        if ret != 0:
            rknn.release()
            raise RuntimeError(f"RKNNLite failed to initialize runtime: ret={ret}")

        self._rknn = rknn
        self._loaded = True

    def detect(self, frame: np.ndarray, face_bbox: tuple[int, int, int, int] | None = None) -> Any:
        raise NotImplementedError(
            "RKNN face landmark postprocess is not implemented yet. "
            "Select a landmark model and map its outputs to FaceLandmarkResult first."
        )

    def close(self) -> None:
        if self._rknn is not None:
            self._rknn.release()
        self._rknn = None
        self._loaded = False
