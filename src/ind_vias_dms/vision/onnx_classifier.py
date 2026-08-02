from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ClassifierPrediction:
    label: str = "unknown"
    confidence: float = 0.0
    probabilities: dict[str, float] | None = None
    backend_status: str = "NOT_CONFIGURED"


class ONNXImageClassifier:
    """Small static-shape ONNX image classifier with explicit metadata.

    The runtime intentionally owns only preprocessing and inference. Crop
    validity, temporal confirmation, and application-specific UNKNOWN states
    remain in the eye-state and seat-belt modules.
    """

    def __init__(self, config: dict[str, Any] | None) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.model_path = Path(str(cfg.get("model_path", "")))
        self.metadata_path = Path(str(cfg.get("metadata_path", "")))
        self.input_width = int(cfg.get("input_width", cfg.get("input_size", 224)))
        self.input_height = int(cfg.get("input_height", cfg.get("input_size", 224)))
        self.mean = _float_triplet(cfg.get("mean"), IMAGENET_MEAN)
        self.std = _float_triplet(cfg.get("std"), IMAGENET_STD)
        self.class_names = _class_names(cfg)
        self.allow_missing_model = bool(cfg.get("allow_missing_model", True))
        self.net: Any | None = None
        self.backend_status = "DISABLED" if not self.enabled else "MODEL_MISSING"

        if self.metadata_path.is_file():
            self._load_metadata()
        if self.enabled:
            self._load_model()

    @property
    def ready(self) -> bool:
        return self.net is not None and self.backend_status == "OK"

    def predict(self, bgr: np.ndarray) -> ClassifierPrediction:
        if not self.enabled:
            return ClassifierPrediction(backend_status="DISABLED")
        if self.net is None:
            return ClassifierPrediction(backend_status=self.backend_status)
        if bgr is None or bgr.size == 0 or bgr.ndim != 3 or bgr.shape[2] != 3:
            return ClassifierPrediction(backend_status="INVALID_INPUT")

        try:
            tensor = self.preprocess(bgr)
            self.net.setInput(tensor)
            raw = np.asarray(self.net.forward(), dtype=np.float32).reshape(-1)
        except (cv2.error, ValueError):
            self.backend_status = "MODEL_ERROR"
            return ClassifierPrediction(backend_status=self.backend_status)

        if raw.size != len(self.class_names) or raw.size < 2:
            self.backend_status = "OUTPUT_MISMATCH"
            return ClassifierPrediction(backend_status=self.backend_status)
        probabilities = _softmax(raw)
        index = int(np.argmax(probabilities))
        result = {
            label: float(probabilities[position])
            for position, label in enumerate(self.class_names)
        }
        self.backend_status = "OK"
        return ClassifierPrediction(
            label=self.class_names[index],
            confidence=float(probabilities[index]),
            probabilities=result,
            backend_status="OK",
        )

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_AREA if max(bgr.shape[:2]) > max(self.input_height, self.input_width) else cv2.INTER_LINEAR,
        )
        tensor = rgb.astype(np.float32) / 255.0
        tensor = (tensor - np.asarray(self.mean, dtype=np.float32)) / np.asarray(
            self.std, dtype=np.float32
        )
        return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])

    def _load_model(self) -> None:
        if not self.model_path.is_file():
            self.backend_status = "MODEL_MISSING"
            if not self.allow_missing_model:
                raise FileNotFoundError(f"Classifier model not found: {self.model_path}")
            return
        if len(self.class_names) < 2:
            self.backend_status = "CLASS_MAP_MISSING"
            if not self.allow_missing_model:
                raise ValueError("Classifier requires at least two ordered class names")
            return
        try:
            self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self.backend_status = "OK"
        except cv2.error as exc:
            self.backend_status = "MODEL_LOAD_FAILED"
            if not self.allow_missing_model:
                raise RuntimeError(f"Failed to load classifier model: {self.model_path}") from exc

    def _load_metadata(self) -> None:
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return
        class_to_idx = payload.get("class_to_idx", {})
        if isinstance(class_to_idx, dict) and class_to_idx:
            ordered = sorted(class_to_idx.items(), key=lambda item: int(item[1]))
            self.class_names = [str(label) for label, _ in ordered]
        image_size = payload.get("img_size", payload.get("input_size"))
        if isinstance(image_size, int) and image_size > 0:
            self.input_width = self.input_height = image_size
        self.mean = _float_triplet(payload.get("mean"), self.mean)
        self.std = _float_triplet(payload.get("std"), self.std)


def _class_names(config: dict[str, Any]) -> list[str]:
    raw = config.get("class_names", [])
    if isinstance(raw, dict):
        return [str(value) for _, value in sorted(raw.items(), key=lambda item: int(item[0]))]
    if isinstance(raw, (list, tuple)):
        return [str(value) for value in raw]
    return []


def _float_triplet(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(float(item) for item in value)
    return tuple(float(item) for item in default)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponents = np.exp(shifted)
    denominator = float(np.sum(exponents))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return np.full_like(values, 1.0 / max(1, values.size), dtype=np.float32)
    return exponents / denominator
