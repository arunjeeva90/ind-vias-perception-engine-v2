from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import cv2
import numpy as np

from ind_vias_dms.vision.eye_crop import (
    LANDMARK_106_IMAGE_LEFT_EYE,
    LANDMARK_106_IMAGE_RIGHT_EYE,
    eye_corners_from_106,
)


@dataclass(frozen=True)
class Landmark106Result:
    points_px: np.ndarray | None = None
    backend_status: str = "DISABLED"
    inference_ms: float = 0.0


@dataclass(frozen=True)
class EyeGeometryAgreement:
    valid: bool
    status: str
    left_normalized_error: float = 0.0
    right_normalized_error: float = 0.0


class ONNXLandmark106:
    """Optional InsightFace 2d106det adapter for internal geometry evidence.

    The pretrained weights remain internal-PoC-only unless separately licensed.
    This backend is disabled by default and is not a second eye-state classifier.
    """

    def __init__(self, config: dict[str, Any] | None) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("landmark_106_enabled", False))
        self.model_path = Path(
            str(cfg.get("landmark_106_model_path", "models/dms/landmark_106.onnx"))
        )
        self.allow_missing = bool(cfg.get("allow_missing_model", True))
        self.session: Any | None = None
        self.input_name = ""
        self.output_names: list[str] = []
        self.input_size = (192, 192)
        self.input_mean = 0.0
        self.input_std = 1.0
        self.backend_status = "DISABLED" if not self.enabled else "MODEL_MISSING"
        if self.enabled:
            self._load()

    backend_name = "ONNX_CPU"
    npu_active = False

    @property
    def ready(self) -> bool:
        return self.session is not None and self.backend_status == "OK"

    def infer(
        self,
        frame: np.ndarray,
        face_bbox: Sequence[float] | None,
    ) -> Landmark106Result:
        if not self.enabled:
            return Landmark106Result(backend_status="DISABLED")
        if not self.ready:
            return Landmark106Result(backend_status=self.backend_status)
        if frame is None or frame.size == 0 or face_bbox is None or len(face_bbox) != 4:
            return Landmark106Result(backend_status="INVALID_INPUT")

        try:
            aligned, matrix = loose_face_crop(
                frame,
                face_bbox,
                output_size=self.input_size[0],
                loose_scale=1.5,
            )
            blob = cv2.dnn.blobFromImage(
                aligned,
                scalefactor=1.0 / self.input_std,
                size=self.input_size,
                mean=(self.input_mean, self.input_mean, self.input_mean),
                swapRB=True,
            )
            started = perf_counter()
            raw = self.session.run(
                self.output_names,
                {self.input_name: blob},
            )[0]
            elapsed_ms = (perf_counter() - started) * 1000.0
            points = decode_landmark_106(raw, matrix, self.input_size[0])
        except (cv2.error, ValueError, RuntimeError):
            self.backend_status = "INFERENCE_ERROR"
            return Landmark106Result(backend_status=self.backend_status)

        if points is None:
            return Landmark106Result(
                backend_status="INVALID_OUTPUT",
                inference_ms=elapsed_ms,
            )
        self.backend_status = "OK"
        return Landmark106Result(
            points_px=points,
            backend_status="OK",
            inference_ms=elapsed_ms,
        )

    def _load(self) -> None:
        if not self.model_path.is_file():
            if not self.allow_missing:
                raise FileNotFoundError(
                    f"106-point landmark model not found: {self.model_path}"
                )
            return
        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
            model_input = self.session.get_inputs()[0]
            self.input_name = model_input.name
            height, width = int(model_input.shape[2]), int(model_input.shape[3])
            self.input_size = (width, height)
            self.output_names = [item.name for item in self.session.get_outputs()]
            self.input_mean, self.input_std = _insightface_input_normalization(
                self.model_path
            )
            self.backend_status = "OK"
        except (ImportError, OSError, ValueError, RuntimeError):
            self.session = None
            self.backend_status = "MODEL_LOAD_FAILED"
            if not self.allow_missing:
                raise

    def close(self) -> None:
        self.session = None


class RKNNLandmark106:
    """Driver-only InsightFace 106-point adapter for RK3588 RKNNLite.

    The backend is deliberately fail-safe: a missing toolkit, kernel driver, or
    model leaves the existing MediaPipe/EAR path active and reports the exact
    backend state instead of falling back silently.
    """

    backend_name = "RKNN_NPU"

    def __init__(self, config: dict[str, Any] | None) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("landmark_106_enabled", False))
        self.model_path = Path(
            str(
                cfg.get(
                    "landmark_106_rknn_model_path",
                    "models/dms/landmark_106_rk3588.rknn",
                )
            )
        )
        self.allow_missing = bool(cfg.get("allow_missing_model", True))
        self.rknnlite_site_packages = Path(
            str(
                cfg.get(
                    "landmark_106_rknnlite_site_packages",
                    ".venv-rknn/lib/python3.10/site-packages",
                )
            )
        )
        self.input_size = (192, 192)
        self.rknn: Any | None = None
        self.backend_status = "DISABLED" if not self.enabled else "MODEL_MISSING"
        self.npu_active = False
        if self.enabled:
            self._load()

    @property
    def ready(self) -> bool:
        return (
            self.rknn is not None
            and self.backend_status == "OK"
            and self.npu_active
        )

    def infer(
        self,
        frame: np.ndarray,
        face_bbox: Sequence[float] | None,
    ) -> Landmark106Result:
        if not self.enabled:
            return Landmark106Result(backend_status="DISABLED")
        if not self.ready:
            return Landmark106Result(backend_status=self.backend_status)
        if frame is None or frame.size == 0 or face_bbox is None or len(face_bbox) != 4:
            return Landmark106Result(backend_status="INVALID_INPUT")

        try:
            aligned, matrix = loose_face_crop(
                frame,
                face_bbox,
                output_size=self.input_size[0],
                loose_scale=1.5,
            )
            # RKNN conversion preserves the model's internal Sub/Mul input
            # normalization. RKNNLite accepts the image as NHWC uint8.
            rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            tensor = np.ascontiguousarray(rgb[None, ...], dtype=np.uint8)
            started = perf_counter()
            outputs = self.rknn.inference(inputs=[tensor])
            elapsed_ms = (perf_counter() - started) * 1000.0
            raw = _select_landmark_106_output(outputs)
            points = (
                decode_landmark_106(raw, matrix, self.input_size[0])
                if raw is not None
                else None
            )
        except (cv2.error, ValueError, RuntimeError, TypeError):
            self.backend_status = "INFERENCE_ERROR"
            return Landmark106Result(backend_status=self.backend_status)

        if points is None:
            return Landmark106Result(
                backend_status="INVALID_OUTPUT",
                inference_ms=elapsed_ms,
            )
        return Landmark106Result(
            points_px=points,
            backend_status="OK",
            inference_ms=elapsed_ms,
        )

    def _load(self) -> None:
        if not self.model_path.is_file():
            if not self.allow_missing:
                raise FileNotFoundError(
                    f"106-point RKNN model not found: {self.model_path}"
                )
            return
        try:
            try:
                from rknnlite.api import RKNNLite
            except ImportError:
                # Append rather than prepend: importing the complete RKNN
                # virtualenv ahead of the DMS environment can replace
                # MediaPipe's compatible protobuf package.
                if self.rknnlite_site_packages.is_dir():
                    site_path = str(self.rknnlite_site_packages.resolve())
                    if site_path not in sys.path:
                        sys.path.append(site_path)
                from rknnlite.api import RKNNLite

            rknn = RKNNLite()
            if rknn.load_rknn(str(self.model_path)) != 0:
                self.backend_status = "MODEL_LOAD_FAILED"
                rknn.release()
                return
            if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
                self.backend_status = "NPU_RUNTIME_UNAVAILABLE"
                rknn.release()
                return
            self.rknn = rknn
            self.backend_status = "OK"
            self.npu_active = True
        except (ImportError, OSError, RuntimeError):
            self.rknn = None
            self.backend_status = "RKNN_RUNTIME_UNAVAILABLE"
            self.npu_active = False
            if not self.allow_missing:
                raise

    def close(self) -> None:
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None
        self.npu_active = False


def create_landmark_106_backend(
    config: dict[str, Any] | None,
) -> ONNXLandmark106 | RKNNLandmark106:
    cfg = dict(config or {})
    backend = str(cfg.get("landmark_106_backend", "onnx")).strip().lower()
    if backend == "rknn":
        return RKNNLandmark106(cfg)
    return ONNXLandmark106(cfg)


def _select_landmark_106_output(outputs: object) -> np.ndarray | None:
    if not isinstance(outputs, (list, tuple)):
        return None
    for output in outputs:
        values = np.asarray(output, dtype=np.float32)
        if values.size == 212:
            return values
    return None


def loose_face_crop(
    frame: np.ndarray,
    face_bbox: Sequence[float],
    *,
    output_size: int = 192,
    loose_scale: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the affine crop used by InsightFace's 2d106det adapter."""

    x1, y1, x2, y2 = (float(value) for value in face_bbox)
    width, height = x2 - x1, y2 - y1
    if width <= 1.0 or height <= 1.0 or output_size <= 0 or loose_scale <= 0:
        raise ValueError("invalid face bounding box")
    center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    scale = output_size / (max(width, height) * loose_scale)
    matrix = np.asarray(
        [
            [scale, 0.0, output_size * 0.5 - scale * center_x],
            [0.0, scale, output_size * 0.5 - scale * center_y],
        ],
        dtype=np.float64,
    )
    aligned = cv2.warpAffine(
        frame,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return aligned, matrix


def decode_landmark_106(
    raw: np.ndarray,
    face_transform: np.ndarray,
    input_size: int = 192,
) -> np.ndarray | None:
    values = np.asarray(raw, dtype=np.float32).reshape(-1)
    if values.size != 212 or not np.isfinite(values).all():
        return None
    points = values.reshape(106, 2)
    inside = np.logical_and(points >= -1.25, points <= 1.25)
    if float(np.mean(inside)) < 0.95:
        return None
    aligned = (points + 1.0) * (input_size * 0.5)
    inverse = cv2.invertAffineTransform(np.asarray(face_transform, dtype=np.float64))
    homogeneous = np.column_stack(
        [aligned.astype(np.float64), np.ones(len(aligned), dtype=np.float64)]
    )
    return (inverse @ homogeneous.T).T.astype(np.float32)


def compare_eye_geometry(
    media_left_corners: Sequence[Sequence[float]],
    media_right_corners: Sequence[Sequence[float]],
    landmark_106_points: np.ndarray | None,
    *,
    max_normalized_error: float = 0.35,
) -> EyeGeometryAgreement:
    """Compare two independent landmark estimates without classifying closure."""

    if landmark_106_points is None:
        return EyeGeometryAgreement(False, "LANDMARK_106_UNAVAILABLE")
    left_106 = eye_corners_from_106(
        landmark_106_points,
        LANDMARK_106_IMAGE_RIGHT_EYE,
    )
    right_106 = eye_corners_from_106(
        landmark_106_points,
        LANDMARK_106_IMAGE_LEFT_EYE,
    )
    if left_106 is None or right_106 is None:
        return EyeGeometryAgreement(False, "LANDMARK_106_INVALID_EYES")
    left_error = _corner_pair_error(media_left_corners, left_106)
    right_error = _corner_pair_error(media_right_corners, right_106)
    valid = left_error <= max_normalized_error and right_error <= max_normalized_error
    return EyeGeometryAgreement(
        valid,
        "OK" if valid else "LANDMARK_GEOMETRY_DISAGREEMENT",
        left_error,
        right_error,
    )


def _corner_pair_error(
    reference: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
) -> float:
    ref = np.asarray(reference, dtype=np.float32).reshape(2, 2)
    other = np.asarray(candidate, dtype=np.float32).reshape(2, 2)
    eye_width = float(np.linalg.norm(ref[0] - ref[1]))
    if eye_width <= 1e-6:
        return float("inf")
    direct = float(np.mean(np.linalg.norm(ref - other, axis=1)))
    reverse = float(np.mean(np.linalg.norm(ref - other[::-1], axis=1)))
    return min(direct, reverse) / eye_width


def _insightface_input_normalization(model_path: Path) -> tuple[float, float]:
    """Mirror InsightFace's first-node normalization detection."""

    try:
        import onnx

        graph = onnx.load(str(model_path)).graph
        names = [node.name for node in graph.node[:8]]
        find_sub = any(
            name.startswith("Sub") or name.startswith("_minus") for name in names
        )
        find_mul = any(
            name.startswith("Mul") or name.startswith("_mul") for name in names
        )
        if any(index < 3 and name == "bn_data" for index, name in enumerate(names)):
            find_sub = find_mul = True
        if find_sub and find_mul:
            return 0.0, 1.0
    except (ImportError, OSError, ValueError):
        pass
    return 127.5, 128.0
