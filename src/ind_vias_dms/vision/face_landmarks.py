from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.vision.face_proposals import FaceProposal, FaceProposalDetector, expand_box
from ind_vias_dms.vision.nir_preprocess import preprocess_for_face_detection


MEDIAPIPE_REQUIRED_MESSAGE = (
    "MediaPipe is required for the default DMS v0.1 face backend. "
    "Install using: pip install mediapipe"
)


@dataclass
class FaceQualityResult:
    proposal_confidence: float = 0.0
    landmark_count: int = 0
    landmark_coverage_score: float = 0.0
    face_box_area_norm: float = 0.0
    face_aspect_ratio: float = 0.0
    left_eye_visible: bool = False
    right_eye_visible: bool = False
    nose_visible: bool = False
    mouth_visible: bool = False
    chin_visible: bool = False
    both_eyes_available: bool = False
    face_completeness_score: float = 0.0
    is_partial_face: bool = True
    is_side_profile: bool = False
    is_valid_driver_face: bool = False
    validation_state: str = "PROPOSAL_ONLY"
    rejection_reason_codes: list[str] | None = None


@dataclass
class FaceLandmarkResult:
    face_found: bool
    bbox: tuple[int, int, int, int] | None = None
    landmarks_px: dict[int, tuple[float, float]] | None = None
    confidence: float = 0.0
    center: tuple[float, float] | None = None
    area: float = 0.0
    box_norm: tuple[float, float, float, float] | None = None
    quality: FaceQualityResult | None = None


class FaceLandmarkBackend:
    def __init__(
        self,
        backend: str = "mediapipe",
        max_num_faces: int = 1,
        config: DMSConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or DMSConfig(face_backend=backend, max_num_faces=max_num_faces)
        self._face_mesh: Any | None = None
        self._proposal_detector: FaceProposalDetector | None = None
        self.last_proposals: list[FaceProposal] = []
        self.last_backend_used = "FaceMesh direct"
        self.last_nir_mode = "BGR"
        self.last_face_mesh_attempt_count = 0
        self.last_face_mesh_success_attempt = "NONE"
        self.last_face_mesh_failure_reason = "NONE"
        self.last_crop_margin_used = 0.0
        self.last_crop_upscale_used = 1.0
        self.last_crop_bbox_norm: tuple[float, float, float, float] | None = None
        if backend != "mediapipe":
            raise ValueError(f"Unsupported DMS face backend: {backend}")
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            raise RuntimeError(MEDIAPIPE_REQUIRED_MESSAGE) from exc
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._proposal_detector = FaceProposalDetector(self.config)

    def process(self, frame_bgr: np.ndarray) -> FaceLandmarkResult:
        faces = self.process_all(frame_bgr)
        return faces[0] if faces else FaceLandmarkResult(face_found=False)

    def process_all(self, frame_bgr: np.ndarray) -> list[FaceLandmarkResult]:
        self.last_face_mesh_attempt_count = 0
        self.last_face_mesh_success_attempt = "NONE"
        self.last_face_mesh_failure_reason = "NONE"
        self.last_crop_margin_used = 0.0
        self.last_crop_upscale_used = 1.0
        self.last_crop_bbox_norm = None
        preprocessed = preprocess_for_face_detection(frame_bgr, self.config)
        detection_frame = preprocessed.frame_bgr
        self.last_nir_mode = preprocessed.mode
        self.last_proposals = []
        if self.config.face_mesh_on_crops and self._proposal_detector is not None:
            self.last_proposals = self._proposal_detector.detect(detection_frame)
            if self.last_proposals:
                faces = self._process_proposal_crops(detection_frame, frame_bgr.shape, self.last_proposals)
                if faces:
                    self.last_backend_used = "FaceDetection crop"
                    return faces
                if self.config.face_mesh_full_frame_fallback:
                    fallback_faces = self._process_full_frame(detection_frame)
                    self.last_face_mesh_attempt_count += 1
                    if fallback_faces:
                        self.last_backend_used = "FaceMesh full-frame fallback"
                        self.last_face_mesh_success_attempt = "FULL_FRAME_FALLBACK"
                        return fallback_faces
                self.last_backend_used = "FaceDetection proposal"
                if self.last_face_mesh_failure_reason == "NONE":
                    self.last_face_mesh_failure_reason = "FACE_MESH_FAILED_ALL_RETRIES"
                return [
                    self._proposal_only_result(proposal, frame_bgr.shape)
                    for proposal in self.last_proposals
                    if self._proposal_area_ok(proposal, frame_bgr.shape)
                ]

        faces = self._process_full_frame(detection_frame)
        self.last_backend_used = "FaceMesh direct"
        return faces

    def _process_full_frame(self, frame_bgr: np.ndarray) -> list[FaceLandmarkResult]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return []
        height, width = frame_bgr.shape[:2]
        observations = []
        for face in result.multi_face_landmarks:
            observation = self._landmark_result(
                face.landmark,
                frame_bgr.shape,
                (width, height),
                (0, 0),
                1.0,
                0.85,
            )
            if self._face_area_ok(observation, frame_bgr.shape):
                observations.append(observation)
        return observations

    def _process_proposal_crops(
        self,
        detection_frame: np.ndarray,
        original_shape: tuple[int, int, int],
        proposals: list[FaceProposal],
    ) -> list[FaceLandmarkResult]:
        faces: list[FaceLandmarkResult] = []
        for proposal in proposals:
            if not self._proposal_area_ok(proposal, original_shape):
                continue
            margins = list(self.config.face_mesh_retry_crop_margins) if self.config.face_mesh_retry_enabled else [self.config.face_crop_margin]
            upscales = list(self.config.face_mesh_retry_upscale_factors) if self.config.face_mesh_retry_enabled else [self.config.face_crop_upscale]
            if self.config.face_crop_margin not in margins:
                margins.insert(0, self.config.face_crop_margin)
            if self.config.face_crop_upscale not in upscales:
                upscales.insert(0, self.config.face_crop_upscale)
            for margin in margins:
                x1, y1, x2, y2 = expand_box(proposal.bbox, detection_frame.shape, float(margin))
                if self.config.face_mesh_retry_square_crop:
                    x1, y1, x2, y2 = self._square_box((x1, y1, x2, y2), detection_frame.shape)
                if min(x2 - x1, y2 - y1) < self.config.face_crop_min_size_px:
                    self.last_face_mesh_failure_reason = "FACE_CROP_TOO_SMALL"
                    continue
                crop = detection_frame[y1:y2, x1:x2]
                if crop.size == 0:
                    self.last_face_mesh_failure_reason = "EMPTY_FACE_CROP"
                    continue
                if self.config.face_mesh_retry_clahe:
                    crop = self._enhance_crop(crop)
                for scale_value in upscales:
                    scale = max(1.0, float(scale_value))
                    self.last_face_mesh_attempt_count += 1
                    crop_for_mesh = cv2.resize(
                        crop,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_CUBIC,
                    )
                    rgb = cv2.cvtColor(crop_for_mesh, cv2.COLOR_BGR2RGB)
                    result = self._face_mesh.process(rgb)
                    if not result.multi_face_landmarks:
                        self.last_face_mesh_failure_reason = "FACE_MESH_FAILED_ALL_RETRIES"
                        continue
                    face = result.multi_face_landmarks[0]
                    height, width = original_shape[:2]
                    self.last_face_mesh_success_attempt = "FACE_MESH_CROP_RETRY"
                    self.last_crop_margin_used = float(margin)
                    self.last_crop_upscale_used = scale
                    self.last_crop_bbox_norm = (x1 / width, y1 / height, x2 / width, y2 / height)
                    faces.append(
                        self._landmark_result(
                            face.landmark,
                            original_shape,
                            (crop_for_mesh.shape[1], crop_for_mesh.shape[0]),
                            (x1, y1),
                            scale,
                            max(0.65, proposal.confidence),
                        )
                    )
                    break
                if faces:
                    break
        return faces

    @staticmethod
    def _square_box(
        box: tuple[int, int, int, int],
        frame_shape: tuple[int, int, int],
    ) -> tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = box
        size = max(x2 - x1, y2 - y1)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half = size // 2
        return (
            max(0, cx - half),
            max(0, cy - half),
            min(width - 1, cx + half),
            min(height - 1, cy + half),
        )

    @staticmethod
    def _enhance_crop(crop_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(l_chan)
        return cv2.cvtColor(cv2.merge((enhanced, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

    def _landmark_result(
        self,
        landmark_list: Any,
        frame_shape: tuple[int, int, int],
        source_size: tuple[int, int],
        offset: tuple[int, int],
        scale: float,
        confidence: float,
    ) -> FaceLandmarkResult:
        height, width = frame_shape[:2]
        source_width, source_height = source_size
        ox, oy = offset
        landmarks: dict[int, tuple[float, float]] = {}
        for idx, lm in enumerate(landmark_list):
            x = (lm.x * source_width) / scale + ox
            y = (lm.y * source_height) / scale + oy
            landmarks[idx] = (x, y)
        xs = [point[0] for point in landmarks.values()]
        ys = [point[1] for point in landmarks.values()]
        x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
        x2, y2 = min(width - 1, int(max(xs))), min(height - 1, int(max(ys)))
        area = float(max(0, x2 - x1) * max(0, y2 - y1))
        result = FaceLandmarkResult(
            face_found=True,
            bbox=(x1, y1, x2, y2),
            landmarks_px=landmarks,
            confidence=confidence,
            center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            area=area,
            box_norm=(x1 / width, y1 / height, x2 / width, y2 / height),
        )
        result.quality = evaluate_face_quality(result, frame_shape, self.config)
        return result

    def _proposal_only_result(
        self,
        proposal: FaceProposal,
        frame_shape: tuple[int, int, int],
    ) -> FaceLandmarkResult:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = proposal.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        area = float(max(0, x2 - x1) * max(0, y2 - y1))
        result = FaceLandmarkResult(
            face_found=True,
            bbox=(x1, y1, x2, y2),
            landmarks_px=None,
            confidence=proposal.confidence,
            center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            area=area,
            box_norm=(x1 / width, y1 / height, x2 / width, y2 / height),
        )
        result.quality = evaluate_face_quality(result, frame_shape, self.config)
        return result

    def _proposal_area_ok(self, proposal: FaceProposal, frame_shape: tuple[int, int, int]) -> bool:
        height, width = frame_shape[:2]
        return proposal.area / float(max(1, width * height)) >= self.config.min_face_box_area_norm

    def _face_area_ok(self, face: FaceLandmarkResult, frame_shape: tuple[int, int, int]) -> bool:
        height, width = frame_shape[:2]
        return face.area / float(max(1, width * height)) >= self.config.min_face_box_area_norm

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
        if self._proposal_detector is not None:
            self._proposal_detector.close()


def evaluate_face_quality(
    face: FaceLandmarkResult,
    frame_shape: tuple[int, int, int],
    config: DMSConfig,
) -> FaceQualityResult:
    height, width = frame_shape[:2]
    frame_area = float(max(1, width * height))
    box = face.box_norm
    if box is None or face.bbox is None:
        return FaceQualityResult(
            proposal_confidence=face.confidence,
            rejection_reason_codes=["FACE_PROPOSAL_NOT_VALIDATED"],
        )
    x1, y1, x2, y2 = box
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    area_norm = face.area / frame_area
    aspect = box_w / max(box_h, 1e-6)
    landmarks = face.landmarks_px or {}
    landmark_count = len(landmarks)
    left_eye = any(idx in landmarks for idx in (33, 133, 159, 145, 468))
    right_eye = any(idx in landmarks for idx in (263, 362, 386, 374, 473))
    nose = any(idx in landmarks for idx in (1, 2, 4, 5))
    mouth = any(idx in landmarks for idx in (13, 14, 61, 291))
    chin = any(idx in landmarks for idx in (152, 199, 200))
    lower = mouth or chin
    coverage = 0.0
    if landmarks:
        xs = [point[0] / max(1, width) for point in landmarks.values()]
        ys = [point[1] / max(1, height) for point in landmarks.values()]
        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)
        coverage = min(1.0, (spread_x / max(box_w, 1e-6) + spread_y / max(box_h, 1e-6)) / 2.0)
    completeness_parts = [left_eye, right_eye, nose, mouth, chin]
    completeness = sum(1.0 for item in completeness_parts if item) / len(completeness_parts)
    is_eye_only = (left_eye or right_eye) and not nose and not lower
    is_ear_only = landmark_count < max(20, config.driver_min_landmark_count // 3) and not nose and not mouth
    is_partial = completeness < config.driver_min_face_completeness_score or coverage < config.driver_min_landmark_coverage_score
    reasons: list[str] = []
    if landmark_count == 0:
        reasons.append("FACE_PROPOSAL_NOT_VALIDATED")
    if landmark_count < config.driver_min_landmark_count:
        reasons.append("LOW_LANDMARK_COVERAGE")
    if area_norm < config.driver_min_face_area_norm:
        reasons.append("FACE_BOX_TOO_SMALL")
    if aspect < config.driver_face_aspect_min or aspect > config.driver_face_aspect_max:
        reasons.append("FACE_ASPECT_INVALID")
    if is_eye_only:
        reasons.append("EYE_ONLY_CROP_REJECTED")
    if is_ear_only:
        reasons.append("EAR_ONLY_CROP_REJECTED")
    if is_partial:
        reasons.append("PARTIAL_FACE_CROP")
    valid = (
        landmark_count >= config.driver_min_landmark_count
        and area_norm >= config.driver_min_face_area_norm
        and config.driver_face_aspect_min <= aspect <= config.driver_face_aspect_max
        and nose
        and (left_eye or right_eye)
        and lower
        and completeness >= config.driver_min_face_completeness_score
        and coverage >= config.driver_min_landmark_coverage_score
        and not is_eye_only
        and not is_ear_only
    )
    return FaceQualityResult(
        proposal_confidence=face.confidence,
        landmark_count=landmark_count,
        landmark_coverage_score=coverage,
        face_box_area_norm=area_norm,
        face_aspect_ratio=aspect,
        left_eye_visible=left_eye,
        right_eye_visible=right_eye,
        nose_visible=nose,
        mouth_visible=mouth,
        chin_visible=chin,
        both_eyes_available=left_eye and right_eye,
        face_completeness_score=completeness,
        is_partial_face=not valid,
        is_side_profile=(left_eye != right_eye) and nose and lower,
        is_valid_driver_face=valid,
        validation_state="VALIDATED" if valid else ("PARTIAL_FACE" if landmark_count else "PROPOSAL_ONLY"),
        rejection_reason_codes=list(dict.fromkeys(reasons)),
    )
