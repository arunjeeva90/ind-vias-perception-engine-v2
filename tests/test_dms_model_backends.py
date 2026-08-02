from __future__ import annotations

import numpy as np

from ind_vias_dms.core.config import DMSConfig
from ind_vias_dms.vision.eye_crop import (
    LANDMARK_106_IMAGE_LEFT_EYE,
    LANDMARK_106_IMAGE_RIGHT_EYE,
    aligned_eye_crop,
    eye_corners_from_106,
)
from ind_vias_dms.vision.eye_state import EyeStateEstimator, _crop_eye
from ind_vias_dms.vision.landmark_106 import (
    Landmark106Result,
    compare_eye_geometry,
    create_landmark_106_backend,
    decode_landmark_106,
    loose_face_crop,
)
from ind_vias_dms.vision.onnx_classifier import (
    ClassifierPrediction,
    ONNXImageClassifier,
)
from ind_vias_dms.vision.seatbelt import (
    SeatbeltDetectionPlaceholder,
    face_to_torso_box,
)


def test_onnx_classifier_disabled_and_preprocessing_contract():
    classifier = ONNXImageClassifier(
        {
            "enabled": False,
            "input_size": 96,
            "class_names": ["eye_closed", "eye_open"],
        }
    )
    tensor = classifier.preprocess(np.full((40, 60, 3), 127, dtype=np.uint8))

    assert classifier.backend_status == "DISABLED"
    assert tensor.shape == (1, 3, 96, 96)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


def test_onnx_classifier_missing_model_is_explicit_and_safe(tmp_path):
    classifier = ONNXImageClassifier(
        {
            "enabled": True,
            "model_path": str(tmp_path / "missing.onnx"),
            "class_names": ["a", "b"],
            "allow_missing_model": True,
        }
    )

    prediction = classifier.predict(np.zeros((32, 32, 3), dtype=np.uint8))

    assert not classifier.ready
    assert prediction.backend_status == "MODEL_MISSING"


def test_onnx_classifier_corrupt_model_is_explicit_and_safe(tmp_path):
    corrupt = tmp_path / "corrupt.onnx"
    corrupt.write_bytes(b"not an ONNX graph")

    classifier = ONNXImageClassifier(
        {
            "enabled": True,
            "model_path": str(corrupt),
            "class_names": ["a", "b"],
            "allow_missing_model": True,
        }
    )

    assert not classifier.ready
    assert classifier.backend_status == "MODEL_LOAD_FAILED"


def test_onnx_classifier_uses_checkpoint_metadata_class_order(tmp_path):
    metadata = tmp_path / "model.metadata.json"
    metadata.write_text(
        '{"class_to_idx":{"eye_open":1,"eye_closed":0},"img_size":96}',
        encoding="utf-8",
    )

    classifier = ONNXImageClassifier(
        {
            "enabled": False,
            "metadata_path": str(metadata),
            "class_names": ["wrong", "order"],
        }
    )

    assert classifier.class_names == ["eye_closed", "eye_open"]
    assert classifier.input_width == classifier.input_height == 96


def test_eye_state_preserves_ear_fallback_when_classifier_disabled():
    landmarks = {
        33: (10.0, 10.0),
        160: (13.0, 8.0),
        158: (17.0, 8.0),
        133: (20.0, 10.0),
        153: (17.0, 12.0),
        144: (13.0, 12.0),
        362: (30.0, 10.0),
        385: (33.0, 8.0),
        387: (37.0, 8.0),
        263: (40.0, 10.0),
        373: (37.0, 12.0),
        380: (33.0, 12.0),
    }

    state = EyeStateEstimator(0.21).estimate(landmarks)

    assert state.classification_source == "LANDMARK_EAR"
    assert state.model_backend_status == "DISABLED"
    assert state.openness > 0.21
    assert not state.is_closed


def test_eye_crop_clamps_to_frame_and_rejects_degenerate_geometry():
    frame = np.zeros((40, 50, 3), dtype=np.uint8)
    valid = [(1.0, 8.0), (3.0, 5.0), (7.0, 5.0), (12.0, 8.0), (7.0, 11.0), (3.0, 11.0)]
    degenerate = [(1.0, 1.0)] * 6

    crop = _crop_eye(frame, valid, padding_x=0.5, padding_y=1.0)

    assert crop is not None
    assert crop.ndim == 3
    assert _crop_eye(frame, degenerate, 0.5, 1.0) is None


def test_aligned_eye_crop_matches_reviewed_96_square_contract():
    checker = np.indices((120, 160)).sum(axis=0) % 2
    frame = np.repeat((checker * 255).astype(np.uint8)[..., None], 3, axis=2)

    result = aligned_eye_crop(
        frame,
        (40.0, 55.0),
        (80.0, 65.0),
        min_eye_width=18.0,
        min_blur=1.0,
    )

    assert result.valid
    assert result.reason == "OK"
    assert result.image is not None
    assert result.image.shape == (96, 96, 3)
    assert result.eye_width > 41.0
    assert 13.0 < result.rotation_degrees < 15.0
    assert result.padding_fraction == 0.0


def test_aligned_eye_crop_abstains_on_bad_geometry_and_exposure():
    dark = np.zeros((100, 100, 3), dtype=np.uint8)

    tiny = aligned_eye_crop(dark, (10.0, 10.0), (20.0, 10.0))
    too_dark = aligned_eye_crop(
        dark,
        (25.0, 50.0),
        (65.0, 50.0),
        min_blur=0.0,
    )

    assert not tiny.valid
    assert tiny.reason == "EYE_TOO_SMALL"
    assert not too_dark.valid
    assert too_dark.reason == "TOO_DARK"


def test_landmark_106_eye_groups_are_complete_and_exclude_eyebrows():
    assert LANDMARK_106_IMAGE_LEFT_EYE == tuple(range(33, 43))
    assert LANDMARK_106_IMAGE_RIGHT_EYE == tuple(range(87, 97))
    assert not set(LANDMARK_106_IMAGE_RIGHT_EYE) & set(range(97, 106))

    points = np.zeros((106, 2), dtype=np.float32)
    for offset, index in enumerate(LANDMARK_106_IMAGE_RIGHT_EYE):
        points[index] = (20.0 + offset, 50.0 + (offset % 2))
    corners = eye_corners_from_106(points, LANDMARK_106_IMAGE_RIGHT_EYE)

    assert corners is not None
    assert corners[0][0] == 20.0
    assert corners[1][0] == 29.0


def test_landmark_106_loose_crop_and_decode_round_trip():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    _, matrix = loose_face_crop(frame, (80.0, 40.0, 240.0, 200.0))
    source = np.tile(np.asarray([[160.0, 120.0]], dtype=np.float32), (106, 1))
    homogeneous = np.column_stack([source, np.ones(106, dtype=np.float32)])
    aligned = (matrix @ homogeneous.T).T
    raw = aligned / 96.0 - 1.0

    decoded = decode_landmark_106(raw.reshape(1, 212), matrix)

    assert decoded is not None
    assert np.allclose(decoded, source, atol=1e-4)


def test_landmark_106_geometry_agreement_is_scale_normalized():
    points = np.zeros((106, 2), dtype=np.float32)
    for index in LANDMARK_106_IMAGE_LEFT_EYE:
        points[index] = (20.0 + index - 33, 50.0)
    for index in LANDMARK_106_IMAGE_RIGHT_EYE:
        points[index] = (80.0 + index - 87, 50.0)

    agreement = compare_eye_geometry(
        ((80.0, 50.0), (89.0, 50.0)),
        ((20.0, 50.0), (29.0, 50.0)),
        points,
    )

    assert agreement.valid
    assert agreement.status == "OK"
    assert agreement.left_normalized_error == 0.0
    assert agreement.right_normalized_error == 0.0


def test_disabled_rknn_106_backend_does_not_touch_runtime():
    backend = create_landmark_106_backend(
        {
            "landmark_106_enabled": False,
            "landmark_106_backend": "rknn",
        }
    )

    assert backend.backend_name == "RKNN_NPU"
    assert backend.backend_status == "DISABLED"
    assert backend.npu_active is False


class _FakeLandmark106Backend:
    enabled = True
    ready = True
    backend_name = "RKNN_NPU"
    backend_status = "OK"
    npu_active = True

    def __init__(self, points):
        self.points = points

    def infer(self, _frame, _bbox):
        return Landmark106Result(
            points_px=self.points,
            backend_status="OK",
            inference_ms=7.5,
        )

    def close(self):
        pass


def test_106_geometry_can_add_confidence_while_eye_cnn_is_disabled():
    points = np.zeros((106, 2), dtype=np.float32)
    for offset, index in enumerate(range(87, 97)):
        points[index] = (80.0 + offset * (40.0 / 9.0), 50.0)
    for offset, index in enumerate(range(33, 43)):
        points[index] = (20.0 + offset * (40.0 / 9.0), 50.0)
    estimator = EyeStateEstimator(0.21)
    estimator.landmark_106 = _FakeLandmark106Backend(points)

    state = estimator.estimate(
        _wide_open_landmarks(),
        np.full((140, 150, 3), 127, dtype=np.uint8),
        (10, 20, 130, 130),
    )

    assert state.model_backend_status == "DISABLED"
    assert state.landmark_geometry_agreement is True
    assert state.landmark_106_status == "RKNN_NPU_OK"
    assert state.landmark_106_inference_ms == 7.5
    assert state.classification_source == "LANDMARK_EAR_RKNN_NPU_AGREEMENT"
    assert state.confidence >= 0.90


class _FakeEyeClassifier:
    ready = True
    backend_status = "OK"
    input_width = 96

    def __init__(self, predictions):
        self.predictions = iter(predictions)

    def predict(self, _crop):
        return next(self.predictions)


def _eye_prediction(closed: float) -> ClassifierPrediction:
    return ClassifierPrediction(
        label="eye_closed" if closed > 0.5 else "eye_open",
        confidence=max(closed, 1.0 - closed),
        probabilities={"eye_closed": closed, "eye_open": 1.0 - closed},
        backend_status="OK",
    )


def _wide_open_landmarks() -> dict[int, tuple[float, float]]:
    return {
        33: (20.0, 50.0),
        160: (30.0, 42.0),
        158: (40.0, 42.0),
        133: (60.0, 50.0),
        153: (40.0, 58.0),
        144: (30.0, 58.0),
        362: (80.0, 50.0),
        385: (90.0, 42.0),
        387: (100.0, 42.0),
        263: (120.0, 50.0),
        373: (100.0, 58.0),
        380: (90.0, 58.0),
    }


def test_eye_fusion_requires_bilateral_and_geometry_agreement():
    checker = np.indices((140, 150)).sum(axis=0) % 2
    frame = np.repeat((checker * 255).astype(np.uint8)[..., None], 3, axis=2)
    estimator = EyeStateEstimator(0.21)
    estimator.classifier = _FakeEyeClassifier(
        [_eye_prediction(0.05), _eye_prediction(0.08)]
    )
    estimator.eye_crop_min_blur = 0.0

    state = estimator.estimate(_wide_open_landmarks(), frame)

    assert not state.is_closed
    assert state.left_eye_state == "OPEN"
    assert state.right_eye_state == "OPEN"
    assert state.confidence > 0.9
    assert state.classification_source == "ONNX_BILATERAL_GEOMETRY_AGREEMENT"


def test_eye_fusion_abstains_on_bilateral_disagreement():
    checker = np.indices((140, 150)).sum(axis=0) % 2
    frame = np.repeat((checker * 255).astype(np.uint8)[..., None], 3, axis=2)
    estimator = EyeStateEstimator(0.21)
    estimator.classifier = _FakeEyeClassifier(
        [_eye_prediction(0.05), _eye_prediction(0.95)]
    )
    estimator.eye_crop_min_blur = 0.0
    estimator.geometry_agreement_enabled = False

    state = estimator.estimate(_wide_open_landmarks(), frame)

    assert state.confidence == 0.0
    assert state.classification_source == "UNKNOWN_BILATERAL_DISAGREEMENT"


def test_handoff_face_to_torso_geometry_is_bounded_and_unknown_safe():
    box = face_to_torso_box((250, 50, 350, 150), (480, 640, 3))

    assert box is not None
    x1, y1, x2, y2 = box
    assert 0 <= x1 < x2 <= 640
    assert 0 <= y1 < y2 <= 480
    assert face_to_torso_box((1, 1, 5, 5), (100, 100, 3)) is None


class _FakeSeatbeltClassifier:
    ready = True
    backend_status = "OK"

    def predict(self, _crop):
        return ClassifierPrediction(
            label="seat_belt_on",
            confidence=0.95,
            probabilities={"no_seat_belt": 0.05, "seat_belt_on": 0.95},
            backend_status="OK",
        )


def test_seatbelt_classifier_requires_temporal_confirmation():
    config = DMSConfig(
        seatbelt_detection={
            "enabled": True,
            "class_names": ["no_seat_belt", "seat_belt_on"],
            "confirm_ms": 100,
            "min_blur": 0.0,
            "min_brightness": 0.0,
            "max_brightness": 255.0,
        }
    )
    detector = SeatbeltDetectionPlaceholder(config)
    detector.classifier = _FakeSeatbeltClassifier()
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    face = (250, 50, 350, 150)

    candidate = detector.process(frame, face, timestamp_ms=1000)
    confirmed = detector.process(frame, face, timestamp_ms=1100)

    assert candidate.final_state == "UNKNOWN"
    assert candidate.visual_belt_path == "CANDIDATE"
    assert confirmed.final_state == "WORN"
    assert confirmed.visual_belt_path == "WORN"


def test_seatbelt_missing_backend_returns_unknown():
    detector = SeatbeltDetectionPlaceholder(DMSConfig())

    result = detector.process(
        np.zeros((480, 640, 3), dtype=np.uint8),
        (250, 50, 350, 150),
        timestamp_ms=0,
    )

    assert result.final_state == "UNKNOWN"
    assert detector.last_status == "DISABLED"
