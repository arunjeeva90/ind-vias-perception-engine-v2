from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from rknnlite.api import RKNNLite


EYENET_CLASSES = [
    "bad_crop",
    "eye_closed",
    "eye_open",
]


class EyeNetRKNNLiteClassifier:
    """
    RKNNLite eye-state classifier.

    Input:
        BGR or RGB eye crop image

    Output:
        class name, confidence, probability dictionary
    """

    def __init__(
        self,
        model_path: str,
        img_size: int = 96,
        input_color: str = "bgr",
        enhance: bool = True,
        gamma: float = 0.55,
        brightness_beta: int = 30,
        contrast_alpha: float = 1.35,
        clahe_clip: float = 3.0,
    ):
        self.model_path = Path(model_path)
        self.img_size = img_size
        self.input_color = input_color.lower()

        self.enhance = enhance
        self.gamma = gamma
        self.brightness_beta = brightness_beta
        self.contrast_alpha = contrast_alpha
        self.clahe_clip = clahe_clip

        if not self.model_path.exists():
            raise FileNotFoundError(f"EyeNet RKNN model not found: {self.model_path}")

        self.rknn = RKNNLite()

        ret = self.rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {self.model_path}")

        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("Failed to initialize RKNNLite runtime")

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    def _enhance_eye_crop(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance only the eye crop.

        Useful for backlight/window glare cases:
        - face appears dark
        - eyes have low contrast
        - glasses reflection reduces eyelid visibility
        """

        if img is None or img.size == 0:
            return img

        # Mild contrast + brightness.
        img = cv2.convertScaleAbs(
            img,
            alpha=float(self.contrast_alpha),
            beta=int(self.brightness_beta),
        )

        # CLAHE on luminance channel.
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=float(self.clahe_clip),
            tileGridSize=(4, 4),
        )
        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Gamma < 1 brightens shadows.
        gamma = max(0.20, float(self.gamma))

        table = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)]
        ).astype("uint8")

        img = cv2.LUT(img, table)

        return img

    def preprocess(self, eye_crop: np.ndarray) -> np.ndarray:
        if eye_crop is None or eye_crop.size == 0:
            raise ValueError("Invalid empty eye crop")

        img = eye_crop.copy()

        if self.input_color == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.enhance:
            img = self._enhance_eye_crop(img)

        img = cv2.resize(
            img,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_LINEAR,
        )

        # RKNN input is NHWC uint8.
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, eye_crop: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        inp = self.preprocess(eye_crop)

        outputs = self.rknn.inference(
            inputs=[inp],
            data_format=["nhwc"],
        )

        logits = outputs[0].reshape(-1)
        probs = self._softmax(logits)

        pred_idx = int(np.argmax(probs))
        pred_class = EYENET_CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        prob_dict = {
            cls: float(prob)
            for cls, prob in zip(EYENET_CLASSES, probs)
        }

        return pred_class, confidence, prob_dict

    def release(self):
        self.rknn.release()
