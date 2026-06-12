from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ind_vias_dms.core.config import DMSConfig


@dataclass
class NIRPreprocessResult:
    frame_bgr: np.ndarray
    mode: str = "BGR"
    grayscale_like: bool = False


def preprocess_for_face_detection(frame: np.ndarray, config: DMSConfig) -> NIRPreprocessResult:
    bgr = _ensure_bgr(frame)
    grayscale_like = is_grayscale_like(bgr)
    if not config.enable_nir_preprocessing:
        return NIRPreprocessResult(bgr, "BGR", grayscale_like)
    if config.nir_mode.lower() not in {"auto", "on", "nir"}:
        return NIRPreprocessResult(bgr, "BGR", grayscale_like)
    if config.nir_mode.lower() == "auto" and not grayscale_like:
        return NIRPreprocessResult(bgr, "BGR", False)

    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    if config.nir_clahe_enabled:
        tile = max(2, int(config.nir_clahe_tile_grid_size))
        clahe = cv2.createCLAHE(
            clipLimit=float(config.nir_clahe_clip_limit),
            tileGridSize=(tile, tile),
        )
        y = clahe.apply(y)
    elif config.nir_equalize_hist_fallback:
        y = cv2.equalizeHist(y)

    enhanced = cv2.merge((y, cr, cb))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2BGR)
    enhanced = _apply_gamma(enhanced, config.nir_gamma)
    return NIRPreprocessResult(enhanced, "NIR_PREPROCESSED", True)


def is_grayscale_like(frame_bgr: np.ndarray) -> bool:
    if frame_bgr.ndim == 2:
        return True
    if frame_bgr.shape[2] == 1:
        return True
    channels = cv2.split(frame_bgr)
    diff_rg = np.mean(np.abs(channels[2].astype(np.float32) - channels[1].astype(np.float32)))
    diff_bg = np.mean(np.abs(channels[0].astype(np.float32) - channels[1].astype(np.float32)))
    overall_std = float(np.std(frame_bgr))
    return (diff_rg + diff_bg) < max(4.0, overall_std * 0.08)


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def _apply_gamma(frame_bgr: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0 or abs(gamma - 1.0) < 1e-6:
        return frame_bgr
    inv_gamma = 1.0 / gamma
    table = np.array([(idx / 255.0) ** inv_gamma * 255.0 for idx in range(256)]).astype("uint8")
    return cv2.LUT(frame_bgr, table)
