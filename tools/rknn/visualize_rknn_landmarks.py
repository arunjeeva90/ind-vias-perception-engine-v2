#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/rknn_landmark_overlay.jpg")
    parser.add_argument("--input-size", nargs=2, type=int, default=[160, 160])
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_w, input_h = args.input_size

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    original_h, original_w = image.shape[:2]

    resized = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_AREA)

    # Current RKNN smoke test used NHWC uint8 successfully.
    input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)

    rknn = RKNNLite()
    try:
        ret = rknn.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")

        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")

        outputs = rknn.inference(inputs=[input_tensor])
        if outputs is None or len(outputs) < 3:
            raise RuntimeError(f"Expected at least 3 outputs, got: {None if outputs is None else len(outputs)}")

        landmarks = np.asarray(outputs[2]).reshape(-1)

        if landmarks.size != 136:
            raise RuntimeError(f"Expected 136 landmark values, got {landmarks.size}")

        points = landmarks.reshape(68, 2)

        # Model appears normalized 0..1 based on output min/max.
        overlay = image.copy()
        for idx, (x, y) in enumerate(points):
            px = int(round(float(x) * original_w))
            py = int(round(float(y) * original_h))

            px = max(0, min(original_w - 1, px))
            py = max(0, min(original_h - 1, py))

            cv2.circle(overlay, (px, py), 2, (0, 255, 0), -1)
            cv2.putText(
                overlay,
                str(idx),
                (px + 2, py - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.imwrite(str(output_path), overlay)
        print(f"[PASS] Saved overlay: {output_path}")
        print(f"Landmarks min={landmarks.min():.4f} max={landmarks.max():.4f}")

    finally:
        rknn.release()


if __name__ == "__main__":
    main()
