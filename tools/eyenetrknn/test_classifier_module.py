from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from ind_vias_dms.eyenetrknn.rknnlite_classifier import EyeNetRKNNLiteClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/eyenetrknn/eyenetrknn_mnv3s_96_int8.rknn")
    parser.add_argument("--image", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    img = cv2.imread(str(image_path))

    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    clf = EyeNetRKNNLiteClassifier(args.model)

    pred_class, confidence, probs = clf.predict(img)

    result = {
        "image": str(image_path),
        "pred_class": pred_class,
        "confidence": confidence,
        "probs": probs,
    }

    print(json.dumps(result, indent=2))

    clf.release()


if __name__ == "__main__":
    main()
