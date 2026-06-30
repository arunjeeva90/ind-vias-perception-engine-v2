from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


CLASSES = [
    "bad_crop",
    "eye_closed",
    "eye_open",
]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/eyenetrknn/eyenetrknn_mnv3s_96_int8.rknn")
    parser.add_argument("--image", required=True)
    parser.add_argument("--img-size", type=int, default=96)
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)

    inp = np.expand_dims(img, axis=0)

    rknn = RKNNLite()

    print("Loading RKNN model...")
    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError("load_rknn failed")

    print("Initializing RKNNLite runtime...")
    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError("init_runtime failed")

    print("Running inference...")
    outputs = rknn.inference(inputs=[inp], data_format=["nhwc"])

    logits = outputs[0].reshape(-1)
    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))

    result = {
        "image": str(image_path),
        "pred_class": CLASSES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs": {
            cls: float(prob)
            for cls, prob in zip(CLASSES, probs)
        },
        "raw_logits": logits.tolist(),
    }

    print(json.dumps(result, indent=2))

    rknn.release()


if __name__ == "__main__":
    main()
