#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_INPUT_SIZE = (192, 192)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a smoke-test inference for an RKNN face landmark model."
    )
    parser.add_argument("--model", required=True, type=Path, help="Input .rknn model path.")
    parser.add_argument("--image", required=True, type=Path, help="Input image path.")
    return parser.parse_args()


def check_ret(ret: int, step: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{step} failed with RKNNLite return code {ret}")


def load_image(image_path: Path) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("cv2 is required to preprocess the RKNN test image.") from exc

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Unable to read image: {image_path}")

    width, height = DEFAULT_INPUT_SIZE
    image_bgr = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return np.expand_dims(image_rgb, axis=0)


def describe_output(index: int, output: np.ndarray) -> None:
    output_array = np.asarray(output)
    if output_array.size == 0:
        print(f"Output[{index}]: shape={output_array.shape} dtype={output_array.dtype} empty")
        return
    print(
        f"Output[{index}]: "
        f"shape={output_array.shape} "
        f"dtype={output_array.dtype} "
        f"min={output_array.min()} "
        f"max={output_array.max()}"
    )


def main() -> int:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"RKNN model not found: {args.model}")
    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    try:
        from rknnlite.api import RKNNLite  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "rknnlite.api is unavailable. Activate .venv-rknn or install RKNNLite."
        ) from exc

    print("========================================")
    print(" RKNNLite Landmark Runtime Test")
    print("========================================")
    print(f"Model:      {args.model}")
    print(f"Image:      {args.image}")
    print(f"Input size: {DEFAULT_INPUT_SIZE[0]}x{DEFAULT_INPUT_SIZE[1]}")

    input_tensor = load_image(args.image)
    print(f"Input tensor: shape={input_tensor.shape} dtype={input_tensor.dtype}")

    rknn = RKNNLite()
    try:
        print("\n[1/3] Loading RKNN model")
        check_ret(rknn.load_rknn(str(args.model)), "RKNNLite.load_rknn")

        print("[2/3] Initializing runtime")
        check_ret(rknn.init_runtime(), "RKNNLite.init_runtime")

        print("[3/3] Running inference")
        outputs = rknn.inference(inputs=[input_tensor])
        if outputs is None:
            raise RuntimeError("RKNNLite inference returned no outputs")

        print("\n--- Outputs ---")
        for index, output in enumerate(outputs):
            describe_output(index, output)
    finally:
        print("Releasing RKNNLite runtime")
        rknn.release()

    print("\n[PASS] RKNNLite inference completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
