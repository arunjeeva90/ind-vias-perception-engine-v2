from __future__ import annotations

import argparse
from pathlib import Path

from rknn.api import RKNN


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--onnx",
        type=str,
        default="models/eyenetrknn/eyenetrknn_mnv3s_96_simplified.onnx",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="models/eyenetrknn/eyenetrknn_mnv3s_96_int8.rknn",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="models/eyenetrknn/rknn_calib_dataset.txt",
    )

    parser.add_argument(
        "--target",
        type=str,
        default="rk3588",
        help="Use rk3588 for Vicharak AXON / RK3588 class boards",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    onnx_path = Path(args.onnx)
    out_path = Path(args.out)
    dataset_path = Path(args.dataset)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Calibration dataset not found: {dataset_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=True)

    print("1. Configuring RKNN...")

    ret = rknn.config(
        target_platform=args.target,
        mean_values=[[123.675, 116.28, 103.53]],
        std_values=[[58.395, 57.12, 57.375]],
        quantized_dtype="asymmetric_quantized-8",
        optimization_level=3,
    )

    if ret != 0:
        raise RuntimeError("RKNN config failed")

    print("2. Loading ONNX...")
    ret = rknn.load_onnx(model=str(onnx_path))

    if ret != 0:
        raise RuntimeError("RKNN load_onnx failed")

    print("3. Building RKNN INT8...")
    ret = rknn.build(
        do_quantization=True,
        dataset=str(dataset_path),
    )

    if ret != 0:
        raise RuntimeError("RKNN build failed")

    print("4. Exporting RKNN...")
    ret = rknn.export_rknn(str(out_path))

    if ret != 0:
        raise RuntimeError("RKNN export failed")

    rknn.release()

    print(f"RKNN export complete: {out_path}")


if __name__ == "__main__":
    main()
