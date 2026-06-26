#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an ONNX face landmark model to RKNN for AXON RK3588."
    )
    parser.add_argument("--onnx", required=True, type=Path, help="Input ONNX model path.")
    parser.add_argument("--output", required=True, type=Path, help="Output RKNN model path.")
    parser.add_argument(
        "--input-name",
        default=None,
        help="Optional ONNX graph input name. Defaults to the first graph input.",
    )
    parser.add_argument(
        "--target-platform",
        default="rk3588",
        help="RKNN target platform (default: rk3588).",
    )
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(192, 192),
        help="Expected model input size, recorded for the PoC workflow (default: 192 192).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional calibration dataset path for future quantized builds.",
    )
    return parser.parse_args()


def check_ret(ret: int, step: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{step} failed with RKNN return code {ret}")


def resolve_input_name(onnx_path: Path, requested_input_name: str | None) -> str:
    if requested_input_name:
        return requested_input_name

    try:
        import onnx  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "onnx is unavailable. Install ONNX or pass --input-name explicitly."
        ) from exc

    model = onnx.load(str(onnx_path))
    if not model.graph.input:
        raise RuntimeError(f"ONNX model has no graph inputs: {onnx_path}")
    return model.graph.input[0].name


def main() -> int:
    args = parse_args()

    if not args.onnx.exists():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    input_width, input_height = args.input_size
    input_name = resolve_input_name(args.onnx, args.input_name)
    fixed_input_shape = [1, 3, input_height, input_width]

    try:
        from rknn.api import RKNN  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "rknn.api is unavailable. Activate .venv-rknn or install the RKNN toolkit."
        ) from exc

    print("========================================")
    print(" RKNN Landmark ONNX Converter")
    print("========================================")
    print(f"ONNX:            {args.onnx}")
    print(f"Output:          {args.output}")
    print(f"Target platform: {args.target_platform}")
    print(f"Input size:      {args.input_size[0]}x{args.input_size[1]}")
    print(f"Input name:      {input_name}")
    print(f"Fixed shape:     {fixed_input_shape}")
    print("Quantization:    disabled for first FP16/FP32 PoC")
    if args.dataset is not None:
        print(f"Dataset:         {args.dataset} (not used while quantization is disabled)")

    rknn = RKNN(verbose=True)
    try:
        print("\n[1/4] Configuring RKNN")
        check_ret(rknn.config(target_platform=args.target_platform), "rknn.config")

        print("[2/4] Loading ONNX")

        print(f"    Input override: {input_name} -> {fixed_input_shape}")
        check_ret(
            rknn.load_onnx(
                model=str(args.onnx),
                inputs=[input_name],
                input_size_list=[fixed_input_shape],
            ),
            "rknn.load_onnx",
        )

        print("[3/4] Building RKNN graph")
        check_ret(rknn.build(do_quantization=False), "rknn.build")

        print("[4/4] Exporting RKNN")
        check_ret(rknn.export_rknn(str(args.output)), "rknn.export_rknn")
    finally:
        print("Releasing RKNN converter")
        rknn.release()

    print("\n[PASS] RKNN model exported successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
