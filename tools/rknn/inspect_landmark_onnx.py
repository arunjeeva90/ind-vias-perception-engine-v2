#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect ONNX landmark model inputs and outputs")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    return parser.parse_args()


def dim_to_str(dim):
    if dim.HasField("dim_value"):
        return str(dim.dim_value)
    if dim.HasField("dim_param"):
        return dim.dim_param
    return "?"


def value_info_shape(value_info):
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return "?"
    dims = [dim_to_str(dim) for dim in tensor_type.shape.dim]
    return "[" + ", ".join(dims) + "]"


def main():
    args = parse_args()
    model_path = Path(args.onnx)
    if not model_path.exists():
        raise SystemExit(f"ONNX model not found: {model_path}")

    model = onnx.load(str(model_path))
    size_mb = model_path.stat().st_size / (1024.0 * 1024.0)
    opsets = ", ".join(
        f"{opset.domain or 'ai.onnx'}:{opset.version}" for opset in model.opset_import
    )

    print("========================================")
    print(" ONNX Landmark Model Inspection")
    print("========================================")
    print(f"Path:              {model_path}")
    print(f"File size:         {size_mb:.2f} MB")
    print(f"Opset version(s):  {opsets}")
    print(f"Initializer count: {len(model.graph.initializer)}")
    print("")
    print("Inputs:")
    for value_info in model.graph.input:
        print(f"  - {value_info.name}: {value_info_shape(value_info)}")
    print("")
    print("Outputs:")
    for value_info in model.graph.output:
        print(f"  - {value_info.name}: {value_info_shape(value_info)}")
    print("========================================")


if __name__ == "__main__":
    main()
