#!/usr/bin/env python3
from __future__ import annotations

import glob
import importlib
import platform
import sys


def print_section(title: str) -> None:
    print("")
    print(f"--- {title} ---")


def check_devices(patterns: list[str]) -> None:
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            print(f"[PASS] {pattern}:")
            for path in matches:
                print(f"  {path}")
        else:
            print(f"[WARN] {pattern}: none found")


def import_module(name: str) -> object | None:
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        print(f"[FAIL] import {name}: {exc}")
        return None
    version = getattr(module, "__version__", "version unknown")
    print(f"[PASS] import {name}: {version}")
    return module


def main() -> int:
    print("========================================")
    print(" AXON RKNN/NPU Probe")
    print("========================================")

    print_section("Python")
    print(f"Executable: {sys.executable}")
    print(f"Version:    {sys.version}")

    print_section("Platform")
    print(f"platform: {platform.platform()}")
    print(f"uname:    {platform.uname()}")

    print_section("Accelerator Devices")
    check_devices(["/dev/dri*", "/dev/mali*", "/dev/rknpu*", "/dev/dma_heap*"])

    print_section("Python Packages")
    rknn = import_module("rknn.api")
    rknnlite = import_module("rknnlite.api")
    onnx = import_module("onnx")
    onnxruntime = import_module("onnxruntime")
    cv2 = import_module("cv2")

    print_section("OpenCV OpenCL")
    if cv2 is None:
        print("[FAIL] cv2 unavailable; cannot query OpenCL")
    else:
        opencl = cv2.ocl
        print(f"haveOpenCL: {opencl.haveOpenCL()}")
        print(f"useOpenCL:  {opencl.useOpenCL()}")

    required = [rknn, rknnlite, onnx, onnxruntime, cv2]
    return 0 if all(module is not None for module in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
