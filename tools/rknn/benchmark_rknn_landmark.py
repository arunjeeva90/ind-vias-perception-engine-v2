#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--input-size", nargs=2, type=int, default=[160, 160])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--loops", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    input_w, input_h = args.input_size

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_AREA)
    tensor = np.expand_dims(img, axis=0).astype(np.uint8)

    rknn = RKNNLite()
    try:
        ret = rknn.load_rknn(args.model)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")

        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")

        for _ in range(args.warmup):
            rknn.inference(inputs=[tensor])

        times = []
        t_total0 = time.perf_counter()
        for _ in range(args.loops):
            t0 = time.perf_counter()
            rknn.inference(inputs=[tensor])
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        t_total1 = time.perf_counter()

        arr = np.asarray(times, dtype=np.float32)

        print("========================================")
        print(" RKNN Landmark Benchmark")
        print("========================================")
        print(f"Model:        {args.model}")
        print(f"Image:        {args.image}")
        print(f"Input:        {input_w}x{input_h}")
        print(f"Warmup:       {args.warmup}")
        print(f"Loops:        {args.loops}")
        print(f"Mean latency: {arr.mean():.3f} ms")
        print(f"Min latency:  {arr.min():.3f} ms")
        print(f"Max latency:  {arr.max():.3f} ms")
        print(f"P50 latency:  {np.percentile(arr, 50):.3f} ms")
        print(f"P90 latency:  {np.percentile(arr, 90):.3f} ms")
        print(f"P99 latency:  {np.percentile(arr, 99):.3f} ms")
        print(f"Throughput:   {args.loops / (t_total1 - t_total0):.2f} inf/s")
        print("========================================")

    finally:
        rknn.release()


if __name__ == "__main__":
    main()
