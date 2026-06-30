from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ind_vias_dms.eyenetrknn.model import build_eyenetrknn_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        default="models/eyenetrknn/eyenetrknn_best.pt",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="models/eyenetrknn/eyenetrknn_mnv3s_96.onnx",
    )

    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--num-classes", type=int, default=5)

    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(weights_path, map_location="cpu")

    model = build_eyenetrknn_model(
        num_classes=args.num_classes,
        pretrained=False,
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, args.img_size, args.img_size)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes=None,
    )

    print(f"Exported ONNX: {out_path}")


if __name__ == "__main__":
    main()
