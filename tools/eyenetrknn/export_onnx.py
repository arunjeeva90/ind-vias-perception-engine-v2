from __future__ import annotations

import argparse
import json
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

    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Optional safety override; normally derived from checkpoint metadata",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(weights_path, map_location="cpu")

    checkpoint_metadata = checkpoint.get("metadata", {})
    class_to_idx = checkpoint.get("class_to_idx") or checkpoint_metadata.get(
        "class_to_idx"
    )
    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError("Checkpoint is missing the required class_to_idx metadata")
    num_classes = len(class_to_idx)
    if args.num_classes is not None and args.num_classes != num_classes:
        raise ValueError(
            f"--num-classes={args.num_classes} conflicts with checkpoint "
            f"class_to_idx ({num_classes})"
        )
    img_size = int(
        args.img_size
        or checkpoint.get("img_size")
        or checkpoint_metadata.get("img_size", 96)
    )

    model = build_eyenetrknn_model(num_classes=num_classes, pretrained=False)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )

    print(f"Exported ONNX: {out_path}")
    metadata = {
        "class_to_idx": class_to_idx,
        "img_size": img_size,
        "mean": checkpoint.get("mean", [0.485, 0.456, 0.406]),
        "std": checkpoint.get("std", [0.229, 0.224, 0.225]),
        "source_checkpoint": str(weights_path),
    }
    metadata_path = out_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Exported metadata: {metadata_path}")


if __name__ == "__main__":
    main()
