#!/usr/bin/env python3
"""Staged pretrained classifier training for the reviewed 96px eye dataset.

This command deliberately stops at native PyTorch qualification.  Deployment
exports are a separate, gated step after crop and video evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


CLASS_NAMES = ["eye_closed", "eye_open"]
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 96
AUTHORITATIVE_ROOT = Path(
    "/home/vicharak/Mobility_ADAS/ADVIS/DMS/"
    "DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/01_Eye_state_dataset"
)
PRETRAINED_URL = (
    "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
)
PRETRAINED_TORCHVISION_ID = "MobileNet_V3_Small_Weights.IMAGENET1K_V1"
EFFICIENTNET_LITE0_URL = (
    "https://github.com/rwightman/pytorch-image-models/releases/download/"
    "v0.1-weights/tf_efficientnet_lite0-0aa007d2.pth"
)
EFFICIENTNET_LITE0_ID = "timm/tf_efficientnet_lite0.in1k"
SIDE_PATTERN = re.compile(r"/(left|right)/")
FLIP_LEFT_RIGHT = getattr(Image, "Transpose", Image).FLIP_LEFT_RIGHT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=["mobilenet_v3_small", "tf_efficientnet_lite0"],
        default="mobilenet_v3_small",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pretrained-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--orientation",
        choices=["original", "mirror_right"],
        default="original",
    )
    parser.add_argument(
        "--imbalance",
        choices=[
            "weighted_ce",
            "balanced_sampler",
            "weighted_ce_balanced_sampler",
        ],
        default="weighted_ce",
    )
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def eye_side(row: dict[str, str]) -> str:
    match = SIDE_PATTERN.search(row["image_path"])
    return match.group(1) if match else "unknown"


def load_rows(manifest: Path) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["task"] == "eye_state" and row["split"] in rows:
                rows[row["split"]].append(row)
    if len(rows["train"]) != 4113 or len(rows["val"]) != 1016:
        raise ValueError(
            "Accepted eye split changed: expected 4113/1016, got "
            f"{len(rows['train'])}/{len(rows['val'])}"
        )
    expected = {
        ("train", "eye_closed"): 375,
        ("train", "eye_open"): 3738,
        ("val", "eye_closed"): 70,
        ("val", "eye_open"): 946,
    }
    actual = Counter(
        (split, row["class_name"])
        for split, split_rows in rows.items()
        for row in split_rows
    )
    if actual != expected:
        raise ValueError(f"Accepted class counts changed: {actual}")

    allowed_root = AUTHORITATIVE_ROOT.resolve()
    hashes: dict[str, set[str]] = {"train": set(), "val": set()}
    sources: dict[str, set[str]] = {"train": set(), "val": set()}
    for split, split_rows in rows.items():
        for row in split_rows:
            image_path = Path(row["image_path"]).resolve()
            if allowed_root not in image_path.parents:
                raise ValueError(f"Non-authoritative image: {image_path}")
            if row["class_name"] not in CLASS_TO_IDX:
                raise ValueError(f"Unexpected eye class: {row['class_name']}")
            actual_hash = sha256_file(image_path)
            if actual_hash != row["sha256"]:
                raise ValueError(f"Image hash mismatch: {image_path}")
            hashes[split].add(actual_hash)
            sources[split].add(row["source_group"])
    if hashes["train"] & hashes["val"]:
        raise ValueError("Exact train/validation hash leakage detected")
    if sources["train"] & sources["val"]:
        raise ValueError("Source-video train/validation leakage detected")
    return rows


def load_provenance(manifest: Path) -> dict[str, Any]:
    metadata_path = manifest.parent / "preparation_metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(manifest)
    if payload["prepared_manifest_sha256"] != manifest_hash:
        raise ValueError("Prepared manifest hash differs from preparation metadata")
    recorded_root = payload["authoritative_image_roots"]["eye_state"]
    if Path(recorded_root).resolve() != AUTHORITATIVE_ROOT.resolve():
        raise ValueError("Authoritative eye root differs from preparation metadata")
    return {
        "validation_description": (
            "source-video-exclusive held-out reviewed crop validation"
        ),
        "authoritative_source_path": str(AUTHORITATIVE_ROOT.resolve()),
        "reviewed_label_manifest": payload["source_manifests"]["eye_state"],
        "prepared_manifest_path": str(manifest),
        "prepared_manifest_sha256": manifest_hash,
        "split_sha256": payload["split_sha256"]["eye_state"],
        "split_policy": payload["split_policy"],
        "exclusions": payload["exclusions"]["eye_state"],
        "counts": payload["counts"]["eye_state"],
        "integrity": {
            "readable_and_hash_verified": 5129,
            "exact_train_val_hash_overlap": 0,
            "source_video_train_val_overlap": 0,
        },
        "raw_video_policy": {
            "prohibited_raw_videos": ["dmsNit207.mp4", "mob_belt.mp4"],
            "reviewed_derivative_crops_permitted": True,
            "raw_video_reprocessing_permitted": False,
        },
    }


class AddGaussianNoise:
    def __init__(self, probability: float, sigma_max: float) -> None:
        self.probability = probability
        self.sigma_max = sigma_max

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.probability:
            return tensor
        sigma = torch.empty(1).uniform_(0.002, self.sigma_max).item()
        return torch.clamp(tensor + torch.randn_like(tensor) * sigma, 0.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(probability={self.probability}, "
            f"sigma_max={self.sigma_max})"
        )


def make_transforms() -> tuple[Any, Any, dict[str, Any]]:
    augmentation = {
        "resize": [IMAGE_SIZE, IMAGE_SIZE],
        "random_affine": {
            "degrees": 5.0,
            "translate": [0.03, 0.03],
            "scale": [0.96, 1.04],
            "interpolation": "bilinear",
        },
        "color_jitter": {
            "brightness": 0.18,
            "contrast": 0.18,
            "saturation": 0.08,
            "hue": 0.02,
        },
        "gaussian_blur": {
            "probability": 0.12,
            "kernel_size": 3,
            "sigma": [0.1, 0.8],
        },
        "sensor_noise": {"probability": 0.15, "sigma_max": 0.018},
        "random_horizontal_flip": False,
        "validation_augmentation": False,
    }
    train = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomAffine(
                degrees=5.0,
                translate=(0.03, 0.03),
                scale=(0.96, 1.04),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(
                brightness=0.18,
                contrast=0.18,
                saturation=0.08,
                hue=0.02,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(3, sigma=(0.1, 0.8))],
                p=0.12,
            ),
            transforms.ToTensor(),
            AddGaussianNoise(probability=0.15, sigma_max=0.018),
            transforms.Normalize(MEAN, STD),
        ]
    )
    validation = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train, validation, augmentation


class EyeDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        transform: Any,
        orientation: str,
    ) -> None:
        self.rows = rows
        self.transform = transform
        self.orientation = orientation

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            if self.orientation == "mirror_right" and eye_side(row) == "right":
                image = image.transpose(FLIP_LEFT_RIGHT)
            tensor = self.transform(image)
        return tensor, CLASS_TO_IDX[row["class_name"]], index


def metrics_for_threshold(
    truth: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
) -> dict[str, Any]:
    matrix = np.zeros((2, 2), dtype=np.int64)
    for expected, probability in zip(truth, probabilities):
        predicted = 0 if probability >= threshold else 1
        matrix[expected, predicted] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(CLASS_NAMES):
        tp = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted_count = int(matrix[:, index].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    balanced_accuracy = (
        per_class["eye_closed"]["recall"] + per_class["eye_open"]["recall"]
    ) / 2.0
    closed_f1 = per_class["eye_closed"]["f1"]
    return {
        "decision_threshold": threshold,
        "accuracy": float(np.trace(matrix) / max(1, matrix.sum())),
        "balanced_accuracy": balanced_accuracy,
        "false_open_rate": 1.0 - per_class["eye_closed"]["recall"],
        "false_closure_rate": 1.0 - per_class["eye_open"]["recall"],
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "safety_score": (balanced_accuracy + closed_f1) / 2.0,
    }


def threshold_sweep(
    truth: list[int], probabilities: list[float]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    thresholds = np.linspace(0.01, 0.99, 99)
    sweep = [
        metrics_for_threshold(truth, probabilities, float(threshold))
        for threshold in thresholds
    ]
    selected = max(
        sweep,
        key=lambda item: (
            item["safety_score"],
            item["per_class"]["eye_closed"]["recall"],
            item["per_class"]["eye_closed"]["precision"],
            -item["false_closure_rate"],
        ),
    )
    return sweep, selected


def gate_result(metrics: dict[str, Any]) -> dict[str, Any]:
    closed = metrics["per_class"]["eye_closed"]
    checks = {
        "closed_recall_gte_0_85": closed["recall"] >= 0.85,
        "closed_precision_gte_0_75": closed["precision"] >= 0.75,
        "closed_f1_gte_0_80": closed["f1"] >= 0.80,
        "balanced_accuracy_gte_0_85": metrics["balanced_accuracy"] >= 0.85,
        "exact_train_val_hash_overlap_eq_0": True,
        "source_video_train_val_overlap_eq_0": True,
    }
    return {"passed": all(checks.values()), "checks": checks}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, list[int], list[float], list[int]]:
    model.eval()
    total_loss = 0.0
    truth: list[int] = []
    probabilities: list[float] = []
    indices: list[int] = []
    with torch.inference_mode():
        for inputs, labels, row_indices in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            total_loss += float(loss_fn(logits, labels).item()) * labels.size(0)
            truth.extend(labels.cpu().tolist())
            probabilities.extend(
                torch.softmax(logits, dim=1)[:, 0].cpu().tolist()
            )
            indices.extend(row_indices.tolist())
    return total_loss / len(loader.dataset), truth, probabilities, indices


def configure_trainable(
    model: nn.Module,
    stage: str,
    architecture: str,
) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable_stages: list[str]
    if stage == "head":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        trainable_stages = ["classifier"]
    elif stage == "final_backbone":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        if architecture == "mobilenet_v3_small":
            modules = list(model.features.children())[-4:]
            trainable_stages = ["features[-4:]", "classifier"]
        else:
            modules = [
                *list(model.blocks.children())[-2:],
                model.conv_head,
                model.bn2,
            ]
            trainable_stages = [
                "blocks[-2:]",
                "conv_head",
                "bn2",
                "classifier",
            ]
        for block in modules:
            for parameter in block.parameters():
                parameter.requires_grad = True
    else:
        raise ValueError(stage)
    return trainable_stages


def checkpoint_payload(
    model: nn.Module,
    *,
    epoch: int,
    stage: str,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    provenance: dict[str, Any],
    pretrained_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "model_state": model.state_dict(),
        "architecture": args.architecture,
        "pretrained_weight_id": _pretrained_metadata(args.architecture)[0],
        "pretrained_weight_url": _pretrained_metadata(args.architecture)[1],
        "pretrained_weight_sha256": pretrained_sha256,
        "class_to_idx": CLASS_TO_IDX,
        "input_size": [3, IMAGE_SIZE, IMAGE_SIZE],
        "colour_order": "RGB",
        "mean": MEAN,
        "std": STD,
        "orientation": args.orientation,
        "imbalance": args.imbalance,
        "seed": args.seed,
        "stage": stage,
        "epoch": epoch,
        "metrics": metrics,
        "provenance": provenance,
    }


def train_stage(
    *,
    model: nn.Module,
    stage: str,
    epochs: int,
    learning_rate: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
    provenance: dict[str, Any],
    pretrained_sha256: str,
    history: list[dict[str, Any]],
    global_best: dict[str, Any],
) -> None:
    trainable_stages = configure_trainable(model, stage, args.architecture)
    parameters = [item for item in model.parameters() if item.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    stale = 0
    for epoch in range(1, epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        training_loss = 0.0
        for inputs, labels, _ in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            training_loss += float(loss.item()) * labels.size(0)
        validation_loss, truth, probabilities, _ = evaluate(
            model, val_loader, loss_fn, device
        )
        _, selected = threshold_sweep(truth, probabilities)
        scheduler.step(selected["safety_score"])
        record = {
            "stage": stage,
            "epoch": epoch,
            "trainable_stages": trainable_stages,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": training_loss / len(train_loader.dataset),
            "validation_loss": validation_loss,
            "duration_seconds": time.monotonic() - epoch_started,
            **selected,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if selected["safety_score"] > global_best["score"] + 1e-8:
            global_best["score"] = selected["safety_score"]
            global_best["stage"] = stage
            global_best["epoch"] = epoch
            stale = 0
            torch.save(
                checkpoint_payload(
                    model,
                    epoch=epoch,
                    stage=stage,
                    metrics=selected,
                    args=args,
                    provenance=provenance,
                    pretrained_sha256=pretrained_sha256,
                ),
                output_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break


def subgroup_metrics(
    rows: list[dict[str, str]],
    truth_by_index: dict[int, int],
    probability_by_index: dict[int, float],
    threshold: float,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    definitions = {
        "eye_side": lambda row: eye_side(row),
        "source_video": lambda row: row["source_group"],
    }
    for dimension, getter in definitions.items():
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            grouped[getter(row)].append(index)
        groups[dimension] = {}
        for name, indices in sorted(grouped.items()):
            groups[dimension][name] = metrics_for_threshold(
                [truth_by_index[index] for index in indices],
                [probability_by_index[index] for index in indices],
                threshold,
            )
    return groups


def save_evaluation(
    *,
    output_dir: Path,
    rows: list[dict[str, str]],
    truth: list[int],
    probabilities: list[float],
    indices: list[int],
    validation_loss: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    truth_by_index = dict(zip(indices, truth))
    probability_by_index = dict(zip(indices, probabilities))
    ordered_truth = [truth_by_index[index] for index in range(len(rows))]
    ordered_probabilities = [
        probability_by_index[index] for index in range(len(rows))
    ]
    sweep, selected = threshold_sweep(ordered_truth, ordered_probabilities)
    selected["validation_loss"] = validation_loss
    selected["acceptance"] = gate_result(selected)
    selected["subgroups"] = subgroup_metrics(
        rows,
        truth_by_index,
        probability_by_index,
        selected["decision_threshold"],
    )
    with (output_dir / "threshold_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = [
            "threshold",
            "closed_precision",
            "closed_recall",
            "closed_f1",
            "open_precision",
            "open_recall",
            "open_f1",
            "balanced_accuracy",
            "false_open_rate",
            "false_closure_rate",
            "safety_score",
            "gates_passed",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in sweep:
            writer.writerow(
                {
                    "threshold": item["decision_threshold"],
                    "closed_precision": item["per_class"]["eye_closed"][
                        "precision"
                    ],
                    "closed_recall": item["per_class"]["eye_closed"]["recall"],
                    "closed_f1": item["per_class"]["eye_closed"]["f1"],
                    "open_precision": item["per_class"]["eye_open"]["precision"],
                    "open_recall": item["per_class"]["eye_open"]["recall"],
                    "open_f1": item["per_class"]["eye_open"]["f1"],
                    "balanced_accuracy": item["balanced_accuracy"],
                    "false_open_rate": item["false_open_rate"],
                    "false_closure_rate": item["false_closure_rate"],
                    "safety_score": item["safety_score"],
                    "gates_passed": gate_result(item)["passed"],
                }
            )

    prediction_fields = [
        "row_index",
        "packaged_path",
        "source_video",
        "eye_side",
        "sha256",
        "truth",
        "p_closed",
        "predicted",
        "correct",
    ]
    errors: list[dict[str, Any]] = []
    with (output_dir / "validation_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as prediction_stream, (output_dir / "validation_errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as error_stream:
        prediction_writer = csv.DictWriter(
            prediction_stream, fieldnames=prediction_fields
        )
        error_writer = csv.DictWriter(error_stream, fieldnames=prediction_fields)
        prediction_writer.writeheader()
        error_writer.writeheader()
        threshold = selected["decision_threshold"]
        for index, row in enumerate(rows):
            probability = ordered_probabilities[index]
            predicted_index = 0 if probability >= threshold else 1
            record = {
                "row_index": index,
                "packaged_path": row["packaged_path"],
                "source_video": row["source_group"],
                "eye_side": eye_side(row),
                "sha256": row["sha256"],
                "truth": CLASS_NAMES[ordered_truth[index]],
                "p_closed": probability,
                "predicted": CLASS_NAMES[predicted_index],
                "correct": predicted_index == ordered_truth[index],
            }
            prediction_writer.writerow(record)
            if not record["correct"]:
                error_writer.writerow(record)
                errors.append(record)
    write_json(output_dir / "crop_evaluation.json", selected)
    return selected, errors


def _pretrained_metadata(architecture: str) -> tuple[str, str]:
    if architecture == "tf_efficientnet_lite0":
        return EFFICIENTNET_LITE0_ID, EFFICIENTNET_LITE0_URL
    return PRETRAINED_TORCHVISION_ID, PRETRAINED_URL


def _build_model(
    architecture: str,
    pretrained_weights: Path,
) -> nn.Module:
    pretrained_state = torch.load(
        pretrained_weights,
        map_location="cpu",
        weights_only=True,
    )
    if architecture == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=None)
        model.load_state_dict(pretrained_state)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_features, len(CLASS_NAMES))
        return model

    try:
        import timm
    except ImportError as exc:
        raise RuntimeError(
            "tf_efficientnet_lite0 requires timm; use an isolated PYTHONPATH"
        ) from exc
    model = timm.create_model("tf_efficientnet_lite0", pretrained=False)
    model.load_state_dict(pretrained_state)
    model.reset_classifier(len(CLASS_NAMES))
    return model


def main() -> int:
    args = parse_args()
    if min(
        args.head_epochs,
        args.finetune_epochs,
        args.batch_size,
        args.threads,
    ) < 1:
        raise ValueError("Epoch, batch, and thread values must be positive")
    started = time.monotonic()
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    manifest = args.manifest.resolve()
    pretrained_weights = args.pretrained_weights.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = load_rows(manifest)
    provenance = load_provenance(manifest)
    pretrained_sha256 = sha256_file(pretrained_weights)
    if args.architecture == "tf_efficientnet_lite0":
        MEAN[:] = [0.5, 0.5, 0.5]
        STD[:] = [0.5, 0.5, 0.5]

    train_transform, validation_transform, augmentation = make_transforms()
    training_dataset = EyeDataset(
        rows["train"], train_transform, args.orientation
    )
    validation_dataset = EyeDataset(
        rows["val"], validation_transform, args.orientation
    )
    counts = Counter(row["class_name"] for row in rows["train"])
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    shuffle = True
    if args.imbalance in {
        "balanced_sampler",
        "weighted_ce_balanced_sampler",
    }:
        sample_weights = [
            1.0 / counts[row["class_name"]] for row in rows["train"]
        ]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.workers,
        generator=generator,
        worker_init_fn=worker_seed,
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        worker_init_fn=worker_seed,
        persistent_workers=args.workers > 0,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = _build_model(args.architecture, pretrained_weights)
    model.to(device)

    inverse_class_weights = torch.tensor(
        [
            len(rows["train"]) / (2.0 * counts[name])
            for name in CLASS_NAMES
        ],
        dtype=torch.float32,
        device=device,
    )
    if args.imbalance == "weighted_ce_balanced_sampler":
        class_weights = torch.sqrt(inverse_class_weights)
    else:
        class_weights = inverse_class_weights
    if args.imbalance in {"weighted_ce", "weighted_ce_balanced_sampler"}:
        loss_fn: nn.Module = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    history: list[dict[str, Any]] = []
    global_best: dict[str, Any] = {"score": -1.0, "stage": None, "epoch": None}
    train_stage(
        model=model,
        stage="head",
        epochs=args.head_epochs,
        learning_rate=args.head_learning_rate,
        train_loader=training_loader,
        val_loader=validation_loader,
        loss_fn=loss_fn,
        device=device,
        output_dir=output_dir,
        args=args,
        provenance=provenance,
        pretrained_sha256=pretrained_sha256,
        history=history,
        global_best=global_best,
    )
    best = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    train_stage(
        model=model,
        stage="final_backbone",
        epochs=args.finetune_epochs,
        learning_rate=args.finetune_learning_rate,
        train_loader=training_loader,
        val_loader=validation_loader,
        loss_fn=loss_fn,
        device=device,
        output_dir=output_dir,
        args=args,
        provenance=provenance,
        pretrained_sha256=pretrained_sha256,
        history=history,
        global_best=global_best,
    )
    best = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    torch.save(
        checkpoint_payload(
            model,
            epoch=best["epoch"],
            stage=best["stage"],
            metrics=best["metrics"],
            args=args,
            provenance=provenance,
            pretrained_sha256=pretrained_sha256,
        ),
        output_dir / "selected_native.pt",
    )
    validation_loss, truth, probabilities, indices = evaluate(
        model, validation_loader, loss_fn, device
    )
    evaluation, errors = save_evaluation(
        output_dir=output_dir,
        rows=rows["val"],
        truth=truth,
        probabilities=probabilities,
        indices=indices,
        validation_loss=validation_loss,
    )
    write_json(output_dir / "history.json", history)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    selected_path = output_dir / "selected_native.pt"
    result = {
        "experiment_id": output_dir.name,
        "architecture": args.architecture,
        "pretrained_weight": {
            "id": _pretrained_metadata(args.architecture)[0],
            "url": _pretrained_metadata(args.architecture)[1],
            "local_input_path": str(pretrained_weights),
            "sha256": pretrained_sha256,
        },
        "seed": args.seed,
        "input_size": [3, IMAGE_SIZE, IMAGE_SIZE],
        "colour_order": "RGB",
        "normalization": {"mean": MEAN, "std": STD},
        "anatomical_normalization": args.orientation,
        "training_stages": [
            {
                "name": "head",
                "trainable": ["classifier"],
                "learning_rate": args.head_learning_rate,
                "epoch_limit": args.head_epochs,
            },
            {
                "name": "final_backbone",
                "trainable": (
                    ["features[-4:]", "classifier"]
                    if args.architecture == "mobilenet_v3_small"
                    else ["blocks[-2:]", "conv_head", "bn2", "classifier"]
                ),
                "learning_rate": args.finetune_learning_rate,
                "epoch_limit": args.finetune_epochs,
            },
        ],
        "imbalance": args.imbalance,
        "loss": (
            "weighted_cross_entropy"
            if args.imbalance in {
                "weighted_ce",
                "weighted_ce_balanced_sampler",
            }
            else "cross_entropy"
        ),
        "class_weights": {
            name: float(class_weights[index].item())
            for index, name in enumerate(CLASS_NAMES)
        },
        "sampler": (
            "weighted_random_sampler"
            if args.imbalance in {
                "balanced_sampler",
                "weighted_ce_balanced_sampler",
            }
            else "shuffle"
        ),
        "augmentation": augmentation,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau(mode=max,factor=0.5,patience=2)",
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "early_stopping_patience": args.patience,
        "best_stage": best["stage"],
        "best_epoch": best["epoch"],
        "crop_evaluation": evaluation,
        "validation_error_count": len(errors),
        "parameter_count": parameter_count,
        "trainable_parameter_count_at_end": trainable_count,
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "native_model_size_bytes": selected_path.stat().st_size,
        "training_duration_seconds": time.monotonic() - started,
        "provenance": provenance,
        "deployment_export_performed": False,
        "command": " ".join(sys.argv),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "numpy": np.__version__,
            "pillow": __import__("PIL").__version__,
            "platform": platform.platform(),
            "device": str(device),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        },
    }
    write_json(output_dir / "training_result.json", result)
    print(json.dumps({"final": result}, sort_keys=True), flush=True)
    return 0 if evaluation["acceptance"]["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
