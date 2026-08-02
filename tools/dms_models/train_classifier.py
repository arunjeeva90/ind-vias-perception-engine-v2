#!/usr/bin/env python3
"""Train an explicit-class MobileNetV3 classifier from a prepared manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from ind_vias_dms.eyenetrknn.model import build_dms_classifier


TASKS = {
    "eye_state": {
        "class_names": ["eye_closed", "eye_open"],
        "img_size": 96,
    },
    "seat_belt": {
        "class_names": ["no_seat_belt", "seat_belt_on"],
        "img_size": 224,
    },
    "phone_classifier": {
        "class_names": ["no_phone_hard_negative", "phone"],
        "img_size": 224,
    },
}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
AUTHORITATIVE_TASK_ROOTS = {
    "eye_state": Path(
        "/home/vicharak/Mobility_ADAS/ADVIS/DMS/"
        "DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/01_Eye_state_dataset"
    ),
    "seat_belt": Path(
        "/home/vicharak/Mobility_ADAS/ADVIS/DMS/"
        "DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/02_Seat_Belt_detection"
    ),
    "phone_classifier": Path(
        "/home/vicharak/Mobility_ADAS/ADVIS/DMS/"
        "DMS_VICHARAK_HANDOFF_2026_0730/01_IMAGES/03_Phone_detection"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Data-loader workers; zero is the safe AXON/container default",
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--architecture",
        choices=["tiny_cnn", "mobilenet_v3_small"],
        default="tiny_cnn",
    )
    parser.add_argument(
        "--loss",
        choices=["weighted_ce", "ce", "focal"],
        default="weighted_ce",
    )
    parser.add_argument(
        "--sampler",
        choices=["shuffle", "balanced"],
        default="shuffle",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class ManifestDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        class_to_idx: dict[str, int],
        transform: Any,
    ) -> None:
        self.rows = rows
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            rgb = image.convert("RGB")
            tensor = self.transform(rgb)
        return tensor, self.class_to_idx[row["class_name"]]


def read_rows(manifest: Path, task: str) -> dict[str, list[dict[str, str]]]:
    selected: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["task"] == task and row["split"] in selected:
                selected[row["split"]].append(row)
    if not selected["train"] or not selected["val"]:
        raise ValueError(
            f"{task} requires non-empty train and val rows; got "
            f"{len(selected['train'])}/{len(selected['val'])}"
        )
    allowed_root = AUTHORITATIVE_TASK_ROOTS[task].resolve()
    for split, split_rows in selected.items():
        for row in split_rows:
            image_path = Path(row["image_path"]).resolve()
            if allowed_root not in image_path.parents:
                raise ValueError(
                    f"{split} image is outside authoritative {task} source: "
                    f"{image_path}; required root: {allowed_root}"
                )
    return selected


def load_preparation_provenance(
    manifest: Path, task: str
) -> dict[str, Any]:
    metadata_path = manifest.parent / "preparation_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Required preparation metadata is missing: {metadata_path}"
        )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_manifest_hash = sha256_file(manifest)
    if payload.get("prepared_manifest_sha256") != actual_manifest_hash:
        raise ValueError("Prepared manifest hash does not match preparation metadata")
    authoritative_source = str(AUTHORITATIVE_TASK_ROOTS[task].resolve())
    recorded_source = payload.get("authoritative_image_roots", {}).get(task)
    if recorded_source != authoritative_source:
        raise ValueError(
            f"Preparation metadata source mismatch for {task}: "
            f"{recorded_source!r} != {authoritative_source!r}"
        )
    source_manifest = payload.get("source_manifests", {}).get(task, {})
    split_hash = payload.get("split_sha256", {}).get(task)
    if not source_manifest.get("sha256") or not split_hash:
        raise ValueError(f"Incomplete preparation provenance for {task}")
    return {
        "authoritative_source_path": authoritative_source,
        "final_manifest_path": source_manifest.get("path"),
        "final_manifest_sha256": source_manifest["sha256"],
        "prepared_manifest_sha256": actual_manifest_hash,
        "split_sha256": split_hash,
        "exclusions": payload.get("exclusions", {}).get(task, {}),
        "prepared_class_counts": payload.get("counts", {}).get(task, {}),
    }


def make_transforms(img_size: int) -> tuple[Any, Any]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(
                degrees=5.0,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.08,
            ),
            transforms.RandomHorizontalFlip(p=0.50),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_transform, val_transform


def classification_metrics(
    truth: list[int], predicted: list[int], class_names: list[str]
) -> dict[str, Any]:
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for expected, actual in zip(truth, predicted):
        matrix[expected, actual] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    for index, name in enumerate(class_names):
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
        if support:
            recalls.append(recall)
    return {
        "accuracy": float(np.trace(matrix) / max(1, matrix.sum())),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    truth: list[int] = []
    closed_probabilities: list[float] = []
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            total_loss += float(loss_fn(logits, labels).item()) * labels.size(0)
            truth.extend(labels.cpu().tolist())
            closed_probabilities.extend(
                torch.softmax(logits, dim=1)[:, 0].cpu().tolist()
            )
    threshold_candidates = np.linspace(0.10, 0.90, 33)
    sweep = []
    for threshold in threshold_candidates:
        predicted = [
            0 if probability >= threshold else 1
            for probability in closed_probabilities
        ]
        metrics = classification_metrics(truth, predicted, class_names)
        closed_f1 = float(metrics["per_class"][class_names[0]]["f1"])
        metrics["decision_threshold"] = float(threshold)
        metrics["safety_score"] = (
            float(metrics["balanced_accuracy"]) + closed_f1
        ) / 2.0
        sweep.append(metrics)
    selected = max(
        sweep,
        key=lambda metrics: (
            float(metrics["safety_score"]),
            float(metrics["per_class"][class_names[0]]["recall"]),
            float(metrics["per_class"][class_names[0]]["precision"]),
        ),
    )
    return total_loss / len(loader.dataset), selected


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probabilities = torch.log_softmax(logits, dim=1)
        probabilities = torch.exp(log_probabilities)
        indices = torch.arange(targets.shape[0], device=targets.device)
        target_log = log_probabilities[indices, targets]
        target_probability = probabilities[indices, targets]
        loss = -((1.0 - target_probability) ** self.gamma) * target_log
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


def acceptance_result(task: str, metrics: dict[str, Any]) -> dict[str, Any]:
    per_class = metrics["per_class"]
    if task == "eye_state":
        closed = per_class["eye_closed"]
        checks = {
            "closed_recall_gte_0_85": closed["recall"] >= 0.85,
            "closed_precision_gte_0_75": closed["precision"] >= 0.75,
            "closed_f1_gte_0_80": closed["f1"] >= 0.80,
            "balanced_accuracy_gte_0_85": metrics["balanced_accuracy"] >= 0.85,
        }
    elif task == "seat_belt":
        checks = {
            "no_seat_belt_recall_gte_0_85": (
                per_class["no_seat_belt"]["recall"] >= 0.85
            ),
            "seat_belt_on_recall_gte_0_85": (
                per_class["seat_belt_on"]["recall"] >= 0.85
            ),
            "balanced_accuracy_gte_0_85": metrics["balanced_accuracy"] >= 0.85,
        }
    else:
        checks = {}
    return {"passed": bool(checks) and all(checks.values()), "checks": checks}


def onnx_parity(
    model: nn.Module,
    onnx_path: Path,
    img_size: int,
    device: torch.device,
) -> dict[str, Any]:
    sample = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.inference_mode():
        torch_output = model(sample).cpu().numpy()
    try:
        import onnx

        onnx.checker.check_model(onnx.load(str(onnx_path)))
        checker = "ok"
    except ImportError:
        checker = "onnx_not_installed"
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        runtime_output = session.run(
            None, {session.get_inputs()[0].name: sample.cpu().numpy()}
        )[0]
        max_abs_error = float(np.max(np.abs(torch_output - runtime_output)))
        return {
            "onnx_checker": checker,
            "onnxruntime": "ok",
            "max_abs_error": max_abs_error,
            "argmax_match": bool(
                np.argmax(torch_output, axis=1)[0]
                == np.argmax(runtime_output, axis=1)[0]
            ),
        }
    except ImportError:
        return {
            "onnx_checker": checker,
            "onnxruntime": "not_installed",
        }


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    seed_everything(args.seed)
    spec = TASKS[args.task]
    class_names = list(spec["class_names"])
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    img_size = int(spec["img_size"])
    manifest_path = args.manifest.resolve()
    rows = read_rows(manifest_path, args.task)
    provenance = load_preparation_provenance(manifest_path, args.task)
    unexpected = sorted(
        {
            row["class_name"]
            for split_rows in rows.values()
            for row in split_rows
            if row["class_name"] not in class_to_idx
        }
    )
    if unexpected:
        raise ValueError(f"Unexpected classes for {args.task}: {unexpected}")
    for required_class in class_names:
        if not any(row["class_name"] == required_class for row in rows["train"]):
            raise ValueError(f"Training split is missing class {required_class}")

    train_transform, val_transform = make_transforms(img_size)
    train_dataset = ManifestDataset(rows["train"], class_to_idx, train_transform)
    val_dataset = ManifestDataset(rows["val"], class_to_idx, val_transform)
    train_counts = Counter(row["class_name"] for row in rows["train"])
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    shuffle = args.sampler == "shuffle"
    if args.sampler == "balanced":
        sample_weights = [
            1.0 / train_counts[row["class_name"]]
            for row in rows["train"]
        ]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = build_dms_classifier(
        architecture=args.architecture,
        num_classes=len(class_names),
        pretrained=args.pretrained,
    ).to(device)
    weights = torch.tensor(
        [
            len(rows["train"]) / (len(class_names) * train_counts[name])
            for name in class_names
        ],
        dtype=torch.float32,
        device=device,
    )
    if args.loss == "weighted_ce":
        loss_fn: nn.Module = nn.CrossEntropyLoss(weight=weights)
    elif args.loss == "focal":
        loss_fn = FocalLoss(alpha=weights)
    else:
        loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    history: list[dict[str, Any]] = []
    best_safety_score = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        training_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            training_loss += float(loss.item()) * labels.size(0)
        val_loss, metrics = evaluate(
            model, val_loader, loss_fn, device, class_names
        )
        safety_score = float(metrics["safety_score"])
        scheduler.step(safety_score)
        record = {
            "epoch": epoch,
            "train_loss": training_loss / len(train_dataset),
            "val_loss": val_loss,
            **metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        if safety_score > best_safety_score + 1e-8:
            best_safety_score = safety_score
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "img_size": img_size,
                    "mean": MEAN,
                    "std": STD,
                    "task": args.task,
                    "architecture": args.architecture,
                    "loss": args.loss,
                    "sampler": args.sampler,
                    "seed": args.seed,
                    "manifest_sha256": sha256_file(manifest_path),
                    "provenance": provenance,
                    "metrics": metrics,
                    "epoch": epoch,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    acceptance = acceptance_result(args.task, checkpoint["metrics"])
    training_result = {
        "task": args.task,
        "architecture": args.architecture,
        "loss": args.loss,
        "sampler": args.sampler,
        "class_to_idx": class_to_idx,
        "best_epoch": checkpoint["epoch"],
        "best_metrics": checkpoint["metrics"],
        "acceptance": acceptance,
        "provenance": provenance,
        "history": history,
    }
    training_result_path = output_dir / "training_result.json"
    training_result_path.write_text(
        json.dumps(training_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not acceptance["passed"]:
        print(f"Acceptance gates failed: {training_result_path}")
        return 3
    onnx_path = output_dir / "model.onnx"
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    parity = onnx_parity(model, onnx_path, img_size, device)
    metadata = {
        "schema_version": 1,
        "task": args.task,
        "architecture": args.architecture,
        "loss": args.loss,
        "sampler": args.sampler,
        "class_to_idx": class_to_idx,
        "img_size": img_size,
        "mean": MEAN,
        "std": STD,
        "seed": args.seed,
        "prepared_manifest": str(manifest_path),
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "provenance": provenance,
        "training_counts": dict(train_counts),
        "validation_counts": dict(
            Counter(row["class_name"] for row in rows["val"])
        ),
        "class_weights": {
            name: float(weights[index].item())
            for index, name in enumerate(class_names)
        },
        "best_epoch": checkpoint["epoch"],
        "best_metrics": checkpoint["metrics"],
        "decision_threshold": checkpoint["metrics"]["decision_threshold"],
        "acceptance": acceptance,
        "history": history,
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_parity": parity,
        "command": " ".join(sys.argv),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    metadata_path = output_dir / "model.metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Best checkpoint: {best_path}")
    print(f"ONNX: {onnx_path}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
