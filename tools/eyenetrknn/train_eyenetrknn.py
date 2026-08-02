from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from ind_vias_dms.eyenetrknn.model import build_eyenetrknn_model


CLASSES = [
    "eye_closed",
    "eye_open",
]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="datasets/eye_state")
    parser.add_argument("--out", type=str, default="models/eyenetrknn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    return parser.parse_args()


def build_loaders(data_dir: Path, img_size: int, batch_size: int, num_workers: int):
    train_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.25,
                        contrast=0.25,
                        saturation=0.10,
                        hue=0.03,
                    )
                ],
                p=0.5,
            ),
            transforms.RandomRotation(degrees=8),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.06, 0.06),
                scale=(0.90, 1.10),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN,
                std=STD,
            ),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN,
                std=STD,
            ),
        ]
    )

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_tfms)

    print("Train images:", len(train_ds))
    print("Val images:", len(val_ds))
    print("Class mapping:", train_ds.class_to_idx)
    expected = {name: index for index, name in enumerate(CLASSES)}
    if train_ds.class_to_idx != expected:
        raise ValueError(
            f"Expected the reviewed binary class map {expected}, got "
            f"{train_ds.class_to_idx}. Do not add a synthetic bad_crop class."
        )
    if val_ds.class_to_idx != train_ds.class_to_idx:
        raise ValueError("Training and validation class maps differ")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, train_ds.class_to_idx


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
        desc = "train"
    else:
        model.eval()
        desc = "val"

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)

    return avg_loss, avg_acc


def main():
    args = parse_args()
    raise RuntimeError(
        "This ImageFolder trainer is retained for implementation history but "
        "disabled to prevent legacy dataset mixing. Use "
        "tools/dms_models/train_classifier.py with the verified prepared "
        "manifest and --task eye_state."
    )

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader, val_loader, class_to_idx = build_loaders(
        data_dir=data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_eyenetrknn_model(
        num_classes=len(class_to_idx),
        pretrained=args.pretrained,
    ).to(device)

    class_counts = torch.bincount(
        torch.tensor(train_loader.dataset.targets),
        minlength=len(class_to_idx),
    ).float()
    class_weights = len(train_loader.dataset) / (
        len(class_to_idx) * class_counts.clamp_min(1.0)
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.05,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    metadata = {
        "model": "MobileNetV3-Small",
        "task": "eye_state_classification",
        "img_size": args.img_size,
        "class_to_idx": class_to_idx,
        "idx_to_class": {v: k for k, v in class_to_idx.items()},
        "mean": MEAN,
        "std": STD,
        "class_weights": class_weights.tolist(),
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )

        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            train=False,
        )

        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        ckpt = {
            "model_state": model.state_dict(),
            "metadata": metadata,
            "class_to_idx": class_to_idx,
            "img_size": args.img_size,
            "mean": MEAN,
            "std": STD,
            "epoch": epoch,
            "val_acc": val_acc,
        }

        torch.save(ckpt, out_dir / "eyenetrknn_last.pt")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, out_dir / "eyenetrknn_best.pt")
            print(f"Saved best model: val_acc={best_val_acc:.4f}")

    print("Training complete.")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
