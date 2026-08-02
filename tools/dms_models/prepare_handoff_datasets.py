#!/usr/bin/env python3
"""Verify and prepare the final-reviewed 2026-07-30 DMS handoff.

This script never changes the handoff. It resolves only package-relative paths,
validates every selected image against manifest dimensions and SHA-256, excludes
invalid/duplicate/uncertain rows, assigns deterministic source-grouped splits,
and optionally materializes a YOLO dataset as symlinks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2


AUTHORITATIVE_HANDOFF_ROOT = Path(
    "/home/vicharak/Mobility_ADAS/ADVIS/DMS/DMS_VICHARAK_HANDOFF_2026_0730"
)
TASK_IMAGE_DIRS = {
    "eye_state": "01_IMAGES/01_Eye_state_dataset",
    "seat_belt": "01_IMAGES/02_Seat_Belt_detection",
    "phone_classifier": "01_IMAGES/03_Phone_detection",
    "phone_object_detection": "01_IMAGES/03_Phone_detection",
}
MANIFESTS = {
    "eye_state": "02_ANNOTATIONS_MANIFESTS/eye_state/final_reviewed_eye_labels.csv",
    "seat_belt": "02_ANNOTATIONS_MANIFESTS/seat_belt/final_reviewed_seatbelt_labels.csv",
    "phone_classifier": (
        "02_ANNOTATIONS_MANIFESTS/phone_detection/final_phone_classifier_labels.csv"
    ),
    "phone_object_detection": (
        "02_ANNOTATIONS_MANIFESTS/phone_detection/final_yolo_frame_manifest.csv"
    ),
}
CLASS_MAPS = {
    "eye_state": {"closed": "eye_closed", "open": "eye_open"},
    "seat_belt": {
        "no_seat_belt": "no_seat_belt",
        "seat_belt_on": "seat_belt_on",
    },
    "phone_classifier": {
        "no_phone_hard_negative": "no_phone_hard_negative",
        "phone": "phone",
    },
    "phone_object_detection": {
        "hard_negative": "hard_negative",
        "phone": "phone",
    },
}
OUTPUT_FIELDS = [
    "task",
    "split",
    "class_name",
    "final_label",
    "image_path",
    "packaged_path",
    "label_path",
    "source_video",
    "source_group",
    "sha256",
    "width",
    "height",
    "channels",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--handoff-root",
        type=Path,
        default=AUTHORITATIVE_HANDOFF_ROOT,
        help="Must be the authoritative 2026-07-30 handoff root",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("local_data/dms_handoff_20260730")
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Write a clean manifest while excluding invalid rows; otherwise exit 2",
    )
    parser.add_argument(
        "--materialize-yolo",
        action="store_true",
        help="Create a YOLO-compatible symlink tree from verified detector rows",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_package_path(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError("path is empty or absolute")
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError("path escapes handoff root")
    return candidate


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def require_task_source(
    task: str, path: Path, handoff_root: Path, kind: str = "image"
) -> None:
    allowed_root = (handoff_root / TASK_IMAGE_DIRS[task]).resolve()
    resolved = path.resolve()
    if resolved != allowed_root and allowed_root not in resolved.parents:
        raise ValueError(
            f"{kind} is outside authoritative {task} source: "
            f"{resolved} (required root: {allowed_root})"
        )


def grouped_split(
    rows: list[dict[str, Any]], seed: int, val_fraction: float
) -> None:
    """Assign whole known source videos to validation; unknown sources train-only."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    totals = Counter(row["class_name"] for row in rows)
    for row in rows:
        source = str(row["source_video"]).strip()
        if source:
            row["source_group"] = source
            groups[source].append(row)
        else:
            row["source_group"] = "__unknown_source_train_only__"
            row["split"] = "train"

    target = {label: totals[label] * val_fraction for label in totals}
    group_counts = {
        group: Counter(row["class_name"] for row in items)
        for group, items in groups.items()
    }
    validation: set[str] = set()
    validation_counts: Counter[str] = Counter()

    def score(counts: Counter[str]) -> float:
        return sum(
            abs(float(counts[label]) - target[label]) / max(1.0, target[label])
            for label in totals
        )

    candidates = sorted(groups, key=lambda group: stable_rank(seed, group))
    changed = True
    while changed:
        changed = False
        best_group = ""
        best_score = score(validation_counts)
        for group in candidates:
            if group in validation:
                continue
            proposed = validation_counts + group_counts[group]
            proposed_score = score(proposed)
            if proposed_score + 1e-12 < best_score:
                best_group, best_score = group, proposed_score
        if best_group:
            validation.add(best_group)
            validation_counts += group_counts[best_group]
            changed = True

    # Where possible, make every class measurable in validation.
    for label in totals:
        if validation_counts[label] > 0:
            continue
        choices = [
            group
            for group in candidates
            if group not in validation and group_counts[group][label] > 0
        ]
        if choices:
            best_group = min(
                choices,
                key=lambda group: (
                    score(validation_counts + group_counts[group]),
                    stable_rank(seed, group),
                ),
            )
            validation.add(best_group)
            validation_counts += group_counts[best_group]

    # Never consume every known group; training must remain possible.
    if groups and validation == set(groups):
        removed = max(
            validation,
            key=lambda group: (
                sum(group_counts[group].values()),
                stable_rank(seed, group),
            ),
        )
        validation.remove(removed)

    for group, items in groups.items():
        split = "val" if group in validation else "train"
        for row in items:
            row["split"] = split


def verify_manifest(
    task: str, manifest: Path, root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, str]], Counter[str]]:
    accepted: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    exclusions: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        for line_number, source in enumerate(csv.DictReader(stream), start=2):
            if not truthy(source.get("manual_reviewed", "")):
                exclusions["not_manual_reviewed"] += 1
                continue
            if not truthy(source.get("recommended_for_training", "")):
                exclusions["not_recommended"] += 1
                continue
            final_label = source.get("final_label", "").strip()
            class_name = CLASS_MAPS[task].get(final_label)
            if class_name is None:
                exclusions["unsupported_or_uncertain_label"] += 1
                continue
            relative = source.get("packaged_path", "").strip()
            try:
                image_path = safe_package_path(root, relative)
                require_task_source(task, image_path, root)
            except ValueError as exc:
                errors.append(
                    {
                        "task": task,
                        "line": str(line_number),
                        "path": relative,
                        "error": str(exc),
                    }
                )
                continue
            try:
                actual_hash = sha256_file(image_path)
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError("image decode failed")
                height, width = image.shape[:2]
                channels = 1 if image.ndim == 2 else image.shape[2]
                expected = (
                    int(source["width"]),
                    int(source["height"]),
                    int(source["channels"]),
                )
                actual = (width, height, channels)
                if actual_hash.lower() != source["sha256"].strip().lower():
                    raise ValueError(
                        f"sha256 mismatch expected={source['sha256']} actual={actual_hash}"
                    )
                if actual != expected:
                    raise ValueError(
                        f"image shape mismatch expected={expected} actual={actual}"
                    )
            except (OSError, ValueError, KeyError) as exc:
                errors.append(
                    {
                        "task": task,
                        "line": str(line_number),
                        "path": relative,
                        "error": str(exc),
                    }
                )
                continue
            if actual_hash in seen_hashes:
                exclusions["duplicate_sha256"] += 1
                continue
            seen_hashes.add(actual_hash)
            label_relative = source.get("yolo_label_path", "").strip()
            label_path = ""
            if label_relative:
                try:
                    resolved_label = safe_package_path(root, label_relative)
                    require_task_source(task, resolved_label, root, kind="label")
                    if not resolved_label.is_file():
                        raise ValueError("YOLO label file missing")
                    label_path = str(resolved_label)
                except ValueError as exc:
                    errors.append(
                        {
                            "task": task,
                            "line": str(line_number),
                            "path": label_relative,
                            "error": str(exc),
                        }
                    )
                    continue
            accepted.append(
                {
                    "task": task,
                    "split": source.get("split", "").strip(),
                    "class_name": class_name,
                    "final_label": final_label,
                    "image_path": str(image_path),
                    "packaged_path": relative,
                    "label_path": label_path,
                    "source_video": source.get("source_video", "").strip(),
                    "source_group": "",
                    "sha256": actual_hash,
                    "width": width,
                    "height": height,
                    "channels": channels,
                }
            )
    return accepted, errors, exclusions


def materialize_yolo(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    yolo_root = output_dir / "phone_yolo"
    for row in rows:
        split = row["split"]
        image_dir = yolo_root / "images" / split
        label_dir = yolo_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image_target = Path(row["image_path"])
        label_target = Path(row["label_path"])
        image_link = image_dir / image_target.name
        label_link = label_dir / label_target.name
        for link, target in ((image_link, image_target), (label_link, label_target)):
            if link.is_symlink() and link.resolve() == target.resolve():
                continue
            if link.exists() or link.is_symlink():
                raise FileExistsError(f"Refusing to replace generated path: {link}")
            os.symlink(target.resolve(), link)
    data_yaml = yolo_root / "data.yaml"
    data_yaml.write_text(
        f"path: {yolo_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: phone\n",
        encoding="utf-8",
    )
    return data_yaml


def main() -> int:
    args = parse_args()
    root = args.handoff_root.resolve()
    required_root = AUTHORITATIVE_HANDOFF_ROOT.resolve()
    if root != required_root:
        raise ValueError(
            f"Refusing non-authoritative handoff root {root}; required "
            f"{required_root}"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    all_errors: list[dict[str, str]] = []
    exclusions: dict[str, dict[str, int]] = {}
    source_manifests: dict[str, dict[str, str]] = {}

    for task, relative in MANIFESTS.items():
        manifest = safe_package_path(root, relative)
        source_manifests[task] = {
            "path": relative,
            "sha256": sha256_file(manifest),
        }
        rows, errors, excluded = verify_manifest(task, manifest, root)
        if task == "phone_object_detection":
            for row in rows:
                row["source_group"] = row["source_video"] or "__unknown_source__"
                if row["split"] not in {"train", "val"}:
                    errors.append(
                        {
                            "task": task,
                            "line": "",
                            "path": row["packaged_path"],
                            "error": f"invalid supplied split {row['split']!r}",
                        }
                    )
        else:
            grouped_split(rows, args.seed, args.val_fraction)
        all_rows.extend(rows)
        all_errors.extend(errors)
        exclusions[task] = dict(excluded)

    all_rows.sort(
        key=lambda row: (
            row["task"],
            row["split"],
            row["class_name"],
            row["packaged_path"],
        )
    )
    manifest_path = output_dir / "prepared_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    counts: dict[str, dict[str, int]] = defaultdict(dict)
    counted = Counter(
        (row["task"], row["split"], row["class_name"]) for row in all_rows
    )
    for (task, split, label), count in sorted(counted.items()):
        counts[task][f"{split}/{label}"] = count
    split_hashes = {}
    for task in TASK_IMAGE_DIRS:
        payload = "".join(
            (
                f"{row['split']}\t{row['class_name']}\t{row['sha256']}\t"
                f"{row['packaged_path']}\n"
            )
            for row in all_rows
            if row["task"] == task
        )
        split_hashes[task] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    metadata = {
        "schema_version": 1,
        "handoff_root": str(root),
        "source_manifests": source_manifests,
        "authoritative_image_roots": {
            task: str((root / relative).resolve())
            for task, relative in TASK_IMAGE_DIRS.items()
        },
        "prepared_manifest": manifest_path.name,
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "split_sha256": split_hashes,
        "seed": args.seed,
        "validation_fraction": args.val_fraction,
        "split_policy": (
            "source_video-grouped; unknown-source classifier rows train-only; "
            "phone detector supplied split preserved"
        ),
        "counts": counts,
        "exclusions": exclusions,
        "validation_errors": all_errors,
        "versions": {
            "python": sys.version.split()[0],
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
    }
    metadata_path = output_dir / "preparation_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.materialize_yolo:
        detector_rows = [
            row for row in all_rows if row["task"] == "phone_object_detection"
        ]
        data_yaml = materialize_yolo(detector_rows, output_dir)
        print(f"YOLO data: {data_yaml}")
    print(f"Prepared manifest: {manifest_path}")
    print(json.dumps(counts, indent=2, sort_keys=True))
    if all_errors:
        print(f"Validation errors: {len(all_errors)} (see {metadata_path})")
        return 0 if args.allow_invalid else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
