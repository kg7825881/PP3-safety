"""Audit a YOLO-format PPE dataset and prepare leak-resistant train/val/test splits.

The source dataset ships train and val only. We keep val for model selection and
carve a TEST set out of train, so the final score comes from images that influenced
neither the weights nor the choice of epoch.

Near-duplicate photos (burst shots, video frames) are grouped with a perceptual hash:
a group never straddles two splits, otherwise the model is graded on images it has
effectively already seen.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Our label space: only what the compliance rules consume. Order = class id.
TARGET_CLASSES = ["person", "helmet", "vest", "no_helmet", "no_vest"]


@dataclass(frozen=True, slots=True)
class Box:
    """YOLO label: class id plus center/size normalized to 0..1."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float


def parse_labels(text: str) -> tuple[list[Box], list[str]]:
    """Return (valid boxes, problems). Bad lines are reported, never silently kept."""
    boxes: list[Box] = []
    problems: list[str] = []
    seen: set[Box] = set()
    for n, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            problems.append(f"line {n}: expected 5 fields, got {len(parts)}")
            continue
        try:
            values = [float(p) for p in parts]
        except ValueError:
            problems.append(f"line {n}: not numeric")
            continue
        if not values[0].is_integer() or values[0] < 0:
            problems.append(f"line {n}: bad class id")
            continue
        box = Box(int(values[0]), *values[1:])
        if not (0 <= box.cx <= 1 and 0 <= box.cy <= 1 and 0 < box.w <= 1 and 0 < box.h <= 1):
            problems.append(f"line {n}: box outside 0..1")
        elif box in seen:
            problems.append(f"line {n}: duplicate box")
        else:
            seen.add(box)
            boxes.append(box)
    return boxes, problems


def class_mapping(source_names: dict[int, str]) -> dict[int, int]:
    """Map source class ids to TARGET_CLASSES ids by name; other classes are dropped."""
    by_name = {name.lower(): idx for idx, name in source_names.items()}
    missing = [c for c in TARGET_CLASSES if c not in by_name]
    if missing:
        raise ValueError(f"Source dataset lacks required classes: {missing}")
    return {by_name[name]: new_id for new_id, name in enumerate(TARGET_CLASSES)}


def dhash(image: np.ndarray) -> int:
    """64-bit difference hash: survives resizing and recompression."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def group_duplicates(hashes: list[int], max_distance: int = 4) -> list[int]:
    """Union-find over hashes within `max_distance` bits; returns a group id per image."""
    parent = list(range(len(hashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if (hashes[i] ^ hashes[j]).bit_count() <= max_distance:
                parent[find(j)] = find(i)
    return [find(i) for i in range(len(hashes))]


def assign_splits(
    source_splits: list[str], groups: list[int], test_count: int, seed: int
) -> list[str]:
    """Official val stays val; train images duplicating val are dropped; test comes from train."""
    val_groups = {g for s, g in zip(source_splits, groups, strict=True) if s == "val"}
    out: list[str] = []
    members: dict[int, list[int]] = defaultdict(list)
    for i, (split, group) in enumerate(zip(source_splits, groups, strict=True)):
        if split == "val":
            out.append("val")
        elif group in val_groups:
            out.append("dropped")
        else:
            out.append("train")
            members[group].append(i)

    order = sorted(members)
    random.Random(seed).shuffle(order)
    taken = 0
    for group in order:
        if taken >= test_count:
            break
        for i in members[group]:
            out[i] = "test"
        taken += len(members[group])
    return out


def prepare(
    source: Path,
    source_names: dict[int, str],
    out: Path,
    report_path: Path,
    test_count: int = 120,
    seed: int = 42,
) -> dict:
    """Audit `source`, write the prepared dataset to `out`, and return the audit report."""
    mapping = class_mapping(source_names)
    paths: list[Path] = []
    splits: list[str] = []
    hashes: list[int] = []
    labels: list[list[Box]] = []
    problems: Counter[str] = Counter()

    for split in ("train", "val"):
        for path in sorted((source / "images" / split).rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(path))
            if image is None:
                problems["unreadable image"] += 1
                continue
            rel = path.relative_to(source / "images")
            label_path = (source / "labels" / rel).with_suffix(".txt")
            if not label_path.exists():
                # Unlabeled is not the same as "no objects": training on it as
                # background would teach the model to ignore real workers.
                problems["missing label file"] += 1
                continue
            boxes, issues = parse_labels(label_path.read_text())
            for issue in issues:
                problems[issue.split(": ", 1)[1]] += 1
            for b in boxes:
                if b.class_id not in source_names:
                    problems["unknown class id"] += 1
            paths.append(path)
            splits.append(split)
            hashes.append(dhash(image))
            labels.append(boxes)

    if not paths:
        raise FileNotFoundError(f"No labeled images found under {source / 'images'}")

    groups = group_duplicates(hashes)
    assigned = assign_splits(splits, groups, test_count, seed)

    if out.exists():
        # Only ever delete a directory this function generated.
        if not (out / "data.yaml").exists() or out.resolve() == source.resolve():
            raise FileExistsError(f"{out} exists and is not a prepared dataset; not deleting it")
        shutil.rmtree(out)
    counts: dict[str, Counter] = defaultdict(Counter)
    images: Counter[str] = Counter()
    manifest: dict[str, list[str]] = defaultdict(list)
    source_counts = Counter(source_names.get(b.class_id, "?") for bs in labels for b in bs)

    for i, (path, boxes, split) in enumerate(zip(paths, labels, assigned, strict=True)):
        images[split] += 1
        manifest[split].append(str(path.relative_to(source)))
        if split == "dropped":
            continue
        stem = f"{i:05d}_{path.stem}"  # index prefix avoids same-name collisions
        img_out = out / "images" / split / f"{stem}{path.suffix.lower()}"
        lbl_out = out / "labels" / split / f"{stem}.txt"
        img_out.parent.mkdir(parents=True, exist_ok=True)
        lbl_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, img_out)  # copy: training tools may rewrite damaged JPEGs in place
        kept = [(mapping[b.class_id], b) for b in boxes if b.class_id in mapping]
        for cls, _ in kept:
            counts[split][TARGET_CLASSES[cls]] += 1
        lbl_out.write_text(
            "".join(f"{c} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}\n" for c, b in kept)
        )

    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": dict(enumerate(TARGET_CLASSES)),
            },
            sort_keys=False,
        )
    )

    sizes = Counter(groups)
    report = {
        "seed": seed,
        "source_images": len(paths),
        "images_per_split": dict(images),
        "instances_per_split": {s: dict(c) for s, c in counts.items()},
        "source_instances": dict(source_counts.most_common()),
        "label_problems": dict(problems),
        "near_duplicate_groups": sum(1 for n in sizes.values() if n > 1),
        "largest_duplicate_group": max(sizes.values()),
        "manifest": {s: sorted(v) for s, v in manifest.items()},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1))
    return report
