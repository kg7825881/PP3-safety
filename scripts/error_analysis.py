"""Recall by object size and confidence threshold, at IoU >= 0.5.

Uses VALIDATION data by default: thresholds and model choices must never be tuned on test.
Objects are bucketed by the square root of their box area after scaling to the model input.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ppe_guard.config import ModelConfig
from ppe_guard.dataset import TARGET_CLASSES, parse_labels
from ppe_guard.detector import Detector

THRESHOLDS = [0.1, 0.15, 0.25, 0.35, 0.5]


def iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def match(gts, preds) -> tuple[set[int], int]:
    """Greedy matching, most confident first. Returns (matched gt indices, false positives)."""
    used: set[int] = set()
    fp = 0
    for det in sorted(preds, key=lambda d: -d.confidence):
        box = (det.x1, det.y1, det.x2, det.y2)
        best = max(
            ((iou(g, box), i) for i, g in enumerate(gts) if i not in used), default=(0.0, -1)
        )
        if best[0] >= 0.5:
            used.add(best[1])
        else:
            fp += 1
    return used, fp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="weights/ppe_yolo26n.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tag", default="errors_val")
    args = ap.parse_args()

    root = Path("data/datasets/ppe5")
    det = Detector(
        ModelConfig(weights=args.weights, image_size=args.imgsz, conf_threshold=min(THRESHOLDS))
    )
    tally = defaultdict(lambda: [0, 0, 0])  # (class, threshold) -> tp, fp, gt
    by_size = defaultdict(lambda: [0, 0])  # (class, bucket) -> found, total at conf 0.25
    for path in sorted((root / "images" / args.split).iterdir()):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        scale = 640 / max(h, w)  # fixed reference so buckets compare across input sizes
        boxes, _ = parse_labels((root / "labels" / args.split / f"{path.stem}.txt").read_text())
        preds = det.predict(img).detections
        for cls, name in enumerate(TARGET_CLASSES):
            gts = [
                ((b.cx - b.w / 2) * w, (b.cy - b.h / 2) * h,
                 (b.cx + b.w / 2) * w, (b.cy + b.h / 2) * h)
                for b in boxes if b.class_id == cls
            ]
            for t in THRESHOLDS:
                used, fp = match(
                    gts, [p for p in preds if p.class_id == cls and p.confidence >= t]
                )
                row = tally[(name, t)]
                row[0] += len(used)
                row[1] += fp
                row[2] += len(gts)
                if t == 0.25:
                    for i, g in enumerate(gts):
                        side = np.sqrt((g[2] - g[0]) * (g[3] - g[1])) * scale
                        bucket = "<16px" if side < 16 else "16-32px" if side < 32 else ">=32px"
                        by_size[(name, bucket)][1] += 1
                        by_size[(name, bucket)][0] += i in used

    out = {"weights": args.weights, "split": args.split, "imgsz": args.imgsz,
           "thresholds": {}, "recall_by_size_at_0.25": {}}
    for (name, t), (tp, fp, n) in tally.items():
        out["thresholds"].setdefault(name, {})[str(t)] = {
            "recall": round(tp / n, 3), "precision": round(tp / max(tp + fp, 1), 3)}
    for (name, bucket), (found, n) in sorted(by_size.items()):
        out["recall_by_size_at_0.25"].setdefault(name, {})[bucket] = f"{found}/{n}"
    Path(f"benchmarks/{args.tag}.json").write_text(json.dumps(out, indent=2))
    for name, rows in out["thresholds"].items():
        print(name, {t: (r["recall"], r["precision"]) for t, r in rows.items()})
    for name, rows in out["recall_by_size_at_0.25"].items():
        print(name, rows)


if __name__ == "__main__":
    main()
