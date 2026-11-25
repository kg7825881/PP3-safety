"""Compare two models on VALIDATION at equal precision, with a paired image bootstrap.

Fixed-threshold comparisons mislead: a model can gain recall just by firing more often.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from error_analysis import iou

from ppe_guard.config import ModelConfig
from ppe_guard.dataset import TARGET_CLASSES, parse_labels
from ppe_guard.detector import Detector


def collect(weights: str, imgsz: int, split: str):
    """Per class: one (confidence, is_true_positive) array and one label count per image."""
    root = Path("data/datasets/ppe5")
    det = Detector(ModelConfig(weights=weights, image_size=imgsz, conf_threshold=0.01))
    preds = {c: [] for c in TARGET_CLASSES}
    counts = {c: [] for c in TARGET_CLASSES}
    for path in sorted((root / "images" / split).iterdir()):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        boxes, _ = parse_labels((root / "labels" / split / f"{path.stem}.txt").read_text())
        found = det.predict(img).detections
        for cls, name in enumerate(TARGET_CLASSES):
            gts = [
                ((b.cx - b.w / 2) * w, (b.cy - b.h / 2) * h,
                 (b.cx + b.w / 2) * w, (b.cy + b.h / 2) * h)
                for b in boxes if b.class_id == cls
            ]
            used: set[int] = set()
            rows = []
            for d in sorted((p for p in found if p.class_id == cls), key=lambda p: -p.confidence):
                score, i = max(
                    ((iou(g, (d.x1, d.y1, d.x2, d.y2)), i)
                     for i, g in enumerate(gts) if i not in used),
                    default=(0.0, -1),
                )
                if score >= 0.5:
                    used.add(i)
                rows.append((d.confidence, score >= 0.5))
            preds[name].append(np.array(rows, dtype=float).reshape(-1, 2))
            counts[name].append(len(gts))
    return preds, counts


def recall_at(preds, counts, idx, target: float) -> tuple[float, float]:
    """Highest recall whose precision is >= target, and the confidence where it occurs."""
    rows = np.concatenate([preds[i] for i in idx])
    rows = rows[np.argsort(-rows[:, 0], kind="stable")]
    tp = np.cumsum(rows[:, 1])
    ok = np.flatnonzero(tp / np.arange(1, len(rows) + 1) >= target)
    if not len(ok):
        return 0.0, 1.0
    return tp[ok[-1]] / sum(counts[i] for i in idx), rows[ok[-1], 0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", nargs=2, default=["weights/ppe_yolo26n.pt", "640"])
    ap.add_argument("--b", nargs=2, default=["weights/ppe_yolo26n_960.pt", "960"])
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--precision", type=float, default=0.9)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--tag", default="compare_val")
    args = ap.parse_args()

    a, n = collect(args.a[0], int(args.a[1]), args.split)
    b, _ = collect(args.b[0], int(args.b[1]), args.split)
    rng = np.random.default_rng(0)
    out = {"a": args.a, "b": args.b, "split": args.split, "precision": args.precision,
           "classes": {}}
    for c in TARGET_CLASSES:
        images = np.arange(len(n[c]))
        ra, ta = recall_at(a[c], n[c], images, args.precision)
        rb, tb = recall_at(b[c], n[c], images, args.precision)
        diffs = []
        for _ in range(args.resamples):
            s = rng.choice(images, len(images))
            diffs.append(
                recall_at(b[c], n[c], s, args.precision)[0]
                - recall_at(a[c], n[c], s, args.precision)[0]
            )
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out["classes"][c] = {
            "recall_a": round(ra, 3), "recall_b": round(rb, 3), "gain": round(rb - ra, 3),
            "gain_ci95": [round(lo, 3), round(hi, 3)],
            "conf_a": round(ta, 3), "conf_b": round(tb, 3),
        }
        print(c, out["classes"][c])
    Path(f"benchmarks/{args.tag}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
