"""Per-class evaluation of a trained detector on a chosen split."""

from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

from ppe_guard.devices import resolve_device


def evaluate(
    weights: str,
    data: str,
    split: str = "test",
    device: str = "auto",
    image_size: int = 640,
    out_dir: str | Path = "benchmarks",
    tag: str = "eval",
) -> dict:
    """Compute precision, recall, mAP50 and mAP50-95 for every class."""
    metrics = YOLO(weights).val(
        data=data,
        split=split,
        device=resolve_device(device),
        imgsz=image_size,
        plots=True,
        project=str(Path("runs/eval").resolve()),
        name=tag,
        exist_ok=True,
        verbose=False,
    )
    box = metrics.box
    per_class = {}
    # box.ap_class_index lists only classes present in this split.
    for i, cls_id in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        per_class[metrics.names[int(cls_id)]] = {
            "precision": round(float(p), 3),
            "recall": round(float(r), 3),
            "map50": round(float(ap50), 3),
            "map50_95": round(float(ap), 3),
        }
    report = {
        "weights": weights,
        "split": split,
        "image_size": image_size,
        "overall": {
            "precision": round(float(box.mp), 3),
            "recall": round(float(box.mr), 3),
            "map50": round(float(box.map50), 3),
            "map50_95": round(float(box.map), 3),
        },
        "per_class": per_class,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(report, indent=2))

    rows = [("**all**", report["overall"]), *per_class.items()]
    table = "| Class | Precision | Recall | mAP50 | mAP50-95 |\n|---|---|---|---|---|\n" + "".join(
        f"| {n} | {m['precision']} | {m['recall']} | {m['map50']} | {m['map50_95']} |\n"
        for n, m in rows
    )
    (out / f"{tag}.md").write_text(f"`{weights}` on **{split}** split\n\n{table}")
    return report
