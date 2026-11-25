"""Fine-tune YOLO on the prepared PPE dataset."""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from ppe_guard.devices import resolve_device


def train(
    data: str,
    weights: str,
    name: str,
    epochs: int = 100,
    patience: int = 20,
    batch: int = 16,
    image_size: int = 640,
    device: str = "auto",
    seed: int = 0,
    fraction: float = 1.0,
) -> Path:
    """Transfer learning: start from COCO weights, replace the head, train end to end.

    Returns the path of best.pt, the epoch with the best VALIDATION fitness.
    """
    model = YOLO(weights)
    model.train(
        data=data,
        epochs=epochs,
        patience=patience,  # early stopping: quit after N epochs without improvement
        batch=batch,
        imgsz=image_size,
        device=resolve_device(device),
        seed=seed,
        fraction=fraction,
        workers=4,
        project=str(Path("runs/train").resolve()),
        name=name,
        exist_ok=True,
        plots=True,
    )
    return Path(model.trainer.best)
