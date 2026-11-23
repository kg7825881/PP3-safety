"""Object detector: wraps a YOLO model behind a small, stable interface.

The rest of the system only calls `Detector.predict(frame) -> FrameResult`.
Today the backend is Ultralytics/PyTorch; on Day 5-6 we add CoreML and ONNX
backends behind the same interface.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from ppe_guard.config import ModelConfig
from ppe_guard.devices import resolve_device
from ppe_guard.schema import Detection, FrameResult, StageTimings


class Detector:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        # YOLO() downloads official weights by name (e.g. "yolo11n.pt") on first use,
        # or loads a local file (our fine-tuned PPE model later).
        self.model = YOLO(cfg.weights)
        self.class_names: dict[int, str] = self.model.names
        # FP16 helps on GPU; on CPU it is unsupported/slow, so silently disable.
        self.half = cfg.half and self.device != "cpu"

    def warmup(self, runs: int = 3) -> None:
        """Run a few dummy frames.

        The first inferences are slow (kernel compilation, memory allocation, Metal
        shader caching). Warming up keeps that one-time cost out of benchmarks and
        out of the first seconds of a live stream.
        """
        dummy = np.zeros((self.cfg.image_size, self.cfg.image_size, 3), dtype=np.uint8)
        for _ in range(runs):
            self.predict(dummy)

    def predict(self, frame: np.ndarray, frame_index: int = 0) -> FrameResult:
        """Detect objects in one BGR frame (OpenCV's channel order)."""
        kwargs: dict = {}
        if self.half:  # only pass when enabled; the flag is deprecated in Ultralytics 8.4
            kwargs["half"] = True
        result = self.model.predict(
            frame,
            imgsz=self.cfg.image_size,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            classes=self.cfg.classes,
            device=self.device,
            verbose=False,
            **kwargs,
        )[0]

        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            # One bulk GPU->CPU copy per tensor instead of one per box.
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids, strict=True):
                detections.append(
                    Detection(
                        x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                        confidence=float(conf),
                        class_id=int(cls_id),
                        label=self.class_names.get(int(cls_id), str(cls_id)),
                    )
                )

        # Ultralytics measures each stage for us (milliseconds).
        speed = result.speed
        timings = StageTimings(
            preprocess_ms=speed.get("preprocess", 0.0),
            inference_ms=speed.get("inference", 0.0),
            postprocess_ms=speed.get("postprocess", 0.0),
        )
        return FrameResult(frame_index=frame_index, detections=detections, timings=timings)
