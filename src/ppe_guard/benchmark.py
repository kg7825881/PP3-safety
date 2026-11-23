"""Reproducible latency benchmark across models and devices.

Method (what to say in an interview):
  1. Same input frames for every run (a fixed image or the first N video frames).
  2. Warm-up runs are excluded (first runs pay one-time compile/alloc cost).
  3. Report mean, p50, p95, p99 and max of model latency, split into pre/infer/post.
  4. Throughput (FPS) = 1000 / mean latency of the model alone, single stream.

Caveat on the stage split for GPU (MPS) runs: GPU work is queued asynchronously,
so "inference" can return before the GPU finishes; the wait is then paid when
results are copied to the CPU, which lands in "postprocess". The TOTAL is
accurate on every device; the split is only exact on CPU.
Results are written to benchmarks/ as JSON (machine-readable) and Markdown (for docs).
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics.utils import ASSETS

from ppe_guard.config import ModelConfig
from ppe_guard.detector import Detector
from ppe_guard.metrics import LatencyTracker


@dataclass
class BenchResult:
    weights: str
    device: str
    image_size: int
    runs: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    fps: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    detections_per_frame: float


def load_frames(source: str | None, n: int) -> list[np.ndarray]:
    """Frames to benchmark on. Default: Ultralytics' bundled street image (has people)."""
    if source is None:
        img = cv2.imread(str(ASSETS / "bus.jpg"))
        return [img] * n
    cap = cv2.VideoCapture(source)
    frames: list[np.ndarray] = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {source}")
    return frames


def bench_one(weights: str, device: str, frames: list[np.ndarray], image_size: int) -> BenchResult:
    det = Detector(ModelConfig(weights=weights, device=device, image_size=image_size))
    det.warmup(runs=10)

    total = LatencyTracker(window=len(frames))
    pre, inf, post, n_det = [], [], [], []
    for i, frame in enumerate(frames):
        r = det.predict(frame, frame_index=i)
        total.record(r.timings.total_ms)
        pre.append(r.timings.preprocess_ms)
        inf.append(r.timings.inference_ms)
        post.append(r.timings.postprocess_ms)
        n_det.append(len(r.detections))

    mean = total.mean_ms
    return BenchResult(
        weights=weights,
        device=det.device,
        image_size=image_size,
        runs=len(frames),
        mean_ms=round(mean, 2),
        p50_ms=round(total.percentile(50), 2),
        p95_ms=round(total.percentile(95), 2),
        p99_ms=round(total.percentile(99), 2),
        max_ms=round(total.percentile(100), 2),
        fps=round(1000 / mean, 1) if mean else 0.0,
        preprocess_ms=round(float(np.mean(pre)), 2),
        inference_ms=round(float(np.mean(inf)), 2),
        postprocess_ms=round(float(np.mean(post)), 2),
        detections_per_frame=round(float(np.mean(n_det)), 1),
    )


def environment() -> dict[str, str]:
    return {
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "macos": platform.mac_ver()[0] or platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }


def to_markdown(results: list[BenchResult], env: dict[str, str]) -> str:
    head = (
        "| Model | Device | imgsz | mean ms | p50 | p95 | p99 | max | FPS | pre | infer | post |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = "".join(
        f"| {Path(r.weights).name} | {r.device} | {r.image_size} | {r.mean_ms} | {r.p50_ms} "
        f"| {r.p95_ms} | {r.p99_ms} | {r.max_ms} | {r.fps} "
        f"| {r.preprocess_ms} | {r.inference_ms} | {r.postprocess_ms} |\n"
        for r in results
    )
    env_line = ", ".join(f"{k}: {v}" for k, v in env.items())
    return f"{head}{rows}\n_Environment: {env_line}_\n"


def run_benchmarks(
    weights: list[str],
    devices: list[str],
    source: str | None = None,
    runs: int = 200,
    image_size: int = 640,
    out_dir: str | Path = "benchmarks",
    tag: str = "baseline",
) -> list[BenchResult]:
    frames = load_frames(source, runs)
    results = [bench_one(w, d, frames, image_size) for w in weights for d in devices]

    env = environment()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": tag,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": env,
        "results": [asdict(r) for r in results],
    }
    (out / f"{tag}.json").write_text(json.dumps(payload, indent=2))
    (out / f"{tag}.md").write_text(to_markdown(results, env))
    return results
