# PPE Guard

**Real-time PPE (helmet / safety-vest) compliance detection, optimized for edge inference on Apple Silicon.**

PPE Guard watches a video feed (CCTV, RTSP stream, webcam or file), detects workers and
their protective equipment, tracks each person across frames, and raises an alert when
someone is in a hazard zone without the required PPE for longer than a grace period.

> Status: **Day 1 of 12 — baseline detection pipeline + benchmarks.** See [Roadmap](#roadmap).

---

## Why this project is different

Most object-detection portfolio projects stop at "run YOLO on a webcam". PPE Guard is built
as a system, with every design decision measured:

| Typical demo | PPE Guard |
|---|---|
| Pretrained model, generic classes | Fine-tuned on a PPE dataset, evaluated with mAP and per-class recall |
| "It runs fast" | Reproducible benchmarks: p50/p95/p99 latency, per-stage breakdown, every backend |
| Runs wherever PyTorch runs | Exported to **CoreML** to use the Apple **Neural Engine** (edge deployment) |
| Boxes on a screen | Tracking + business rules: *"person without helmet in Zone A for > 3 s"* → alert |
| Script | Typed config, tests, CI, Docker, metrics (Prometheus), REST + WebSocket API, dashboard |

## Quick start

Requires macOS on Apple Silicon (Linux/CPU also works) and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"

# Webcam (press q to quit)
.venv/bin/ppe-guard run

# A video file, headless, saving the annotated result
.venv/bin/ppe-guard run -s path/to/video.mp4 --no-display --save outputs/annotated.mp4

# Benchmark models x devices -> benchmarks/baseline.{json,md}
.venv/bin/ppe-guard benchmark

# Tests
.venv/bin/pytest
```

Official model weights download automatically into `weights/` on first use.

## Current results (Day 1 baseline, MacBook Air M3, 16 GB)

Pretrained COCO models, 640×640 input, 300 timed runs after warm-up:

| Model | Device | mean ms | p95 ms | FPS |
|---|---|---|---|---|
| YOLO11n | CPU | 38.7 | 47.9 | 25.9 |
| YOLO11n | GPU (MPS) | 13.8 | 14.2 | 72.4 |
| YOLO26n | CPU | 35.7 | 42.3 | 28.0 |
| YOLO26n | GPU (MPS) | 15.9 | 26.6 | 62.8 |

Takeaways: the M3 GPU is **~2.5× faster** than CPU. YOLO11n vs YOLO26n is within run-to-run
noise in PyTorch on this fanless laptop; the comparison that matters comes after CoreML export.
Full table and methodology: [benchmarks/baseline.md](benchmarks/baseline.md) and
[docs/01-baseline.md](docs/01-baseline.md).

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — the whole system, data flow, and design decisions
- [Day 1 — Baseline pipeline](docs/01-baseline.md) — code walkthrough, benchmark method, interview Q&A

## Project layout

```
configs/         YAML runtime configs
src/ppe_guard/   the package (one module per pipeline stage)
tests/           unit tests (no model weights needed -> fast in CI)
scripts/         helper scripts (test video generation, later: dataset prep)
benchmarks/      committed benchmark results (JSON + Markdown)
docs/            architecture and per-phase write-ups
weights/ data/ outputs/   large/local files, git-ignored
```

## Roadmap

- [x] **Day 1–2** Repo, architecture, baseline detector, benchmark harness
- [ ] **Day 3–4** PPE dataset, fine-tune on MPS, evaluation (mAP, per-class recall)
- [ ] **Day 5–6** CoreML (Neural Engine) + ONNX export, FP16/INT8, benchmark every backend
- [ ] **Day 7–8** Multi-object tracking (ByteTrack), zones, compliance rules, alerts
- [ ] **Day 9–10** FastAPI service, WebSocket live stream, dashboard
- [ ] **Day 11–12** Docker, GitHub Actions CI, Prometheus metrics, final docs
