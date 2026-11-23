# PPE Guard — Architecture

This document describes the complete target system. Sections marked _(planned)_ are built
in later phases; each phase gets its own write-up in `docs/NN-*.md`.

## 1. Problem statement

Construction sites, warehouses and factories require helmets and high-visibility vests.
Compliance is checked manually, which is intermittent and expensive. PPE Guard turns
existing cameras into a continuous, automatic compliance monitor.

**Functional requirements**
1. Detect people, helmets and vests in live video.
2. Decide per person whether they are compliant (helmet on head, vest on torso).
3. Raise an alert only when a person is non-compliant **inside a hazard zone** for longer
   than a grace period (avoids alert spam from one bad frame).
4. Expose live annotated video, events and statistics through an API and dashboard.

**Non-functional requirements**
- Real-time: ≥ 25 FPS per stream on a laptop-class device (MacBook Air M3).
- Edge-first: runs fully on-device; video never has to leave the site (privacy, bandwidth).
- Observable: latency/FPS/alert metrics exported for monitoring.
- Reproducible: configs, pinned dependencies, benchmark results committed to git.

## 2. High-level data flow

```
 ┌──────────┐   frames   ┌──────────┐ detections ┌──────────┐  tracks   ┌──────────┐
 │  Video   │──────────▶│ Detector │───────────▶│ Tracker  │─────────▶│  Rules   │
 │  Source  │ (bounded   │ (YOLO on │            │(ByteTrack│ (stable   │ engine   │
 │ file/cam/│  queue)    │ CoreML / │            │ IDs over │  person   │ zones +  │
 │  RTSP    │            │ MPS/CPU) │            │  frames) │  IDs)     │ timers)  │
 └──────────┘            └──────────┘            └──────────┘           └────┬─────┘
                                                                             │ events
            ┌─────────────────────────────┬──────────────────────────────────┤
            ▼                             ▼                                  ▼
   ┌─────────────────┐         ┌────────────────────┐            ┌────────────────────┐
   │ Annotated video │         │  Event store /     │            │ Metrics (Prometheus│
   │ (WebSocket MJPEG│         │  alert queue       │            │ FPS, latency, GPU, │
   │  → dashboard)   │         │  (planned)         │            │ alert counts)      │
   └─────────────────┘         └────────────────────┘            └────────────────────┘
```

Every arrow carries a type from `src/ppe_guard/schema.py` (`Detection`, `FrameResult`,
later `Track` and `ComplianceEvent`). Stages never pass framework objects (torch tensors,
Ultralytics `Results`) to each other. **That is what makes backends swappable**: replacing
PyTorch with CoreML touches only the detector.

## 3. Components

| Stage | Module | Responsibility | Status |
|---|---|---|---|
| Config | `config.py` | Typed, validated YAML config (pydantic) | Day 1 ✅ |
| Source | `video.py` | Unified file / webcam / RTSP reader, MP4 writer | Day 1 ✅ |
| Detector | `detector.py` | Model wrapper: `predict(frame) -> FrameResult` | Day 1 ✅ (PyTorch) |
| Metrics | `metrics.py` | Rolling p50/p95 latency and FPS | Day 1 ✅ |
| Benchmark | `benchmark.py` | Reproducible model × device latency study | Day 1 ✅ |
| Pipeline | `pipeline.py` | The real-time loop | Day 1 ✅ (synchronous) |
| Training | `train.py` | Fine-tune on PPE dataset, evaluate | Day 3–4 _(planned)_ |
| Export | `export.py` | CoreML / ONNX, FP16 / INT8 | Day 5–6 _(planned)_ |
| Tracker | `tracking.py` | ByteTrack: persistent IDs | Day 7–8 _(planned)_ |
| Rules | `rules.py` | PPE-to-person association, zones, timers | Day 7–8 _(planned)_ |
| API | `api/` | FastAPI REST + WebSocket stream | Day 9–10 _(planned)_ |
| Dashboard | `dashboard/` | Live view, events, stats | Day 9–10 _(planned)_ |
| Ops | `Dockerfile`, CI, Prometheus | Packaging, CI, monitoring | Day 11–12 _(planned)_ |

## 4. Key design decisions

### 4.1 Model: YOLO (one-stage detector), nano size
- **One-stage** detectors (YOLO, SSD) predict boxes in a single pass, so they are fast enough
  for real time. Two-stage detectors (Faster R-CNN) are more accurate on small objects
  but much slower.
- **Nano** variant: helmets/vests are mid-sized objects at CCTV distances, and the edge
  latency budget is tight. We will measure whether "small" is worth the extra cost.
- **YOLO26 vs YOLO11**: YOLO26 is designed end-to-end **without NMS** (Non-Maximum
  Suppression) post-processing, which simplifies export to CoreML/ONNX and removes a CPU-side
  step. In PyTorch on the M3 the two are within noise (see `benchmarks/baseline.md`); the
  decision is finalized after export benchmarks.

### 4.2 Hardware target: Apple Silicon, three compute units
| Unit | Reached via | Expected role |
|---|---|---|
| CPU | PyTorch `cpu`, ONNX Runtime | Fallback, Docker/CI |
| GPU | PyTorch `mps` | Training, development inference |
| Neural Engine (ANE) | CoreML only | Production inference (fastest, most power-efficient) |

Docker on macOS runs a Linux VM that cannot see the GPU or ANE, so the **containerized**
deployment uses ONNX Runtime on CPU, and the **native** deployment uses CoreML. This trade-off
is deliberate and documented in the benchmark results.

### 4.3 Compliance logic separate from detection
The model only answers "where are people / helmets / vests?". Whether a person is
*compliant* is decided by a rules engine (box association + zones + time). Benefits:
rules change without retraining; behavior is explainable and testable with plain unit tests.

### 4.4 Synchronous first, then a bounded queue
Day 1's loop is single-threaded to establish an honest baseline. For live streams, capture
and inference will run on separate threads connected by a **bounded queue that drops the
oldest frame**: with a live camera, processing a stale frame is worse than skipping it.

## 5. Repository conventions
- `src/` layout, installable package, CLI entry point `ppe-guard`.
- Large binaries (weights, datasets, videos, outputs) are git-ignored.
- Benchmark results **are** committed: they are part of the project's claims.
- `ruff` for lint, `pytest` for tests; unit tests must not need model weights so CI is fast.
