# Day 1 — Baseline Detection Pipeline

**Goal:** a working real-time detector plus a reproducible way to measure it. Every later
optimization (fine-tuning, CoreML, INT8) gets compared against these numbers.

## 1. What was built

```
VideoSource ──frame──▶ Detector.predict ──FrameResult──▶ draw_detections / draw_hud ──▶ window / VideoSink
                             │
                             └── timings ──▶ LatencyTracker (model) ;  wall clock ──▶ LatencyTracker (end-to-end)
```

At this stage the model is the pretrained COCO YOLO, filtered to class 0 (`person`).
Helmet/vest classes need fine-tuning (Day 3–4).

## 2. Tech stack used today, and why

| Tool | What it is | Why we use it |
|---|---|---|
| **PyTorch** | Deep learning framework | Runs YOLO; its `mps` backend uses the M3 GPU |
| **Ultralytics** | Library for YOLO training/inference/export | Mature, one API for train → evaluate → export (CoreML/ONNX) |
| **YOLO11n / YOLO26n** | Pretrained nano object detectors (COCO, 80 classes) | Fast enough for real time on a laptop; good starting point for fine-tuning |
| **OpenCV** | Computer vision library | Reads video files / webcams / RTSP, draws, writes MP4 |
| **pydantic** | Data validation from type hints | Config errors fail at startup with a clear message |
| **uv** | Fast Python package/venv manager | Reproducible environment, pinned Python 3.12 |
| **ruff / pytest** | Linter / test runner | Code quality gates (run in CI later) |

Why Python 3.12 and not the system 3.14? ML libraries (especially `coremltools`) lag behind
new Python releases. 3.12 is the safe, widely supported version.

## 3. Code walkthrough (read in this order)

### `schema.py`: the language between stages
- `Detection`: one box. **Format xyxy** = top-left `(x1, y1)` and bottom-right `(x2, y2)` in
  original-frame pixels. (Other formats: xywh = center + size, used in YOLO label files.)
  `frozen=True` makes it immutable and `slots=True` makes it smaller/faster; we create
  thousands per second.
- `StageTimings`: preprocess / inference / postprocess milliseconds.
- `FrameResult`: frame index + detections + timings.

### `config.py`: typed configuration
- Nested pydantic models mirror `configs/default.yaml`. `Field(ge=0, le=1)` enforces ranges.
- `load_config(path)` → YAML → `AppConfig.model_validate`. CLI flags override YAML values.
- Key knobs: `conf_threshold` (drop low-confidence boxes), `iou_threshold` (NMS overlap),
  `classes` (filter), `image_size` (model input size, a multiple of 32 because the network
  downsamples by 32).

### `devices.py`: picking CPU vs GPU
`resolve_device("auto")` returns `mps` when Metal is available. The Neural Engine is not
reachable from PyTorch, only through CoreML (Day 5–6).

### `detector.py`: the model wrapper
1. `YOLO(weights)` loads the network. A known model name is downloaded to that path once.
2. `warmup()` runs dummy frames. The first calls pay one-time costs (Metal shader compilation,
   memory allocation) that must not pollute measurements.
3. `predict(frame)`:
   - Ultralytics **preprocess**: letterbox-resize to 640×640 (keeps aspect ratio, pads the
     rest), BGR→RGB, HWC→CHW, scale 0–255 → 0–1, move to device.
   - **inference**: forward pass through backbone (features) → neck (multi-scale fusion) →
     head (box + class predictions).
   - **postprocess**: confidence filtering, NMS for YOLO11 (YOLO26 is NMS-free), and scaling
     boxes back to original-frame coordinates.
   - We convert tensors to `Detection` objects with **one bulk `.cpu()` copy per tensor**,
     not one per box. Each GPU→CPU copy forces a sync, so this matters.

### `video.py`: input and output
- `VideoSource` wraps `cv2.VideoCapture`: `"0"` → webcam 0; otherwise a file path or URL.
  It is an iterator (`for frame in source`) and a context manager (`with ...`), so the
  camera is always released, even on errors.
- `VideoSink` writes MP4 with the `mp4v` codec.

### `metrics.py`: measuring correctly
- `LatencyTracker` keeps a **rolling window** (`deque(maxlen=N)`) so the overlay reflects
  current performance, not the average since startup.
- Reports **p50** (typical frame) and **p95** (bad frames). Users notice stutter, which
  averages hide.

### `pipeline.py`: the loop
read → predict → draw → show/save, timing both the model alone and the whole frame
(end-to-end). `q` quits. A `finally` block guarantees the writer and window are closed.

### `benchmark.py` + `cli.py`
`ppe-guard benchmark` runs every (model × device) pair over the same frames, after 10
warm-up runs, and writes `benchmarks/<tag>.json` (machine-readable) and `.md` (docs).

## 4. Results

MacBook Air M3 (16 GB, fanless), 640×640, 300 timed runs, pretrained COCO weights:

| Model | Device | mean ms | p50 | p95 | p99 | max | FPS |
|---|---|---|---|---|---|---|---|
| YOLO11n | CPU | 38.65 | 37.35 | 47.91 | 50.15 | 55.24 | 25.9 |
| YOLO11n | MPS | 13.81 | 13.24 | 14.20 | 14.71 | 172.33 | 72.4 |
| YOLO26n | CPU | 35.71 | 34.67 | 42.30 | 48.11 | 51.82 | 28.0 |
| YOLO26n | MPS | 15.93 | 14.15 | 26.56 | 35.38 | 61.29 | 62.8 |

End-to-end on a 960×540 video (read + detect + draw + encode), YOLO26n on MPS:
**model 13.9 ms, end-to-end 20.3 ms ≈ 49 FPS**.

### How to read these numbers
1. **GPU ≈ 2.5× faster than CPU.** Expected for convolution-heavy networks.
2. **YOLO11n vs YOLO26n is within noise here.** An earlier identical run measured YOLO26n/MPS
   at 14.1 ms mean / 14.1 ms p95. The MacBook Air has **no fan**: under sustained load the chip
   heats and macOS throttles it, so results drift between runs. Conclusion: don't pick a model
   on a 2 ms PyTorch difference. Compare after export, over repeated runs.
3. **Spikes (YOLO11n/MPS max = 172 ms).** One frame out of 300. Typical causes: a background
   process, a memory allocation, a GPU shader recompile. This is exactly why we report p99
   and max: it's the frame a viewer sees as a freeze.
4. **The stage split is misleading on MPS.** Postprocess is ~7–8 ms on MPS but ~0.5 ms on CPU.
   GPU work is **asynchronous**: the "inference" call returns once work is *queued*, and the
   CPU actually waits for the GPU when results are copied back, which is counted under
   postprocess. **Totals are correct; only the split moves.** On CPU the split is exact.
5. **End-to-end (20 ms) > model (14 ms).** The extra ~6 ms is video decode, drawing and
   MP4 encode on the CPU. Later we overlap these with inference using a capture thread.

## 5. How to reproduce

```bash
.venv/bin/ppe-guard benchmark --runs 300 --tag baseline
.venv/bin/python scripts/make_test_video.py
.venv/bin/ppe-guard run -s data/videos/test_pan.mp4 --no-display --save outputs/annotated.mp4
```

The synthetic test video is a pan across a bundled image, used only to smoke-test the
pipeline. Accuracy is judged on real footage and a held-out dataset split (Day 3–4).

## 6. Interview questions this phase prepares you for

**Q: Why YOLO and not Faster R-CNN?**
YOLO is one-stage: one forward pass predicts all boxes, so it's real-time. Faster R-CNN first
proposes regions, then classifies each one: more accurate on tiny objects, but several times
slower. For live CCTV at ≥ 25 FPS on edge hardware, one-stage is the right trade-off.

**Q: What is NMS and why does it matter for deployment?**
A detector outputs many overlapping boxes for the same object. Non-Maximum Suppression keeps
the highest-confidence box and removes others whose IoU (overlap) exceeds a threshold. It's
sequential, data-dependent work that runs outside the network graph, so it's awkward to
export and accelerate. NMS-free models (YOLO26) learn to output one box per object, which
makes the exported model a pure graph that runs fully on the accelerator.

**Q: What is IoU?**
Intersection over Union: overlap area divided by combined area of two boxes, from 0 to 1.
It's used in NMS, in matching predictions to ground truth for mAP, and in tracking.

**Q: How did you benchmark, and why percentiles?**
Same inputs for each configuration, warm-up excluded, 300 timed runs, mean/p50/p95/p99/max
reported, results committed with environment info. Percentiles because real-time quality is
set by the slow frames: a good mean can hide stutter.

**Q: Why warm up?**
The first inferences include one-time costs: kernel and shader compilation, memory pool
allocation, lazy initialization. Including them inflates the mean and misrepresents
steady-state performance.

**Q: Your GPU postprocess time is 15× the CPU's. Is NMS slow on the GPU?**
No. GPU execution is asynchronous, so the measured "inference" only covers queueing the work.
The CPU blocks when it copies results back, and that wait gets attributed to postprocess.
To time stages precisely on a GPU you must synchronize (`torch.mps.synchronize()`) at each
boundary. The total is what users experience, and it's correct.

**Q: Why are results different between runs?**
Thermal throttling on a fanless laptop, background processes, and memory state. Mitigations:
warm-up, many runs, percentiles, repeat and report variance, and avoid deciding on
differences smaller than the noise.

**Q: Why separate detection from compliance rules?**
The model answers *what is where*; rules answer *is this a violation*. Separating them means
rules can change (new zone, longer grace period) without retraining, and they can be unit
tested deterministically.
