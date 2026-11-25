| Model | Device | imgsz | mean ms | p50 | p95 | p99 | max | FPS | pre | infer | post |
|---|---|---|---|---|---|---|---|---|---|---|---|
| yolo26n.pt | mps | 640 | 14.0 | 13.49 | 13.71 | 13.91 | 184.28 | 71.4 | 0.48 | 6.45 | 7.07 |
| ppe_yolo26n.pt | mps | 640 | 13.4 | 13.36 | 13.74 | 13.93 | 29.6 | 74.6 | 0.47 | 5.92 | 7.0 |

_Environment: machine: arm64, processor: arm, macos: 27.2.1, python: 3.12.16, torch: 2.15.0, ultralytics: 8.4.216_
