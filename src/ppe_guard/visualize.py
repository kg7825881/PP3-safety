"""Drawing boxes and a performance overlay onto frames (in place)."""

from __future__ import annotations

import cv2
import numpy as np

from ppe_guard.schema import Detection


def _color_for(class_id: int) -> tuple[int, int, int]:
    """Deterministic, distinct-ish BGR color per class id."""
    rng = np.random.default_rng(class_id * 7919)
    return tuple(int(c) for c in rng.integers(64, 256, size=3))  # type: ignore[return-value]


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    for det in detections:
        color = _color_for(det.class_id)
        p1, p2 = (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        text = f"{det.label} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(frame, text, (p1[0] + 2, p1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, stats: dict[str, float], device: str) -> np.ndarray:
    """Top-left overlay: device, FPS and latency percentiles."""
    lines = [
        f"device: {device}",
        f"fps: {stats.get('fps', 0):.1f}",
        f"latency p50/p95: {stats.get('p50_ms', 0):.1f} / {stats.get('p95_ms', 0):.1f} ms",
    ]
    cv2.rectangle(frame, (0, 0), (330, 22 * len(lines) + 8), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (8, 22 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
