"""Rolling latency / throughput statistics.

Why percentiles and not just the average?
  A real-time system is judged by its WORST frames. An average of 20 ms can hide
  a p95 of 80 ms - which a viewer sees as stutter. We report p50 (typical) and
  p95 (tail) like production services do.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np


class LatencyTracker:
    """Keeps the last `window` samples so stats reflect current, not lifetime, performance."""

    def __init__(self, window: int = 120) -> None:
        self._samples: deque[float] = deque(maxlen=window)
        self._frame_times: deque[float] = deque(maxlen=window)

    def record(self, latency_ms: float) -> None:
        self._samples.append(latency_ms)
        self._frame_times.append(time.perf_counter())

    @property
    def count(self) -> int:
        return len(self._samples)

    def percentile(self, q: float) -> float:
        return float(np.percentile(self._samples, q)) if self._samples else 0.0

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self._samples)) if self._samples else 0.0

    @property
    def fps(self) -> float:
        """End-to-end frames per second measured from wall-clock arrival times."""
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "mean_ms": round(self.mean_ms, 2),
            "p50_ms": round(self.percentile(50), 2),
            "p95_ms": round(self.percentile(95), 2),
            "fps": round(self.fps, 1),
        }
