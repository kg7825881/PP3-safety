"""The real-time loop: read frame -> detect -> annotate -> show/save.

Day 1 version is synchronous (one thread). That is intentional: it gives us an
honest baseline. Later we decouple capture from inference with a bounded queue
so a slow model drops stale frames instead of falling further behind live video.
"""

from __future__ import annotations

import logging
import time

import cv2

from ppe_guard.config import AppConfig
from ppe_guard.detector import Detector
from ppe_guard.metrics import LatencyTracker
from ppe_guard.video import VideoSink, VideoSource
from ppe_guard.visualize import draw_detections, draw_hud

log = logging.getLogger(__name__)


def run(cfg: AppConfig) -> dict[str, float]:
    detector = Detector(cfg.model)
    log.info("Loaded %s on %s", cfg.model.weights, detector.device)
    detector.warmup()

    model_latency = LatencyTracker()  # detector only
    e2e_latency = LatencyTracker()    # read + detect + draw + write
    frames_processed = 0              # trackers keep a rolling window, so count separately

    with VideoSource(cfg.source.uri, cfg.source.max_frames) as source:
        save_path = cfg.output.save_path
        sink = VideoSink(save_path, source.fps, source.size) if save_path else None
        try:
            frame_start = time.perf_counter()
            for idx, frame in enumerate(source):
                result = detector.predict(frame, frame_index=idx)
                model_latency.record(result.timings.total_ms)
                frames_processed += 1

                draw_detections(frame, result.detections)
                draw_hud(frame, e2e_latency.summary(), detector.device)

                if sink:
                    sink.write(frame)
                if cfg.output.display:
                    cv2.imshow("PPE Guard", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                now = time.perf_counter()
                e2e_latency.record((now - frame_start) * 1000)
                frame_start = now
        finally:
            if sink:
                sink.close()
            if cfg.output.display:
                cv2.destroyAllWindows()

    summary = {
        "frames": frames_processed,
        **{f"model_{k}": v for k, v in model_latency.summary().items() if k != "fps"},
        **{f"e2e_{k}": v for k, v in e2e_latency.summary().items()},
    }
    log.info("Run summary: %s", summary)
    return summary
