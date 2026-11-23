"""Fast unit tests that need no model weights (they run in CI on every push)."""

import pytest
from pydantic import ValidationError

from ppe_guard.config import AppConfig, ModelConfig, load_config
from ppe_guard.metrics import LatencyTracker
from ppe_guard.schema import Detection, StageTimings


def test_detection_geometry():
    d = Detection(10, 20, 50, 100, 0.9, 0, "person")
    assert d.width == 40
    assert d.height == 80
    assert d.center == (30, 60)


def test_stage_timings_total():
    assert StageTimings(1.0, 2.5, 0.5).total_ms == pytest.approx(4.0)


def test_latency_percentiles():
    t = LatencyTracker(window=100)
    for ms in range(1, 101):  # 1..100 ms
        t.record(float(ms))
    assert t.count == 100
    assert t.percentile(50) == pytest.approx(50.5)
    assert t.percentile(95) == pytest.approx(95.05)
    assert t.mean_ms == pytest.approx(50.5)


def test_latency_window_drops_old_samples():
    t = LatencyTracker(window=3)
    for ms in [100.0, 1.0, 1.0, 1.0]:
        t.record(ms)
    assert t.mean_ms == pytest.approx(1.0)


def test_empty_tracker_is_safe():
    t = LatencyTracker()
    assert t.summary() == {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "fps": 0.0}


def test_config_rejects_invalid_threshold():
    with pytest.raises(ValidationError):
        ModelConfig(conf_threshold=1.5)


def test_default_yaml_loads():
    cfg = load_config("configs/default.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.model.classes == [0]


def test_missing_config_uses_defaults():
    assert load_config(None).model.image_size == 640
