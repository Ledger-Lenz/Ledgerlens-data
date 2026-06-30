"""Unit tests for monitoring.capacity_metrics (Issue #242)."""

import importlib
import sys


def _reload_module():
    """Re-import the module so metric registration is exercised."""
    mod_name = "monitoring.capacity_metrics"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


def test_metrics_registered_on_import():
    """Prometheus metrics are registered as Gauge instances when prometheus_client is available."""
    pytest = __import__("pytest")
    cm = _reload_module()
    try:
        from prometheus_client import REGISTRY
    except ImportError:
        pytest.skip("prometheus_client not installed")

    collector_names = set(REGISTRY._names_to_collectors.keys())  # type: ignore[attr-defined]
    assert "ledgerlens_cpu_usage_ratio" in collector_names
    assert "ledgerlens_memory_usage_bytes" in collector_names
    assert "ledgerlens_trades_per_second" in collector_names


def test_set_cpu_usage_updates_gauge():
    """set_cpu_usage sets the labelled gauge value without raising."""
    cm = _reload_module()
    try:
        from prometheus_client import REGISTRY
    except ImportError:
        import pytest; pytest.skip("prometheus_client not installed")

    cm.set_cpu_usage("benford", 0.42)
    gauge = cm.CPU_USAGE_RATIO
    assert gauge is not None
    sample = next(
        s for s in gauge.collect()[0].samples if s.labels.get("component") == "benford"
    )
    assert abs(sample.value - 0.42) < 1e-9


def test_set_memory_usage_updates_gauge():
    """set_memory_usage sets the bytes gauge without raising."""
    cm = _reload_module()
    try:
        from prometheus_client import REGISTRY
    except ImportError:
        import pytest; pytest.skip("prometheus_client not installed")

    cm.set_memory_usage(123_456_789)
    gauge = cm.MEMORY_USAGE_BYTES
    assert gauge is not None
    sample = gauge.collect()[0].samples[0]
    assert sample.value == 123_456_789


def test_set_trades_per_second_updates_gauge():
    """set_trades_per_second sets the labelled gauge without raising."""
    cm = _reload_module()
    try:
        from prometheus_client import REGISTRY
    except ImportError:
        import pytest; pytest.skip("prometheus_client not installed")

    cm.set_trades_per_second("XLM/USDC", 17.5)
    gauge = cm.TRADES_PER_SECOND
    assert gauge is not None
    sample = next(
        s for s in gauge.collect()[0].samples if s.labels.get("asset_pair") == "XLM/USDC"
    )
    assert abs(sample.value - 17.5) < 1e-9
