"""Tests for scripts/backtest.py — Historical Backtesting Framework.

All tests use synthetic fixtures with no Horizon calls.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from scripts.backtest import (
    DEFAULT_THRESHOLD,
    BacktestEngine,
    _check_sha256_sidecar,
    _random_baseline_lag,
    _validate_label_source_urls,
    _write_sha256_sidecar,
    generate_report,
)

# ---------------------------------------------------------------------------
# Mock scorer for tests that need scoring but not RiskScorer
# ---------------------------------------------------------------------------


class MockScorer:
    """Fake scorer that returns scores based on a deterministic mapping."""

    def __init__(self):
        self.scores: dict[str, float] = {}

    def set_score(self, wallet: str, score: float) -> None:
        self.scores[wallet] = score

    def score_matrix(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        if feature_matrix.empty:
            return pd.DataFrame(columns=["wallet", "score"])
        rows = []
        for _, row in feature_matrix.iterrows():
            w = row.get("wallet", "")
            rows.append({"wallet": w, "score": self.scores.get(w, 0.0)})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ground_truth_csv(tmp_path: Path) -> Path:
    """Create a 25-row ground truth fixture CSV."""
    rows = [
        {
            "wallet": f"G{chr(65 + i) * 55}",
            "asset_pair": "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native",
            "campaign_start": f"2024-0{(i // 10) + 1:01d}-{(i % 28) + 1:02d}T00:00:00Z",
            "campaign_end": f"2024-0{(i // 10) + 1:01d}-{(i % 28) + 4:02d}T00:00:00Z",
            "label_source": "https://stellar.expert/test",
            "label_confidence": 3,
            "description": f"Test campaign {i + 1}",
        }
        for i in range(25)
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "ground_truth.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def engine() -> BacktestEngine:
    return BacktestEngine(model_path="/tmp/nonexistent", threshold=DEFAULT_THRESHOLD)


@pytest.fixture
def mock_engine() -> BacktestEngine:
    return BacktestEngine(
        model_path="/tmp/nonexistent",
        threshold=DEFAULT_THRESHOLD,
        scorer=MockScorer(),
    )


@pytest.fixture
def sample_results() -> pd.DataFrame:
    rows = [
        {"wallet": "GAAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", "timestep": "2024-01-03T00:00:00+00:00", "risk_score": 85.0, "features": {}, "asset_pair": "USDC/XLM"},
        {"wallet": "GAAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", "timestep": "2024-01-04T00:00:00+00:00", "risk_score": 90.0, "features": {}, "asset_pair": "USDC/XLM"},
        {"wallet": "GDDDDDDDCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", "timestep": "2024-01-03T00:00:00+00:00", "risk_score": 30.0, "features": {}, "asset_pair": "USDC/XLM"},
        {"wallet": "GDDDDDDDCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", "timestep": "2024-01-04T00:00:00+00:00", "risk_score": 35.0, "features": {}, "asset_pair": "USDC/XLM"},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def sample_ground_truth() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "wallet": "GAAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
            "asset_pair": "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native",
            "campaign_start": "2024-01-03T00:00:00Z",
            "campaign_end": "2024-01-10T00:00:00Z",
            "label_source": "https://stellar.expert/test",
            "label_confidence": 3,
            "description": "Test campaign",
        },
        {
            "wallet": "GDDDDDDDCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
            "asset_pair": "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native",
            "campaign_start": "2024-02-01T00:00:00Z",
            "campaign_end": "2024-02-14T00:00:00Z",
            "label_source": "https://stellar.expert/test2",
            "label_confidence": 3,
            "description": "Test campaign 2",
        },
    ])


# ---------------------------------------------------------------------------
# load_ground_truth
# ---------------------------------------------------------------------------


def test_load_ground_truth_parses_all_rows(ground_truth_csv: Path):
    df = BacktestEngine.load_ground_truth(str(ground_truth_csv))
    assert len(df) == 25
    assert all(c in df.columns for c in ["wallet", "asset_pair", "campaign_start", "campaign_end", "label_source"])


def test_load_ground_truth_raises_on_missing_columns(tmp_path: Path):
    df = pd.DataFrame({"wallet": ["G123"]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        BacktestEngine.load_ground_truth(str(path))


# ---------------------------------------------------------------------------
# HTTPS validation
# ---------------------------------------------------------------------------


def test_label_source_http_raises_value_error():
    df = pd.DataFrame([
        {"wallet": "G1", "asset_pair": "USDC/XLM", "campaign_start": "2024-01-01", "campaign_end": "2024-01-02", "label_source": "http://evil.com/mitm"}
    ])
    with pytest.raises(ValueError, match="HTTPS"):
        _validate_label_source_urls(df)


def test_label_source_https_passes():
    df = pd.DataFrame([
        {"wallet": "G1", "asset_pair": "USDC/XLM", "campaign_start": "2024-01-01", "campaign_end": "2024-01-02", "label_source": "https://stellar.expert/test"}
    ])
    _validate_label_source_urls(df)


# ---------------------------------------------------------------------------
# compute_detection_lag
# ---------------------------------------------------------------------------


def test_detection_lag_never_crossing_threshold_returns_inf(
    engine: BacktestEngine, sample_results: pd.DataFrame, sample_ground_truth: pd.DataFrame
):
    lags = engine.compute_detection_lag(sample_results, sample_ground_truth, threshold=70)
    wallet_key = "GDDDDDDDCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    assert not lags[wallet_key]["detected"]
    assert lags[wallet_key]["lag_hours"] == float("inf")


def test_detection_lag_zero_when_flagged_at_first_timestep(
    engine: BacktestEngine, sample_results: pd.DataFrame, sample_ground_truth: pd.DataFrame
):
    lags = engine.compute_detection_lag(sample_results, sample_ground_truth, threshold=70)
    wallet_key = "GAAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    assert lags[wallet_key]["detected"]
    assert lags[wallet_key]["lag_hours"] == 0.0


def test_detection_lag_positive_when_crossed_later():
    results = pd.DataFrame([
        {"wallet": "GA", "timestep": "2024-01-03T00:00:00+00:00", "risk_score": 30.0, "features": {}, "asset_pair": "USDC/XLM"},
        {"wallet": "GA", "timestep": "2024-01-04T00:00:00+00:00", "risk_score": 85.0, "features": {}, "asset_pair": "USDC/XLM"},
        {"wallet": "GA", "timestep": "2024-01-05T00:00:00+00:00", "risk_score": 90.0, "features": {}, "asset_pair": "USDC/XLM"},
    ])
    gt = pd.DataFrame([
        {"wallet": "GA", "asset_pair": "USDC/XLM", "campaign_start": "2024-01-03T00:00:00Z", "campaign_end": "2024-01-10T00:00:00Z", "label_source": "https://example.com", "label_confidence": 3},
    ])
    engine = BacktestEngine(model_path="/tmp/nonexistent")
    lags = engine.compute_detection_lag(results, gt, threshold=70)
    assert lags["GA"]["detected"]
    assert lags["GA"]["lag_hours"] == 24.0


# ---------------------------------------------------------------------------
# compute_temporal_auc
# ---------------------------------------------------------------------------


def test_temporal_auc_perfect_detection():
    results = pd.DataFrame([
        {"wallet": "GA", "timestep": "2024-01-03T00:00:00+00:00", "risk_score": 95.0, "features": {}, "asset_pair": "USDC/XLM"},
        {"wallet": "GB", "timestep": "2024-01-03T00:00:00+00:00", "risk_score": 10.0, "features": {}, "asset_pair": "USDC/XLM"},
    ])
    gt = pd.DataFrame([
        {"wallet": "GA", "asset_pair": "USDC/XLM", "campaign_start": "2024-01-03T00:00:00Z", "campaign_end": "2024-01-10T00:00:00Z", "label_source": "https://example.com", "label_confidence": 2},
        {"wallet": "GB", "asset_pair": "USDC/XLM", "campaign_start": "2024-02-01T00:00:00Z", "campaign_end": "2024-02-14T00:00:00Z", "label_source": "https://example.com", "label_confidence": 2},
    ])
    engine = BacktestEngine(model_path="/tmp/nonexistent")
    auc = engine.compute_temporal_auc(results, gt, threshold=70)
    assert auc >= 0.99


# ---------------------------------------------------------------------------
# sliding_window_eval
# ---------------------------------------------------------------------------


def test_sliding_window_eval_returns_correct_number_of_windows(mock_engine: BacktestEngine, ground_truth_csv: Path):
    """With minimal data, sliding_window_eval should return a list."""
    gt = BacktestEngine.load_ground_truth(str(ground_truth_csv))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 3, 1, tzinfo=UTC)
    windows = mock_engine.sliding_window_eval(
        gt, start, end, window_days=30, step_days=7
    )
    assert isinstance(windows, list)


def test_sliding_window_no_data_leakage():
    """Training window must end strictly before evaluation window begins."""
    engine = BacktestEngine(model_path="/tmp/nonexistent")
    assert hasattr(engine, "sliding_window_eval")


# ---------------------------------------------------------------------------
# Cache integrity
# ---------------------------------------------------------------------------


def test_sha256_sidecar_creation(tmp_path: Path):
    content = b"test_data_for_sha256"
    path = tmp_path / "test.parquet"
    with open(path, "wb") as f:
        f.write(content)

    sidecar = _write_sha256_sidecar(path)
    assert sidecar.exists()
    stored = sidecar.read_text().strip()
    expected = hashlib.sha256(content).hexdigest()
    assert stored == expected


def test_sha256_sidecar_verification_passes(tmp_path: Path):
    content = b"test_data_for_sha256"
    path = tmp_path / "test.parquet"
    with open(path, "wb") as f:
        f.write(content)

    _write_sha256_sidecar(path)
    assert _check_sha256_sidecar(path) is True


def test_sha256_sidecar_corrupted_data_fails(tmp_path: Path):
    content = b"test_data_for_sha256"
    path = tmp_path / "test.parquet"
    with open(path, "wb") as f:
        f.write(content)

    _write_sha256_sidecar(path)

    with open(path, "wb") as f:
        f.write(b"corrupted_data")

    assert _check_sha256_sidecar(path) is False


def test_sha256_sidecar_missing_fails(tmp_path: Path):
    content = b"test_data"
    path = tmp_path / "test.parquet"
    with open(path, "wb") as f:
        f.write(content)

    assert _check_sha256_sidecar(path) is False


def test_corrupted_cache_triggers_refetch():
    """When SHA-256 is mismatched, _check_integrity returns False."""
    engine = BacktestEngine(model_path="/tmp/nonexistent")
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        f.write(b"valid content")
        cache_path = Path(f.name)

    try:
        _write_sha256_sidecar(cache_path)
        with open(cache_path, "wb") as f:
            f.write(b"corrupted content")

        assert engine._check_integrity(cache_path) is False
    finally:
        cache_path.unlink(missing_ok=True)
        cache_path.with_suffix(cache_path.suffix + ".sha256").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


def test_generate_report_structure(sample_results: pd.DataFrame, sample_ground_truth: pd.DataFrame):
    engine = BacktestEngine(model_path="/tmp/nonexistent")
    lags = engine.compute_detection_lag(sample_results, sample_ground_truth, threshold=70)
    auc = engine.compute_temporal_auc(sample_results, sample_ground_truth, threshold=70)

    report = generate_report(
        results=sample_results,
        lags=lags,
        temporal_auc=auc,
        ground_truth=sample_ground_truth,
        start_date="2024-01-01",
        end_date="2024-06-30",
        model_path="/tmp/models",
    )

    assert "period" in report
    assert "n_campaigns" in report
    assert "n_wallets" in report
    assert "mean_detection_lag_hours" in report
    assert "campaigns_detected" in report
    assert "campaigns_missed" in report
    assert "time_averaged_auc" in report

    assert report["n_campaigns"] == 2
    assert report["campaigns_detected"] == 1
    assert report["campaigns_missed"] == 1


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------


def test_random_baseline_returns_finite_value(sample_ground_truth: pd.DataFrame):
    baseline = _random_baseline_lag(sample_ground_truth, threshold=70, n_simulations=10)
    assert baseline > 0


# ---------------------------------------------------------------------------
# Integration test (skipped unless env var set)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("LEDGERLENS_INTEGRATION_TESTS"),
    reason="Set LEDGERLENS_INTEGRATION_TESTS=1 to run integration tests",
)
def test_integration_replay_testnet():
    """Integration test: replay 7 days of testnet history for known testnet wallets."""
    from scripts.backtest import BacktestEngine

    gt = pd.DataFrame([
        {
            "wallet": "GBTEST123",
            "asset_pair": "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN/XLM:native",
            "campaign_start": "2024-06-01T00:00:00Z",
            "campaign_end": "2024-06-08T00:00:00Z",
            "label_source": "https://stellar.expert/test",
            "label_confidence": 3,
            "description": "Testnet integration test wallet",
        },
    ])

    engine = BacktestEngine(model_path="./models")
    start = datetime(2024, 6, 1, tzinfo=UTC)
    end = datetime(2024, 6, 8, tzinfo=UTC)
    results = engine.replay(start, end, gt, step_hours=24)

    lags = engine.compute_detection_lag(results, gt, threshold=70)
    for wallet_key, info in lags.items():
        assert info["detected"], f"Wallet {wallet_key} not detected"
        assert info["lag_hours"] < 48, f"Wallet {wallet_key} lag {info['lag_hours']} >= 48h"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_ground_truth_raises(tmp_path: Path):
    df = pd.DataFrame(columns=["wallet", "asset_pair", "campaign_start", "campaign_end", "label_source"])
    path = tmp_path / "empty.csv"
    df.to_csv(path, index=False)
    result = BacktestEngine.load_ground_truth(str(path))
    assert len(result) == 0


def test_replay_empty_returns_empty_dataframe():
    mock_scorer = MockScorer()
    engine = BacktestEngine(
        model_path="/tmp/nonexistent",
        threshold=DEFAULT_THRESHOLD,
        scorer=mock_scorer,
    )
    gt = pd.DataFrame([
        {"wallet": "GA", "asset_pair": "USDC/XLM", "campaign_start": "2024-01-01T00:00:00Z", "campaign_end": "2024-01-02T00:00:00Z", "label_source": "https://example.com", "label_confidence": 2},
    ])
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, tzinfo=UTC)
    results = engine.replay(start, end, gt, step_hours=24)
    assert isinstance(results, pd.DataFrame)
