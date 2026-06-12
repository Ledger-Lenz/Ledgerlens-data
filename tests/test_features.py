import pandas as pd

from detection.feature_engineering import (
    build_feature_matrix,
    compute_trade_pattern_features,
    compute_volume_timing_features,
)


def sample_trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_id": "1",
                "ledger_close_time": "2024-01-01T00:00:00Z",
                "base_account": "A",
                "counter_account": "B",
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "price": 0.1,
            },
            {
                "trade_id": "2",
                "ledger_close_time": "2024-01-01T00:01:00Z",
                "base_account": "B",
                "counter_account": "A",
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "price": 0.1,
            },
        ]
    )


def test_compute_trade_pattern_features_empty():
    features = compute_trade_pattern_features("A", pd.DataFrame())
    assert features["counterparty_concentration_ratio"] == 0.0
    assert features["round_trip_frequency"] == 0.0


def test_compute_trade_pattern_features_concentration():
    df = sample_trades()
    features = compute_trade_pattern_features("A", df)
    # All of A's volume is with counterparty B -> full concentration
    assert features["counterparty_concentration_ratio"] == 1.0


def test_compute_volume_timing_features_empty():
    features = compute_volume_timing_features(pd.DataFrame())
    assert features["volume_per_counterparty_ratio"] == 0.0


def test_build_feature_matrix_returns_row_per_wallet():
    df = sample_trades()
    matrix = build_feature_matrix(df)

    assert set(matrix["wallet"]) == {"A", "B"}
    assert "benford_chi_square_1h" in matrix.columns
    assert "counterparty_concentration_ratio" in matrix.columns


def test_build_feature_matrix_empty_input():
    matrix = build_feature_matrix(pd.DataFrame())
    assert matrix.empty
