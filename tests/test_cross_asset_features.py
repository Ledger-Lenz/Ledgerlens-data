"""Tests for cross-asset coordination detection features."""

import pandas as pd

from detection.benford_engine import cross_pair_benford_consistency
from detection.feature_engineering import (
    build_feature_matrix,
    compute_cross_asset_features,
)


def test_cross_pair_synchrony_high_for_coordinated_trades():
    """Trades within synchrony window on different pairs should increase synchrony."""
    trades = []
    base_time = pd.Timestamp("2024-01-01T00:00:00Z")

    # Wallet A trades on pair 1 and pair 2 within 10 seconds
    for i in range(10):
        trade_time = base_time + pd.Timedelta(seconds=i)
        trades.append(
            {
                "trade_id": f"1_{i}",
                "ledger_close_time": trade_time.isoformat(),
                "base_account": "A",
                "counter_account": "B",
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "pair_id": "pair1",
            }
        )
        trades.append(
            {
                "trade_id": f"2_{i}",
                "ledger_close_time": (trade_time + pd.Timedelta(seconds=5)).isoformat(),
                "base_account": "A",
                "counter_account": "C",
                "base_asset": "BTC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "pair_id": "pair2",
            }
        )

    df = pd.DataFrame(trades)
    features = compute_cross_asset_features("A", df)
    assert features["cross_pair_trade_synchrony"] >= 0.7


def test_net_flow_near_zero_for_closed_cycle():
    """Buy asset on one pair, sell on another = net flow near zero."""
    trades = pd.DataFrame(
        [
            {
                "trade_id": "1",
                "ledger_close_time": "2024-01-01T00:00:00Z",
                "base_account": "A",
                "counter_account": "B",
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "pair_id": "pair1",
            },
            {
                "trade_id": "2",
                "ledger_close_time": "2024-01-01T00:01:00Z",
                "base_account": "B",
                "counter_account": "A",
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "pair_id": "pair2",
            },
        ]
    )
    features = compute_cross_asset_features("A", trades)
    # Net flow: A gets 100 XLM from trade 1, sends 100 USDC from trade 2
    # Then A gets 100 USDC and sends 100 XLM from trade 2
    # So A's net flow in both is ~0
    assert features["net_asset_flow_deviation"] < 0.15


def test_net_flow_nonzero_for_legitimate_inventory():
    """Consistent buying of one asset = high net flow."""
    trades = []
    for i in range(10):
        trades.append(
            {
                "trade_id": f"{i}",
                "ledger_close_time": f"2024-01-01T{i:02d}:00:00Z",
                "base_account": f"B_{i}",
                "counter_account": "A",
                "base_asset": "XLM:native",
                "counter_asset": "USDC:issuer",
                "amount": 100.0,
                "pair_id": "pair1",
            }
        )

    df = pd.DataFrame(trades)
    features = compute_cross_asset_features("A", df)
    # A consistently receives USDC, so net_flow[USDC] >> 0
    assert features["net_asset_flow_deviation"] > 0.5


def test_counterparty_overlap_high_for_shared_counterparties():
    """Same counterparties across pairs = high overlap."""
    trades = pd.DataFrame(
        [
            {
                "trade_id": "1",
                "ledger_close_time": "2024-01-01T00:00:00Z",
                "base_account": "A",
                "counter_account": "CP1",
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "pair_id": "pair1",
            },
            {
                "trade_id": "2",
                "ledger_close_time": "2024-01-01T00:01:00Z",
                "base_account": "A",
                "counter_account": "CP2",
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "pair_id": "pair1",
            },
            {
                "trade_id": "3",
                "ledger_close_time": "2024-01-01T00:02:00Z",
                "base_account": "A",
                "counter_account": "CP3",
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "pair_id": "pair1",
            },
            {
                "trade_id": "4",
                "ledger_close_time": "2024-01-01T00:03:00Z",
                "base_account": "CP1",
                "counter_account": "A",
                "base_asset": "BTC:issuer",
                "counter_asset": "XLM:native",
                "amount": 50.0,
                "pair_id": "pair2",
            },
            {
                "trade_id": "5",
                "ledger_close_time": "2024-01-01T00:04:00Z",
                "base_account": "CP2",
                "counter_account": "A",
                "base_asset": "BTC:issuer",
                "counter_asset": "XLM:native",
                "amount": 50.0,
                "pair_id": "pair2",
            },
            {
                "trade_id": "6",
                "ledger_close_time": "2024-01-01T00:05:00Z",
                "base_account": "CP3",
                "counter_account": "A",
                "base_asset": "BTC:issuer",
                "counter_asset": "XLM:native",
                "amount": 50.0,
                "pair_id": "pair2",
            },
        ]
    )
    features = compute_cross_asset_features("A", trades)
    # All 3 counterparties on both pairs: perfect overlap
    assert features["cross_pair_counterparty_overlap"] == 1.0


def test_counterparty_overlap_zero_for_disjoint():
    """Completely different counterparties across pairs = zero overlap."""
    trades = pd.DataFrame(
        [
            {
                "trade_id": "1",
                "ledger_close_time": "2024-01-01T00:00:00Z",
                "base_account": "A",
                "counter_account": "CP1",
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "pair_id": "pair1",
            },
            {
                "trade_id": "2",
                "ledger_close_time": "2024-01-01T00:01:00Z",
                "base_account": "A",
                "counter_account": "CP2",
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "pair_id": "pair1",
            },
            {
                "trade_id": "3",
                "ledger_close_time": "2024-01-01T00:02:00Z",
                "base_account": "CP3",
                "counter_account": "A",
                "base_asset": "BTC:issuer",
                "counter_asset": "XLM:native",
                "amount": 50.0,
                "pair_id": "pair2",
            },
            {
                "trade_id": "4",
                "ledger_close_time": "2024-01-01T00:03:00Z",
                "base_account": "CP4",
                "counter_account": "A",
                "base_asset": "BTC:issuer",
                "counter_asset": "XLM:native",
                "amount": 50.0,
                "pair_id": "pair2",
            },
        ]
    )
    features = compute_cross_asset_features("A", trades)
    # No overlap between {CP1, CP2} and {CP3, CP4}
    assert features["cross_pair_counterparty_overlap"] == 0.0


def test_volume_correlation_high_for_synchronised_spikes():
    """Volume spikes at same time across pairs = high correlation."""
    trades = []
    base_time = pd.Timestamp("2024-01-01T00:00:00Z")

    # Synchronized volume spikes in both pairs at same minutes
    for minute in range(10):
        minute_time = base_time + pd.Timedelta(minutes=minute)
        # High volume spike in pair1
        trades.append(
            {
                "trade_id": f"p1_{minute}",
                "ledger_close_time": minute_time.isoformat(),
                "base_account": "A",
                "counter_account": "B",
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 1000.0 if minute % 2 == 0 else 100.0,
                "pair_id": "pair1",
            }
        )
        # High volume spike in pair2 at same time
        trades.append(
            {
                "trade_id": f"p2_{minute}",
                "ledger_close_time": minute_time.isoformat(),
                "base_account": "A",
                "counter_account": "C",
                "base_asset": "BTC:issuer",
                "counter_asset": "XLM:native",
                "amount": 2000.0 if minute % 2 == 0 else 200.0,
                "pair_id": "pair2",
            }
        )

    df = pd.DataFrame(trades)
    features = compute_cross_asset_features("A", df)
    # Both spike at even minutes, should be highly correlated
    assert features["cross_pair_volume_correlation"] > 0.7


def test_defaults_with_single_pair():
    """Single pair = all cross-pair features should be defaults."""
    trades = pd.DataFrame(
        [
            {
                "trade_id": "1",
                "ledger_close_time": "2024-01-01T00:00:00Z",
                "base_account": "A",
                "counter_account": "B",
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "pair_id": "pair1",
            },
        ]
    )
    features = compute_cross_asset_features("A", trades)
    assert features["cross_pair_trade_synchrony"] == 0.0
    assert features["net_asset_flow_deviation"] == 1.0
    assert features["cross_pair_counterparty_overlap"] == 0.0
    assert features["cross_pair_volume_correlation"] == 0.0
    assert features["pair_diversity_score"] == 0.0


def test_cross_pair_mad_std_zero_for_identical_distributions():
    """Same Benford metrics on both pairs = MAD std ~0."""
    # Create trades that would produce identical Benford distributions
    trades = []
    for pair_id in ["pair1", "pair2"]:
        # Create 100 trades with similar digit distribution
        for i in range(1, 10):
            for j in range(30):  # 30 of each digit
                trades.append(
                    {
                        "trade_id": f"{pair_id}_{i}_{j}",
                        "ledger_close_time": f"2024-01-01T{i % 24:02d}:{j % 60:02d}:00Z",
                        "base_account": "A",
                        "counter_account": "B",
                        "base_asset": "USDC:issuer",
                        "counter_asset": "XLM:native",
                        "amount": float(i * 10 ** (j % 5)),
                        "pair_id": pair_id,
                    }
                )

    df = pd.DataFrame(trades)
    features = compute_cross_asset_features("A", df)
    # MAD std should be very close to 0 since both pairs have identical distributions
    assert features["cross_pair_mad_std"] < 0.01


def test_feature_matrix_includes_cross_asset_columns():
    """build_feature_matrix with multi-pair data should include all cross-asset columns."""
    trades = pd.DataFrame(
        [
            {
                "trade_id": "1",
                "ledger_close_time": "2024-01-01T00:00:00Z",
                "base_account": "A",
                "counter_account": "B",
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "pair_id": "pair1",
            },
            {
                "trade_id": "2",
                "ledger_close_time": "2024-01-01T00:01:00Z",
                "base_account": "A",
                "counter_account": "C",
                "base_asset": "BTC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "pair_id": "pair2",
            },
        ]
    )

    feature_matrix = build_feature_matrix(trades, all_pairs_df=trades)

    expected_cols = [
        "cross_pair_trade_synchrony",
        "net_asset_flow_deviation",
        "cross_pair_counterparty_overlap",
        "cross_pair_volume_correlation",
        "pair_diversity_score",
        "cross_pair_mad_std",
    ]

    for col in expected_cols:
        assert col in feature_matrix.columns, f"Missing column: {col}"


def test_synthetic_dataset_includes_cross_asset_columns():
    """Regenerate synthetic dataset and verify cross-asset columns present."""
    from scripts.generate_synthetic_dataset import generate_synthetic_dataset

    df = generate_synthetic_dataset(n_wallets=100)

    expected_cols = [
        "cross_pair_trade_synchrony",
        "net_asset_flow_deviation",
        "cross_pair_counterparty_overlap",
        "cross_pair_volume_correlation",
        "pair_diversity_score",
        "cross_pair_mad_std",
    ]

    for col in expected_cols:
        assert col in df.columns, f"Missing column in synthetic dataset: {col}"

    # Check that label 0 and 1 have different distributions
    for col in expected_cols:
        if col != "cross_pair_mad_std":  # Skip MAD std which might have less variance
            label_0_mean = df[df["label"] == 0][col].mean()
            label_1_mean = df[df["label"] == 1][col].mean()
            # At least some difference expected
            assert (
                abs(label_0_mean - label_1_mean) > 0.01
            ), f"Feature {col} should differ between labels"


def test_cross_pair_benford_consistency():
    """Test the cross_pair_benford_consistency function directly."""
    # Two pairs with identical MAD = std 0
    metrics1 = {"mad": 0.01, "chi_square": 5.0}
    metrics2 = {"mad": 0.01, "chi_square": 5.0}
    std = cross_pair_benford_consistency({"pair1": metrics1, "pair2": metrics2})
    assert std < 0.001

    # Two pairs with different MAD = higher std
    metrics3 = {"mad": 0.05, "chi_square": 20.0}
    metrics4 = {"mad": 0.01, "chi_square": 5.0}
    std = cross_pair_benford_consistency({"pair1": metrics3, "pair2": metrics4})
    assert std > 0.01


def test_empty_dataframe():
    """Empty DataFrame should return default features."""
    df = pd.DataFrame()
    features = compute_cross_asset_features("A", df)

    defaults = {
        "cross_pair_trade_synchrony": 0.0,
        "net_asset_flow_deviation": 1.0,
        "cross_pair_counterparty_overlap": 0.0,
        "cross_pair_volume_correlation": 0.0,
        "pair_diversity_score": 0.0,
        "cross_pair_mad_std": 0.0,
    }

    for key, value in defaults.items():
        assert features[key] == value


def test_missing_timestamps():
    """Trades with NaT timestamps should be handled gracefully."""
    trades = pd.DataFrame(
        [
            {
                "trade_id": "1",
                "ledger_close_time": pd.NaT,
                "base_account": "A",
                "counter_account": "B",
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "pair_id": "pair1",
            },
            {
                "trade_id": "2",
                "ledger_close_time": "2024-01-01T00:00:00Z",
                "base_account": "A",
                "counter_account": "C",
                "base_asset": "BTC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "pair_id": "pair2",
            },
        ]
    )

    # Should not raise an error
    features = compute_cross_asset_features("A", trades)
    assert isinstance(features, dict)
    assert "cross_pair_trade_synchrony" in features
