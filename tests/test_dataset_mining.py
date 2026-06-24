"""Tests for the dataset mining pipeline (issue #014).

All six tests required by the issue specification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.build_labelled_dataset import apply_labelling_rules
from scripts.mine_roundtrips import detect_roundtrip_pairs

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_trades(
    trade_a_time: str,
    trade_b_time: str,
    amount_a: float = 100.0,
    amount_b: float = 100.0,
) -> pd.DataFrame:
    """Two-row trade DataFrame: A→B then B→A on USDC:issuer."""
    return pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "ledger_close_time": trade_a_time,
                "base_account": "WALLET_A",
                "counter_account": "WALLET_B",
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": amount_a,
                "price": 0.1,
            },
            {
                "trade_id": "t2",
                "ledger_close_time": trade_b_time,
                "base_account": "WALLET_B",
                "counter_account": "WALLET_A",
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": amount_b,
                "price": 0.1,
            },
        ]
    )


def _wallet_trades_map(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    wallets = ["WALLET_A", "WALLET_B"]
    result = {}
    for w in wallets:
        mask = (trades["base_account"] == w) | (trades["counter_account"] == w)
        result[w] = trades[mask]
    return result


# ── Test 1 ────────────────────────────────────────────────────────────────────


def test_mine_roundtrips_detects_known_pattern():
    """Round-trip within 5 minutes and ±5% amounts must be detected."""
    trades = _make_trades(
        trade_a_time="2024-01-01T00:00:00Z",
        trade_b_time="2024-01-01T00:04:00Z",  # 4 minutes later
        amount_a=100.0,
        amount_b=102.0,  # within 5%
    )
    pairs = detect_roundtrip_pairs(trades, max_ledger_window=100, amount_tolerance=0.05)
    assert len(pairs) == 1
    assert pairs.iloc[0]["wallet_a"] == "WALLET_A"
    assert pairs.iloc[0]["wallet_b"] == "WALLET_B"


# ── Test 2 ────────────────────────────────────────────────────────────────────


def test_mine_roundtrips_ignores_slow_return():
    """Return trade more than 100 ledger closes later must NOT be detected."""
    # 100 ledger closes × 5 s = 500 s ≈ 8.3 min. Use 2 hours to exceed window.
    trades = _make_trades(
        trade_a_time="2024-01-01T00:00:00Z",
        trade_b_time="2024-01-01T02:00:00Z",  # 2 hours later
        amount_a=100.0,
        amount_b=100.0,
    )
    pairs = detect_roundtrip_pairs(trades, max_ledger_window=100, amount_tolerance=0.05)
    assert len(pairs) == 0


# ── Test 3 ────────────────────────────────────────────────────────────────────


def test_labelling_rule_requires_both_signals():
    """Wallet flagged only by Signal 1 (round-trip) must get label = NaN."""
    # Build a minimal trades set with enough rows for wallet_trades map
    base_time = "2024-01-01T00:00:00Z"
    rows = []
    for i in range(60):
        rows.append(
            {
                "trade_id": f"t{i}",
                "ledger_close_time": base_time,
                "base_account": "WALLET_A",
                "counter_account": f"CP{i:03d}",
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "price": 0.1,
            }
        )
    trades = pd.DataFrame(rows)
    wt_map = {"WALLET_A": trades}

    # Only Signal 1 fires; Signal 2 (graph) does NOT flag WALLET_A
    result = apply_labelling_rules(
        wallets=["WALLET_A"],
        roundtrip_flagged={"WALLET_A"},
        graph_flagged=set(),
        wallet_trades=wt_map,
    )
    row = result[result["wallet"] == "WALLET_A"].iloc[0]
    assert pd.isna(row["label"]), f"Expected NaN label, got {row['label']}"
    assert row["labelling_signal"] == "roundtrip_only"


# ── Test 4 ────────────────────────────────────────────────────────────────────


def test_labelling_rule_positive():
    """Wallet flagged by BOTH signals must get label = 1."""
    base_time = "2024-01-01T00:00:00Z"
    rows = [
        {
            "trade_id": f"t{i}",
            "ledger_close_time": base_time,
            "base_account": "WALLET_A",
            "counter_account": "WALLET_B",
            "base_asset": "USDC:issuer",
            "counter_asset": "XLM:native",
            "amount": 100.0,
            "price": 0.1,
        }
        for i in range(5)
    ]
    trades = pd.DataFrame(rows)
    wt_map = {"WALLET_A": trades}

    result = apply_labelling_rules(
        wallets=["WALLET_A"],
        roundtrip_flagged={"WALLET_A"},
        graph_flagged={"WALLET_A"},
        wallet_trades=wt_map,
    )
    row = result[result["wallet"] == "WALLET_A"].iloc[0]
    assert row["label"] == 1
    assert row["labelling_signal"] == "roundtrip_and_graph"


# ── Test 5 ────────────────────────────────────────────────────────────────────


def test_dataset_schema_matches_feature_matrix():
    """Columns in the synthetic dataset (excl. label) must match build_feature_matrix output."""
    from detection.feature_engineering import build_feature_matrix

    synthetic_path = Path("data/synthetic_dataset.parquet")
    if not synthetic_path.exists():
        # Generate the synthetic dataset on the fly
        from scripts.generate_synthetic_dataset import generate_synthetic_dataset

        df = generate_synthetic_dataset(n_wallets=50, seed=0)
        synthetic_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(synthetic_path)

    synthetic_df = pd.read_parquet(synthetic_path)

    # Columns in the released labelled dataset but NOT in the raw feature matrix
    extra_cols = {
        "label",
        "labelling_signal",
        "review_notes",
        "data_window_start",
        "data_window_end",
        "n_trades",
    }
    synthetic_feature_cols = set(synthetic_df.columns) - extra_cols

    # Build a minimal feature matrix to get its column set
    sample_trades = pd.DataFrame(
        [
            {
                "trade_id": "x1",
                "ledger_close_time": "2024-01-01T00:00:00Z",
                "base_account": "A",
                "counter_account": "B",
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "price": 0.1,
            }
        ]
    )
    feature_matrix = build_feature_matrix(sample_trades)
    feature_matrix_cols = set(feature_matrix.columns)

    # Stale parquet files may predate GNN embedding columns; regenerate if needed.
    gnn_only = {c for c in feature_matrix_cols if c.startswith("gnn_")}
    if gnn_only and gnn_only.isdisjoint(synthetic_feature_cols):
        from scripts.generate_synthetic_dataset import generate_synthetic_dataset

        df = generate_synthetic_dataset(n_wallets=50, seed=0)
        synthetic_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(synthetic_path)
        synthetic_df = df
        synthetic_feature_cols = set(synthetic_df.columns) - extra_cols

    # GNN placeholders are zero-filled when no encoder is loaded.
    gnn_cols = {c for c in feature_matrix_cols if c.startswith("gnn_")}
    synthetic_feature_cols -= gnn_cols
    feature_matrix_cols -= gnn_cols

    assert synthetic_feature_cols == feature_matrix_cols, (
        f"Column mismatch.\n"
        f"  In synthetic but not feature matrix: {synthetic_feature_cols - feature_matrix_cols}\n"
        f"  In feature matrix but not synthetic: {feature_matrix_cols - synthetic_feature_cols}"
    )


# ── Test 6 ────────────────────────────────────────────────────────────────────


def test_build_config_json_exists_and_is_valid():
    """data/build_config.json must exist, be valid JSON, and contain required keys."""
    config_path = Path("data/build_config.json")
    assert config_path.exists(), "data/build_config.json does not exist"

    with config_path.open() as f:
        cfg = json.load(f)

    required_keys = {"date_range_start", "date_range_end", "asset_pairs", "thresholds"}
    missing = required_keys - cfg.keys()
    assert not missing, f"build_config.json is missing keys: {missing}"
