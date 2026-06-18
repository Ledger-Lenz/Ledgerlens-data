"""Tests for per-pair feature attribution in the detection pipeline (issue #4)."""

import logging
from unittest.mock import MagicMock, patch

import pandas as pd

import run_pipeline
from detection.feature_engineering import build_feature_matrix
from ingestion.historical_loader import load_pair_to_dataframe

# ---------------------------------------------------------------------------
# 1. load_pair_to_dataframe filters to the requested pair only
# ---------------------------------------------------------------------------


@patch("ingestion.historical_loader._fetch_page")
def test_load_pair_to_dataframe_filters_correctly(mock_fetch):
    """load_pair_to_dataframe must return only the pair requested, not all trades."""
    from stellar_sdk import Asset as SdkAsset

    usdc_issuer = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
    xlm = SdkAsset.native()
    usdc = SdkAsset("USDC", usdc_issuer)

    usdc_record = {
        "id": "t1",
        "paging_token": "1",
        "ledger_close_time": "2024-01-01T00:00:00Z",
        "base_account": "GA",
        "counter_account": "GB",
        "base_asset_type": "credit_alphanum4",
        "base_asset_code": "USDC",
        "base_asset_issuer": usdc_issuer,
        "counter_asset_type": "native",
        "counter_asset_code": "",  # native: code is empty, _to_trade uses `or "XLM"`
        "base_amount": "100.0",
        "counter_amount": "50.0",
        "price": {"n": 1, "d": 2},
    }

    mock_fetch.side_effect = [
        {"_embedded": {"records": [usdc_record]}, "_links": {"next": {"href": ""}}},
    ]

    result = load_pair_to_dataframe(usdc, xlm)

    assert not result.empty
    assert len(result) == 1
    assert result.iloc[0]["base_account"] == "GA"
    assert result.iloc[0]["counter_account"] == "GB"


# ---------------------------------------------------------------------------
# 2. build_feature_matrix returns distinct results for each pair DataFrame
# ---------------------------------------------------------------------------


def test_feature_matrix_built_per_pair():
    """Each pair's DataFrame produces an independent feature matrix."""
    ts = pd.Timestamp("2024-01-01", tz="UTC")

    df_pair_a = pd.DataFrame(
        {
            "trade_id": ["t1"],
            "ledger_close_time": [ts],
            "base_account": ["GA"],
            "counter_account": ["GB"],
            "base_asset": ["USDC:GISSUER"],
            "counter_asset": ["XLM:native"],
            "amount": [1000.0],
            "price": [0.5],
        }
    )
    df_pair_b = pd.DataFrame(
        {
            "trade_id": ["t2"],
            "ledger_close_time": [ts],
            "base_account": ["GC"],
            "counter_account": ["GD"],
            "base_asset": ["BTC:GISSUER2"],
            "counter_asset": ["XLM:native"],
            "amount": [0.001],
            "price": [30000.0],
        }
    )

    fm_a = build_feature_matrix(df_pair_a)
    fm_b = build_feature_matrix(df_pair_b)

    assert set(fm_a["wallet"]) == {"GA", "GB"}
    assert set(fm_b["wallet"]) == {"GC", "GD"}
    assert set(fm_a["wallet"]).isdisjoint(set(fm_b["wallet"]))


# ---------------------------------------------------------------------------
# 3. Pipeline upserts one record per (wallet, pair_id) — two pairs × two wallets
# ---------------------------------------------------------------------------


def test_pipeline_upserts_one_record_per_wallet_per_pair():
    """One RiskScore row is upserted per (wallet, pair_id) tuple."""
    ts = pd.Timestamp("2024-01-01", tz="UTC")

    def make_trades(w1, w2):
        return pd.DataFrame(
            {
                "base_account": [w1],
                "counter_account": [w2],
                "ledger_close_time": [ts],
                "amount": [100.0],
            }
        )

    pair_a_trades = make_trades("GA", "GB")
    pair_b_trades = make_trades("GC", "GD")

    call_count = [0]

    def fake_load_pair(asset, xlm, start_time=None):
        call_count[0] += 1
        return pair_a_trades if call_count[0] == 1 else pair_b_trades

    upserted: list[tuple[str, str]] = []

    def fake_upsert(wallet, asset_pair, risk_score):
        upserted.append((wallet, asset_pair))

    feat_a = pd.DataFrame({"wallet": ["GA", "GB"], "benford_mad_1h": [0.0, 0.0]})
    feat_b = pd.DataFrame({"wallet": ["GC", "GD"], "benford_mad_1h": [0.0, 0.0]})
    feat_call = [0]

    def fake_build_feature_matrix(trades_df, **kwargs):
        feat_call[0] += 1
        return feat_a if feat_call[0] == 1 else feat_b

    def make_scored(wallets):
        return pd.DataFrame(
            {
                "wallet": wallets,
                "score": [10] * len(wallets),
                "benford_flag": [False] * len(wallets),
                "ml_flag": [False] * len(wallets),
                "confidence": [50] * len(wallets),
            }
        )

    score_call = [0]
    fake_scorer = MagicMock()

    def fake_score_matrix(fm):
        score_call[0] += 1
        return make_scored(["GA", "GB"] if score_call[0] == 1 else ["GC", "GD"])

    fake_scorer.score_matrix.side_effect = fake_score_matrix

    # Use valid Stellar issuer public keys
    usdc_issuer = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
    btc_issuer = "GBVOL67TMUQBGL4TZYNMY3ZQ5WGQYFPFD5VJRWXR72VA33VFNL225PL5"

    with (
        patch("sys.argv", ["run_pipeline.py", "--no-orderbook"]),
        patch.object(run_pipeline, "load_pair_to_dataframe", side_effect=fake_load_pair),
        patch.object(run_pipeline, "build_feature_matrix", side_effect=fake_build_feature_matrix),
        patch("detection.model_inference.RiskScorer", return_value=fake_scorer),
        patch.object(run_pipeline.RiskScoreStore, "upsert", side_effect=fake_upsert),
        patch.object(
            run_pipeline.config,
            "WATCHED_ASSET_PAIRS",
            [("USDC", usdc_issuer), ("BTC", btc_issuer)],
        ),
    ):
        run_pipeline.main()

    # 2 wallets × 2 pairs = 4 upserts
    assert len(upserted) == 4
    pair_ids = {ap for _, ap in upserted}
    assert len(pair_ids) == 2  # two distinct pair_ids


# ---------------------------------------------------------------------------
# 4. watched_pairs_label is no longer present in run_pipeline
# ---------------------------------------------------------------------------


def test_watched_pairs_label_removed():
    """watched_pairs_label must not exist in run_pipeline after the refactor."""
    assert not hasattr(
        run_pipeline, "watched_pairs_label"
    ), "watched_pairs_label should have been removed from run_pipeline.py"


# ---------------------------------------------------------------------------
# 5. pair_id appears in log output during stage 1
# ---------------------------------------------------------------------------


def test_per_pair_logging(caplog):
    """The pair_id string must appear in log output during stage 1 trade loading."""
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    trades = pd.DataFrame(
        {
            "base_account": ["GA"],
            "counter_account": ["GB"],
            "ledger_close_time": [ts],
            "amount": [100.0],
        }
    )
    feature_matrix = pd.DataFrame({"wallet": ["GA", "GB"], "benford_mad_1h": [0.0, 0.0]})
    scored = pd.DataFrame(
        {
            "wallet": ["GA", "GB"],
            "score": [10, 20],
            "benford_flag": [False, False],
            "ml_flag": [False, False],
            "confidence": [50, 50],
        }
    )
    fake_scorer = MagicMock()
    fake_scorer.score_matrix.return_value = scored

    usdc_issuer = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
    expected_pair_id = f"USDC:{usdc_issuer}/XLM:native"

    with caplog.at_level(logging.INFO):
        with (
            patch("sys.argv", ["run_pipeline.py", "--no-orderbook", "--no-persist"]),
            patch.object(run_pipeline, "load_pair_to_dataframe", return_value=trades),
            patch.object(run_pipeline, "build_feature_matrix", return_value=feature_matrix),
            patch("detection.model_inference.RiskScorer", return_value=fake_scorer),
            patch.object(
                run_pipeline.config,
                "WATCHED_ASSET_PAIRS",
                [("USDC", usdc_issuer)],
            ),
        ):
            run_pipeline.main()

    assert expected_pair_id in caplog.text
