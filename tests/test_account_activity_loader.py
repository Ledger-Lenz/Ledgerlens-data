"""Tests for ingestion/account_activity_loader.py (issue #5)."""

from unittest.mock import MagicMock, patch

import run_pipeline
from detection.feature_engineering import build_feature_matrix
from detection.wallet_graph import build_funding_graph
from ingestion.account_activity_loader import load_account_activity, load_accounts_activity
from ingestion.data_models import AccountActivity


def _make_effect_page(records: list[dict]) -> dict:
    return {"_embedded": {"records": records}}


def _account_created_record(account: str, funder: str, created_at: str) -> dict:
    return {
        "type": "account_created",
        "account": account,
        "funder": funder,
        "created_at": created_at,
        "starting_balance": "1.0000000",
    }


# ---------------------------------------------------------------------------
# 1. load_account_activity returns correct AccountActivity model
# ---------------------------------------------------------------------------


@patch("ingestion.account_activity_loader.Server")
def test_load_account_activity_returns_model(mock_server_cls):
    record = _account_created_record("GABC", "GFUNDER", "2024-01-15T10:30:00Z")
    page = _make_effect_page([record])

    mock_call_builder = MagicMock()
    mock_call_builder.call.return_value = page
    mock_call_builder.for_account.return_value = mock_call_builder
    mock_call_builder.limit.return_value = mock_call_builder
    mock_call_builder.order.return_value = mock_call_builder

    mock_server = MagicMock()
    mock_server.effects.return_value = mock_call_builder
    mock_server_cls.return_value = mock_server

    result = load_account_activity("GABC")

    assert result is not None
    assert isinstance(result, AccountActivity)
    assert result.account_id == "GABC"
    assert result.funding_account == "GFUNDER"
    assert result.account_created_at.year == 2024


# ---------------------------------------------------------------------------
# 2. load_account_activity returns None for an account with no creation record
# ---------------------------------------------------------------------------


@patch("ingestion.account_activity_loader.Server")
def test_load_account_activity_returns_none_for_missing_account(mock_server_cls):
    page = _make_effect_page([])

    mock_call_builder = MagicMock()
    mock_call_builder.call.return_value = page
    mock_call_builder.for_account.return_value = mock_call_builder
    mock_call_builder.limit.return_value = mock_call_builder
    mock_call_builder.order.return_value = mock_call_builder

    mock_server = MagicMock()
    mock_server.effects.return_value = mock_call_builder
    mock_server_cls.return_value = mock_server

    result = load_account_activity("GNOGENESIS")

    assert result is None


# ---------------------------------------------------------------------------
# 3. load_accounts_activity batches correctly (3 accounts → 3 results)
# ---------------------------------------------------------------------------


@patch("ingestion.account_activity_loader.load_account_activity")
def test_load_accounts_activity_batches_correctly(mock_load):
    mock_load.side_effect = [
        AccountActivity(
            account_id="GA", account_created_at="2024-01-01T00:00:00Z", funding_account="GF"
        ),
        AccountActivity(
            account_id="GB", account_created_at="2024-01-02T00:00:00Z", funding_account="GF"
        ),
        AccountActivity(
            account_id="GC", account_created_at="2024-01-03T00:00:00Z", funding_account="GF"
        ),
    ]

    results = load_accounts_activity(["GA", "GB", "GC"])

    assert len(results) == 3
    assert {r.account_id for r in results} == {"GA", "GB", "GC"}


# ---------------------------------------------------------------------------
# 4. load_accounts_activity tolerates individual failure
# ---------------------------------------------------------------------------


@patch("ingestion.account_activity_loader.load_account_activity")
def test_load_accounts_activity_tolerates_individual_failure(mock_load):
    mock_load.side_effect = [
        AccountActivity(
            account_id="GA", account_created_at="2024-01-01T00:00:00Z", funding_account="GF"
        ),
        ConnectionError("timeout"),
        AccountActivity(
            account_id="GC", account_created_at="2024-01-03T00:00:00Z", funding_account="GF"
        ),
    ]

    results = load_accounts_activity(["GA", "GB", "GC"])

    assert len(results) == 2
    assert {r.account_id for r in results} == {"GA", "GC"}


# ---------------------------------------------------------------------------
# 5. Funding graph features are non-zero when a graph is provided
# ---------------------------------------------------------------------------


def test_funding_graph_features_nonzero_when_graph_provided():
    """Two wallets sharing a funder → funding_source_similarity and
    network_centrality must both be > 0 in the feature matrix."""
    import pandas as pd

    activities = [
        AccountActivity(account_id="GF", account_created_at="2020-01-01T00:00:00Z"),
        AccountActivity(
            account_id="GA", account_created_at="2021-01-01T00:00:00Z", funding_account="GF"
        ),
        AccountActivity(
            account_id="GB", account_created_at="2021-01-02T00:00:00Z", funding_account="GF"
        ),
    ]
    funding_graph = build_funding_graph(activities)

    trades_df = pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "ledger_close_time": pd.Timestamp("2024-01-01", tz="UTC"),
                "base_account": "GA",
                "counter_account": "GB",
                "amount": 100.0,
            }
        ]
    )

    feature_matrix = build_feature_matrix(trades_df, funding_graph=funding_graph)

    row_a = feature_matrix[feature_matrix["wallet"] == "GA"].iloc[0]
    assert row_a["funding_source_similarity"] > 0
    assert row_a["network_centrality"] > 0


# ---------------------------------------------------------------------------
# 6. --no-graph flag means load_accounts_activity is never called
# ---------------------------------------------------------------------------


def test_pipeline_no_graph_flag_skips_activity_load():
    """When --no-graph is passed, load_accounts_activity must not be called."""
    import sys
    from types import ModuleType

    import pandas as pd

    # Build a scored DataFrame with the columns run_pipeline expects
    scored_df = pd.DataFrame(
        {"wallet": [], "score": [], "benford_flag": [], "ml_flag": [], "confidence": []}
    )

    # Inject a fake detection.model_inference module so the lazy import inside
    # run_pipeline.main() succeeds and returns our scored_df.
    fake_module = ModuleType("detection.model_inference")
    mock_scorer = MagicMock()
    mock_scorer.return_value.score_matrix.return_value = scored_df
    fake_module.RiskScorer = mock_scorer  # type: ignore[attr-defined]

    original = sys.modules.get("detection.model_inference")
    sys.modules["detection.model_inference"] = fake_module
    try:
        with (
            patch("run_pipeline.load_pair_to_dataframe") as mock_trades,
            patch("run_pipeline.load_accounts_orderbook_events"),
            patch("run_pipeline.load_accounts_activity") as mock_activity,
            patch("run_pipeline.build_feature_matrix") as mock_feat,
            patch("run_pipeline.RiskScoreStore"),
            patch.object(
                run_pipeline.config,
                "WATCHED_ASSET_PAIRS",
                [("USDC", "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN")],
            ),
        ):
            mock_trades.return_value = scored_df
            mock_feat.return_value = scored_df

            sys.argv = ["run_pipeline.py", "--no-graph", "--no-persist"]

            from run_pipeline import main

            main()

        mock_activity.assert_not_called()
    finally:
        if original is None:
            sys.modules.pop("detection.model_inference", None)
        else:
            sys.modules["detection.model_inference"] = original
