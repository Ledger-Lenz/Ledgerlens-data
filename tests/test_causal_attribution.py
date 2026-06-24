import datetime

import pandas as pd
import pytest

from detection.causal_attribution import CounterfactualAttributor
from detection.feature_engineering import build_feature_vector
from detection.causal_forensic_report import CausalForensicReportGenerator
from detection.wallet_graph import build_funding_graph
from ingestion.data_models import AccountActivity
from scripts.score_wallet import _parse_remove_trade_ids

WALLET_A = "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF"
WALLET_B = "GBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBWHF"
WALLET_C = "GCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCWHF"
WALLET_D = "GDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDWHF"


def _wallet_trades() -> pd.DataFrame:
    rows = []
    for idx, amount in enumerate([100.0, 101.0, 102.0], start=1):
        rows.append(
            {
                "trade_id": f"wash-{idx}",
                "ledger_close_time": datetime.datetime(2024, 6, 1, 12, idx, tzinfo=datetime.UTC),
                "base_account": WALLET_A,
                "counter_account": WALLET_A,
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": amount,
                "price": 1.0,
            }
        )
    for idx, amount in enumerate([9.0, 8.0], start=1):
        rows.append(
            {
                "trade_id": f"legit-{idx}",
                "ledger_close_time": datetime.datetime(2024, 6, 1, 13, idx, tzinfo=datetime.UTC),
                "base_account": WALLET_A,
                "counter_account": WALLET_C,
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": amount,
                "price": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _wash_only_trades() -> pd.DataFrame:
    rows = []
    for idx, amount in enumerate([100.0, 101.0, 102.0], start=1):
        rows.append(
            {
                "trade_id": f"wash-only-{idx}",
                "ledger_close_time": datetime.datetime(2024, 6, 1, 12, idx, tzinfo=datetime.UTC),
                "base_account": WALLET_A,
                "counter_account": WALLET_A,
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": amount,
                "price": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _funding_graph() -> tuple[list[AccountActivity], object]:
    activities = [
        AccountActivity(account_id=WALLET_D, account_created_at=datetime.datetime(2020, 1, 1)),
        AccountActivity(
            account_id=WALLET_A,
            account_created_at=datetime.datetime(2024, 1, 1),
            funding_account=WALLET_D,
        ),
        AccountActivity(
            account_id=WALLET_B,
            account_created_at=datetime.datetime(2024, 1, 2),
            funding_account=WALLET_D,
        ),
        AccountActivity(
            account_id=WALLET_C,
            account_created_at=datetime.datetime(2024, 1, 3),
            funding_account=WALLET_D,
        ),
    ]
    return activities, build_funding_graph(activities)


class FakeScorer:
    def __init__(self):
        self.models = {"fake": object()}

    def score(self, feature_row: pd.Series) -> dict:
        score = int(round(feature_row.get("round_trip_frequency", 0.0) * 40))
        score += int(round(feature_row.get("self_matching_rate", 0.0) * 40))
        score += int(round(feature_row.get("benford_mad_24h", 0.0) * 50))
        return {
            "score": score,
            "benford_flag": bool(feature_row.get("benford_mad_24h", 0.0) > 0.015),
            "ml_flag": bool(score >= 50),
            "confidence": min(score, 100),
        }


def test_counterfactual_score_identity_and_trade_removal():
    trades = _wallet_trades()
    attributor = CounterfactualAttributor(FakeScorer())

    feature_row = build_feature_vector(WALLET_A, trades)
    expected = attributor._scorer.score(pd.Series(feature_row))

    result = attributor.counterfactual_score(WALLET_A, trades, [])
    assert result["original_score"] == expected["score"]
    assert result["counterfactual_score"] == expected["score"]

    zero_result = attributor.counterfactual_score(
        WALLET_A,
        _wash_only_trades(),
        [f"wash-only-{i}" for i in range(1, 4)],
    )
    assert zero_result["counterfactual_score"] <= result["original_score"]
    assert zero_result["counterfactual_score"] < 10


def test_minimal_exonerating_set_finds_obvious_wash_trades():
    trades = _wallet_trades()
    attributor = CounterfactualAttributor(FakeScorer())

    result = attributor.minimal_exonerating_set(WALLET_A, trades, threshold=11)
    assert result is not None
    assert set(result) == {"wash-1", "wash-2", "wash-3"}


def test_root_cause_wallet_prefers_highest_shared_counterparty():
    trades = pd.DataFrame(
        [
            {
                "trade_id": "r1",
                "ledger_close_time": datetime.datetime(2024, 6, 1, 12, 1, tzinfo=datetime.UTC),
                "base_account": WALLET_A,
                "counter_account": WALLET_B,
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 100.0,
                "price": 1.0,
            },
            {
                "trade_id": "r2",
                "ledger_close_time": datetime.datetime(2024, 6, 1, 12, 2, tzinfo=datetime.UTC),
                "base_account": WALLET_A,
                "counter_account": WALLET_B,
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 101.0,
                "price": 1.0,
            },
            {
                "trade_id": "r3",
                "ledger_close_time": datetime.datetime(2024, 6, 1, 12, 3, tzinfo=datetime.UTC),
                "base_account": WALLET_A,
                "counter_account": WALLET_B,
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 102.0,
                "price": 1.0,
            },
            {
                "trade_id": "r4",
                "ledger_close_time": datetime.datetime(2024, 6, 1, 13, 1, tzinfo=datetime.UTC),
                "base_account": WALLET_A,
                "counter_account": WALLET_C,
                "base_asset": "USDC:issuer",
                "counter_asset": "XLM:native",
                "amount": 10.0,
                "price": 1.0,
            },
        ]
    )
    graph = build_funding_graph(
        [
            AccountActivity(account_id=WALLET_D, account_created_at=datetime.datetime(2020, 1, 1)),
            AccountActivity(
                account_id=WALLET_A,
                account_created_at=datetime.datetime(2024, 1, 1),
                funding_account=WALLET_D,
            ),
            AccountActivity(
                account_id=WALLET_B,
                account_created_at=datetime.datetime(2024, 1, 2),
                funding_account=WALLET_D,
            ),
            AccountActivity(
                account_id=WALLET_C,
                account_created_at=datetime.datetime(2024, 1, 3),
                funding_account=WALLET_D,
            ),
        ]
    )
    attributor = CounterfactualAttributor(FakeScorer())

    result = attributor.root_cause_wallet(WALLET_A, trades, graph)
    assert result == WALLET_B


def test_causal_chain_respects_max_hops_and_cycles():
    graph = build_funding_graph(
        [
            AccountActivity(
                account_id=WALLET_A,
                account_created_at=datetime.datetime(2024, 1, 1),
                funding_account=WALLET_B,
            ),
            AccountActivity(
                account_id=WALLET_B,
                account_created_at=datetime.datetime(2024, 1, 2),
                funding_account=WALLET_C,
            ),
            AccountActivity(
                account_id=WALLET_C,
                account_created_at=datetime.datetime(2024, 1, 3),
                funding_account=WALLET_A,
            ),
        ]
    )
    attributor = CounterfactualAttributor(FakeScorer())

    chain = attributor.causal_chain(WALLET_A, graph, max_hops=2)
    assert len(chain) <= 3
    assert chain[0]["role"] == "primary"


def test_interventional_score_propagates_downstream_features():
    trades = _wallet_trades()
    activities, graph = _funding_graph()
    attributor = CounterfactualAttributor(FakeScorer())
    scm = attributor.build_scm(WALLET_A, trades, activities=activities, funding_graph=graph)

    original = attributor.interventional_score(WALLET_A, scm, {})
    intervened = attributor.interventional_score(WALLET_A, scm, {"benford_chi_square_24h": 0.0})

    assert intervened["score"] <= original["score"]
    assert "benford_chi_square_24h" in intervened["features_changed"]
    assert intervened["features_changed"].get("net_roundtrip_ratio") is not None


def test_parse_remove_trade_ids_validates_history():
    trades = _wallet_trades()
    with pytest.raises(ValueError):
        _parse_remove_trade_ids("not-a-trade", trades, WALLET_A)


def test_forensic_report_generator_populates_causal_attribution():
    trades = _wallet_trades()
    activities, graph = _funding_graph()
    feature_row = pd.Series(build_feature_vector(WALLET_A, trades))
    generator = CausalForensicReportGenerator(scorer=FakeScorer())

    report = generator.generate(
        WALLET_A,
        "USDC:issuer/XLM:native",
        feature_row,
        trades,
        activity=activities[1],
        funding_graph=graph,
        causal=True,
    )

    assert report.causal_attribution is not None
    assert isinstance(report.causal_attribution.minimal_exonerating_trades, list)
