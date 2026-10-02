from __future__ import annotations

import fakeredis
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from ci_metrics.store import MetricsStore
from detection.adversarial.graph_attacks import EdgeInsertionEvasion, NodeSplittingEvasion
from detection.score_normaliser import (
    DIRECT_COMPARISON_MATRIX,
    REGISTERED_MODEL_TYPES,
    SCORE_NORM_MIN_SAMPLES,
    PerPairScoreNormaliser,
    is_directly_comparable,
)
from scripts.adversarial_wash_trade_simulator import (
    make_cross_chain_bridge_wash_trade,
    make_cross_venue_round_trip,
)

ASSET_PAIR = "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"


def _normaliser() -> PerPairScoreNormaliser:
    return PerPairScoreNormaliser(fakeredis.FakeRedis(decode_responses=False))


@given(
    scores=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=SCORE_NORM_MIN_SAMPLES,
        max_size=100,
        unique=True,
    ),
    probes=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=12,
    ),
)
def test_normalised_scores_are_monotonic_and_bounded_for_every_model_type(scores, probes):
    for model_type in REGISTERED_MODEL_TYPES:
        normaliser = _normaliser()
        for score in scores:
            normaliser.add_score(ASSET_PAIR, score)
        ordered = sorted(probes)
        values = [
            normaliser.normalise(ASSET_PAIR, probe).normalised_risk_score for probe in ordered
        ]
        assert values == sorted(values)
        upper = (len(set(scores)) + 0.5) / len(set(scores))
        assert all(0.0 < value <= upper for value in values)
        assert is_directly_comparable(model_type, model_type)


def test_cross_model_compatibility_matrix_is_diagonal_only():
    assert set(DIRECT_COMPARISON_MATRIX) == set(REGISTERED_MODEL_TYPES)
    for left in REGISTERED_MODEL_TYPES:
        for right in REGISTERED_MODEL_TYPES:
            assert is_directly_comparable(left, right) is (left == right)


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trade_id": "1", "base_account": "A", "counter_account": "B", "amount": 100.0},
            {"trade_id": "2", "base_account": "B", "counter_account": "C", "amount": 50.0},
        ]
    )


def test_edge_insertion_adds_two_hop_decoys_without_removing_original():
    original = _trades()
    attacked = EdgeInsertionEvasion(n_edges=1).perturb(original)
    assert len(attacked) == len(original) + 2
    assert set(original.trade_id).issubset(set(attacked.trade_id))
    assert any(str(value).startswith("GEDGE_DECOY_") for value in attacked.counter_account)


def test_node_splitting_preserves_rows_and_splits_target_degree():
    original = _trades()
    attacked = NodeSplittingEvasion(n_sybils=2, target_wallet="B").perturb(original)
    assert len(attacked) == len(original)
    assert "B" not in set(attacked.base_account) | set(attacked.counter_account)
    assert len(set(attacked.base_account) | set(attacked.counter_account)) > len(
        set(original.base_account) | set(original.counter_account)
    )


def test_cross_venue_round_trip_has_two_venues_and_cycles():
    trades = make_cross_venue_round_trip(3, seed=1)
    assert len(trades) == 6
    assert set(trades["venue"]) == {"venue_a", "venue_b"}
    assert trades.groupby("cycle_id").size().tolist() == [2, 2, 2]


def test_cross_chain_bridge_pattern_links_origin_and_return_legs():
    trades = make_cross_chain_bridge_wash_trade(3, seed=1)
    assert len(trades) == 6
    assert set(trades["chain"]) == {"stellar", "ethereum"}
    assert trades.groupby("bridge_id").size().eq(2).all()


def test_robustness_history_store_round_trips(tmp_path):
    store = MetricsStore(tmp_path / "history.jsonl")
    assert store.all() == []
    # The CI entry point uses the same store contract for baseline and latest.
    from ci_metrics import CIRunRecord, MetricSnapshot

    record = CIRunRecord(
        run_id="test",
        commit_sha="abc",
        branch="main",
        timestamp_utc="2026-01-01T00:00:00Z",
        metrics=[MetricSnapshot("adversarial_robustness_score", 70.0)],
    )
    store.append(record)
    assert store.metric_series("adversarial_robustness_score") == [("test", 70.0)]
