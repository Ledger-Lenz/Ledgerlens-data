"""Tests for payment path analysis and multi-hop wash trade detection.

Tests cover path reconstruction, round-trip detection, and feature engineering
integration for payment path operations on Stellar DEX.
"""

from datetime import datetime

import pytest

from detection.feature_engineering import compute_payment_path_features
from ingestion.exceptions import RecordValidationError
from ingestion.payment_path_analyzer import (
    MAX_PATH_LENGTH,
    ReconstructedPathFlow,
    compute_path_payment_round_trip_frequency,
    merge_path_flows,
    path_has_cycle,
    reconstruct_path_flow,
    validate_path_schema,
)


def sample_path_payment_strict_send() -> dict:
    """Create a sample path_payment_strict_send operation."""
    return {
        "type": "path_payment_strict_send",
        "source_account": "GAAAA" + "A" * 49,
        "destination_account": "GBBBB" + "B" * 49,
        "transaction_id": "tx-1",
        "created_at": datetime(2024, 1, 1, 12, 0, 0),
        "amount": 100.0,  # source amount (exact)
        "destination_amount": 95.0,  # destination amount (variable)
        "asset_path": [{"code": "USDC", "issuer": "GA5Z"}],
    }


def sample_path_payment_strict_receive() -> dict:
    """Create a sample path_payment_strict_receive operation."""
    return {
        "type": "path_payment_strict_receive",
        "source_account": "GAAAA" + "A" * 49,
        "destination_account": "GBBBB" + "B" * 49,
        "transaction_id": "tx-2",
        "created_at": datetime(2024, 1, 1, 13, 0, 0),
        "amount_sent": 105.0,  # source amount (variable)
        "amount": 100.0,  # destination amount (exact)
        "asset_path": [{"code": "USDT", "issuer": "GA5Z"}, {"code": "USDC", "issuer": "GA5Z"}],
    }


def sample_round_trip_path() -> dict:
    """Create a round-trip path payment (A -> B -> A)."""
    return {
        "type": "path_payment_strict_send",
        "source_account": "GAAAA" + "A" * 49,
        "destination_account": "GAAAA" + "A" * 49,  # Same as source = round-trip
        "transaction_id": "tx-roundtrip",
        "created_at": datetime(2024, 1, 1, 12, 0, 0),
        "amount": 100.0,
        "destination_amount": 95.0,
        "asset_path": [{"code": "USDC", "issuer": "GA5Z"}, {"code": "XLM", "issuer": None}],
    }


class TestReconstructPathFlow:
    """Tests for path flow reconstruction."""

    def test_reconstruct_strict_send_basic(self):
        """Test basic 1-hop strict send reconstruction."""
        op = sample_path_payment_strict_send()
        flow = reconstruct_path_flow(op)

        assert flow is not None
        assert flow["source_wallet"] == "GAAAA" + "A" * 49
        assert flow["destination_wallet"] == "GBBBB" + "B" * 49
        assert flow["source_amount"] == 100.0
        assert flow["destination_amount"] == 95.0
        assert flow["hop_count"] == 1
        assert flow["is_round_trip"] is False
        assert flow["path_payment_ids"] == ["tx-1"]

    def test_reconstruct_strict_receive_2hops(self):
        """Test 2-hop strict receive reconstruction."""
        op = sample_path_payment_strict_receive()
        flow = reconstruct_path_flow(op)

        assert flow is not None
        assert flow["source_amount"] == 105.0
        assert flow["destination_amount"] == 100.0
        assert flow["hop_count"] == 2
        assert flow["is_round_trip"] is False

    def test_reconstruct_round_trip(self):
        """Test round-trip path (A -> B -> A)."""
        op = sample_round_trip_path()
        flow = reconstruct_path_flow(op)

        assert flow is not None
        assert flow["source_wallet"] == flow["destination_wallet"]
        assert flow["is_round_trip"] is True
        assert flow["hop_count"] == 2

    def test_reconstruct_3hop_round_trip(self):
        """Test 3-hop round-trip path (A -> B -> C -> A)."""
        op = {
            "type": "path_payment_strict_send",
            "source_account": "GAAAA" + "A" * 49,
            "destination_account": "GAAAA" + "A" * 49,
            "transaction_id": "tx-3hop",
            "created_at": datetime(2024, 1, 1, 12, 0, 0),
            "amount": 100.0,
            "destination_amount": 88.0,
            "asset_path": [
                {"code": "USDC", "issuer": "GA5Z"},
                {"code": "XLM", "issuer": None},
                {"code": "USDT", "issuer": "GA5Z"},
            ],
        }
        flow = reconstruct_path_flow(op)

        assert flow is not None
        assert flow["hop_count"] == 3
        assert flow["is_round_trip"] is True

    def test_reject_invalid_operation_type(self):
        """Test that invalid operation types raise ValueError."""
        op = sample_path_payment_strict_send()
        op["type"] = "trade"

        with pytest.raises(ValueError, match="not a path payment variant"):
            reconstruct_path_flow(op)

    def test_reject_missing_required_fields(self):
        """Missing required fields raise RecordValidationError.

        Deliberately not a ``KeyError``: ``RecordValidationError`` also wraps
        pydantic failures elsewhere, and inheriting ``KeyError`` would let those
        satisfy unrelated ``except KeyError`` control flow (see
        ``ingestion/sketches.py``'s wallet-lock lookup).
        """
        op = sample_path_payment_strict_send()
        del op["source_account"]

        with pytest.raises(RecordValidationError, match="required fields"):
            reconstruct_path_flow(op)

    def test_missing_required_fields_is_not_key_error(self):
        op = sample_path_payment_strict_send()
        del op["source_account"]

        with pytest.raises(RecordValidationError) as excinfo:
            reconstruct_path_flow(op)

        assert not isinstance(excinfo.value, KeyError)
        assert excinfo.value.source == "payment_path_analyzer.reconstruct_path_flow"

    def test_reject_oversized_path(self):
        """Test that paths exceeding MAX_PATH_LENGTH are rejected."""
        op = sample_path_payment_strict_send()
        op["asset_path"] = [
            {"code": f"ASS{i}", "issuer": "GA5Z"} for i in range(MAX_PATH_LENGTH + 1)
        ]

        flow = reconstruct_path_flow(op)
        assert flow is None

    def test_handle_max_path_length(self):
        """Test that paths at exactly MAX_PATH_LENGTH are accepted."""
        op = sample_path_payment_strict_send()
        op["asset_path"] = [{"code": f"ASS{i}", "issuer": "GA5Z"} for i in range(MAX_PATH_LENGTH)]

        flow = reconstruct_path_flow(op)
        assert flow is not None
        assert flow["hop_count"] == MAX_PATH_LENGTH


class TestCyclicalPaths:
    """Payment paths may legitimately revisit an asset they already passed through."""

    def test_path_has_cycle_detects_revisited_asset(self):
        cyclical = [
            {"code": "USDC", "issuer": "GA5Z"},
            {"code": "XLM", "issuer": None},
            {"code": "USDC", "issuer": "GA5Z"},
        ]
        assert path_has_cycle(cyclical) is True

    def test_path_has_cycle_false_for_acyclic_path(self):
        acyclic = [{"code": "USDC", "issuer": "GA5Z"}, {"code": "XLM", "issuer": None}]
        assert path_has_cycle(acyclic) is False

    def test_path_has_cycle_handles_empty_path(self):
        assert path_has_cycle([]) is False

    def test_reconstruct_cyclical_path_terminates_with_result(self):
        """A cyclical path returns a sane flow rather than hanging or recursing.

        Reconstruction never walks the hops recursively, so the cycle is only
        flagged. If this ever regressed into a graph walk the test would hang
        or raise RecursionError instead of asserting.
        """
        op = sample_path_payment_strict_send()
        op["asset_path"] = [
            {"code": "USDC", "issuer": "GA5Z"},
            {"code": "XLM", "issuer": None},
            {"code": "USDC", "issuer": "GA5Z"},  # revisits the first asset
        ]

        flow = reconstruct_path_flow(op)

        assert flow is not None
        assert flow["hop_count"] == 3
        assert flow["is_cyclical"] is True

    def test_acyclic_path_is_not_flagged_as_cyclical(self):
        flow = reconstruct_path_flow(sample_path_payment_strict_receive())

        assert flow is not None
        assert flow["is_cyclical"] is False


class TestValidatePathSchema:
    """Tests for path operation schema validation."""

    def test_valid_strict_send(self):
        """Test validation of valid strict_send operation."""
        op = sample_path_payment_strict_send()
        assert validate_path_schema(op) is True

    def test_valid_strict_receive(self):
        """Test validation of valid strict_receive operation."""
        op = sample_path_payment_strict_receive()
        assert validate_path_schema(op) is True

    def test_reject_invalid_type(self):
        """Test rejection of invalid operation type."""
        op = sample_path_payment_strict_send()
        op["type"] = "trade"
        assert validate_path_schema(op) is False

    def test_reject_missing_fields(self):
        """Test rejection of operations with missing required fields."""
        op = sample_path_payment_strict_send()
        del op["source_account"]
        assert validate_path_schema(op) is False

    def test_reject_invalid_account_format(self):
        """Test rejection of malformed account IDs."""
        op = sample_path_payment_strict_send()
        op["source_account"] = "invalid_account"
        assert validate_path_schema(op) is False

    def test_reject_oversized_path(self):
        """Test rejection of paths exceeding MAX_PATH_LENGTH."""
        op = sample_path_payment_strict_send()
        op["asset_path"] = [
            {"code": f"ASS{i}", "issuer": "GA5Z"} for i in range(MAX_PATH_LENGTH + 1)
        ]
        assert validate_path_schema(op) is False


class TestComputePathPaymentRoundTripFrequency:
    """Tests for round-trip frequency computation."""

    def test_empty_flows(self):
        """Test that empty flows return 0.0."""
        wallet = "GAAAA" + "A" * 49
        freq = compute_path_payment_round_trip_frequency(wallet, [])
        assert freq == 0.0

    def test_single_round_trip(self):
        """Test wallet with single round-trip flow."""
        wallet = "GAAAA" + "A" * 49
        flow: ReconstructedPathFlow = {
            "source_wallet": wallet,
            "destination_wallet": wallet,
            "source_amount": 100.0,
            "destination_amount": 95.0,
            "hop_count": 2,
            "path_payment_ids": ["tx-1"],
            "execution_time": datetime(2024, 1, 1, 12, 0, 0),
            "is_round_trip": True,
        }

        freq = compute_path_payment_round_trip_frequency(wallet, [flow])
        assert freq == 1.0  # 100% of volume is round-trip

    def test_no_round_trips(self):
        """Test wallet with no round-trip flows."""
        wallet = "GAAAA" + "A" * 49
        flow: ReconstructedPathFlow = {
            "source_wallet": wallet,
            "destination_wallet": "GBBBB" + "B" * 49,  # Different destination
            "source_amount": 100.0,
            "destination_amount": 95.0,
            "hop_count": 1,
            "path_payment_ids": ["tx-1"],
            "execution_time": datetime(2024, 1, 1, 12, 0, 0),
            "is_round_trip": False,
        }

        freq = compute_path_payment_round_trip_frequency(wallet, [flow])
        assert freq == 0.0

    def test_mixed_flows(self):
        """Test wallet with both round-trip and direct flows."""
        wallet = "GAAAA" + "A" * 49
        flows: list[ReconstructedPathFlow] = [
            {
                "source_wallet": wallet,
                "destination_wallet": wallet,
                "source_amount": 100.0,
                "destination_amount": 95.0,
                "hop_count": 2,
                "path_payment_ids": ["tx-1"],
                "execution_time": datetime(2024, 1, 1, 12, 0, 0),
                "is_round_trip": True,
            },
            {
                "source_wallet": wallet,
                "destination_wallet": "GBBBB" + "B" * 49,
                "source_amount": 200.0,
                "destination_amount": 190.0,
                "hop_count": 1,
                "path_payment_ids": ["tx-2"],
                "execution_time": datetime(2024, 1, 1, 13, 0, 0),
                "is_round_trip": False,
            },
        ]

        freq = compute_path_payment_round_trip_frequency(wallet, flows)
        assert freq == pytest.approx(100.0 / 300.0)  # 100 round-trip of 300 total

    def test_wallet_not_source(self):
        """Test that only flows where wallet is source are counted."""
        wallet = "GAAAA" + "A" * 49
        other = "GBBBB" + "B" * 49
        flow: ReconstructedPathFlow = {
            "source_wallet": other,  # Different source
            "destination_wallet": wallet,
            "source_amount": 100.0,
            "destination_amount": 95.0,
            "hop_count": 1,
            "path_payment_ids": ["tx-1"],
            "execution_time": datetime(2024, 1, 1, 12, 0, 0),
            "is_round_trip": False,
        }

        freq = compute_path_payment_round_trip_frequency(wallet, [flow])
        assert freq == 0.0


class TestMergePathFlows:
    """Tests for merging multiple path flow lists."""

    def test_merge_empty_lists(self):
        """Test merging empty flow lists."""
        result = merge_path_flows([], [])
        assert result == []

    def test_merge_no_duplicates(self):
        """Test merging flows with no overlapping routes."""
        flow1: ReconstructedPathFlow = {
            "source_wallet": "GAAAA" + "A" * 49,
            "destination_wallet": "GBBBB" + "B" * 49,
            "source_amount": 100.0,
            "destination_amount": 95.0,
            "hop_count": 1,
            "path_payment_ids": ["tx-1"],
            "execution_time": datetime(2024, 1, 1, 12, 0, 0),
            "is_round_trip": False,
        }
        flow2: ReconstructedPathFlow = {
            "source_wallet": "GCCCC" + "C" * 49,
            "destination_wallet": "GDDDD" + "D" * 49,
            "source_amount": 200.0,
            "destination_amount": 190.0,
            "hop_count": 2,
            "path_payment_ids": ["tx-2"],
            "execution_time": datetime(2024, 1, 1, 13, 0, 0),
            "is_round_trip": False,
        }

        result = merge_path_flows([flow1], [flow2])
        assert len(result) == 2

    def test_merge_consolidates_duplicates(self):
        """Test that duplicate routes are consolidated."""
        flow1: ReconstructedPathFlow = {
            "source_wallet": "GAAAA" + "A" * 49,
            "destination_wallet": "GBBBB" + "B" * 49,
            "source_amount": 100.0,
            "destination_amount": 95.0,
            "hop_count": 1,
            "path_payment_ids": ["tx-1"],
            "execution_time": datetime(2024, 1, 1, 12, 0, 0),
            "is_round_trip": False,
        }
        flow2: ReconstructedPathFlow = {
            "source_wallet": "GAAAA" + "A" * 49,
            "destination_wallet": "GBBBB" + "B" * 49,
            "source_amount": 150.0,
            "destination_amount": 140.0,
            "hop_count": 1,
            "path_payment_ids": ["tx-2"],
            "execution_time": datetime(2024, 1, 1, 13, 0, 0),
            "is_round_trip": False,
        }

        result = merge_path_flows([flow1], [flow2])
        assert len(result) == 1
        assert result[0]["source_amount"] == 250.0
        assert result[0]["destination_amount"] == 235.0
        assert set(result[0]["path_payment_ids"]) == {"tx-1", "tx-2"}


class TestComputePaymentPathFeatures:
    """Tests for feature engineering integration."""

    def test_empty_path_flows(self):
        """Test feature computation with no path flows."""
        wallet = "GAAAA" + "A" * 49
        features = compute_payment_path_features(wallet, None)

        assert "path_payment_round_trip_frequency" in features
        assert features["path_payment_round_trip_frequency"] == 0.0

    def test_round_trip_frequency_feature(self):
        """Test that round-trip frequency is correctly computed as feature."""
        wallet = "GAAAA" + "A" * 49
        flow: ReconstructedPathFlow = {
            "source_wallet": wallet,
            "destination_wallet": wallet,
            "source_amount": 100.0,
            "destination_amount": 95.0,
            "hop_count": 2,
            "path_payment_ids": ["tx-1"],
            "execution_time": datetime(2024, 1, 1, 12, 0, 0),
            "is_round_trip": True,
        }

        features = compute_payment_path_features(wallet, [flow])
        assert features["path_payment_round_trip_frequency"] == 1.0

    def test_feature_is_float(self):
        """Test that all returned features are floats."""
        wallet = "GAAAA" + "A" * 49
        flows: list[ReconstructedPathFlow] = [
            {
                "source_wallet": wallet,
                "destination_wallet": wallet,
                "source_amount": 100.0,
                "destination_amount": 95.0,
                "hop_count": 2,
                "path_payment_ids": ["tx-1"],
                "execution_time": datetime(2024, 1, 1, 12, 0, 0),
                "is_round_trip": True,
            }
        ]

        features = compute_payment_path_features(wallet, flows)
        assert isinstance(features["path_payment_round_trip_frequency"], float)


# ---------------------------------------------------------------------------
# Bounded payment-graph traversal (issue #916)
# ---------------------------------------------------------------------------


def _payment(src: str, dst: str, tx: str) -> dict:
    return {"source_account": src, "destination_account": dst, "transaction_id": tx}


def test_trace_linear_chain_is_unchanged_by_bounds():
    """Regression: a well-formed path yields the same output under any bounds."""
    from ingestion.payment_path_analyzer import trace_payment_paths

    payments = [_payment("A", "B", "t1"), _payment("B", "C", "t2"), _payment("C", "D", "t3")]
    default = trace_payment_paths("A", payments)
    generous = trace_payment_paths("A", payments, max_depth=100, max_branching=100)

    assert default.paths == generous.paths == [["A", "B", "C", "D"]]
    assert default.payment_ids == [["t1", "t2", "t3"]]
    assert not default.truncated
    assert not default.has_cycle


def test_trace_branching_graph_enumerates_all_paths_in_order():
    from ingestion.payment_path_analyzer import trace_payment_paths

    payments = [_payment("A", "B", "t1"), _payment("A", "C", "t2"), _payment("B", "D", "t3")]
    result = trace_payment_paths("A", payments)
    assert result.paths == [["A", "B", "D"], ["A", "C"]]
    assert not result.truncated


def test_trace_detects_simple_cycle_and_terminates():
    from ingestion.payment_path_analyzer import trace_payment_paths

    payments = [_payment("A", "B", "t1"), _payment("B", "C", "t2"), _payment("C", "A", "t3")]
    result = trace_payment_paths("A", payments)
    assert result.cycles == [["A", "B", "C", "A"]]
    assert result.payment_ids == [["t1", "t2", "t3"]]
    assert not result.truncated


def test_trace_self_loop_is_a_cycle():
    from ingestion.payment_path_analyzer import trace_payment_paths

    result = trace_payment_paths("A", [_payment("A", "A", "t1")])
    assert result.cycles == [["A", "A"]]


def test_trace_adversarial_complete_graph_is_bounded(caplog):
    """A fully connected cyclic graph has factorially many simple paths; the
    trace must stop at its bounds quickly and log the truncation."""
    import logging
    import time

    from ingestion.payment_path_analyzer import trace_payment_paths

    n = 40
    wallets = [f"W{i}" for i in range(n)]
    payments = [_payment(a, b, f"tx-{a}-{b}") for a in wallets for b in wallets if a != b]

    started = time.monotonic()
    with caplog.at_level(logging.WARNING):
        result = trace_payment_paths("W0", payments, max_depth=8, max_branching=10, max_paths=2_000)
    elapsed = time.monotonic() - started

    assert elapsed < 5.0
    assert result.truncated
    assert len(result.paths) <= 2_000
    assert all(len(p) <= 8 + 1 for p in result.paths)
    assert {"max_branching", "max_paths"} <= result.truncation_reasons
    assert result.truncated_payment_ids
    assert "wallet=W0" in caplog.text
    assert "max_branching" in caplog.text


def test_trace_deep_chain_truncated_at_max_depth(caplog):
    import logging

    from ingestion.payment_path_analyzer import trace_payment_paths

    payments = [_payment(f"W{i}", f"W{i + 1}", f"t{i}") for i in range(10_000)]
    with caplog.at_level(logging.WARNING):
        result = trace_payment_paths("W0", payments, max_depth=5)
    assert result.paths == [[f"W{i}" for i in range(6)]]
    assert result.truncation_reasons == {"max_depth"}
    assert result.truncated_payment_ids == ["t5"]
    assert "t5" in caplog.text


def test_trace_rejects_invalid_bounds():
    from ingestion.exceptions import InvalidInputError
    from ingestion.payment_path_analyzer import trace_payment_paths

    with pytest.raises(InvalidInputError):
        trace_payment_paths("A", [], max_depth=0)
