"""Tests for payment path analysis and multi-hop wash trade detection.

Tests cover path reconstruction, round-trip detection, and feature engineering
integration for payment path operations on Stellar DEX.
"""

import pytest
from datetime import datetime

from ingestion.payment_path_analyzer import (
    ReconstructedPathFlow,
    compute_path_payment_round_trip_frequency,
    reconstruct_path_flow,
    validate_path_schema,
    merge_path_flows,
    MAX_PATH_LENGTH,
)
from detection.feature_engineering import compute_payment_path_features


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
        "asset_path": [
            {"code": "USDC", "issuer": "GA5Z"}
        ],
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
        "asset_path": [
            {"code": "USDT", "issuer": "GA5Z"},
            {"code": "USDC", "issuer": "GA5Z"}
        ],
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
        "asset_path": [
            {"code": "USDC", "issuer": "GA5Z"},
            {"code": "XLM", "issuer": None}
        ],
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
                {"code": "USDT", "issuer": "GA5Z"}
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
        """Test that missing required fields raise KeyError."""
        op = sample_path_payment_strict_send()
        del op["source_account"]
        
        with pytest.raises(KeyError, match="required fields"):
            reconstruct_path_flow(op)

    def test_reject_oversized_path(self):
        """Test that paths exceeding MAX_PATH_LENGTH are rejected."""
        op = sample_path_payment_strict_send()
        op["asset_path"] = [{"code": f"ASS{i}", "issuer": "GA5Z"} for i in range(MAX_PATH_LENGTH + 1)]
        
        flow = reconstruct_path_flow(op)
        assert flow is None

    def test_handle_max_path_length(self):
        """Test that paths at exactly MAX_PATH_LENGTH are accepted."""
        op = sample_path_payment_strict_send()
        op["asset_path"] = [{"code": f"ASS{i}", "issuer": "GA5Z"} for i in range(MAX_PATH_LENGTH)]
        
        flow = reconstruct_path_flow(op)
        assert flow is not None
        assert flow["hop_count"] == MAX_PATH_LENGTH


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
        op["asset_path"] = [{"code": f"ASS{i}", "issuer": "GA5Z"} for i in range(MAX_PATH_LENGTH + 1)]
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
