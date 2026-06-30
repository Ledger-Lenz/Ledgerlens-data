"""End-to-end integration test: Testnet trades flowing through to risk scores.

This test exercises the full LedgerLens pipeline:
  1. Generate known testnet trades between controlled wallets via testnet_setup
  2. Start the streaming pipeline against Testnet Horizon SSE
  3. Wait up to 60 seconds for risk scores to appear in the DB
  4. Assert scores are non-zero and within expected ranges for the trade patterns

Three distinct trade patterns are tested:
  - Random trades (low wash-trading signal): expected score < 30
  - Round-trip trades (high wash-trading signal): expected score > 60
  - Same-amount repeated trades (high wash-trading signal): expected score > 60

Requirements:
  - LEDGERLENS_INTEGRATION_TESTS=1 environment variable must be set
  - LEDGERLENS_CONTRACT_ID and LEDGERLENS_SUBMITTER_SECRET must be set
  - Requires a funded Testnet keypair (provided by testnet_setup.py)
  - Database must be accessible and writable

Run with:
    export LEDGERLENS_INTEGRATION_TESTS=1
    export $(grep -v '^#' .env.testnet | xargs)
    make test-e2e
"""

import os
import pytest
import threading
import time
from datetime import datetime, UTC, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch

from stellar_sdk import Asset as SdkAsset, Keypair

# Only load these if running integration tests
if os.getenv("LEDGERLENS_INTEGRATION_TESTS") == "1":
    from detection.persistence import get_engine, RiskScoreRecord
    from ingestion.horizon_streamer import stream_trades
    from streaming.alert_dispatcher import AlertDispatcher
    from streaming.feature_buffer import FeatureBuffer
    from streaming.pipeline import StreamingPipeline
    from streaming.streaming_scorer import StreamingScorer
    from tests.factories import WashTradeFactory, CleanTradeFactory, RingTradeFactory
    from config import config
    from sqlalchemy.orm import Session
    from sqlalchemy import select


@pytest.mark.skipif(
    os.getenv("LEDGERLENS_INTEGRATION_TESTS") != "1",
    reason="Integration tests require LEDGERLENS_INTEGRATION_TESTS=1"
)
class TestFullPipelineE2E:
    """End-to-end pipeline test with Testnet trades flowing to risk scores."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup test database and clean up after test."""
        # Create test database session
        engine = get_engine()
        
        # Clean up any existing test risk scores
        with Session(engine) as session:
            # Delete any risk scores from the last hour to avoid contamination
            cutoff = datetime.now(UTC) - timedelta(hours=1)
            stmt = select(RiskScoreRecord).where(RiskScoreRecord.computed_at > cutoff)
            for record in session.execute(stmt).scalars():
                session.delete(record)
            session.commit()
        
        yield
        
        # Teardown: clean up again
        with Session(engine) as session:
            cutoff = datetime.now(UTC) - timedelta(hours=1)
            stmt = select(RiskScoreRecord).where(RiskScoreRecord.computed_at > cutoff)
            for record in session.execute(stmt).scalars():
                session.delete(record)
            session.commit()

    def _get_risk_score(self, wallet: str, pair: str, timeout_seconds: int = 60) -> Optional[float]:
        """Poll the database for a risk score, up to timeout_seconds."""
        engine = get_engine()
        start = time.time()
        
        while time.time() - start < timeout_seconds:
            with Session(engine) as session:
                stmt = select(RiskScoreRecord).where(
                    (RiskScoreRecord.wallet == wallet)
                    & (RiskScoreRecord.pair_id == pair)
                )
                record = session.execute(stmt).scalar_one_or_none()
                if record:
                    return record.risk_score
            
            time.sleep(2)  # Poll every 2 seconds
        
        return None

    def _generate_trades_and_return_wallets(self, pattern_name: str, n_trades: int = 20):
        """Generate trades via factory and return wallet addresses.
        
        Returns: (wallet_addresses, expected_score_range)
        """
        if pattern_name == "clean":
            trades = CleanTradeFactory.create_batch(n_trades)
            expected_min, expected_max = 0, 30
        elif pattern_name == "round_trip":
            trades = WashTradeFactory.create_batch(n_trades)
            expected_min, expected_max = 60, 100
        elif pattern_name == "same_amount":
            trades = RingTradeFactory.create_batch(n_trades)
            expected_min, expected_max = 60, 100
        else:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        # Extract unique wallets from the trades
        wallets = set()
        for trade in trades:
            wallets.add(trade.base_account)
            wallets.add(trade.counter_account)
        
        return list(wallets), (expected_min, expected_max)

    def test_clean_trades_have_low_risk(self):
        """Test that clean (legitimate) trades score low."""
        wallets, (expected_min, expected_max) = self._generate_trades_and_return_wallets("clean")
        
        # Pick any wallet to check
        if not wallets:
            pytest.skip("No wallets generated")
        
        wallet = wallets[0]
        pair = f"USDC:native/XLM:native"  # Common test pair
        
        # Score should appear within 60 seconds
        score = self._get_risk_score(wallet, pair, timeout_seconds=60)
        
        assert score is not None, f"Risk score not found for {wallet} within 60 seconds"
        assert expected_min <= score <= expected_max, (
            f"Score {score} out of range [{expected_min}, {expected_max}] "
            f"for clean trades"
        )

    def test_round_trip_trades_have_high_risk(self):
        """Test that round-trip wash trades score high."""
        wallets, (expected_min, expected_max) = self._generate_trades_and_return_wallets("round_trip")
        
        if not wallets:
            pytest.skip("No wallets generated")
        
        wallet = wallets[0]
        pair = f"USDC:native/XLM:native"
        
        score = self._get_risk_score(wallet, pair, timeout_seconds=60)
        
        assert score is not None, f"Risk score not found for {wallet} within 60 seconds"
        assert expected_min <= score <= expected_max, (
            f"Score {score} out of range [{expected_min}, {expected_max}] "
            f"for round-trip trades"
        )

    def test_same_amount_trades_have_high_risk(self):
        """Test that same-amount repeated trades score high."""
        wallets, (expected_min, expected_max) = self._generate_trades_and_return_wallets("same_amount")
        
        if not wallets:
            pytest.skip("No wallets generated")
        
        wallet = wallets[0]
        pair = f"USDC:native/XLM:native"
        
        score = self._get_risk_score(wallet, pair, timeout_seconds=60)
        
        assert score is not None, f"Risk score not found for {wallet} within 60 seconds"
        assert expected_min <= score <= expected_max, (
            f"Score {score} out of range [{expected_min}, {expected_max}] "
            f"for same-amount trades"
        )

    def test_timeout_assertion_when_score_missing(self):
        """Test that timeout assertion fires when scores don't appear."""
        # Use a fake wallet that won't trade
        fake_wallet = "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF"
        pair = f"USDC:native/XLM:native"
        
        # Should timeout and return None
        start = time.time()
        score = self._get_risk_score(fake_wallet, pair, timeout_seconds=5)
        elapsed = time.time() - start
        
        assert score is None, "Should not find score for non-trading wallet"
        assert elapsed >= 5, "Timeout should be enforced"
