"""Tests for stream replay capability (Issue #015).

Tests verify:
  1. Replay mode does not trigger alert dispatcher
  2. Dry-run mode produces logs but zero DB writes
  3. Timestamp argument parsing
  4. Trade reconstruction from Kafka payload
  5. Score storage with replay tag
"""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestNoOpAlertDispatcher:
    """Test no-op alert dispatcher used in replay mode."""

    def test_noop_dispatch_does_nothing(self):
        """No-op dispatcher should not raise even with high scores."""
        from scripts.replay_stream import NoOpAlertDispatcher

        dispatcher = NoOpAlertDispatcher()

        # Should not raise or do anything
        dispatcher.dispatch(
            wallet="GA111",
            risk_score={"score": 100, "benford_flag": True, "ml_flag": True, "confidence": 100},
            pair_id="USDC:native/XLM:native",
        )


class TestStreamReplayerInitialization:
    """Test StreamReplayer initialization."""

    def test_replayer_initializes_with_noop_dispatcher(self):
        """Replayer should use no-op dispatcher in replay mode."""
        from scripts.replay_stream import StreamReplayer, NoOpAlertDispatcher

        with patch("scripts.replay_stream.KafkaConsumer"):
            with patch("scripts.replay_stream.RiskScorer"):
                with patch("scripts.replay_stream.StreamingScorer"):
                    replayer = StreamReplayer(
                        topic="trades",
                        dry_run=False,
                    )

                    assert isinstance(replayer.dispatcher, NoOpAlertDispatcher)

    def test_replayer_initializes_without_store_in_dry_run(self):
        """Replayer should not initialize RiskScoreStore in dry-run mode."""
        from scripts.replay_stream import StreamReplayer

        with patch("scripts.replay_stream.KafkaConsumer"):
            with patch("scripts.replay_stream.RiskScorer"):
                with patch("scripts.replay_stream.StreamingScorer"):
                    replayer = StreamReplayer(
                        topic="trades",
                        dry_run=True,
                    )

                    assert replayer.store is None


class TestTimestampParsing:
    """Test timestamp argument parsing."""

    def test_parse_relative_timestamp_offset(self):
        """Parse relative timestamp offset (e.g., -86400 for 24h ago)."""
        from scripts.replay_stream import _get_timestamp_from_arg

        now = datetime(2024, 1, 2, 12, 0, 0, tzinfo=UTC)
        ts = _get_timestamp_from_arg("-86400", now)

        assert ts is not None
        assert ts < int(now.timestamp())

    def test_parse_absolute_timestamp(self):
        """Parse absolute Unix timestamp."""
        from scripts.replay_stream import _get_timestamp_from_arg

        now = datetime.now(UTC)
        ts = _get_timestamp_from_arg("1704067200", now)

        assert ts == 1704067200

    def test_parse_none_returns_none(self):
        """None argument should return None."""
        from scripts.replay_stream import _get_timestamp_from_arg

        now = datetime.now(UTC)
        ts = _get_timestamp_from_arg(None, now)

        assert ts is None

    def test_parse_invalid_timestamp_returns_none(self):
        """Invalid timestamp should return None."""
        from scripts.replay_stream import _get_timestamp_from_arg

        now = datetime.now(UTC)
        ts = _get_timestamp_from_arg("invalid", now)

        assert ts is None


class TestTradeReconstruction:
    """Test Trade object reconstruction from Kafka payload."""

    def test_reconstruct_valid_trade(self):
        """Reconstruct Trade object from valid Kafka payload."""
        from scripts.replay_stream import StreamReplayer

        with patch("scripts.replay_stream.KafkaConsumer"):
            with patch("scripts.replay_stream.RiskScorer"):
                with patch("scripts.replay_stream.StreamingScorer"):
                    replayer = StreamReplayer(topic="trades")

                    payload = {
                        "trade_id": "123",
                        "ledger_close_time": "2024-01-01T00:00:00",
                        "base_account": "GA111",
                        "counter_account": "GA222",
                        "base_asset_code": "USDC",
                        "base_asset_issuer": "native",
                        "counter_asset_code": "XLM",
                        "counter_asset_issuer": "native",
                        "base_amount": 100.0,
                        "counter_amount": 200.0,
                        "price": 2.0,
                    }

                    trade = replayer._reconstruct_trade(payload)

                    assert trade.trade_id == "123"
                    assert trade.base_account == "GA111"
                    assert trade.counter_account == "GA222"
                    assert trade.base_amount == 100.0

    def test_reconstruct_trade_with_missing_issuer(self):
        """Handle trade with missing issuer (should default to None)."""
        from scripts.replay_stream import StreamReplayer

        with patch("scripts.replay_stream.KafkaConsumer"):
            with patch("scripts.replay_stream.RiskScorer"):
                with patch("scripts.replay_stream.StreamingScorer"):
                    replayer = StreamReplayer(topic="trades")

                    payload = {
                        "trade_id": "456",
                        "ledger_close_time": "2024-01-01T00:00:00",
                        "base_account": "GA111",
                        "counter_account": "GA222",
                        "base_asset_code": "BTC",
                        # base_asset_issuer omitted
                        "counter_asset_code": "XLM",
                        "counter_asset_issuer": "native",
                        "base_amount": 50.0,
                        "counter_amount": 100.0,
                        "price": 2.0,
                    }

                    trade = replayer._reconstruct_trade(payload)

                    assert trade.base_asset.code == "BTC"
                    assert trade.base_asset.issuer is None


class TestDryRunMode:
    """Test dry-run mode behavior."""

    def test_dry_run_logs_scores_without_db_write(self):
        """Dry-run mode should log scores without writing to DB."""
        from scripts.replay_stream import StreamReplayer

        with patch("scripts.replay_stream.KafkaConsumer"):
            with patch("scripts.replay_stream.RiskScorer"):
                with patch("scripts.replay_stream.StreamingScorer"):
                    replayer = StreamReplayer(topic="trades", dry_run=True)

                    # Store should be None in dry-run
                    assert replayer.store is None

                    # Mock logger to capture logs
                    with patch("scripts.replay_stream.logger") as mock_logger:
                        replayer._store_replay_score(
                            wallet="GA111",
                            risk_score={"score": 75, "benford_flag": False, "ml_flag": True},
                            pair_id="USDC:native/XLM:native",
                        )

                        # Should have logged
                        mock_logger.info.assert_called()


class TestReplayScoreStorage:
    """Test replay score storage with model version tag."""

    def test_store_replay_score_adds_tag(self):
        """Store replay score should include replay_model_version tag."""
        from scripts.replay_stream import StreamReplayer

        with patch("scripts.replay_stream.KafkaConsumer"):
            with patch("scripts.replay_stream.RiskScorer"):
                with patch("scripts.replay_stream.StreamingScorer"):
                    mock_store = Mock()
                    replayer = StreamReplayer(topic="trades", dry_run=False)
                    replayer.store = mock_store

                    replayer._store_replay_score(
                        wallet="GA111",
                        risk_score={"score": 75, "benford_flag": False, "ml_flag": True, "confidence": 80},
                        pair_id="USDC:native/XLM:native",
                    )

                    # Verify store.upsert was called
                    mock_store.upsert.assert_called_once()
                    call_args = mock_store.upsert.call_args
                    assert call_args[0][0] == "GA111"
                    assert call_args[0][1] == "USDC:native/XLM:native"
                    
                    # Check that replay_model_version tag was added
                    score_dict = call_args[0][2]
                    assert "replay_model_version" in score_dict
                    assert score_dict["replay_model_version"] == "replay"


class TestConfirmationPrompt:
    """Test replay confirmation prompt."""

    def test_confirm_replay_with_yes(self):
        """Confirmation prompt returns True when user enters 'yes'."""
        from scripts.replay_stream import _confirm_replay

        with patch("builtins.input", return_value="yes"):
            assert _confirm_replay() is True

    def test_confirm_replay_with_no(self):
        """Confirmation prompt returns False when user doesn't enter 'yes'."""
        from scripts.replay_stream import _confirm_replay

        with patch("builtins.input", return_value="no"):
            assert _confirm_replay() is False

    def test_confirm_replay_with_eof(self):
        """Confirmation prompt returns False on EOFError."""
        from scripts.replay_stream import _confirm_replay

        with patch("builtins.input", side_effect=EOFError):
            assert _confirm_replay() is False


class TestArgumentParsing:
    """Test command-line argument parsing."""

    def test_parse_args_with_defaults(self):
        """Parse arguments with default values."""
        from scripts.replay_stream import parse_args

        with patch("sys.argv", ["replay_stream.py"]):
            args = parse_args()

            assert args.topic == "trades"
            assert args.bootstrap_servers == "localhost:9092"
            assert args.dry_run is False
            assert args.confirm is False

    def test_parse_args_with_custom_values(self):
        """Parse arguments with custom values."""
        from scripts.replay_stream import parse_args

        with patch(
            "sys.argv",
            [
                "replay_stream.py",
                "--topic",
                "my-topic",
                "--from-timestamp",
                "-86400",
                "--dry-run",
                "--confirm",
            ],
        ):
            args = parse_args()

            assert args.topic == "my-topic"
            assert args.from_timestamp == "-86400"
            assert args.dry_run is True
            assert args.confirm is True
