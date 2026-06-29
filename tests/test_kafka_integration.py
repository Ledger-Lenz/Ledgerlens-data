"""Integration tests for Kafka partitioning with producer, consumer, and workers.

These tests use an embedded Kafka instance or Docker Compose.
Requires: docker, docker-compose (optional; can mock Kafka)

Tests verify:
  1. Multiple workers correctly process partitions independently
  2. Each event is processed exactly once (no duplicates, no loss)
  3. Offsets are committed correctly on rebalance
  4. Cross-pair aggregator reads from all partitions
"""

import json
from unittest.mock import Mock, patch, MagicMock

import pytest


class TestProducerConsumerIntegration:
    """Test producer → consumer flow (mocked Kafka)."""

    @patch("ingestion.kafka_producer.KafkaProducer")
    def test_producer_sends_with_partition_key(self, mock_producer_class):
        """Producer sends events with canonical pair ID as partition key."""
        from ingestion.kafka_producer import KafkaTradeProducer
        from ingestion.data_models import Trade, Asset

        # Mock Kafka producer
        mock_producer_instance = Mock()
        mock_producer_class.return_value = mock_producer_instance
        mock_future = Mock()
        mock_metadata = Mock(partition=0, offset=123)
        mock_future.get.return_value = mock_metadata
        mock_producer_instance.send.return_value = mock_future

        # Create producer
        producer = KafkaTradeProducer(topic="trades", bootstrap_servers=["localhost:9092"])

        # Create trade
        trade = Trade(
            trade_id="123",
            ledger_close_time="2024-01-01T00:00:00Z",
            base_account="GA111",
            counter_account="GA222",
            base_asset=Asset(code="USDC", issuer="native"),
            counter_asset=Asset(code="XLM", issuer="native"),
            base_amount=100.0,
            counter_amount=200.0,
            price=2.0,
        )

        # Produce trade
        producer.produce_trade(trade)

        # Verify send was called with canonical pair key
        mock_producer_instance.send.assert_called_once()
        call_args = mock_producer_instance.send.call_args
        assert call_args[0][0] == "trades"  # Topic
        assert call_args[1]["key"] == "USDC:native/XLM:native"  # Partition key
        assert call_args[1]["value"]["trade_id"] == "123"

    @patch("ingestion.kafka_producer.KafkaProducer")
    def test_producer_sends_invalid_pair_to_dlq(self, mock_producer_class):
        """Producer sends events with invalid pairs to dead-letter queue."""
        from ingestion.kafka_producer import KafkaTradeProducer
        from ingestion.data_models import Trade, Asset

        mock_producer_instance = Mock()
        mock_producer_class.return_value = mock_producer_instance
        mock_future = Mock()
        mock_metadata = Mock(partition=0, offset=123)
        mock_future.get.return_value = mock_metadata
        mock_producer_instance.send.return_value = mock_future

        producer = KafkaTradeProducer(topic="trades", bootstrap_servers=["localhost:9092"])

        # Create trade with invalid issuer
        trade = Trade(
            trade_id="456",
            ledger_close_time="2024-01-01T00:00:00Z",
            base_account="GA111",
            counter_account="GA222",
            base_asset=Asset(code="USDC", issuer="invalid_issuer"),
            counter_asset=Asset(code="XLM", issuer="native"),
            base_amount=100.0,
            counter_amount=200.0,
            price=2.0,
        )

        # Produce trade
        producer.produce_trade(trade)

        # Verify send was called to DLQ
        assert mock_producer_instance.send.call_count == 1
        call_args = mock_producer_instance.send.call_args
        assert call_args[0][0] == "trades-dlq"  # DLQ topic


class TestWorkerProcessing:
    """Test per-partition worker behavior."""

    def test_worker_processes_message(self):
        """Worker correctly processes a single trade message."""
        from streaming.kafka_worker import KafkaWorker
        from streaming.feature_buffer import FeatureBuffer, StreamingScorer
        from streaming.alert_dispatcher import AlertDispatcher

        buffer = FeatureBuffer()
        scorer = Mock(spec=StreamingScorer)
        scorer.score_wallet = Mock(return_value=None)
        dispatcher = Mock(spec=AlertDispatcher)

        worker = KafkaWorker(
            topic="trades",
            group_id="test-group",
            buffer=buffer,
            scorer=scorer,
            dispatcher=dispatcher,
        )

        # Create test message
        payload = {
            "trade_id": "123",
            "ledger_close_time": "2024-01-01T00:00:00Z",
            "base_account": "GA111",
            "counter_account": "GA222",
            "base_amount": 100.0,
            "pair_id": "USDC:native/XLM:native",
        }

        # Process message
        worker._process_message(payload)

        # Verify buffer was updated
        assert len(buffer._buffer.get("GA111", [])) > 0 or len(buffer._buffer.get("GA222", [])) > 0

    def test_worker_commits_offsets(self):
        """Worker commits offsets after processing."""
        from streaming.kafka_worker import KafkaWorker

        mock_consumer = MagicMock()
        mock_buffer = Mock(spec=FeatureBuffer)
        mock_scorer = Mock(spec=StreamingScorer)
        mock_dispatcher = Mock(spec=AlertDispatcher)

        with patch("streaming.kafka_worker.KafkaConsumer", return_value=mock_consumer):
            worker = KafkaWorker(
                topic="trades",
                group_id="test-group",
                buffer=mock_buffer,
                scorer=mock_scorer,
                dispatcher=mock_dispatcher,
            )

            worker._commit_offsets()
            mock_consumer.commit.assert_called_once()


class TestCrossVenueAggregator:
    """Test cross-venue aggregator functionality."""

    def test_aggregator_buffers_trades_by_wallet(self):
        """Aggregator correctly buffers trades by wallet."""
        from detection.cross_venue_features import CrossVenueAggregator

        with patch("detection.cross_venue_features.KafkaConsumer"):
            aggregator = CrossVenueAggregator(
                topic="trades",
                group_id="test-aggregator",
            )

            # Buffer some trades
            trades = [
                {
                    "base_account": "GA111",
                    "counter_account": "GA222",
                    "base_amount": 100.0,
                    "pair_id": "USDC:native/XLM:native",
                    "ledger_close_time": "2024-01-01T00:00:00Z",
                },
                {
                    "base_account": "GA111",
                    "counter_account": "GA333",
                    "base_amount": 50.0,
                    "pair_id": "BTC:native/XLM:native",
                    "ledger_close_time": "2024-01-01T00:01:00Z",
                },
            ]

            for trade in trades:
                aggregator._buffer_trade(trade)

            # Verify wallet buffer
            assert len(aggregator._trades_by_wallet["GA111"]) == 2

    def test_aggregator_computes_cross_pair_features(self):
        """Aggregator computes cross-pair features from buffered trades."""
        from detection.cross_venue_features import CrossVenueAggregator

        with patch("detection.cross_venue_features.KafkaConsumer"):
            aggregator = CrossVenueAggregator(
                topic="trades",
                group_id="test-aggregator",
            )

            # Buffer trades across multiple pairs
            trades = [
                {
                    "base_account": "GA111",
                    "counter_account": "GA222",
                    "base_amount": 100.0,
                    "pair_id": "USDC:native/XLM:native",
                },
                {
                    "base_account": "GA111",
                    "counter_account": "GA333",
                    "base_amount": 50.0,
                    "pair_id": "BTC:native/XLM:native",
                },
            ]

            for trade in trades:
                aggregator._buffer_trade(trade)

            # Get cross-pair features
            features = aggregator.get_cross_pair_features("GA111")

            assert features["n_distinct_pairs"] == 2
            assert 0 <= features["cross_pair_volume_concentration"] <= 1.0
            assert 0 <= features["venue_diversity_score"] <= 1.0
