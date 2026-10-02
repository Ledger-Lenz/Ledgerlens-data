"""SIGTERM mid-batch drains the in-flight message and commits it (#892)."""

import os
import signal
from unittest.mock import MagicMock

from streaming.kafka_worker import KafkaWorker


def _msg(offset: int) -> MagicMock:
    m = MagicMock()
    m.error.return_value = None
    m.offset.return_value = offset
    return m


def test_sigterm_mid_batch_commits_in_flight_and_stops_consuming():
    batch = [_msg(i) for i in range(5)]
    consumer = MagicMock()
    consumer.poll.side_effect = batch
    worker = KafkaWorker(
        MagicMock(),
        MagicMock(),
        consumer=consumer,
        lag_threshold=1000,
        enable_backpressure=False,
        dedup_cache=MagicMock(),
        drain_timeout=5,
    )
    worker.install_signal_handlers()
    consumed, committed = [], []

    def fake_process(msg):
        consumed.append(msg.offset())
        if msg.offset() == 1:
            os.kill(os.getpid(), signal.SIGTERM)  # arrives mid-processing
        committed.append(msg.offset())

    worker.process_message = fake_process
    try:
        worker.run()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.default_int_handler)

    assert consumed == committed == [0, 1]  # no partial state
    assert consumer.poll.call_count == 2  # no new messages after SIGTERM
    consumer.close.assert_called_once()
