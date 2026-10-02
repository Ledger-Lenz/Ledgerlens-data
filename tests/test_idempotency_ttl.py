"""TTL eviction and store-size bounds for the pipeline idempotency-key store (issue #919)."""

from datetime import UTC, datetime, timedelta

import pytest
from prometheus_client import REGISTRY
from sqlalchemy import update

from pipeline import idempotency
from pipeline.idempotency import CheckpointRecord, CheckpointStore, PipelineCheckpoint

TTL_HOURS = 48
PAIR = "USDC:GA5Z/XLM:native"


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{tmp_path / 'checkpoints.db'}"


def _age(store: CheckpointStore, run_id: str, hours: float) -> None:
    """Backdate every timestamp of *run_id*'s entries by *hours*."""
    ts = datetime.now(UTC) - timedelta(hours=hours)
    with store._session_factory() as session:
        session.execute(
            update(CheckpointRecord)
            .where(CheckpointRecord.run_id == run_id)
            .values(started_at=ts, completed_at=ts)
        )
        session.commit()


def _complete(store: CheckpointStore, run_id: str, stage: str = "ingest") -> None:
    with PipelineCheckpoint(store, run_id, PAIR, stage) as cp:
        cp.set_result({"row_count": 1})


def test_key_older_than_ttl_is_evicted_and_no_longer_blocks_reprocessing(db_url):
    store = CheckpointStore(db_url=db_url, ttl_hours=TTL_HOURS)
    _complete(store, "old-run")
    _age(store, "old-run", TTL_HOURS + 1)

    assert store.evict_expired() == 1
    assert store.list_stages("old-run", PAIR) == []

    ran = False
    with PipelineCheckpoint(store, "old-run", PAIR, "ingest") as cp:
        assert not cp.skip
        ran = True
    assert ran
    assert store.is_complete("old-run", PAIR, "ingest")


def test_key_within_ttl_is_kept_and_still_blocks_reprocessing(db_url):
    store = CheckpointStore(db_url=db_url, ttl_hours=TTL_HOURS)
    _complete(store, "recent-run")
    _age(store, "recent-run", TTL_HOURS - 1)

    assert store.evict_expired() == 0
    with PipelineCheckpoint(store, "recent-run", PAIR, "ingest") as cp:
        assert cp.skip


def test_stale_running_and_failed_entries_are_evicted(db_url):
    store = CheckpointStore(db_url=db_url, ttl_hours=TTL_HOURS)
    store.mark_started("crashed-run", PAIR, "ingest")  # never completed
    store.mark_failed("failed-run", PAIR, "features", error="boom")
    _age(store, "crashed-run", TTL_HOURS + 1)
    _age(store, "failed-run", TTL_HOURS + 1)

    assert store.evict_expired() == 2
    assert sum(store.size().values()) == 0


def test_eviction_is_enforced_when_the_store_is_opened(db_url):
    store = CheckpointStore(db_url=db_url, ttl_hours=TTL_HOURS)
    _complete(store, "old-run")
    _complete(store, "recent-run")
    _age(store, "old-run", TTL_HOURS + 1)

    reopened = CheckpointStore(db_url=db_url, ttl_hours=TTL_HOURS)

    assert reopened.list_stages("old-run", PAIR) == []
    assert len(reopened.list_stages("recent-run", PAIR)) == 1


def test_eviction_is_enforced_periodically_on_stage_start(db_url, monkeypatch):
    store = CheckpointStore(db_url=db_url, ttl_hours=TTL_HOURS)
    _complete(store, "old-run")
    _age(store, "old-run", TTL_HOURS + 1)

    _complete(store, "next-run")  # within the eviction interval: no sweep yet
    assert len(store.list_stages("old-run", PAIR)) == 1

    monkeypatch.setattr(idempotency, "_EVICTION_INTERVAL_SECONDS", 0.0)
    _complete(store, "another-run")
    assert store.list_stages("old-run", PAIR) == []


def test_store_size_and_evictions_are_exported_as_metrics(db_url):
    evicted_before = REGISTRY.get_sample_value("ledgerlens_idempotency_store_evicted_total") or 0
    store = CheckpointStore(db_url=db_url, ttl_hours=TTL_HOURS)
    for stage in ("ingest", "features", "scoring"):
        _complete(store, "run-a", stage)

    assert store.size()["done"] == 3
    assert (
        REGISTRY.get_sample_value("ledgerlens_idempotency_store_entries", {"status": "done"}) == 3
    )

    _age(store, "run-a", TTL_HOURS + 1)
    store.evict_expired()

    assert (
        REGISTRY.get_sample_value("ledgerlens_idempotency_store_entries", {"status": "done"}) == 0
    )
    assert REGISTRY.get_sample_value("ledgerlens_idempotency_store_evicted_total") == (
        evicted_before + 3
    )
