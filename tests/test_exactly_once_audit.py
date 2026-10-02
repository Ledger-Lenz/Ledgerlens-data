"""End-to-end exactly-once audit tooling (issue #918)."""

import json

import fakeredis
import pytest

from pipeline.exactly_once import (
    DedupDecision,
    DedupKey,
    DedupState,
    ExactlyOnceStore,
    RedisExactlyOnceBackend,
    SqlExactlyOnceBackend,
)
from pipeline.exactly_once_audit import (
    AUDIT_PREFIX,
    Invariant,
    StageBoundary,
    audit_pipeline,
)
from scripts import audit_exactly_once


class FailOpenBackend:
    """Misconfiguration: a dedup backend that never remembers anything."""

    def check_and_stage(self, key: DedupKey, ttl_seconds: float) -> DedupDecision:
        return DedupDecision(DedupState.NEW)

    def commit(self, key: DedupKey, payload=None) -> None:
        pass

    def mark_failed(self, key: DedupKey) -> None:
        pass

    def get(self, key: DedupKey) -> DedupDecision:
        return DedupDecision(DedupState.NEW)

    def health_check(self) -> bool:
        return True


class UnreachableRedis:
    def __getattr__(self, name):
        def _fail(*args, **kwargs):
            raise ConnectionError("connection refused")

        return _fail


def _redis_store(ttl_seconds: float = 86400.0) -> ExactlyOnceStore:
    backend = RedisExactlyOnceBackend(
        "redis://unused", client=fakeredis.FakeRedis(decode_responses=True)
    )
    return ExactlyOnceStore(backend, ttl_seconds=ttl_seconds)


@pytest.fixture
def pipeline_boundaries(tmp_path):
    """The pipeline's three boundaries, correctly configured."""
    sql = SqlExactlyOnceBackend(f"sqlite:///{tmp_path / 'alerts.db'}")
    return [
        StageBoundary("ingestion", _redis_store(), "horizon_trade:audit"),
        StageBoundary("feature_scoring", _redis_store(), "kafka_trade"),
        StageBoundary("alerting", ExactlyOnceStore(sql), "alert_delivery"),
    ]


def test_properly_configured_pipeline_passes_cleanly(pipeline_boundaries):
    report = audit_pipeline(pipeline_boundaries, sample_size=3, run_id="clean")

    assert report.ok, report.violations
    assert report.stages == ["ingestion", "feature_scoring", "alerting"]
    assert report.records == [f"{AUDIT_PREFIX}:clean:{i}" for i in range(3)]


def test_repeated_audits_do_not_interfere(pipeline_boundaries):
    assert audit_pipeline(pipeline_boundaries, sample_size=2).ok
    assert audit_pipeline(pipeline_boundaries, sample_size=2).ok


def test_fail_open_stage_is_flagged_as_at_least_once(pipeline_boundaries):
    pipeline_boundaries[1] = StageBoundary(
        "feature_scoring", ExactlyOnceStore(FailOpenBackend()), "kafka_trade"
    )

    report = audit_pipeline(pipeline_boundaries, sample_size=2)

    assert not report.ok
    assert report.violations_for("ingestion") == []
    assert report.violations_for("alerting") == []
    flagged = {(v.invariant, v.record_id) for v in report.violations_for("feature_scoring")}
    assert flagged == {
        (invariant, record)
        for record in report.records
        for invariant in (
            Invariant.INFLIGHT_REDELIVERY_REDONE,
            Invariant.COMMITTED_REDELIVERY_SKIPPED,
        )
    }


def test_ttl_shorter_than_redelivery_window_is_flagged(pipeline_boundaries):
    pipeline_boundaries[0] = StageBoundary(
        "ingestion", _redis_store(ttl_seconds=60), "horizon_trade:audit", min_ttl_seconds=3600
    )

    report = audit_pipeline(pipeline_boundaries, sample_size=1)

    [violation] = report.violations
    assert violation.stage == "ingestion"
    assert violation.invariant is Invariant.TTL_COVERS_REDELIVERY_WINDOW


def test_unreachable_backend_is_flagged_not_treated_as_new(pipeline_boundaries):
    backend = RedisExactlyOnceBackend("redis://unused", client=UnreachableRedis())
    pipeline_boundaries[0] = StageBoundary(
        "ingestion", ExactlyOnceStore(backend), "horizon_trade:audit"
    )

    report = audit_pipeline(pipeline_boundaries, sample_size=2)

    assert [v.invariant for v in report.violations] == [Invariant.BACKEND_AVAILABLE]
    assert report.violations[0].stage == "ingestion"


def test_sample_size_must_be_positive(pipeline_boundaries):
    with pytest.raises(ValueError):
        audit_pipeline(pipeline_boundaries, sample_size=0)


@pytest.fixture
def fake_staging(monkeypatch):
    server = fakeredis.FakeServer()

    def _backend(url, *, key_prefix):
        client = fakeredis.FakeRedis(server=server, decode_responses=True)
        return RedisExactlyOnceBackend(url, key_prefix=key_prefix, client=client)

    monkeypatch.setattr(audit_exactly_once, "RedisExactlyOnceBackend", _backend)


def test_cli_exits_zero_and_writes_report_for_healthy_staging(fake_staging, tmp_path, capsys):
    output = tmp_path / "report.json"
    code = audit_exactly_once.main(
        ["--db-url", f"sqlite:///{tmp_path / 'db.sqlite'}", "--output", str(output)]
    )

    assert code == 0
    report = json.loads(output.read_text())
    assert report["ok"] is True
    assert report["stages"] == ["ingestion", "feature_scoring", "alerting"]
    assert json.loads(capsys.readouterr().out) == report


def test_cli_exits_nonzero_on_violation(fake_staging, tmp_path, capsys):
    code = audit_exactly_once.main(
        [
            "--db-url",
            f"sqlite:///{tmp_path / 'db.sqlite'}",
            "--min-ttl-seconds",
            str(10 * 365 * 86400),
        ]
    )

    assert code == 1
    report = json.loads(capsys.readouterr().out)
    assert {v["invariant"] for v in report["violations"]} == {"ttl_covers_redelivery_window"}
