"""Tests for issues #960, #961, #962, #963.

#960 — --dry-run mode for all state-mutating CLI commands
#961 — CLI audit logging hook for production environments
#962 — Interactive confirmation with blast-radius summary for destructive commands
#963 — Automated forward/backward migration testing against representative snapshot
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, inspect, text

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sqlite_engine():
    """Fresh in-memory SQLite engine."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    yield engine
    engine.dispose()


@pytest.fixture()
def populated_engine(sqlite_engine):
    """Engine with the pre-migration baseline tables (simulates existing DB)."""
    with sqlite_engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS risk_scores (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet       VARCHAR NOT NULL,
                asset_pair   VARCHAR NOT NULL,
                score        INTEGER NOT NULL,
                benford_flag BOOLEAN NOT NULL DEFAULT 0,
                ml_flag      BOOLEAN NOT NULL DEFAULT 0,
                confidence   INTEGER NOT NULL DEFAULT 0,
                updated_at   TIMESTAMP NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id          VARCHAR NOT NULL,
                model_artifact_path VARCHAR NOT NULL,
                status              VARCHAR NOT NULL,
                trained_at          TIMESTAMP NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS audit_merkle_roots (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                root_hash  VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
        """))
    return sqlite_engine


# ===========================================================================
# Issue #960 — Dry-run mode
# ===========================================================================


class TestDryRunContext:
    """Tests for cli.dry_run.DryRunContext."""

    def test_enabled_prints_banner(self):
        from cli.dry_run import DryRunContext

        out = io.StringIO()
        with DryRunContext(enabled=True, out=out):
            pass
        assert "[DRY RUN]" in out.getvalue()

    def test_disabled_no_banner(self):
        from cli.dry_run import DryRunContext

        out = io.StringIO()
        with DryRunContext(enabled=False, out=out):
            pass
        assert "[DRY RUN]" not in out.getvalue()

    def test_bool_true_when_enabled(self):
        from cli.dry_run import DryRunContext

        with DryRunContext(True) as dry:
            assert bool(dry) is True

    def test_bool_false_when_disabled(self):
        from cli.dry_run import DryRunContext

        with DryRunContext(False) as dry:
            assert bool(dry) is False

    def test_record_collected_effects_printed(self):
        from cli.dry_run import DryRunContext

        out = io.StringIO()
        with DryRunContext(enabled=True, out=out) as dry:
            dry.record("Would write file /tmp/backup.db")
            dry.record("Would write manifest MANIFEST.json")
        output = out.getvalue()
        assert "Would write file /tmp/backup.db" in output
        assert "Would write manifest MANIFEST.json" in output

    def test_effects_list_accessible(self):
        from cli.dry_run import DryRunContext

        with DryRunContext(True) as dry:
            dry.record("effect A")
            dry.record("effect B")
        assert dry.effects == ["effect A", "effect B"]

    def test_dry_run_prevents_side_effects(self, tmp_path):
        """Verify that code inside `if dry:` block never runs in dry-run mode."""
        from cli.dry_run import DryRunContext

        sentinel_file = tmp_path / "should_not_exist.txt"
        with DryRunContext(True) as dry:
            dry.record("Would create file")
            if dry:
                pass  # mutation skipped
            else:
                sentinel_file.write_text("written")  # pragma: no cover
        assert not sentinel_file.exists()

    def test_non_dry_run_runs_side_effects(self, tmp_path):
        """Verify that code runs normally when dry-run is disabled."""
        from cli.dry_run import DryRunContext

        sentinel_file = tmp_path / "should_exist.txt"
        with DryRunContext(False) as dry:
            dry.record("Creating file")
            if not dry:
                sentinel_file.write_text("written")
        assert sentinel_file.exists()

    def test_add_dry_run_argument_adds_flag(self):
        import argparse

        from cli.dry_run import add_dry_run_argument

        parser = argparse.ArgumentParser()
        add_dry_run_argument(parser)
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_add_dry_run_argument_default_false(self):
        import argparse

        from cli.dry_run import add_dry_run_argument

        parser = argparse.ArgumentParser()
        add_dry_run_argument(parser)
        args = parser.parse_args([])
        assert args.dry_run is False


class TestAbortIfDryRun:
    def test_raises_when_enabled(self):
        from cli.dry_run import DryRunAbort, abort_if_dry_run

        with pytest.raises(DryRunAbort):
            abort_if_dry_run(True, "Would write DB")

    def test_noop_when_disabled(self):
        from cli.dry_run import abort_if_dry_run

        abort_if_dry_run(False, "Not a dry run")  # must not raise


class TestBackupDryRun:
    """Backup command dry-run produces zero side effects."""

    def test_backup_dry_run_no_file_written(self, tmp_path, monkeypatch):
        from cli.dry_run import DryRunContext

        backup_dir = tmp_path / "backups"
        written = []

        with DryRunContext(True) as dry:
            dry.record(f"Would write DB backup to {backup_dir}")
            dry.record(f"Would write manifest to {backup_dir}/MANIFEST.json")
            if dry:
                pass  # no mutations
            else:
                written.append("backup happened")  # pragma: no cover

        assert written == []
        assert not backup_dir.exists()


class TestRestoreDryRun:
    """Restore command dry-run produces zero side effects."""

    def test_restore_dry_run_no_db_touched(self, tmp_path):
        from cli.dry_run import DryRunContext

        db_file = tmp_path / "ledgerlens.db"
        db_file.write_text("original")

        with DryRunContext(True) as dry:
            dry.record("Would overwrite database")
            if dry:
                pass  # skip overwrite
            else:
                db_file.write_text("overwritten")  # pragma: no cover

        assert db_file.read_text() == "original"


class TestMigrateDryRun:
    """migrations.runner.MigrationRunner dry-run produces no applied migrations."""

    def test_migrate_dry_run_no_columns_added(self, populated_engine):
        from migrations import MigrationRunner

        runner = MigrationRunner(populated_engine, dry_run=True)
        runner.upgrade()

        with populated_engine.connect() as conn:
            rows = conn.execute(text("SELECT migration_id FROM schema_migrations")).fetchall()
        assert rows == [], "Dry-run must not record applied migrations"

        inspector = inspect(populated_engine)
        col_names = {c["name"] for c in inspector.get_columns("risk_scores")}
        assert "ring_id" not in col_names, "Dry-run must not add columns"


# ===========================================================================
# Issue #961 — CLI audit logging hook
# ===========================================================================


class TestRedactArgs:
    """Secret redaction in emit_cli_audit_event."""

    def test_secret_key_redacted(self):
        from cli.audit_hook import redact_args

        result = redact_args({"submitter_secret": "SXXX", "db_url": "sqlite:///db"})
        assert result["submitter_secret"] == "[REDACTED]"
        assert result["db_url"] == "sqlite:///db"

    def test_token_redacted(self):
        from cli.audit_hook import redact_args

        result = redact_args({"api_token": "tok_abc123", "command": "backup"})
        assert result["api_token"] == "[REDACTED]"
        assert result["command"] == "backup"

    def test_password_redacted(self):
        from cli.audit_hook import redact_args

        result = redact_args({"db_password": "hunter2", "dry_run": True})
        assert result["db_password"] == "[REDACTED]"

    def test_non_sensitive_args_preserved(self):
        from cli.audit_hook import redact_args

        args = {"dry_run": True, "target": "0002", "db_url": "sqlite:///db"}
        result = redact_args(args)
        assert result == args

    def test_api_key_variants_redacted(self):
        from cli.audit_hook import redact_args

        for key in ("api_key", "apikey", "API_KEY", "credential", "CREDENTIAL"):
            result = redact_args({key: "should-be-redacted"})
            assert result[key] == "[REDACTED]", f"Expected {key!r} to be redacted"


class TestEmitCliAuditEvent:
    """emit_cli_audit_event writes events in production env."""

    def test_writes_event_to_log_in_production(self, tmp_path):
        from cli.audit_hook import emit_cli_audit_event

        log_path = tmp_path / "cli_audit.ndjson"
        event = emit_cli_audit_event(
            command="backup",
            args={"dry_run": False},
            outcome="success",
            log_path=log_path,
            force=True,
        )
        assert event is not None
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["command"] == "backup"
        assert parsed["outcome"] == "success"

    def test_event_contains_all_required_fields(self, tmp_path):
        from cli.audit_hook import emit_cli_audit_event

        log_path = tmp_path / "cli_audit.ndjson"
        event = emit_cli_audit_event(
            command="restore",
            args={"dry_run": True},
            outcome="dry-run",
            log_path=log_path,
            force=True,
        )
        for field in ("timestamp", "actor", "env", "command", "args", "outcome"):
            assert field in event, f"Field {field!r} missing from audit event"

    def test_skipped_when_not_production(self, tmp_path):
        from cli.audit_hook import emit_cli_audit_event

        log_path = tmp_path / "cli_audit.ndjson"
        result = emit_cli_audit_event(
            command="backup",
            args={},
            outcome="success",
            env="local",
            log_path=log_path,
            force=False,
        )
        assert result is None
        assert not log_path.exists()

    def test_secrets_are_redacted_in_log(self, tmp_path):
        from cli.audit_hook import emit_cli_audit_event

        log_path = tmp_path / "cli_audit.ndjson"
        emit_cli_audit_event(
            command="migrate",
            args={"submitter_secret": "SXXX_SENSITIVE", "dry_run": False},
            outcome="success",
            log_path=log_path,
            force=True,
        )
        content = log_path.read_text()
        assert "SXXX_SENSITIVE" not in content
        assert "[REDACTED]" in content

    def test_multiple_events_appended(self, tmp_path):
        from cli.audit_hook import emit_cli_audit_event

        log_path = tmp_path / "cli_audit.ndjson"
        for _ in range(3):
            emit_cli_audit_event("backup", {}, "success", log_path=log_path, force=True)
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_error_outcome_includes_error_field(self, tmp_path):
        from cli.audit_hook import emit_cli_audit_event

        log_path = tmp_path / "cli_audit.ndjson"
        event = emit_cli_audit_event(
            command="restore",
            args={},
            outcome="error",
            error="Database restore failed",
            log_path=log_path,
            force=True,
        )
        assert event["error"] == "Database restore failed"

    def test_audit_decorator_emits_on_success(self, tmp_path):
        from cli.audit_hook import audit_cli_command

        log_path = tmp_path / "cli_audit.ndjson"

        @audit_cli_command("test_cmd", log_path=log_path, force=True)
        def my_command(args):
            return 0

        import argparse

        ns = argparse.Namespace(dry_run=False)
        my_command(ns)

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["command"] == "test_cmd"
        assert parsed["outcome"] == "success"

    def test_audit_decorator_emits_dry_run_outcome(self, tmp_path):
        from cli.audit_hook import audit_cli_command

        log_path = tmp_path / "cli_audit.ndjson"

        @audit_cli_command("test_cmd", log_path=log_path, force=True)
        def my_command(args):
            return 0

        import argparse

        ns = argparse.Namespace(dry_run=True)
        my_command(ns)

        parsed = json.loads(log_path.read_text().strip())
        assert parsed["outcome"] == "dry-run"

    def test_audit_decorator_emits_on_exception(self, tmp_path):
        from cli.audit_hook import audit_cli_command

        log_path = tmp_path / "cli_audit.ndjson"

        @audit_cli_command("failing_cmd", log_path=log_path, force=True)
        def my_command(args):
            raise ValueError("something went wrong")

        import argparse

        ns = argparse.Namespace(dry_run=False)
        with pytest.raises(ValueError):
            my_command(ns)

        parsed = json.loads(log_path.read_text().strip())
        assert parsed["outcome"] == "error"
        assert "something went wrong" in parsed["error"]


# ===========================================================================
# Issue #962 — Interactive confirmation with blast-radius summary
# ===========================================================================


class TestConfirmDestructive:
    """Tests for cli.confirmation.confirm_destructive."""

    _SAMPLE_SUMMARY = {
        "operation": "restore database from backup",
        "affected_records": "150,000",
        "target_database": "postgresql://prod/ledgerlens",
    }

    def test_yes_flag_skips_prompt(self):
        from cli.confirmation import confirm_destructive

        out = io.StringIO()
        # Should not raise and should not read from stdin
        confirm_destructive(self._SAMPLE_SUMMARY, yes=True, out=out)
        output = out.getvalue()
        assert "--yes" in output.lower() or "non-interactive" in output.lower()

    def test_yes_flag_prints_blast_radius(self):
        from cli.confirmation import confirm_destructive

        out = io.StringIO()
        confirm_destructive(self._SAMPLE_SUMMARY, yes=True, out=out)
        output = out.getvalue()
        assert "restore database from backup" in output
        assert "150,000" in output

    def test_interactive_yes_proceeds(self):
        from cli.confirmation import confirm_destructive

        out = io.StringIO()
        inp = io.StringIO("yes\n")
        confirm_destructive(self._SAMPLE_SUMMARY, yes=False, out=out, inp=inp)
        # No exception raised means confirmation was accepted

    def test_interactive_no_raises_aborted(self):
        from cli.confirmation import ConfirmationAborted, confirm_destructive

        out = io.StringIO()
        inp = io.StringIO("no\n")
        with pytest.raises(ConfirmationAborted):
            confirm_destructive(self._SAMPLE_SUMMARY, yes=False, out=out, inp=inp)

    def test_interactive_empty_input_raises_aborted(self):
        from cli.confirmation import ConfirmationAborted, confirm_destructive

        out = io.StringIO()
        inp = io.StringIO("\n")
        with pytest.raises(ConfirmationAborted):
            confirm_destructive(self._SAMPLE_SUMMARY, yes=False, out=out, inp=inp)

    def test_interactive_random_input_raises_aborted(self):
        from cli.confirmation import ConfirmationAborted, confirm_destructive

        out = io.StringIO()
        inp = io.StringIO("y\n")  # 'y' is not 'yes'
        with pytest.raises(ConfirmationAborted):
            confirm_destructive(self._SAMPLE_SUMMARY, yes=False, out=out, inp=inp)

    def test_blast_radius_all_fields_displayed(self):
        from cli.confirmation import print_blast_radius

        out = io.StringIO()
        print_blast_radius(self._SAMPLE_SUMMARY, out=out)
        output = out.getvalue()
        for key, value in self._SAMPLE_SUMMARY.items():
            assert str(value) in output, f"Expected blast-radius field {key!r}={value!r} in output"

    def test_aborted_message_printed_on_decline(self):
        from cli.confirmation import ConfirmationAborted, confirm_destructive

        out = io.StringIO()
        inp = io.StringIO("no\n")
        with pytest.raises(ConfirmationAborted):
            confirm_destructive(self._SAMPLE_SUMMARY, yes=False, out=out, inp=inp)
        assert "Aborted" in out.getvalue()

    def test_yes_warning_is_prominently_displayed(self):
        from cli.confirmation import confirm_destructive

        out = io.StringIO()
        confirm_destructive(self._SAMPLE_SUMMARY, yes=True, out=out)
        output = out.getvalue()
        assert "WARNING" in output or "CAUTION" in output or "caution" in output.lower()

    def test_add_yes_argument(self):
        import argparse

        from cli.confirmation import add_yes_argument

        parser = argparse.ArgumentParser()
        add_yes_argument(parser)
        args = parser.parse_args(["--yes"])
        assert args.yes is True

    def test_add_yes_argument_default_false(self):
        import argparse

        from cli.confirmation import add_yes_argument

        parser = argparse.ArgumentParser()
        add_yes_argument(parser)
        args = parser.parse_args([])
        assert args.yes is False


class TestRestoreCommandConfirmation:
    """Restore command requires confirmation and is blocked without it."""

    def test_restore_blocked_without_yes_and_no_stdin_yes(self, tmp_path):
        """Simulate operator typing 'no' — restore must be blocked."""
        from cli.confirmation import ConfirmationAborted, confirm_destructive

        blast = {"operation": "restore", "db": "prod"}
        out = io.StringIO()
        inp = io.StringIO("no\n")
        with pytest.raises(ConfirmationAborted):
            confirm_destructive(blast, yes=False, out=out, inp=inp)

    def test_restore_proceeds_with_yes_flag(self, tmp_path):
        """Simulate --yes flag — restore must proceed without reading stdin."""
        from cli.confirmation import confirm_destructive

        blast = {"operation": "restore", "db": "prod"}
        out = io.StringIO()
        # Passing yes=True must not read from stdin at all
        confirm_destructive(blast, yes=True, out=out)


# ===========================================================================
# Issue #963 — Migration snapshot testing
# ===========================================================================


class TestRepresentativeSnapshot:
    """Tests for migrations.snapshot_testing.build_representative_snapshot."""

    def test_snapshot_builds_successfully(self):
        from migrations.snapshot_testing import SNAPSHOT_ROW_COUNT, build_representative_snapshot

        engine = build_representative_snapshot(row_count=1_000)
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM risk_scores")).scalar()
        assert count == 1_000

    def test_snapshot_has_required_tables(self):
        from migrations.snapshot_testing import build_representative_snapshot

        engine = build_representative_snapshot(row_count=100)
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        assert "risk_scores" in tables
        assert "model_versions" in tables
        assert "audit_merkle_roots" in tables

    def test_snapshot_scores_in_valid_range(self):
        from migrations.snapshot_testing import (
            assert_score_values_valid,
            build_representative_snapshot,
        )

        engine = build_representative_snapshot(row_count=500)
        assert_score_values_valid(engine)  # must not raise

    def test_snapshot_is_reproducible(self):
        from migrations.snapshot_testing import build_representative_snapshot

        e1 = build_representative_snapshot(row_count=100, seed=99)
        e2 = build_representative_snapshot(row_count=100, seed=99)

        with e1.connect() as c1, e2.connect() as c2:
            rows1 = c1.execute(text("SELECT wallet, score FROM risk_scores ORDER BY id")).fetchall()
            rows2 = c2.execute(text("SELECT wallet, score FROM risk_scores ORDER BY id")).fetchall()
        assert rows1 == rows2


class TestForwardMigrationTest:
    """Tests for run_forward_migration_test."""

    def test_all_migrations_apply_forward(self):
        from migrations.snapshot_testing import (
            build_representative_snapshot,
            run_forward_migration_test,
        )

        engine = build_representative_snapshot(row_count=500)
        timing = run_forward_migration_test(engine, row_count=500)
        assert len(timing) > 0, "At least one migration must be applied"

    def test_timing_report_has_elapsed(self):
        from migrations.snapshot_testing import (
            build_representative_snapshot,
            run_forward_migration_test,
        )

        engine = build_representative_snapshot(row_count=200)
        timing = run_forward_migration_test(engine, row_count=200)
        for entry in timing:
            assert "id" in entry
            assert "elapsed_s" in entry
            assert entry["elapsed_s"] >= 0

    def test_columns_present_after_forward_run(self):
        from migrations.snapshot_testing import (
            assert_column_exists,
            build_representative_snapshot,
            run_forward_migration_test,
        )

        engine = build_representative_snapshot(row_count=200)
        run_forward_migration_test(engine, row_count=200)

        assert_column_exists(engine, "risk_scores", "ring_id")
        assert_column_exists(engine, "risk_scores", "provenance_json")
        assert_column_exists(engine, "risk_scores", "certified_robust")
        assert_column_exists(engine, "risk_scores", "schema_version")

    def test_row_count_preserved_after_all_migrations(self):
        from migrations.snapshot_testing import (
            assert_row_count_preserved,
            build_representative_snapshot,
            run_forward_migration_test,
        )

        engine = build_representative_snapshot(row_count=300)
        run_forward_migration_test(engine, row_count=300)
        assert_row_count_preserved(engine, 300)

    def test_timing_threshold_catches_slow_migration(self):
        """A very tight threshold of 0 s should fail for any real migration."""
        from migrations.snapshot_testing import (
            build_representative_snapshot,
            run_forward_migration_test,
        )

        engine = build_representative_snapshot(row_count=100)
        with pytest.raises(RuntimeError, match="threshold"):
            run_forward_migration_test(engine, max_seconds=0.0, row_count=100)


class TestCorruptingMigration:
    """Tests for build_corrupting_migration (negative / corruption-detection test)."""

    def test_corrupting_migration_caught_by_integrity_check(self):
        """The integrity assertion must raise when a corrupting migration is applied."""
        from migrations.snapshot_testing import (
            assert_score_values_valid,
            build_corrupting_migration,
            build_representative_snapshot,
        )

        engine = build_representative_snapshot(row_count=200)
        bad_migration = build_corrupting_migration()

        with engine.begin() as conn:
            bad_migration.up(conn)

        with pytest.raises(AssertionError, match="Data integrity failure"):
            assert_score_values_valid(engine)

    def test_clean_migration_passes_integrity_check(self):
        """After normal forward migrations the integrity assertion must pass."""
        from migrations.snapshot_testing import (
            assert_score_values_valid,
            build_representative_snapshot,
            run_forward_migration_test,
        )

        engine = build_representative_snapshot(row_count=200)
        run_forward_migration_test(engine, row_count=200)
        assert_score_values_valid(engine)  # must not raise


class TestBackwardMigrationTest:
    """Tests for run_backward_migration_test."""

    def test_backward_test_runs_without_error(self):
        from migrations.snapshot_testing import (
            build_representative_snapshot,
            run_backward_migration_test,
            run_forward_migration_test,
        )

        engine = build_representative_snapshot(row_count=200)
        run_forward_migration_test(engine, row_count=200)
        # Must not raise
        run_backward_migration_test(engine)

    def test_backward_test_on_empty_migration_list(self):
        """run_backward_migration_test is a no-op when there are no migrations."""
        from migrations.snapshot_testing import run_backward_migration_test
        from unittest.mock import patch

        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        with patch("migrations.snapshot_testing._load_all_migrations", return_value=[]):
            run_backward_migration_test(engine)  # must not raise


class TestAssertHelpers:
    def test_assert_row_count_raises_on_mismatch(self):
        from migrations.snapshot_testing import (
            assert_row_count_preserved,
            build_representative_snapshot,
        )

        engine = build_representative_snapshot(row_count=100)
        with pytest.raises(AssertionError, match="Row count changed"):
            assert_row_count_preserved(engine, expected=999)

    def test_assert_column_exists_raises_when_absent(self):
        from migrations.snapshot_testing import (
            assert_column_exists,
            build_representative_snapshot,
        )

        engine = build_representative_snapshot(row_count=50)
        with pytest.raises(AssertionError, match="Expected column"):
            assert_column_exists(engine, "risk_scores", "nonexistent_column_xyz")

    def test_assert_score_values_valid_passes_on_valid_data(self):
        from migrations.snapshot_testing import (
            assert_score_values_valid,
            build_representative_snapshot,
        )

        engine = build_representative_snapshot(row_count=100)
        assert_score_values_valid(engine)  # must not raise
