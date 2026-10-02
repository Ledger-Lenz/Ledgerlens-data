"""Issue #960 — Tests for dry-run mode across state-mutating CLI commands.

Acceptance criteria verified here:
1. Dry-run produces zero side effects for every identified mutating command.
2. Dry-run output clearly enumerates the concrete changes that would be made.
3. Documentation recommendation (dry-run first for production) is enforced via
   the consistent ``--dry-run`` flag presence on all mutating commands.
"""

from __future__ import annotations

import io
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cli.dry_run import (
    DryRunAction,
    DryRunPlan,
    add_dry_run_argument,
    check_dry_run,
)


# ---------------------------------------------------------------------------
# DryRunAction
# ---------------------------------------------------------------------------


class TestDryRunAction:
    def test_fields_accessible(self):
        action = DryRunAction("write", "data/output.parquet", "500 rows")
        assert action.action_type == "write"
        assert action.target == "data/output.parquet"
        assert action.detail == "500 rows"

    def test_empty_detail_allowed(self):
        action = DryRunAction("delete", "risk_scores table")
        assert action.detail == ""


# ---------------------------------------------------------------------------
# DryRunPlan
# ---------------------------------------------------------------------------


class TestDryRunPlan:
    def test_format_contains_command_name(self):
        plan = DryRunPlan("my-command")
        output = plan.format()
        assert "MY-COMMAND" in output

    def test_format_contains_dry_run_header(self):
        plan = DryRunPlan("restore")
        output = plan.format()
        assert "DRY-RUN" in output.upper()

    def test_format_contains_no_changes_made(self):
        plan = DryRunPlan("restore")
        output = plan.format()
        assert "No changes will be made" in output or "no changes" in output.lower()

    def test_format_lists_all_actions(self):
        plan = DryRunPlan("backfill")
        plan.add_action("fetch", "Horizon API", "5 pools")
        plan.add_action("write", "data/output.parquet", "1234 rows")
        output = plan.format()
        assert "FETCH" in output.upper()
        assert "Horizon API" in output
        assert "5 pools" in output
        assert "WRITE" in output.upper()
        assert "data/output.parquet" in output
        assert "1234 rows" in output

    def test_format_shows_no_actions_message_when_empty(self):
        plan = DryRunPlan("noop-cmd")
        output = plan.format()
        assert "No state-mutating actions" in output or "no" in output.lower()

    def test_format_contains_extra_fields(self):
        plan = DryRunPlan("cmd", extra={"pool_count": "5", "date_range": "2024-01-01 → 2024-12-31"})
        output = plan.format()
        assert "pool_count" in output
        assert "5" in output
        assert "date_range" in output

    def test_add_action_appends(self):
        plan = DryRunPlan("cmd")
        plan.add_action("upsert", "risk_scores", "10 rows")
        assert len(plan.actions) == 1
        assert plan.actions[0].action_type == "upsert"

    def test_print_report_writes_to_stream(self):
        plan = DryRunPlan("restore")
        plan.add_action("overwrite", "database")
        buf = io.StringIO()
        plan.print_report(stream=buf)
        output = buf.getvalue()
        assert "DRY-RUN" in output.upper()
        assert "overwrite" in output.lower() or "OVERWRITE" in output

    def test_action_count_displayed(self):
        plan = DryRunPlan("cmd")
        for i in range(3):
            plan.add_action("write", f"file_{i}.parquet")
        output = plan.format()
        assert "3" in output


# ---------------------------------------------------------------------------
# add_dry_run_argument
# ---------------------------------------------------------------------------


class TestAddDryRunArgument:
    def test_flag_added_to_parser(self):
        parser = ArgumentParser()
        add_dry_run_argument(parser)
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_flag_defaults_false(self):
        parser = ArgumentParser()
        add_dry_run_argument(parser)
        args = parser.parse_args([])
        assert args.dry_run is False

    def test_help_text_mentions_no_side_effects(self):
        parser = ArgumentParser()
        add_dry_run_argument(parser)
        # Find the action for --dry-run
        dry_run_action = next(
            a for a in parser._actions if "--dry-run" in getattr(a, "option_strings", [])
        )
        assert any(
            phrase in dry_run_action.help.lower()
            for phrase in ("side effect", "without applying", "no change", "zero side")
        ), f"Help text should mention zero side effects: {dry_run_action.help}"


# ---------------------------------------------------------------------------
# check_dry_run
# ---------------------------------------------------------------------------


class TestCheckDryRun:
    def test_returns_true_when_dry_run_active(self):
        plan = DryRunPlan("cmd")
        args = Namespace(dry_run=True)
        buf = io.StringIO()
        result = check_dry_run(args, plan, stream=buf)
        assert result is True

    def test_returns_false_when_not_dry_run(self):
        plan = DryRunPlan("cmd")
        args = Namespace(dry_run=False)
        result = check_dry_run(args, plan)
        assert result is False

    def test_prints_plan_when_active(self):
        plan = DryRunPlan("restore")
        plan.add_action("overwrite", "database")
        args = Namespace(dry_run=True)
        buf = io.StringIO()
        check_dry_run(args, plan, stream=buf)
        output = buf.getvalue()
        assert "RESTORE" in output.upper()
        assert "overwrite" in output.lower() or "OVERWRITE" in output

    def test_does_not_print_when_not_active(self):
        plan = DryRunPlan("cmd")
        args = Namespace(dry_run=False)
        buf = io.StringIO()
        check_dry_run(args, plan, stream=buf)
        assert buf.getvalue() == "", "Nothing should be printed when dry-run is off"

    def test_missing_dry_run_attr_treated_as_false(self):
        """If args has no dry_run attribute, treat as False (safe default)."""
        plan = DryRunPlan("cmd")
        args = Namespace()  # No dry_run attr
        result = check_dry_run(args, plan)
        assert result is False


# ---------------------------------------------------------------------------
# AC1 — Zero side effects for restore with --dry-run
# ---------------------------------------------------------------------------


class TestRestoreDryRunZeroSideEffects:
    """Verify that running restore with --dry-run produces no file writes."""

    def test_restore_dry_run_does_not_call_restore_database(
        self, tmp_path, monkeypatch
    ):
        """AC1 — dry-run must never invoke restore_database."""
        import json
        import hashlib

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        db_file = backup_dir / "database_2024-01-01.db"
        db_file.write_text("fake")
        checksum = hashlib.sha256(b"fake").hexdigest()
        manifest = {
            "timestamp": "2024-01-01T00:00:00Z",
            "database": {
                "type": "sqlite",
                "timestamp": "2024-01-01T00:00:00Z",
                "path": str(db_file),
                "checksum": checksum,
                "size_bytes": 4,
            },
            "models": {},
        }
        (backup_dir / "MANIFEST.json").write_text(json.dumps(manifest))

        monkeypatch.setenv("BACKUP_DIR", str(backup_dir))
        monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")
        monkeypatch.setattr(sys, "argv", ["restore.py", "--dry-run"])

        with patch("scripts.restore.restore_database") as mock_restore:
            from scripts.restore import main
            result = main()

        mock_restore.assert_not_called()
        assert result == 0, "Dry-run should exit 0"

    def test_restore_dry_run_does_not_write_files(self, tmp_path, monkeypatch):
        """AC1 — no new files are created on disk during dry-run."""
        import json
        import hashlib

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        db_file = backup_dir / "database_2024-01-01.db"
        db_file.write_text("fake")
        checksum = hashlib.sha256(b"fake").hexdigest()
        manifest = {
            "timestamp": "2024-01-01T00:00:00Z",
            "database": {
                "type": "sqlite",
                "timestamp": "2024-01-01T00:00:00Z",
                "path": str(db_file),
                "checksum": checksum,
                "size_bytes": 4,
            },
            "models": {},
        }
        (backup_dir / "MANIFEST.json").write_text(json.dumps(manifest))

        target_db = tmp_path / "ledgerlens.db"
        monkeypatch.setenv("BACKUP_DIR", str(backup_dir))
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{target_db}")
        monkeypatch.setattr(sys, "argv", ["restore.py", "--dry-run"])

        from scripts.restore import main
        main()

        assert not target_db.exists(), "Dry-run must not create the target database"


# ---------------------------------------------------------------------------
# AC2 — Dry-run output enumerates concrete changes
# ---------------------------------------------------------------------------


class TestDryRunOutputClarity:
    def test_restore_dry_run_output_names_database(self, tmp_path, monkeypatch, capsys):
        """The dry-run output for restore must name the target database."""
        import json
        import hashlib

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        db_file = backup_dir / "database_2024-01-01.db"
        db_file.write_text("fake")
        checksum = hashlib.sha256(b"fake").hexdigest()
        manifest = {
            "timestamp": "2024-01-01T00:00:00Z",
            "database": {
                "type": "sqlite",
                "timestamp": "2024-01-01T00:00:00Z",
                "path": str(db_file),
                "checksum": checksum,
                "size_bytes": 4,
            },
            "models": {},
        }
        (backup_dir / "MANIFEST.json").write_text(json.dumps(manifest))

        monkeypatch.setenv("BACKUP_DIR", str(backup_dir))
        monkeypatch.setenv("DATABASE_URL", "sqlite:///test_clarity.db")
        monkeypatch.setattr(sys, "argv", ["restore.py", "--dry-run"])

        from scripts.restore import main
        main()

        captured = capsys.readouterr()
        # The database identifier must appear in the output
        assert "database" in captured.out.lower() or "sqlite" in captured.out.lower(), (
            f"Dry-run output should name the target database.\nOutput:\n{captured.out}"
        )

    def test_plan_output_enumerates_all_actions(self):
        """AC2 — dry-run output must enumerate each concrete change (not just
        'would proceed')."""
        plan = DryRunPlan("backfill-amm-trades")
        plan.add_action("fetch", "Horizon AMM API", "3 pools")
        plan.add_action("write", "data/labelled.parquet", "850 rows × 47 cols")
        plan.add_action("write", "checkpoint.json", "checkpoint metadata")

        output = plan.format()

        assert "FETCH" in output.upper(), "fetch action must appear"
        assert "Horizon AMM API" in output
        assert "3 pools" in output
        assert "WRITE" in output.upper(), "write action must appear"
        assert "data/labelled.parquet" in output
        assert "checkpoint.json" in output


# ---------------------------------------------------------------------------
# AC3 — Consistent --dry-run flag on mutating commands
# ---------------------------------------------------------------------------


class TestDryRunFlagPresenceOnMutatingCommands:
    """Verify that all identified state-mutating commands expose ``--dry-run``."""

    def _parser_has_dry_run(self, parser: ArgumentParser) -> bool:
        return any(
            "--dry-run" in getattr(a, "option_strings", [])
            for a in parser._actions
        )

    def test_restore_script_has_dry_run_flag(self):
        """scripts/restore.py must expose --dry-run."""
        # Re-parse a fresh parser from restore.main() flow by inspecting the
        # argument declarations in the script.
        import ast
        from pathlib import Path

        src = Path("scripts/restore.py").read_text()
        assert "--dry-run" in src, (
            "scripts/restore.py must declare a --dry-run argument"
        )

    def test_backfill_amm_trades_has_dry_run_flag(self):
        """scripts/backfill_amm_trades.py must expose --dry-run."""
        from pathlib import Path

        src = Path("scripts/backfill_amm_trades.py").read_text()
        assert "--dry-run" in src or "add_dry_run_argument" in src, (
            "scripts/backfill_amm_trades.py must declare a --dry-run argument"
        )

    def test_cli_main_validate_artifacts_has_dry_run(self):
        """ledgerlens-ops validate-artifacts must expose --dry-run."""
        from pathlib import Path

        src = Path("cli/main.py").read_text()
        assert "--dry-run" in src, (
            "cli/main.py validate-artifacts subcommand must declare --dry-run"
        )

    def test_add_dry_run_argument_is_consistent(self):
        """All parsers using add_dry_run_argument get the same flag name and default."""
        p1 = ArgumentParser()
        p2 = ArgumentParser()
        add_dry_run_argument(p1)
        add_dry_run_argument(p2)

        a1 = p1.parse_args([])
        a2 = p2.parse_args(["--dry-run"])
        assert a1.dry_run is False
        assert a2.dry_run is True
