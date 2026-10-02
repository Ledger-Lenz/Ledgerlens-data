"""Issue #962 — Tests for interactive confirmation and blast-radius summary.

Acceptance criteria verified here:
1. Confirmation flow blocks execution without 'yes' answer and proceeds with it.
2. Blast-radius summary is accurate against a test scenario with known scope.
3. Non-interactive override is documented and its warning is surfaced.
"""

from __future__ import annotations

import io
import os
from unittest.mock import patch

import pytest

from cli.confirmation import (
    NON_INTERACTIVE_ENV_VAR,
    BlastRadiusSummary,
    confirm_destructive_action,
)


# ---------------------------------------------------------------------------
# BlastRadiusSummary formatting
# ---------------------------------------------------------------------------


class TestBlastRadiusSummary:
    def test_format_contains_operation_name(self):
        summary = BlastRadiusSummary(operation="Database restore")
        output = summary.format()
        assert "Database restore" in output

    def test_format_contains_record_counts(self):
        summary = BlastRadiusSummary(
            operation="Backfill",
            affected_records={"risk_scores": 50_000, "model_versions": 120},
        )
        output = summary.format()
        assert "risk_scores" in output
        assert "50,000" in output
        assert "model_versions" in output
        assert "120" in output

    def test_format_contains_date_range(self):
        summary = BlastRadiusSummary(
            operation="Restore",
            affected_date_range=("2024-01-01", "2024-06-30"),
        )
        output = summary.format()
        assert "2024-01-01" in output
        assert "2024-06-30" in output

    def test_format_contains_tenants(self):
        summary = BlastRadiusSummary(
            operation="Restore",
            affected_tenants=["production", "staging"],
        )
        output = summary.format()
        assert "production" in output
        assert "staging" in output

    def test_format_contains_extra_fields(self):
        summary = BlastRadiusSummary(
            operation="Restore",
            extra={"backup_timestamp": "2024-06-01T12:00:00Z"},
        )
        output = summary.format()
        assert "backup_timestamp" in output
        assert "2024-06-01T12:00:00Z" in output

    def test_format_contains_destructive_warning_header(self):
        summary = BlastRadiusSummary(operation="Restore")
        output = summary.format()
        assert "DESTRUCTIVE" in output.upper()

    def test_blast_radius_summary_accuracy_known_scope(self):
        """AC2 — blast-radius summary is accurate against a known-scope scenario."""
        summary = BlastRadiusSummary(
            operation="Database restore",
            affected_records={"risk_scores": 75_000, "audit_trail": 12_500},
            affected_date_range=("2024-01-01", "2024-12-31"),
            affected_tenants=["prod"],
            extra={"backup_timestamp": "2024-07-15T08:30:00Z"},
        )
        output = summary.format()

        # All fields must appear verbatim in the output
        assert "75,000" in output, "risk_scores row count must appear"
        assert "12,500" in output, "audit_trail row count must appear"
        assert "2024-01-01" in output, "date range start must appear"
        assert "2024-12-31" in output, "date range end must appear"
        assert "prod" in output, "tenant must appear"
        assert "2024-07-15T08:30:00Z" in output, "backup timestamp must appear"


# ---------------------------------------------------------------------------
# confirm_destructive_action — interactive path
# ---------------------------------------------------------------------------


class TestConfirmDestructiveActionInteractive:
    def _run(self, user_input: str) -> tuple[bool, str]:
        """Run confirm_destructive_action with the given user input.

        Returns (result, stderr_output).
        """
        summary = BlastRadiusSummary(
            operation="Test operation",
            affected_records={"risk_scores": 100},
        )
        out = io.StringIO()
        inp = io.StringIO(user_input + "\n")
        result = confirm_destructive_action(summary, prompt_stream=out, input_stream=inp)
        return result, out.getvalue()

    def test_yes_confirms(self):
        """AC1 — typing 'yes' must allow execution to proceed."""
        result, _ = self._run("yes")
        assert result is True

    def test_no_cancels(self):
        """AC1 — typing anything other than 'yes' must block execution."""
        result, _ = self._run("no")
        assert result is False

    def test_empty_input_cancels(self):
        result, _ = self._run("")
        assert result is False

    def test_uppercase_no_cancels(self):
        result, _ = self._run("YES")
        assert result is False, "'YES' in uppercase must not be accepted — must be exact 'yes'"

    def test_summary_printed_before_prompt(self):
        """The blast-radius summary must be printed to the output stream."""
        summary = BlastRadiusSummary(
            operation="Critical delete",
            affected_records={"risk_scores": 999},
        )
        out = io.StringIO()
        inp = io.StringIO("no\n")
        confirm_destructive_action(summary, prompt_stream=out, input_stream=inp)
        output = out.getvalue()
        assert "Critical delete" in output
        assert "999" in output

    def test_cancel_message_printed_on_no(self):
        _, output = self._run("no")
        assert "Cancelled" in output or "cancelled" in output.lower()

    def test_eof_cancels(self):
        """EOFError (piped empty input) must result in cancellation."""
        summary = BlastRadiusSummary(operation="Test")
        out = io.StringIO()
        inp = io.StringIO("")  # EOF immediately
        result = confirm_destructive_action(summary, prompt_stream=out, input_stream=inp)
        assert result is False


# ---------------------------------------------------------------------------
# confirm_destructive_action — non-interactive override
# ---------------------------------------------------------------------------


class TestConfirmDestructiveActionNonInteractive:
    def test_non_interactive_flag_proceeds_without_prompt(self):
        """AC3 — --yes / non_interactive=True must bypass the prompt."""
        summary = BlastRadiusSummary(
            operation="Automated restore",
            affected_records={"risk_scores": 50_000},
        )
        out = io.StringIO()
        # No input stream needed — should never read from it
        inp = io.StringIO()
        result = confirm_destructive_action(summary, non_interactive=True, prompt_stream=out, input_stream=inp)
        assert result is True

    def test_non_interactive_prints_warning(self):
        """AC3 — the non-interactive bypass must surface a warning."""
        summary = BlastRadiusSummary(operation="Automated restore")
        out = io.StringIO()
        confirm_destructive_action(
            summary, non_interactive=True, prompt_stream=out, input_stream=io.StringIO()
        )
        output = out.getvalue()
        assert any(
            w in output.lower()
            for w in ("caution", "warning", "bypasses", "non-interactive", "skipping")
        ), f"Expected caution language in output, got:\n{output}"

    def test_env_var_override_proceeds(self):
        """Setting LEDGERLENS_YES=1 in the environment must act as --yes."""
        summary = BlastRadiusSummary(operation="CI restore")
        out = io.StringIO()
        with patch.dict(os.environ, {NON_INTERACTIVE_ENV_VAR: "1"}):
            result = confirm_destructive_action(
                summary, non_interactive=False, prompt_stream=out, input_stream=io.StringIO()
            )
        assert result is True

    def test_env_var_true_value_proceeds(self):
        summary = BlastRadiusSummary(operation="CI restore")
        out = io.StringIO()
        with patch.dict(os.environ, {NON_INTERACTIVE_ENV_VAR: "true"}):
            result = confirm_destructive_action(
                summary, non_interactive=False, prompt_stream=out, input_stream=io.StringIO()
            )
        assert result is True

    def test_env_var_unset_still_requires_confirmation(self):
        """Without the env var, the prompt must still be shown."""
        summary = BlastRadiusSummary(operation="Manual restore")
        out = io.StringIO()
        inp = io.StringIO("no\n")
        with patch.dict(os.environ, {}, clear=False):
            # Ensure the env var is NOT set
            os.environ.pop(NON_INTERACTIVE_ENV_VAR, None)
            result = confirm_destructive_action(
                summary, non_interactive=False, prompt_stream=out, input_stream=inp
            )
        assert result is False, "Without --yes/env-var, a 'no' answer must cancel"


# ---------------------------------------------------------------------------
# Integration: restore.py honours confirmation
# ---------------------------------------------------------------------------


class TestRestoreConfirmationIntegration:
    """Verify that scripts/restore.py calls confirm_destructive_action before
    any writes, and that a 'no' answer prevents the restore from proceeding."""

    def test_restore_cancelled_on_no(self, tmp_path, monkeypatch):
        """When the operator declines, restore_database must never be called."""
        import sys

        from unittest.mock import MagicMock, patch

        # Create a minimal manifest so the script can load it
        import json

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        db_file = backup_dir / "database_2024-01-01.db"
        db_file.write_text("fake")

        import hashlib

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
        monkeypatch.setenv("DATABASE_URL", "sqlite:///test_restore.db")

        # Patch confirm to return False (operator said no)
        with patch("cli.confirmation.confirm_destructive_action", return_value=False) as mock_confirm, \
             patch("scripts.restore.restore_database") as mock_restore:
            monkeypatch.setattr(sys, "argv", ["restore.py"])
            from scripts.restore import main
            result = main()

        # Restore must not have been called
        mock_restore.assert_not_called()
        assert result == 1, "Should exit with code 1 when cancelled"

    def test_restore_proceeds_on_yes(self, tmp_path, monkeypatch):
        """When the operator confirms, the restore must proceed."""
        import sys
        import json
        import hashlib
        from unittest.mock import patch

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
        monkeypatch.setenv("DATABASE_URL", "sqlite:///test_restore.db")

        # Patch confirm to return True (operator said yes)
        with patch("cli.confirmation.confirm_destructive_action", return_value=True), \
             patch("scripts.restore.restore_database", return_value=True) as mock_restore:
            monkeypatch.setattr(sys, "argv", ["restore.py"])
            from scripts import restore as restore_mod
            # Reload to pick up monkeypatched env
            result = restore_mod.main()

        mock_restore.assert_called_once()
        assert result == 0
