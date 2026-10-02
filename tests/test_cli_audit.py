"""Issue #961 — Tests for CLI command execution audit logging.

Acceptance criteria verified here:
1. Audit entry is complete and accurate for a test command execution.
2. Secret redaction works for a command invoked with a sensitive argument.
3. Non-production environments do NOT produce audit entries (no-op).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from cli.audit import (
    ACTOR_ENV_VAR,
    AUDIT_ALL_ENV_VAR,
    LEDGERLENS_ENV_VAR,
    audit_cli_command,
    build_cli_audit_entry,
    cli_audit_hook,
    is_production_env,
    redact_secrets,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _read_entries(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    lines = log_path.read_text().strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# is_production_env
# ---------------------------------------------------------------------------


class TestIsProductionEnv:
    def test_production_env_var_true(self, monkeypatch):
        monkeypatch.setenv(LEDGERLENS_ENV_VAR, "production")
        assert is_production_env() is True

    def test_production_env_var_false(self, monkeypatch):
        monkeypatch.setenv(LEDGERLENS_ENV_VAR, "development")
        assert is_production_env() is False

    def test_audit_all_env_var_overrides(self, monkeypatch):
        monkeypatch.setenv(AUDIT_ALL_ENV_VAR, "1")
        monkeypatch.setenv(LEDGERLENS_ENV_VAR, "staging")
        assert is_production_env() is True

    def test_no_env_vars_is_not_production(self, monkeypatch):
        monkeypatch.delenv(LEDGERLENS_ENV_VAR, raising=False)
        monkeypatch.delenv(AUDIT_ALL_ENV_VAR, raising=False)
        assert is_production_env() is False


# ---------------------------------------------------------------------------
# redact_secrets
# ---------------------------------------------------------------------------


class TestRedactSecrets:
    def test_secret_key_redacted(self):
        result = redact_secrets({"secret": "super_secret_value"})
        assert result["secret"] == "[REDACTED]"

    def test_password_key_redacted(self):
        result = redact_secrets({"password": "hunter2"})
        assert result["password"] == "[REDACTED]"

    def test_token_key_redacted(self):
        result = redact_secrets({"api_token": "abc123"})
        assert result["api_token"] == "[REDACTED]"

    def test_private_key_redacted(self):
        result = redact_secrets({"private_key_path": "/etc/keys/private.pem"})
        assert result["private_key_path"] == "[REDACTED]"

    def test_non_secret_key_preserved(self):
        result = redact_secrets({"backup_dir": "/var/backups", "dry_run": True})
        assert result["backup_dir"] == "/var/backups"
        assert result["dry_run"] is True

    def test_mixed_args_redacted_correctly(self):
        """AC2 — secret redaction verified for a command with sensitive arguments."""
        args = {
            "db_url": "postgresql://user:@host/db",
            "secret": "s3kr3t",
            "model_dir": "./models",
            "api_token": "Bearer abc123",
            "dry_run": False,
        }
        result = redact_secrets(args)
        assert result["secret"] == "[REDACTED]"
        assert result["api_token"] == "[REDACTED]"
        assert result["db_url"] == "postgresql://user:@host/db"
        assert result["model_dir"] == "./models"
        assert result["dry_run"] is False

    def test_case_insensitive_redaction(self):
        """Secret key detection is case-insensitive."""
        result = redact_secrets({"SECRET": "val", "PassWord": "val2", "TOKEN": "val3"})
        assert result["SECRET"] == "[REDACTED]"
        assert result["PassWord"] == "[REDACTED]"
        assert result["TOKEN"] == "[REDACTED]"


# ---------------------------------------------------------------------------
# build_cli_audit_entry
# ---------------------------------------------------------------------------


class TestBuildCliAuditEntry:
    def test_entry_contains_command(self):
        entry = build_cli_audit_entry("restore", {}, "success")
        assert entry["command"] == "restore"

    def test_entry_contains_outcome(self):
        entry = build_cli_audit_entry("restore", {}, "success")
        assert entry["outcome"] == "success"

    def test_entry_contains_event_type(self):
        entry = build_cli_audit_entry("restore", {}, "success")
        assert entry["event_type"] == "cli_command"

    def test_entry_contains_actor(self):
        entry = build_cli_audit_entry("restore", {}, "success", actor="ci-bot")
        assert entry["actor"] == "ci-bot"

    def test_entry_contains_timestamp(self):
        entry = build_cli_audit_entry("restore", {}, "success")
        assert "timestamp" in entry
        assert entry["timestamp"].endswith("Z") or "+" in entry["timestamp"]

    def test_entry_contains_error_on_failure(self):
        entry = build_cli_audit_entry(
            "restore", {}, "failure", error="Connection refused"
        )
        assert entry["outcome"] == "failure"
        assert "Connection refused" in entry["error"]

    def test_entry_secrets_are_redacted(self):
        """AC2 — secrets in args are redacted in the audit entry."""
        entry = build_cli_audit_entry(
            "backfill",
            {"db_password": "s3kr3t", "pool_ids": ["abc"]},
            "success",
        )
        assert entry["args"]["db_password"] == "[REDACTED]"
        assert entry["args"]["pool_ids"] == ["abc"]


# ---------------------------------------------------------------------------
# audit_cli_command context manager
# ---------------------------------------------------------------------------


class TestAuditCliCommandContextManager:
    def test_success_entry_written_in_production(self, tmp_path, monkeypatch):
        """AC1 — A complete, accurate audit entry is written for a test command."""
        log_path = tmp_path / "audit.ndjson"
        monkeypatch.setenv(AUDIT_ALL_ENV_VAR, "1")
        monkeypatch.setenv(ACTOR_ENV_VAR, "test-operator")

        with audit_cli_command(
            "validate-artifacts",
            args={"dir": "./models"},
            log_path=str(log_path),
        ):
            pass  # Simulated command body

        entries = _read_entries(log_path)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["command"] == "validate-artifacts"
        assert entry["outcome"] == "success"
        assert entry["actor"] == "test-operator"
        assert entry["args"]["dir"] == "./models"
        assert "timestamp" in entry

    def test_failure_entry_written_on_exception(self, tmp_path, monkeypatch):
        log_path = tmp_path / "audit.ndjson"
        monkeypatch.setenv(AUDIT_ALL_ENV_VAR, "1")

        with pytest.raises(RuntimeError, match="boom"):
            with audit_cli_command("restore", args={}, log_path=str(log_path)):
                raise RuntimeError("boom")

        entries = _read_entries(log_path)
        assert len(entries) == 1
        assert entries[0]["outcome"] == "failure"
        assert "boom" in entries[0]["error"]

    def test_no_entry_written_outside_production(self, tmp_path, monkeypatch):
        log_path = tmp_path / "audit.ndjson"
        monkeypatch.delenv(LEDGERLENS_ENV_VAR, raising=False)
        monkeypatch.delenv(AUDIT_ALL_ENV_VAR, raising=False)

        with audit_cli_command("restore", args={}, log_path=str(log_path)):
            pass

        assert not log_path.exists(), "No audit file should be created in non-production"

    def test_secret_args_redacted_in_written_entry(self, tmp_path, monkeypatch):
        """AC2 — secret args are redacted in the written entry."""
        log_path = tmp_path / "audit.ndjson"
        monkeypatch.setenv(AUDIT_ALL_ENV_VAR, "1")

        with audit_cli_command(
            "backfill",
            args={"api_token": "Bearer secret123", "since": "2024-01-01"},
            log_path=str(log_path),
        ):
            pass

        entries = _read_entries(log_path)
        assert entries[0]["args"]["api_token"] == "[REDACTED]"
        assert entries[0]["args"]["since"] == "2024-01-01"


# ---------------------------------------------------------------------------
# cli_audit_hook decorator
# ---------------------------------------------------------------------------


class TestCliAuditHookDecorator:
    def test_decorator_writes_entry_on_success(self, tmp_path, monkeypatch):
        log_path = tmp_path / "audit.ndjson"
        monkeypatch.setenv(AUDIT_ALL_ENV_VAR, "1")
        monkeypatch.setenv(ACTOR_ENV_VAR, "ci-runner")

        @cli_audit_hook("my-command", log_path=str(log_path))
        def run(opts):
            return 0

        import argparse

        opts = argparse.Namespace(dry_run=True, verbose=False)
        run(opts)

        entries = _read_entries(log_path)
        assert len(entries) == 1
        assert entries[0]["command"] == "my-command"
        assert entries[0]["outcome"] == "success"

    def test_decorator_writes_failure_on_exception(self, tmp_path, monkeypatch):
        log_path = tmp_path / "audit.ndjson"
        monkeypatch.setenv(AUDIT_ALL_ENV_VAR, "1")

        @cli_audit_hook("failing-command", log_path=str(log_path))
        def run(opts):
            raise ValueError("something went wrong")

        import argparse

        opts = argparse.Namespace()
        with pytest.raises(ValueError, match="something went wrong"):
            run(opts)

        entries = _read_entries(log_path)
        assert entries[0]["outcome"] == "failure"
        assert "something went wrong" in entries[0]["error"]

    def test_decorator_redacts_secrets_in_namespace(self, tmp_path, monkeypatch):
        """Secrets in argparse Namespace attrs are redacted by the decorator."""
        log_path = tmp_path / "audit.ndjson"
        monkeypatch.setenv(AUDIT_ALL_ENV_VAR, "1")

        @cli_audit_hook("secure-cmd", log_path=str(log_path))
        def run(opts):
            return 0

        import argparse

        opts = argparse.Namespace(secret="top_secret", output_dir="./out")
        run(opts)

        entries = _read_entries(log_path)
        assert entries[0]["args"]["secret"] == "[REDACTED]"
        assert entries[0]["args"]["output_dir"] == "./out"


# ---------------------------------------------------------------------------
# Actor detection
# ---------------------------------------------------------------------------


class TestActorDetection:
    def test_explicit_actor_env_var_used(self, monkeypatch):
        monkeypatch.setenv(ACTOR_ENV_VAR, "jane.doe")
        entry = build_cli_audit_entry("cmd", {}, "success")
        assert entry["actor"] == "jane.doe"

    def test_fallback_to_user_env(self, monkeypatch):
        monkeypatch.delenv(ACTOR_ENV_VAR, raising=False)
        monkeypatch.setenv("USER", "john")
        entry = build_cli_audit_entry("cmd", {}, "success")
        assert entry["actor"] == "john"

    def test_unknown_when_no_identity(self, monkeypatch):
        monkeypatch.delenv(ACTOR_ENV_VAR, raising=False)
        monkeypatch.delenv("USER", raising=False)
        monkeypatch.delenv("USERNAME", raising=False)
        entry = build_cli_audit_entry("cmd", {}, "success")
        assert entry["actor"] == "unknown"
