"""Tests for the `make check-env` Makefile target.

These tests verify that:
1. `make check-env` exits 0 with all required env vars set.
2. `make check-env` exits non-zero with a specific missing env var and produces the expected error message.
"""

import os
import subprocess
import sys


def _run_check_env(env_vars: dict[str, str]) -> tuple[int, str, str]:
    """Run the check-env script with the given environment variables.

    Returns the exit code, stdout, and stderr.
    """
    env = os.environ.copy()
    env.update(env_vars)

    result = subprocess.run(
        [sys.executable, "scripts/check_env.py"],
        capture_output=True,
        text=True,
        env=env,
    )
    return result.returncode, result.stdout, result.stderr


def test_check_env_passes_with_all_required_vars_set(monkeypatch):
    """make check-env exits 0 with all env vars set."""
    monkeypatch.setenv("WATCHED_ASSET_PAIRS", "USDC:GA5Z...")
    monkeypatch.setenv("RISK_SCORE_DB_URL", "sqlite:///test.db")
    monkeypatch.setenv("MODEL_DIR", "./models")

    exit_code, stdout, _ = _run_check_env(
        {
            "WATCHED_ASSET_PAIRS": "USDC:GA5Z...",
            "RISK_SCORE_DB_URL": "sqlite:///test.db",
            "MODEL_DIR": "./models",
        }
    )

    assert exit_code == 0
    assert "Environment validation passed." in stdout
    assert "✓ WATCHED_ASSET_PAIRS is set." in stdout
    assert "✓ RISK_SCORE_DB_URL is set." in stdout
    assert "✓ MODEL_DIR is set." in stdout


def test_check_env_fails_with_missing_watched_asset_pairs(monkeypatch):
    """make check-env exits non-zero when WATCHED_ASSET_PAIRS is missing."""
    monkeypatch.delenv("WATCHED_ASSET_PAIRS", raising=False)

    exit_code, stdout, _ = _run_check_env(
        {
            "WATCHED_ASSET_PAIRS": "",
            "RISK_SCORE_DB_URL": "sqlite:///test.db",
            "MODEL_DIR": "./models",
        }
    )

    assert exit_code == 1
    assert "Missing required environment variables:" in stdout
    assert "✗ WATCHED_ASSET_PAIRS is not set." in stdout


def test_check_env_fails_with_missing_risk_score_db_url(monkeypatch):
    """make check-env exits non-zero when RISK_SCORE_DB_URL is missing."""
    exit_code, stdout, _ = _run_check_env(
        {
            "WATCHED_ASSET_PAIRS": "USDC:GA5Z...",
            "RISK_SCORE_DB_URL": "",
            "MODEL_DIR": "./models",
        }
    )

    assert exit_code == 1
    assert "✗ RISK_SCORE_DB_URL is not set." in stdout


def test_check_env_fails_with_missing_model_dir(monkeypatch):
    """make check-env exits non-zero when MODEL_DIR is missing."""
    exit_code, stdout, _ = _run_check_env(
        {
            "WATCHED_ASSET_PAIRS": "USDC:GA5Z...",
            "RISK_SCORE_DB_URL": "sqlite:///test.db",
            "MODEL_DIR": "",
        }
    )

    assert exit_code == 1
    assert "✗ MODEL_DIR is not set." in stdout


def test_check_env_reports_multiple_missing_vars(monkeypatch):
    """make check-env reports all missing variables in one run."""
    exit_code, stdout, _ = _run_check_env(
        {
            "WATCHED_ASSET_PAIRS": "",
            "RISK_SCORE_DB_URL": "",
            "MODEL_DIR": "",
        }
    )

    assert exit_code == 1
    assert "✗ WATCHED_ASSET_PAIRS is not set." in stdout
    assert "✗ RISK_SCORE_DB_URL is not set." in stdout
    assert "✗ MODEL_DIR is not set." in stdout
    # Verify it's human-readable, not a Python traceback
    assert "Traceback" not in stdout


def test_check_env_handles_whitespace_as_missing(monkeypatch):
    """make check-env treats whitespace-only values as missing."""
    exit_code, stdout, _ = _run_check_env(
        {
            "WATCHED_ASSET_PAIRS": "   ",
            "RISK_SCORE_DB_URL": "sqlite:///test.db",
            "MODEL_DIR": "./models",
        }
    )

    assert exit_code == 1
    assert "✗ WATCHED_ASSET_PAIRS is not set." in stdout
