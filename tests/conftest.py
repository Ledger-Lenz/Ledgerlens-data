"""Pytest configuration and shared fixtures."""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables for tests
os.environ.setdefault("MODEL_DIR", "./models")
os.environ.setdefault("RISK_SCORE_DB_URL", "sqlite:///:memory:")
os.environ.setdefault("WATCHED_ASSET_PAIRS", "USDC:native,BTC:native,XLM:native")
os.environ.setdefault("BENFORD_WINDOWS_HOURS", "1,4,24,168,720")
os.environ.setdefault("MIN_TRADES_FOR_SCORING", "20")

# Hypothesis configuration for property-based tests (issue #205)
from hypothesis import settings, HealthCheck

# Configure Hypothesis for CI environment
settings.register_profile(
    "ci",
    max_examples=500,           # Reasonable number for CI
    deadline=5000,              # 5 seconds per example
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

settings.register_profile(
    "dev",
    max_examples=50,            # Faster for local development
    deadline=2000,
)

# Select profile based on environment
import os
if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
    settings.load_profile("ci")
else:
    settings.load_profile("dev")
