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
