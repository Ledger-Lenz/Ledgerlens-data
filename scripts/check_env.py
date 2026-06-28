"""Validate required environment variables for the LedgerLens pipeline.

This script is called by `make check-env` and produces clear, human-readable
error messages for any missing configuration variables.
"""

import os
import sys


def check_env() -> int:
    """Check required environment variables and report status.

    Returns:
        0 if all required variables are set
        1 if any required variables are missing
    """
    required_vars = ["WATCHED_ASSET_PAIRS", "RISK_SCORE_DB_URL", "MODEL_DIR"]

    errors = []
    for var in required_vars:
        value = os.getenv(var, "").strip()
        if not value:
            errors.append(f"✗ {var} is not set.")
        else:
            print(f"✓ {var} is set.")

    if errors:
        print()
        print("Missing required environment variables:")
        for error in errors:
            print(error)
        return 1

    print("Environment validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(check_env())
