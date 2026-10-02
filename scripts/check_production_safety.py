#!/usr/bin/env python3
"""Pre-deploy gate: verify all debug/unsafe feature flags are disabled (Issue #955).

Exits with code 0 if the production configuration is safe.
Exits with code 1 if any unsafe flag is enabled.

Usage::

    # Check the current runtime config (reads env vars)
    python scripts/check_production_safety.py

    # Check against a specific environment file
    python scripts/check_production_safety.py --env .env.production

    # Output JSON results (useful in CI pipelines)
    python scripts/check_production_safety.py --json

    # Non-zero exit on violations (default); suppress with --no-fail
    python scripts/check_production_safety.py --no-fail

This script is designed to be run as the first step of any production
deployment pipeline.  If it exits non-zero, the deployment must be aborted
until the flagged configuration is corrected.

What is checked
---------------
Every flag in ``config.deployment_modes.PRODUCTION_UNSAFE_FLAGS`` is inspected
against ``config.Config`` (class-level defaults + env var overrides loaded at
import time).  The canonical list of unsafe flags is defined in
``config/deployment_modes.py`` — add new flags there, not here.

Exit codes
----------
0  All flags are safe.
1  One or more unsafe flags are enabled; violations printed to stderr.
2  Configuration could not be loaded (import error, missing dependency).
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _load_env_file(path: str) -> None:
    """Load KEY=VALUE pairs from *path* into ``os.environ`` (simple dotenv)."""
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
    except FileNotFoundError:
        print(f"ERROR: env file {path!r} not found", file=sys.stderr)
        sys.exit(2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify all debug/unsafe feature flags are disabled before production deploy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--env",
        metavar="PATH",
        default=None,
        help="Load environment variables from this file before checking (optional).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit results as JSON to stdout instead of human-readable text.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        dest="no_fail",
        help="Always exit 0 even when violations are found (CI warning mode).",
    )
    args = parser.parse_args()

    if args.env:
        _load_env_file(args.env)

    try:
        from config.deployment_modes import (
            ProductionSafetyChecker,
            ProductionSafetyError,
            PRODUCTION_UNSAFE_FLAGS,
        )
        from config import Config
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to import config: {exc}", file=sys.stderr)
        sys.exit(2)

    checker = ProductionSafetyChecker()

    try:
        checker.verify_production_safe(Config)
        violations = []
    except ProductionSafetyError as e:
        violations = e.violations

    if args.json_output:
        result = {
            "safe": len(violations) == 0,
            "violations": [
                {
                    "flag": v.attr,
                    "current_value": repr(v.current_value),
                    "description": v.description,
                }
                for v in violations
            ],
            "flags_checked": len(PRODUCTION_UNSAFE_FLAGS),
        }
        json.dump(result, sys.stdout, indent=2)
        print()
    else:
        if violations:
            print(
                f"PRODUCTION SAFETY CHECK FAILED — {len(violations)} unsafe flag(s) detected:\n",
                file=sys.stderr,
            )
            for v in violations:
                print(f"  ✗ {v.attr} = {v.current_value!r}", file=sys.stderr)
                # Indent the description
                for line in v.description.splitlines():
                    print(f"      {line}", file=sys.stderr)
                print(file=sys.stderr)
            print(
                "Disable the flags above before deploying to production.",
                file=sys.stderr,
            )
        else:
            print(
                f"Production safety check PASSED — "
                f"{len(PRODUCTION_UNSAFE_FLAGS)} flag(s) verified safe."
            )

    if violations and not args.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
