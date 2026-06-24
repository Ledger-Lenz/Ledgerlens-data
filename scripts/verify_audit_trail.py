#!/usr/bin/env python3
"""Verify signed forensic audit trail entries."""

from __future__ import annotations

import argparse
import sys

from config import config
from detection.audit_trail import verify_audit_log
from utils.logging import get_logger

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify forensic audit trail signatures")
    parser.add_argument(
        "--log-path",
        default=None,
        help=f"NDJSON audit log path (default: {config.AUDIT_LOG_PATH})",
    )
    parser.add_argument(
        "--public-key-path",
        default=None,
        help="Ed25519 public key PEM for verification (defaults to private key pair)",
    )
    args = parser.parse_args()

    log_path = args.log_path or config.AUDIT_LOG_PATH
    public_key_path = args.public_key_path or config.AUDIT_VERIFY_PUBLIC_KEY_PATH or None

    try:
        valid, failures = verify_audit_log(log_path, public_key_path=public_key_path)
    except FileNotFoundError:
        logger.error("Audit log not found: %s", log_path)
        return 1
    except Exception as exc:
        logger.error("Verification failed: %s", exc)
        return 1

    if failures:
        logger.error(
            "Audit trail verification failed on line(s): %s (%d valid, %d invalid)",
            failures,
            valid,
            len(failures),
        )
        return 1

    logger.info("Audit trail verified: %d entries OK", valid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
