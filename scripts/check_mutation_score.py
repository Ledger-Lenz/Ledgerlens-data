#!/usr/bin/env python3
"""Enforce a minimum mutation score threshold for CI.

Reads the mutmut SQLite results database (`.mutmut-cache`) and computes the
mutation score as:

    killed / (killed + survived) * 100

Exits with code 1 if the score is below the threshold so the CI step fails.
Surviving mutations are printed to stdout so they can be triaged and tracked
as follow-up issues.

Usage:
    python scripts/check_mutation_score.py --threshold 80

Exit codes:
    0  Mutation score meets or exceeds the threshold.
    1  Mutation score is below the threshold (CI failure).
    2  No mutmut cache found — run `mutmut run` first.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enforce a minimum mutmut mutation score threshold."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=80.0,
        help="Minimum required mutation score as a percentage (default: 80).",
    )
    parser.add_argument(
        "--cache",
        default=".mutmut-cache",
        help="Path to the mutmut SQLite cache file (default: .mutmut-cache).",
    )
    return parser.parse_args()


def _load_results(cache_path: Path) -> tuple[int, int, list[dict]]:
    """Return (killed, survived, surviving_mutations) from the mutmut cache.

    Mutmut statuses:
        ok         — mutation was killed (test suite detected the change)
        survived   — mutation was NOT killed (test suite missed it)
        suspicious — tests passed but with timing/output differences
        timeout    — test run exceeded the timeout
        ba_error   — mutmut encountered an error mutating the file

    For the score calculation we treat:
        killed  = ok + suspicious + timeout (test suite ran and caught *something*)
        survived = survived only
    Errors are excluded from both sides (they don't represent a test quality signal).
    """
    conn = sqlite3.connect(str(cache_path))
    try:
        cursor = conn.execute(
            "SELECT id, line, status, filename FROM mutant"
        )
        rows = cursor.fetchall()
    except sqlite3.OperationalError as exc:
        print(f"ERROR: Could not read mutmut cache — {exc}", file=sys.stderr)
        print(
            "Make sure you have run `mutmut run` before calling this script.",
            file=sys.stderr,
        )
        sys.exit(2)
    finally:
        conn.close()

    killed = 0
    survived = 0
    surviving: list[dict] = []

    for mut_id, line, status, filename in rows:
        if status in ("ok", "suspicious", "timeout"):
            killed += 1
        elif status == "survived":
            survived += 1
            surviving.append(
                {
                    "id": mut_id,
                    "filename": filename,
                    "line": line,
                    "status": status,
                }
            )
        # ba_error / other — skip

    return killed, survived, surviving


def main() -> None:
    args = _parse_args()
    cache_path = Path(args.cache)

    if not cache_path.exists():
        print(
            f"ERROR: mutmut cache not found at '{cache_path}'.\n"
            "Run `mutmut run` (or `make mutation-test`) first.",
            file=sys.stderr,
        )
        sys.exit(2)

    killed, survived, surviving = _load_results(cache_path)
    total = killed + survived

    if total == 0:
        print(
            "WARNING: No mutations found in cache. "
            "Check that --paths-to-mutate is correct and mutmut ran successfully.",
            file=sys.stderr,
        )
        # Treat as failure — something is wrong with the setup.
        sys.exit(1)

    score = (killed / total) * 100.0

    print(f"\n{'=' * 60}")
    print(f"  Mutation Testing Report")
    print(f"{'=' * 60}")
    print(f"  Total mutations:  {total:>5}")
    print(f"  Killed:           {killed:>5}  ({score:.1f}%)")
    print(f"  Survived:         {survived:>5}  ({100 - score:.1f}%)")
    print(f"  Threshold:        {args.threshold:.1f}%")
    print(f"{'=' * 60}")

    if surviving:
        print(f"\n  Surviving mutations ({len(surviving)}) — follow-up issues needed:")
        print(f"  {'ID':<6} {'File':<45} {'Line':<6} Status")
        print(f"  {'-'*6} {'-'*45} {'-'*6} {'-'*10}")
        for m in sorted(surviving, key=lambda x: (x["filename"] or "", x["line"] or 0)):
            fname = (m["filename"] or "unknown")
            # Trim long paths for readability
            if len(fname) > 44:
                fname = "..." + fname[-41:]
            print(f"  {m['id']:<6} {fname:<45} {str(m['line'] or '?'):<6} {m['status']}")
        print()
        print(
            "  To inspect a specific surviving mutation run:\n"
            "    mutmut show <ID>\n"
            "  To apply it locally for investigation:\n"
            "    mutmut apply <ID>   # then run pytest manually\n"
            "    mutmut unapply <ID> # restore original\n"
        )

    print(f"\n  Score {score:.1f}% vs threshold {args.threshold:.1f}%  =>  ", end="")

    if score >= args.threshold:
        print("PASS ✓")
        sys.exit(0)
    else:
        print("FAIL ✗")
        print(
            f"\n  Mutation score {score:.1f}% is below the required {args.threshold:.1f}%.\n"
            "  Add tests that kill the surviving mutations listed above,\n"
            "  then re-run `make mutation-test`.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
