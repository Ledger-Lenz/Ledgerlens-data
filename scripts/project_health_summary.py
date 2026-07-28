"""CLI for dashboard-ready project health summaries (Issue #605).

Examples
--------
    # Print pretty JSON to stdout
    python -m scripts.project_health_summary

    # Write artifact for dashboards / CI
    python -m scripts.project_health_summary --output reports/project_health.json

    # Fail CI when overall status is critical
    python -m scripts.project_health_summary --fail-on critical

    # Fail when status is degraded or worse
    python -m scripts.project_health_summary --fail-on degraded --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from monitoring.project_health import (
    STATUS_RANK,
    build_project_health_summary,
    write_project_health_summary,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a dashboard-ready LedgerLens project health summary."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: auto-detect from package layout).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path to write the summary JSON (e.g. reports/project_health.json).",
    )
    parser.add_argument(
        "--fail-on",
        choices=("never", "critical", "degraded", "unknown"),
        default="never",
        help="Exit non-zero when overall_status reaches this severity or worse.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print JSON to stdout (still writes --output if set).",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact JSON (no indentation).",
    )
    return parser.parse_args(argv)


def _should_fail(overall: str, fail_on: str) -> bool:
    if fail_on == "never":
        return False
    return STATUS_RANK.get(overall, 0) >= STATUS_RANK.get(fail_on, 99)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    indent = None if args.compact else 2

    if args.output is not None:
        summary = write_project_health_summary(
            args.output,
            repo_root=args.repo_root,
            indent=indent,
        )
    else:
        summary = build_project_health_summary(repo_root=args.repo_root)

    if not args.quiet:
        print(summary.to_json(indent=indent))

    if _should_fail(summary.overall_status, args.fail_on):
        print(
            f"project health gate failed: overall_status={summary.overall_status!r} "
            f"(fail-on={args.fail_on!r})",
            file=sys.stderr,
        )
        for line in summary.diagnostics[:12]:
            print(f"  - {line}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
