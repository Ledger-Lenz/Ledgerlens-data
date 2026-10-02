"""Issue #960 — Dry-run mode for all state-mutating CLI commands.

Every CLI command that mutates state (backfills, restore, model promotion,
config changes) must support a ``--dry-run`` flag that:

1. Reports *exactly* what would change without applying it.
2. Produces clearly formatted output so operators can review the blast radius
   before committing.
3. Never performs the actual mutation (zero side effects).

Usage::

    from cli.dry_run import DryRunPlan, add_dry_run_argument, check_dry_run

    parser = argparse.ArgumentParser(...)
    add_dry_run_argument(parser)
    args = parser.parse_args()

    plan = DryRunPlan("backfill", [
        DryRunAction("write", "data/labelled_with_cross_venue.parquet",
                     detail="1 234 rows × 47 cols"),
        DryRunAction("upsert", "risk_scores table",
                     detail="12 wallets, score range 45–92"),
    ])

    if check_dry_run(args, plan):
        # dry-run: plan was printed, nothing was written
        sys.exit(0)

    # Normal execution path
    features_df.to_parquet(output_path)
    ...

Documentation
-------------
See ``docs/cli_contracts.md`` for the full dry-run contract and the
recommendation that operators **always run ``--dry-run`` first** before
executing any state-mutating command in production.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DryRunAction:
    """A single concrete change that *would* be made if the command ran for real.

    Attributes
    ----------
    action_type:
        Verb describing the mutation (e.g. ``"write"``, ``"upsert"``,
        ``"delete"``, ``"promote"``, ``"send"``).
    target:
        The resource that would be affected (file path, table name, API
        endpoint, …).
    detail:
        Optional human-readable elaboration (row counts, date range, etc.).
    """

    action_type: str
    target: str
    detail: str = ""


@dataclass
class DryRunPlan:
    """A complete dry-run report for one CLI command invocation.

    Attributes
    ----------
    command:
        The CLI command name (e.g. ``"backfill"``, ``"restore"``).
    actions:
        Ordered list of :class:`DryRunAction` items — the concrete changes
        that *would* be made if ``--dry-run`` were not set.
    extra:
        Optional additional key/value metadata to append to the report.
    """

    command: str
    actions: list[DryRunAction] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def add_action(self, action_type: str, target: str, detail: str = "") -> None:
        """Append a new :class:`DryRunAction` to the plan."""
        self.actions.append(DryRunAction(action_type, target, detail))

    def format(self) -> str:
        """Return a human-readable dry-run report string."""
        lines: list[str] = [
            "",
            "=" * 60,
            f"🔍 DRY-RUN MODE — {self.command.upper()}",
            "   No changes will be made.  Review the plan below.",
            "=" * 60,
        ]

        if not self.actions:
            lines.append("  (No state-mutating actions identified.)")
        else:
            lines.append(f"  {len(self.actions)} action(s) would be performed:")
            for i, action in enumerate(self.actions, start=1):
                line = f"    {i:2d}. [{action.action_type.upper()}] {action.target}"
                if action.detail:
                    line += f"\n        → {action.detail}"
                lines.append(line)

        for key, value in sorted(self.extra.items()):
            lines.append(f"  {key}: {value}")

        lines += [
            "=" * 60,
            "  Re-run without --dry-run to apply these changes.",
            "",
        ]
        return "\n".join(lines)

    def print_report(self, stream=None) -> None:
        """Print the formatted dry-run report to *stream* (default: stdout)."""
        out = stream if stream is not None else sys.stdout
        out.write(self.format())
        out.flush()


# ---------------------------------------------------------------------------
# Argument parser helper
# ---------------------------------------------------------------------------


def add_dry_run_argument(parser) -> None:
    """Add a consistent ``--dry-run`` flag to *parser*.

    All state-mutating CLI commands must call this helper so the flag is
    defined consistently across the CLI surface.
    """
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Report exactly what would change without applying any writes. "
            "Recommended before running any state-mutating command in production. "
            "Produces zero side effects."
        ),
    )


def check_dry_run(args, plan: DryRunPlan, *, stream=None) -> bool:
    """If ``args.dry_run`` is True, print the plan and return True.

    The caller should exit immediately after ``check_dry_run`` returns True::

        if check_dry_run(args, plan):
            sys.exit(0)

    Parameters
    ----------
    args:
        Parsed argument namespace.  Must have a ``dry_run`` attribute.
    plan:
        :class:`DryRunPlan` describing the concrete changes that would occur.
    stream:
        Output stream (default: ``sys.stdout``).

    Returns
    -------
    bool
        ``True`` if dry-run mode is active (caller should skip all writes),
        ``False`` otherwise.
    """
    if not getattr(args, "dry_run", False):
        return False

    plan.print_report(stream=stream)
    return True
