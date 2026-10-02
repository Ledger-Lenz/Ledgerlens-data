"""Issue #962 — Interactive confirmation with blast-radius summary for
destructive CLI commands.

Every destructive CLI command (restore, backfill, large-scale model promotion)
must call :func:`confirm_destructive_action` before executing any writes.  The
function:

1. Collects and displays a *blast-radius summary* — record counts, affected
   date ranges, affected tenants / wallets, etc. — so the operator knows
   exactly what will change.
2. Prompts for explicit ``yes`` / ``no`` confirmation.
3. Accepts a ``--yes`` / ``non_interactive=True`` override for scripted/CI use
   (documented with an explicit caution banner).

Usage example::

    from cli.confirmation import confirm_destructive_action, BlastRadiusSummary

    summary = BlastRadiusSummary(
        operation="Database restore",
        affected_records={"risk_scores": 50_000, "model_versions": 120},
        affected_date_range=("2024-01-01", "2024-12-31"),
        affected_tenants=["prod"],
        extra={"backup_timestamp": "2024-06-01T12:00:00Z"},
    )
    if not confirm_destructive_action(summary, non_interactive=opts.yes):
        sys.exit(1)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Blast-radius data class
# ---------------------------------------------------------------------------


@dataclass
class BlastRadiusSummary:
    """A structured description of what a destructive command will affect.

    Attributes
    ----------
    operation:
        Human-readable name of the operation (e.g. "Database restore").
    affected_records:
        Mapping of table/collection name → estimated row count.
    affected_date_range:
        Optional (start, end) ISO-8601 date strings for the affected data window.
    affected_tenants:
        Optional list of tenant / environment identifiers affected.
    extra:
        Any additional key/value metadata to surface in the summary.
    """

    operation: str
    affected_records: dict[str, int] = field(default_factory=dict)
    affected_date_range: Optional[tuple[str, str]] = None
    affected_tenants: list[str] = field(default_factory=list)
    extra: dict[str, str] = field(default_factory=dict)

    def format(self) -> str:
        """Return a human-readable blast-radius summary string."""
        lines: list[str] = [
            "",
            "=" * 60,
            "⚠️  DESTRUCTIVE OPERATION — BLAST-RADIUS SUMMARY",
            "=" * 60,
            f"  Operation : {self.operation}",
        ]

        if self.affected_records:
            lines.append("  Affected records:")
            for table, count in sorted(self.affected_records.items()):
                lines.append(f"    • {table}: {count:,} rows")

        if self.affected_date_range:
            start, end = self.affected_date_range
            lines.append(f"  Date range: {start} → {end}")

        if self.affected_tenants:
            lines.append(f"  Tenants / environments: {', '.join(self.affected_tenants)}")

        for key, value in sorted(self.extra.items()):
            lines.append(f"  {key}: {value}")

        lines += [
            "=" * 60,
            "",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Confirmation logic
# ---------------------------------------------------------------------------

#: Non-interactive override env variable.  Set to ``1`` in scripted / CI
#: environments where no TTY is available.  Use with extreme caution.
NON_INTERACTIVE_ENV_VAR = "LEDGERLENS_YES"


def confirm_destructive_action(
    summary: BlastRadiusSummary,
    *,
    non_interactive: bool = False,
    prompt_stream=None,
    input_stream=None,
) -> bool:
    """Display a blast-radius summary and ask the operator to confirm.

    Parameters
    ----------
    summary:
        :class:`BlastRadiusSummary` describing what will change.
    non_interactive:
        If ``True`` the confirmation prompt is skipped and the function returns
        ``True`` immediately.  Controlled by ``--yes`` CLI flag or the
        :data:`NON_INTERACTIVE_ENV_VAR` environment variable.

        .. warning::
            Using ``--yes`` / non-interactive mode bypasses the human
            confirmation gate entirely.  Only use it in CI pipelines or
            automation scripts where the blast radius has already been
            reviewed and the operation is known to be safe.  **Never** use
            it as a shortcut during ad-hoc production operations.

    prompt_stream:
        File-like object used for output (default: ``sys.stderr``).
    input_stream:
        File-like object used for input (default: ``sys.stdin``).

    Returns
    -------
    bool
        ``True`` if the operator confirmed (or ``non_interactive=True``),
        ``False`` if they declined.
    """
    out = prompt_stream if prompt_stream is not None else sys.stderr
    inp = input_stream if input_stream is not None else sys.stdin

    # Print the blast-radius summary regardless of interactive / non-interactive
    out.write(summary.format())
    out.flush()

    # Check env-var override first
    env_yes = os.environ.get(NON_INTERACTIVE_ENV_VAR, "").strip().lower() in ("1", "true", "yes")
    if non_interactive or env_yes:
        out.write(
            "⚠️  --yes / non-interactive mode: skipping confirmation prompt.\n"
            "   This bypasses the human safety gate. Proceeding automatically.\n\n"
        )
        out.flush()
        return True

    # Interactive prompt
    try:
        out.write(
            "Type 'yes' and press Enter to proceed, or anything else to cancel: "
        )
        out.flush()
        answer = inp.readline().strip().lower()
    except (EOFError, KeyboardInterrupt):
        out.write("\nAborted.\n")
        out.flush()
        return False

    if answer == "yes":
        out.write("✅ Confirmed. Proceeding.\n\n")
        out.flush()
        return True

    out.write("❌ Cancelled. No changes were made.\n\n")
    out.flush()
    return False
