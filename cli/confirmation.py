"""Interactive confirmation with blast-radius summary for destructive CLI commands.

Issue #962: Destructive CLI commands should present a clear summary of
scope/blast-radius and require explicit confirmation before proceeding, with a
documented non-interactive ``--yes`` override for scripted/CI use.

Design
------
- :func:`confirm_destructive` prints the blast-radius summary and prompts the
  operator with "Proceed? [yes/N]: ".  The command is blocked until the
  operator types the word ``yes`` (case-insensitive) or passes ``--yes``.
- Anything other than ``yes`` aborts with :class:`ConfirmationAborted`.
- The ``--yes`` override is intentionally verbose (full word, not ``-y``) to
  reduce accidental non-interactive execution.  It should always be coupled
  with ``--dry-run`` in runbooks for a first-pass sanity check.

WARNING — non-interactive override (``--yes``)
----------------------------------------------
Passing ``--yes`` bypasses the confirmation prompt entirely.  Use it **only**
in automation (CI, scheduled jobs) where the blast-radius has been reviewed
in advance and the operator has explicit authority to proceed.  **Never** add
``--yes`` to a one-liner shared in chat or issue comments without first
running the command with ``--dry-run``.
"""

from __future__ import annotations

import sys
from typing import Any

_BLAST_RADIUS_HEADER = "⚠️  DESTRUCTIVE OPERATION — blast-radius summary:"
_CONFIRM_PROMPT = "Type 'yes' to proceed, anything else to abort: "
_ABORTED_MSG = "Aborted — no changes were made."


class ConfirmationAborted(Exception):
    """Raised when the operator declines the confirmation prompt."""


def print_blast_radius(summary: dict[str, Any], *, out=None) -> None:
    """Print a structured blast-radius summary to *out* (default stdout).

    Parameters
    ----------
    summary:
        Mapping of label → value describing the scope of the operation, e.g.::

            {
                "affected_records": 150_000,
                "affected_date_range": "2024-01-01 → 2024-06-30",
                "target_database": "postgresql://prod-host/ledgerlens",
                "operation": "restore from backup 2026-09-01",
            }
    """
    out = out or sys.stdout
    print(_BLAST_RADIUS_HEADER, file=out)
    for label, value in summary.items():
        print(f"  {label}: {value}", file=out)
    print("", file=out)


def confirm_destructive(
    summary: dict[str, Any],
    *,
    yes: bool = False,
    out=None,
    inp=None,
) -> None:
    """Print the blast-radius summary and require explicit confirmation.

    Parameters
    ----------
    summary:
        Blast-radius summary passed to :func:`print_blast_radius`.
    yes:
        When ``True`` the prompt is skipped (``--yes`` non-interactive override).
        A warning is still printed to *out* so CI logs are auditable.
    out:
        Output stream (default: ``sys.stdout``).
    inp:
        Input stream for reading the answer (default: ``sys.stdin``).
        Passing a custom stream enables testing without interactive I/O.

    Raises
    ------
    ConfirmationAborted
        If the operator does not confirm (interactive mode only).
    """
    out = out or sys.stdout
    inp = inp or sys.stdin

    print_blast_radius(summary, out=out)

    if yes:
        print(
            "  [--yes] Non-interactive override active — skipping confirmation prompt.",
            file=out,
        )
        print(
            "  ⚠️  WARNING: --yes bypasses the safety prompt. Ensure blast-radius was"
            " reviewed before proceeding.",
            file=out,
        )
        print("", file=out)
        return

    # Interactive prompt
    try:
        print(_CONFIRM_PROMPT, end="", flush=True, file=out)
        answer = inp.readline().strip()
    except (EOFError, KeyboardInterrupt):
        print("\n" + _ABORTED_MSG, file=out)
        raise ConfirmationAborted("Interrupted by user")

    if answer.lower() != "yes":
        print(_ABORTED_MSG, file=out)
        raise ConfirmationAborted(f"Operator declined (answered {answer!r})")


def add_yes_argument(parser) -> None:
    """Add the standard ``--yes`` non-interactive override to an argparse parser.

    This helper ensures every destructive command uses an identical flag name,
    help text, and default.
    """
    parser.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help=(
            "Skip the interactive confirmation prompt (non-interactive override). "
            "USE WITH CAUTION — this bypasses the blast-radius safety check. "
            "Always run with --dry-run first before using --yes in production."
        ),
    )
