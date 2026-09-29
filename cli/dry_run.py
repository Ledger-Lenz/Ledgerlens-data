"""Dry-run support for state-mutating CLI commands.

Every command that mutates state (writes to DB, writes files, calls external
APIs) should call :func:`check_dry_run` or use :class:`DryRunContext` so that
``--dry-run`` mode produces clear enumeration of *what would happen* without
performing any mutation.

Usage pattern::

    from cli.dry_run import DryRunContext

    def run_backup(args):
        with DryRunContext(args.dry_run) as dry:
            # Stage effects before executing them
            dry.record("Would write database backup to /backups/db_2026-01-01.db")
            dry.record("Would write model archive to /backups/models_2026-01-01.tar.gz")
            dry.record("Would write manifest to /backups/MANIFEST.json")

            if dry:   # truthy when in dry-run mode
                return   # skip all mutations

            # Real mutations go here
            ...

The context manager guarantees:

- In dry-run mode all recorded effects are printed to stdout and no mutations
  occur.
- In live mode nothing extra is printed; effects run normally.

Rules enforced
--------------
- ``DryRunContext.__bool__`` returns ``True`` when dry-run is active so you
  can use ``if dry:`` to skip mutation blocks.
- :func:`abort_if_dry_run` raises :class:`DryRunAbort` which the CLI entrypoint
  catches as a clean success — useful when a command has a single, late
  mutation that cannot be conditioned easily.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

_DRY_RUN_BANNER = "[DRY RUN] No data will be written."
_DRY_RUN_PREFIX = "[dry-run] "


class DryRunAbort(Exception):
    """Raised by :func:`abort_if_dry_run` to stop execution cleanly."""


class DryRunContext:
    """Context manager that collects would-be mutations and prints them
    when in dry-run mode without executing them.

    Parameters
    ----------
    enabled:
        ``True`` activates dry-run mode.
    out:
        Output stream for dry-run messages. Defaults to ``sys.stdout``.
    """

    def __init__(self, enabled: bool, out: Any = None) -> None:
        self._enabled = bool(enabled)
        self._out = out or sys.stdout
        self._effects: list[str] = []

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "DryRunContext":
        if self._enabled:
            print(_DRY_RUN_BANNER, file=self._out)
            logger.info(_DRY_RUN_BANNER)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._enabled and not exc_type:
            if self._effects:
                print("Would perform the following operations:", file=self._out)
                for effect in self._effects:
                    print(f"  {_DRY_RUN_PREFIX}{effect}", file=self._out)
            else:
                print(f"  {_DRY_RUN_PREFIX}(no operations to report)", file=self._out)
        return False  # don't suppress exceptions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __bool__(self) -> bool:
        """``True`` when dry-run mode is active."""
        return self._enabled

    def record(self, description: str) -> None:
        """Record an effect that *would* happen (dry-run) or *is* happening (live).

        In dry-run mode the description is collected and printed at context
        exit.  In live mode the description is emitted as a DEBUG log so the
        call is always safe without branching.
        """
        self._effects.append(description)
        if self._enabled:
            logger.debug("%s%s", _DRY_RUN_PREFIX, description)
        else:
            logger.debug("Executing: %s", description)

    @property
    def effects(self) -> list[str]:
        """Recorded effect descriptions (read-only snapshot)."""
        return list(self._effects)


def abort_if_dry_run(dry_run: bool, description: str = "") -> None:
    """Raise :class:`DryRunAbort` when *dry_run* is ``True``.

    Use this at the boundary just before the single irreversible write in a
    command that is difficult to restructure with a full :class:`DryRunContext`.

    Parameters
    ----------
    dry_run:
        Flag from ``argparse``.
    description:
        Human-readable summary of the would-be mutation, printed to stdout.
    """
    if dry_run:
        print(_DRY_RUN_BANNER)
        if description:
            print(f"  {_DRY_RUN_PREFIX}{description}")
        raise DryRunAbort(description)


def add_dry_run_argument(parser) -> None:
    """Add the standard ``--dry-run`` flag to an :class:`argparse.ArgumentParser`.

    This helper ensures every command uses an identical flag name, help text,
    and default so the interface is consistent.
    """
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Report what would change without applying any mutation. "
            "Recommended before running destructive commands in production."
        ),
    )
