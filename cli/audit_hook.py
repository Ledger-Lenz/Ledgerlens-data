"""CLI audit logging hook for production environments.

Every state-mutating CLI command should call :func:`emit_cli_audit_event`
(or use the :func:`audit_cli_command` decorator) so that operations run
directly against production are captured in the same audit trail used by
the API (``detection/audit_trail.py``).

This satisfies issue #961: "CLI commands run directly against production
(model promotion, overrides, backfills) should be logged to the same audit
trail used elsewhere in the project."

Design
------
- Audit events are written as NDJSON lines to ``CLI_AUDIT_LOG_PATH``
  (default ``logs/cli_audit.ndjson``; overridable via env / :mod:`config`).
- Each event captures: timestamp, actor, command, sanitised arguments
  (secrets redacted), environment label (``LEDGERLENS_ENV``), and outcome.
- Secret redaction: any argument whose name contains ``secret``, ``key``,
  ``token``, ``password``, or ``credential`` has its value replaced with
  ``[REDACTED]``.
- When ``LEDGERLENS_ENV`` is not ``production`` the hook is a no-op by
  default.  Pass ``force=True`` to emit in any environment (useful in tests).
- Missing log directory is created automatically.
- Write failures are logged at WARNING level but never propagate so that
  a broken audit log path cannot take down a CLI command.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SECRET_ARG_PATTERN = re.compile(
    r"(secret|key|token|password|credential|passwd|api[_-]?key)",
    re.IGNORECASE,
)
_DEFAULT_LOG_PATH = "logs/cli_audit.ndjson"
_AUDIT_ENABLED_ENVS = {"production", "prod"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_log_path() -> Path:
    """Return the audit log path from config or the default."""
    try:
        from config import config  # noqa: PLC0415

        path_str = getattr(config, "CLI_AUDIT_LOG_PATH", None) or _DEFAULT_LOG_PATH
    except Exception:  # pragma: no cover — config import failure in CI
        path_str = os.getenv("CLI_AUDIT_LOG_PATH", _DEFAULT_LOG_PATH)
    return Path(path_str)


def _current_env() -> str:
    try:
        from config import config  # noqa: PLC0415

        return (getattr(config, "LEDGERLENS_ENV", None) or os.getenv("LEDGERLENS_ENV", "local")).lower()
    except Exception:  # pragma: no cover
        return os.getenv("LEDGERLENS_ENV", "local").lower()


def _resolve_actor() -> str:
    """Return an actor identifier from env / system context."""
    # Prefer an explicit operator identity variable; fall back to Unix user.
    actor = (
        os.getenv("LEDGERLENS_OPERATOR")
        or os.getenv("GITHUB_ACTOR")
        or os.getenv("USER")
        or os.getenv("LOGNAME")
        or "unknown"
    )
    return actor


def redact_args(args: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *args* with sensitive values replaced by ``[REDACTED]``.

    Parameters
    ----------
    args:
        Mapping of argument name → value (e.g. ``vars(parsed_args)``).

    Returns
    -------
    dict
        New mapping with the same keys; sensitive values replaced.
    """
    sanitised: dict[str, Any] = {}
    for k, v in args.items():
        if _SECRET_ARG_PATTERN.search(str(k)):
            sanitised[k] = "[REDACTED]"
        else:
            # Coerce to string so the JSON serialiser doesn't choke on
            # arbitrary objects (e.g. file handles, Namespace objects).
            try:
                json.dumps(v)
                sanitised[k] = v
            except (TypeError, ValueError):
                sanitised[k] = str(v)
    return sanitised


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def emit_cli_audit_event(
    command: str,
    args: dict[str, Any],
    outcome: str,
    *,
    error: str | None = None,
    actor: str | None = None,
    env: str | None = None,
    log_path: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any] | None:
    """Write a single CLI audit event to the NDJSON audit log.

    Parameters
    ----------
    command:
        Name of the CLI subcommand (e.g. ``"backup"``, ``"restore"``).
    args:
        Parsed argument mapping (``vars(opts)``); secrets are redacted
        automatically.
    outcome:
        ``"success"``, ``"dry-run"``, or ``"error"``.
    error:
        Optional error message when *outcome* is ``"error"``.
    actor:
        Identity of the operator.  Defaults to :func:`_resolve_actor`.
    env:
        Environment label.  Defaults to :func:`_current_env`.
    log_path:
        Override the default audit log path.
    force:
        Emit even when not in a production environment (useful for testing).

    Returns
    -------
    dict | None
        The event that was written, or ``None`` if the hook was skipped.
    """
    current_env = env or _current_env()
    if not force and current_env not in _AUDIT_ENABLED_ENVS:
        logger.debug(
            "CLI audit hook skipped (env=%s, not in %s)", current_env, _AUDIT_ENABLED_ENVS
        )
        return None

    event: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "actor": actor or _resolve_actor(),
        "env": current_env,
        "command": command,
        "args": redact_args(args),
        "outcome": outcome,
    }
    if error:
        event["error"] = error

    resolved_path = Path(log_path) if log_path else _resolve_log_path()

    try:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")
        logger.debug("CLI audit event written to %s: command=%s outcome=%s", resolved_path, command, outcome)
    except Exception as exc:  # pragma: no cover — filesystem failures
        logger.warning(
            "Failed to write CLI audit event (path=%s): %s", resolved_path, exc
        )

    return event


def audit_cli_command(command: str, *, log_path: Path | str | None = None, force: bool = False):
    """Decorator that wraps a CLI ``main(args)`` function with audit logging.

    The decorated function must accept ``args`` as its first positional
    argument (an :class:`argparse.Namespace` or dict).

    Example::

        @audit_cli_command("backup")
        def run_backup(args):
            ...

    On success the outcome is ``"dry-run"`` when ``args.dry_run`` is truthy,
    otherwise ``"success"``.  On exception the outcome is ``"error"`` and the
    exception propagates after the event is written.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(args, **kwargs):
            args_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
            dry_run = bool(args_dict.get("dry_run", False))
            try:
                result = fn(args, **kwargs)
                outcome = "dry-run" if dry_run else "success"
                emit_cli_audit_event(
                    command=command,
                    args=args_dict,
                    outcome=outcome,
                    log_path=log_path,
                    force=force,
                )
                return result
            except SystemExit:
                # SystemExit from argparse --help etc. — don't audit
                raise
            except Exception as exc:
                emit_cli_audit_event(
                    command=command,
                    args=args_dict,
                    outcome="error",
                    error=str(exc),
                    log_path=log_path,
                    force=force,
                )
                raise

        return wrapper

    return decorator
