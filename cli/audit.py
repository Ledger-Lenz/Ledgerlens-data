"""Issue #961 — CLI command execution audit logging for production environments.

Every CLI command that mutates state (model promotion, backfills, restore,
config changes) must produce an audit-log entry when running in a
production-configured environment.  The entry is written to the same NDJSON
audit trail used by :mod:`detection.audit_trail` so all mutating events are
traceable from a single log file.

The entry captures:
- actor identity (from environment / auth context — see ``_get_actor()``)
- command name and all arguments (with secrets redacted)
- outcome (``success`` / ``failure`` and optional error message)
- timestamp (UTC ISO-8601)

Usage::

    from cli.audit import audit_cli_command, is_production_env

    # Wrap any mutating command:
    with audit_cli_command("restore", args=vars(opts), outcome_callback=True):
        do_the_work()

Or as a decorator on a ``main()``-style function::

    from cli.audit import cli_audit_hook

    @cli_audit_hook("backfill")
    def main(opts):
        ...

Production detection
--------------------
Auditing is active when :func:`is_production_env` returns ``True``.  This is
the case when:

- ``LEDGERLENS_ENV=production`` is set in the environment, **or**
- ``AUDIT_ALL_ENVIRONMENTS=1`` is set (for testing / staging).

In non-production environments the hook is a no-op (no file I/O, no overhead).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Generator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Environment variable that marks a production deployment.
LEDGERLENS_ENV_VAR: str = "LEDGERLENS_ENV"

#: Set ``AUDIT_ALL_ENVIRONMENTS=1`` to force auditing in non-production
#: environments (useful for staging / integration tests).
AUDIT_ALL_ENV_VAR: str = "AUDIT_ALL_ENVIRONMENTS"

#: Set ``LEDGERLENS_ACTOR`` to explicitly name the actor running the command.
#: Defaults to the value of ``USER``/``USERNAME`` env vars, then ``"unknown"``.
ACTOR_ENV_VAR: str = "LEDGERLENS_ACTOR"

#: Patterns whose values are redacted from argument logs.
#: Any argument key matching one of these patterns has its value replaced with
#: ``"[REDACTED]"``.
SECRET_KEY_PATTERNS: list[re.Pattern] = [
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"key", re.IGNORECASE),
    re.compile(r"credential", re.IGNORECASE),
    re.compile(r"passphrase", re.IGNORECASE),
    re.compile(r"private", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_production_env() -> bool:
    """Return ``True`` when the current deployment is production-configured.

    Production is detected when:
    - ``LEDGERLENS_ENV=production`` is set, **or**
    - ``AUDIT_ALL_ENVIRONMENTS=1`` is set.
    """
    env = os.environ.get(LEDGERLENS_ENV_VAR, "").strip().lower()
    force_all = os.environ.get(AUDIT_ALL_ENV_VAR, "").strip().lower() in ("1", "true", "yes")
    return env == "production" or force_all


def _get_actor() -> str:
    """Derive the actor identity from the environment.

    Priority order:
    1. ``LEDGERLENS_ACTOR`` env var (explicit override)
    2. ``USER`` / ``USERNAME`` (OS login name)
    3. ``"unknown"``
    """
    explicit = os.environ.get(ACTOR_ENV_VAR, "").strip()
    if explicit:
        return explicit
    user = os.environ.get("USER", "") or os.environ.get("USERNAME", "")
    return user.strip() or "unknown"


def redact_secrets(args: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *args* with secret values replaced by ``"[REDACTED]"``.

    Any key whose name matches one of :data:`SECRET_KEY_PATTERNS` has its
    value replaced with the string ``"[REDACTED]"``.  This prevents credential
    values from appearing in the audit log.

    >>> redact_secrets({"db_url": "postgresql://...", "secret": "s3kr3t"})
    {'db_url': 'postgresql://...', 'secret': '[REDACTED]'}
    """
    redacted: dict[str, Any] = {}
    for key, value in args.items():
        if any(pattern.search(str(key)) for pattern in SECRET_KEY_PATTERNS):
            redacted[key] = "[REDACTED]"
        else:
            redacted[key] = value
    return redacted


def _get_log_path() -> str:
    """Return the CLI audit log path.

    Uses ``CLI_AUDIT_LOG_PATH`` env var if set, otherwise falls back to the
    shared :data:`config.config.AUDIT_LOG_PATH` value so CLI entries land in
    the same trail as forensic-report entries.
    """
    custom = os.environ.get("CLI_AUDIT_LOG_PATH", "").strip()
    if custom:
        return custom
    try:
        from config import config as _cfg

        return _cfg.AUDIT_LOG_PATH
    except Exception:
        return "data/audit_trail.ndjson"


def _write_audit_entry(entry: dict[str, Any], log_path: str) -> None:
    """Append *entry* as an NDJSON line to *log_path*."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "ab") as fh:
        fh.write((json.dumps(entry, sort_keys=True) + "\n").encode())


def build_cli_audit_entry(
    command: str,
    args: dict[str, Any],
    outcome: str,
    *,
    actor: str | None = None,
    error: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build a CLI audit log entry dict.

    Parameters
    ----------
    command:
        The CLI command name (e.g. ``"restore"``, ``"backfill"``).
    args:
        Parsed CLI arguments dict.  Secret values are redacted automatically.
    outcome:
        ``"success"`` or ``"failure"``.
    actor:
        Actor identity.  Defaults to :func:`_get_actor`.
    error:
        Error message string if ``outcome == "failure"``.
    timestamp:
        ISO-8601 UTC timestamp.  Defaults to ``datetime.now(UTC).isoformat()``.

    Returns
    -------
    dict
        The entry dict ready to be written to the audit trail.
    """
    entry: dict[str, Any] = {
        "event_type": "cli_command",
        "command": command,
        "actor": actor or _get_actor(),
        "args": redact_secrets(args),
        "outcome": outcome,
        "timestamp": timestamp or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    if error:
        entry["error"] = str(error)
    return entry


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@contextmanager
def audit_cli_command(
    command: str,
    args: dict[str, Any] | None = None,
    *,
    log_path: str | None = None,
    actor: str | None = None,
) -> Generator[None, None, None]:
    """Context manager that writes a CLI audit entry on exit.

    Writes a ``success`` entry if the body completes normally; writes a
    ``failure`` entry (with the exception message) if the body raises.

    Only active when :func:`is_production_env` returns ``True``.

    Parameters
    ----------
    command:
        CLI command name.
    args:
        Arguments dict.  Secrets are redacted automatically.
    log_path:
        Override for the audit log file path.
    actor:
        Override for the actor identity.

    Example::

        with audit_cli_command("restore", args={"backup_dir": "/var/backup"}):
            perform_restore()
    """
    _args = args or {}
    _log_path = log_path or _get_log_path()
    _active = is_production_env()

    exc_info: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        exc_info = exc
        raise
    finally:
        if _active:
            if exc_info is not None:
                entry = build_cli_audit_entry(
                    command,
                    _args,
                    "failure",
                    actor=actor,
                    error=str(exc_info),
                )
            else:
                entry = build_cli_audit_entry(command, _args, "success", actor=actor)
            try:
                _write_audit_entry(entry, _log_path)
            except Exception as write_err:  # pragma: no cover — best-effort
                sys.stderr.write(
                    f"[cli.audit] WARNING: Failed to write audit entry: {write_err}\n"
                )


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def cli_audit_hook(command: str, *, log_path: str | None = None, actor: str | None = None):
    """Decorator that wraps a ``main(opts)``-style function with audit logging.

    Only active in production environments.

    Parameters
    ----------
    command:
        CLI command name to record in the audit trail.
    log_path:
        Override for the audit log file path.
    actor:
        Override for the actor identity.

    Example::

        @cli_audit_hook("backfill")
        def main(opts):
            ...
    """
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*fn_args, **fn_kwargs):
            # Extract args dict from the first positional arg if it is a
            # Namespace (argparse) or dict, otherwise record empty dict.
            raw_args: dict[str, Any] = {}
            if fn_args:
                first = fn_args[0]
                if hasattr(first, "__dict__"):
                    raw_args = vars(first)
                elif isinstance(first, dict):
                    raw_args = first

            with audit_cli_command(command, args=raw_args, log_path=log_path, actor=actor):
                return fn(*fn_args, **fn_kwargs)

        return wrapper

    return decorator
