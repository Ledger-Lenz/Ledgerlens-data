"""Diagnostic health checks for the LedgerLens operational CLI.

Issue #959 — CLI: add structured, machine-readable output mode for all
diagnostic commands.

This module exposes ``run_diagnostics()`` which returns a dict conforming to
the **stable JSON schema** below.  The schema is versioned (``schema_version``)
so consumers can detect breaking changes without re-parsing the payload.

Stable JSON schema (``healthcheck`` subcommand)
-----------------------------------------------
.. code-block:: json

    {
      "schema_version": "1.0",
      "overall_status": "PASS" | "FAIL",
      "checks": {
        "environment": {
          "status": "PASS" | "FAIL",
          "details": {
            "<VAR_NAME>": "<sanitized-value>",
            ...
          },
          "missing": ["<VAR_NAME>", ...]
        },
        "streaming": {
          "status": "PASS" | "FAIL",
          "backend": "sse" | "kafka" | "stdout" | "<other>"
        }
      }
    }

``schema_version`` will be incremented (major bump) for any field removal or
type change.  Additive changes (new keys) increment the minor component only.

Adding a new check
------------------
1. Implement a class with a ``healthcheck() -> ServiceHealth`` method.
2. Add it to ``run_diagnostics()`` under a new key in ``"checks"``.
3. Update the schema docstring above.
4. Run ``python scripts/check_cli_contracts.py`` to verify the CLI surface
   is still documented.
"""

from __future__ import annotations

import os
from typing import Any

from utils.interfaces import ServiceHealth
from utils.secrets import mask_secret, sanitize_url

# Bump this when the schema gains breaking changes (field removal / type change).
DIAGNOSTICS_SCHEMA_VERSION = "1.0"


class EnvironmentHealthCheck:
    """Check that the required environment variables are present and parseable."""

    def healthcheck(self) -> ServiceHealth:
        db_url = os.environ.get("RISK_SCORE_DB_URL", "")
        horizon_url = os.environ.get("HORIZON_URL", "")
        kafka_sasl_pass = os.environ.get("KAFKA_SASL_PASSWORD", "")

        details: dict[str, Any] = {}
        missing: list[str] = []

        if db_url:
            details["RISK_SCORE_DB_URL"] = sanitize_url(db_url)
        else:
            missing.append("RISK_SCORE_DB_URL")

        if horizon_url:
            details["HORIZON_URL"] = sanitize_url(horizon_url)
        else:
            missing.append("HORIZON_URL")

        if kafka_sasl_pass:
            details["KAFKA_SASL_PASSWORD"] = mask_secret(kafka_sasl_pass)

        status = "PASS" if not missing else "FAIL"
        return ServiceHealth(status=status, details=details, missing=missing)


class StreamingHealthCheck:
    """Check that the streaming backend is configured."""

    def healthcheck(self) -> ServiceHealth:
        backend = os.environ.get("STREAMING_BACKEND", "stdout")
        return ServiceHealth(status="PASS", details={"backend": backend})


def run_diagnostics() -> dict[str, Any]:
    """Run all diagnostic checks and return a schema-versioned result dict.

    The return value always conforms to the stable JSON schema documented in
    this module's docstring.  Pass it directly to ``json.dumps()`` to get
    machine-readable output.

    Returns
    -------
    dict
        Keys: ``schema_version``, ``overall_status``, ``checks``.
    """
    env_service = EnvironmentHealthCheck()
    streaming_service = StreamingHealthCheck()

    env_health = env_service.healthcheck()
    streaming_health = streaming_service.healthcheck()

    overall = (
        "PASS"
        if env_health.status == "PASS" and streaming_health.status == "PASS"
        else "FAIL"
    )

    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "overall_status": overall,
        "checks": {
            "environment": {
                "status": env_health.status,
                "details": env_health.details,
                "missing": env_health.missing,
            },
            "streaming": {
                "status": streaming_health.status,
                "backend": streaming_health.details.get("backend", "stdout"),
            },
        },
    }


def json_schema() -> dict[str, Any]:
    """Return the JSON schema descriptor for the ``healthcheck`` output.

    Intended for documentation generators and ``check_cli_contracts.py``.
    """
    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "description": "Output schema for `ledgerlens-ops healthcheck --json`",
        "fields": {
            "schema_version": "string — semantic version of this schema",
            "overall_status": "'PASS' or 'FAIL'",
            "checks.environment.status": "'PASS' or 'FAIL'",
            "checks.environment.details": "dict of VAR_NAME → sanitized value",
            "checks.environment.missing": "list of missing required variable names",
            "checks.streaming.status": "'PASS' or 'FAIL'",
            "checks.streaming.backend": "streaming backend name string",
        },
    }
