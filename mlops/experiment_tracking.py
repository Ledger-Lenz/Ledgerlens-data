"""Reproducible local experiment tracking — with full lineage provenance (Issue #941).

This module intentionally uses append-only JSONL files so model development
metadata remains easy to diff, archive, and inspect in CI without requiring an
external service.

Issue #941 additions
--------------------
Every training run now automatically records:
- ``dataset_snapshot_id`` — identifier of the frozen dataset snapshot used for
  training (from :mod:`data.reproducibility`).
- ``git_commit_hash`` — the resolved HEAD commit SHA at training time.
- ``resolved_hyperparams`` — the full, resolved hyperparameter dict (not just
  the user-supplied overrides).

A :func:`lookup_lineage` function returns the full provenance record for any
production model version in a single call.

A :func:`check_lineage_complete` function can be used as a CI/release gate to
block promotion of models with incomplete lineage metadata.

Incident investigation guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To trace a production model back to its exact training data and code::

    from mlops.experiment_tracking import JsonlExperimentTracker, lookup_lineage

    tracker = JsonlExperimentTracker()
    lineage = lookup_lineage(tracker, run_id="<run_id_from_model_metadata>")
    # lineage contains: dataset_snapshot_id, git_commit_hash,
    #                   resolved_hyperparams, feature_schema_hash, metrics, …

    # Verify the snapshot still exists and is untampered
    from data.reproducibility import DatasetSnapshot
    snap = DatasetSnapshot()
    ok = snap.verify(lineage["dataset_snapshot_id"])

Use ``check_lineage_complete(record)`` in CI to prevent promotion of runs that
are missing any provenance field::

    from mlops.experiment_tracking import check_lineage_complete
    errors = check_lineage_complete(lineage)
    if errors:
        raise SystemExit(f"Lineage incomplete — promotion blocked: {errors}")
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants — required provenance fields for a complete lineage record
# ---------------------------------------------------------------------------

REQUIRED_LINEAGE_FIELDS: tuple[str, ...] = (
    "run_id",
    "name",
    "feature_schema_hash",
    "dataset_sha256",
    "git_sha",
    "dataset_snapshot_id",
    "resolved_hyperparams",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


# ---------------------------------------------------------------------------
# ExperimentRun
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentRun:
    """Immutable metadata for one model-development run.

    Issue #941 fields
    -----------------
    dataset_snapshot_id:
        The snapshot identifier returned by :meth:`~data.reproducibility.DatasetSnapshot.freeze`.
        Records the exact content-addressed copy of the training dataset.
    resolved_hyperparams:
        The full, resolved hyperparameter dict — i.e. defaults merged with any
        user-supplied overrides — so the run is exactly reproducible without
        relying on code defaults that may change between releases.
    """

    name: str
    params: dict[str, Any]
    feature_schema_hash: str
    dataset_sha256: str
    git_sha: str | None = None
    dataset_snapshot_id: str | None = None
    resolved_hyperparams: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ExperimentRun.name must be non-empty")
        if not self.feature_schema_hash.strip():
            raise ValueError("ExperimentRun.feature_schema_hash must be non-empty")
        if not self.dataset_sha256.strip():
            raise ValueError("ExperimentRun.dataset_sha256 must be non-empty")
        if self.created_at.tzinfo is None:
            raise ValueError("ExperimentRun.created_at must be timezone-aware")

    @property
    def run_id(self) -> str:
        payload = {
            "dataset_sha256": self.dataset_sha256,
            "feature_schema_hash": self.feature_schema_hash,
            "git_sha": self.git_sha,
            "name": self.name,
            "params": self.params,
        }
        return hashlib.sha256(_stable_json(payload).encode()).hexdigest()[:16]

    def to_record(
        self, metrics: dict[str, float], artifacts: dict[str, str] | None = None
    ) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "params": self.params,
            "metrics": metrics,
            "artifacts": artifacts or {},
            "feature_schema_hash": self.feature_schema_hash,
            "dataset_sha256": self.dataset_sha256,
            "git_sha": self.git_sha,
            # Issue #941 — full provenance fields
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "resolved_hyperparams": self.resolved_hyperparams or self.params,
        }


# ---------------------------------------------------------------------------
# JsonlExperimentTracker
# ---------------------------------------------------------------------------


class JsonlExperimentTracker:
    """Append-only experiment tracker with deterministic run identifiers."""

    def __init__(self, path: str | Path = "models/experiments.jsonl") -> None:
        self.path = Path(path)

    def log_run(
        self,
        run: ExperimentRun,
        metrics: dict[str, float],
        artifacts: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not metrics:
            raise ValueError("metrics must be non-empty")
        invalid_metrics = [
            name for name, value in metrics.items() if not isinstance(value, int | float)
        ]
        if invalid_metrics:
            raise TypeError(f"metrics must be numeric: {invalid_metrics}")

        record = run.to_record(
            metrics={k: float(v) for k, v in metrics.items()}, artifacts=artifacts
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_stable_json(record) + "\n")
        return record

    # Planned for Issue #858 (DANN training curves):
    #   def log_curve(self, run: ExperimentRun, name: str,
    #                 points: list[dict[str, float]]) -> dict[str, Any]
    # appends one record {"run_id", "type": "curve", "name", "points"} to the
    # same JSONL file. `points` are per-epoch dicts such as
    # {"epoch": 0, "lambda": 0.0, "task_loss": ..., "domain_loss": ...,
    #  "task_accuracy": ..., "domain_accuracy": ...}, and every value is
    # validated as a finite number, like `log_run` does for metrics.
    # `list_runs` skips records with type == "curve", so existing readers see
    # exactly what they see today; a new `list_curves(run_id)` returns them.
    def list_runs(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# Issue #941 — lineage lookup and CI promotion gate
# ---------------------------------------------------------------------------


def lookup_lineage(tracker: JsonlExperimentTracker, run_id: str) -> dict[str, Any]:
    """Return the full lineage record for a given run_id.

    This is the single-call lookup tool for incident forensics and audit.
    Given a ``run_id`` (available in ``model_metadata.json`` for every
    production model), it returns the complete provenance record:
    dataset snapshot, code commit hash, resolved hyperparameters, feature
    schema, and evaluation metrics.

    Parameters
    ----------
    tracker:
        A :class:`JsonlExperimentTracker` pointing at the experiment store.
    run_id:
        The run identifier to look up.

    Returns
    -------
    dict
        The full lineage record as logged by :meth:`JsonlExperimentTracker.log_run`.

    Raises
    ------
    KeyError
        If no run with the given ``run_id`` exists in the tracker store.

    Example
    -------
    During an incident investigation::

        from mlops.experiment_tracking import JsonlExperimentTracker, lookup_lineage
        tracker = JsonlExperimentTracker("models/experiments.jsonl")
        lineage = lookup_lineage(tracker, run_id="<run_id_from_model_metadata>")
        print(lineage["dataset_snapshot_id"])  # verify the training snapshot
        print(lineage["git_sha"])              # check out the exact code version
        print(lineage["resolved_hyperparams"]) # reproduce the training run
    """
    for record in tracker.list_runs():
        if record.get("run_id") == run_id:
            return record
    raise KeyError(
        f"No experiment run found with run_id={run_id!r}. "
        "Check models/experiments.jsonl or re-run training with lineage recording enabled."
    )


def check_lineage_complete(record: dict[str, Any]) -> list[str]:
    """Return a list of missing or empty provenance fields.

    Use this as a CI/release gate before promoting a model to production.
    An empty list means the lineage is complete.

    Parameters
    ----------
    record:
        A lineage record returned by :func:`lookup_lineage` or
        :meth:`JsonlExperimentTracker.log_run`.

    Returns
    -------
    list[str]
        Names of fields that are ``None``, empty, or absent.  An empty list
        means the record is complete and the model may be promoted.

    Example — blocking promotion in CI::

        errors = check_lineage_complete(lineage)
        if errors:
            raise SystemExit(
                f"Lineage incomplete — promotion blocked. Missing: {errors}"
            )
    """
    missing: list[str] = []
    for field_name in REQUIRED_LINEAGE_FIELDS:
        value = record.get(field_name)
        if value is None or value == "" or value == {}:
            missing.append(field_name)
    return missing
