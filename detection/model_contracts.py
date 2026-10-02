"""Lightweight contracts shared by model training and inference."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

FEATURE_COLUMNS_EXCLUDE = {"wallet", "label", "profile"}


def compute_feature_schema_hash(feature_columns: list[str]) -> str:
    """Compute a stable SHA-256 hash of the sorted feature names."""
    schema = "\n".join(sorted(feature_columns))
    return f"sha256:{hashlib.sha256(schema.encode()).hexdigest()}"


@dataclass(frozen=True)
class ModelContract:
    """Declared input/output schema for a registered model."""

    name: str
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]

    def input_schema_hash(self) -> str:
        return compute_feature_schema_hash(list(self.input_columns))

    def output_schema_hash(self) -> str:
        return compute_feature_schema_hash(list(self.output_columns))


@dataclass(frozen=True)
class ConformanceResult:
    """Outcome of checking one model against its declared contract."""

    model: str
    ok: bool
    errors: tuple[str, ...] = field(default_factory=tuple)

    def report(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        lines = [f"[{status}] {self.model}"]
        lines.extend(f"  - {err}" for err in self.errors)
        return "\n".join(lines)


class ContractMismatchError(AssertionError):
    """Raised when one or more models fail contract conformance."""

    def __init__(self, results: Iterable[ConformanceResult]) -> None:
        self.results = tuple(results)
        failures = [r for r in self.results if not r.ok]
        report = "\n".join(r.report() for r in self.results)
        super().__init__(
            f"{len(failures)} model contract mismatch(es):\n{report}"
        )


def _as_columns(value: Any) -> tuple[str, ...] | None:
    """Normalize a model's declared schema into a tuple of column names."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = value.get("columns", value.get("feature_columns"))
        if value is None:
            return None
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(col) for col in value)
    except TypeError:
        return None


def check_model_conformance(
    contract: ModelContract,
    model: Any,
) -> ConformanceResult:
    """Check a single model artifact against its declared contract."""
    errors: list[str] = []

    actual_input = _as_columns(
        getattr(model, "input_columns", None)
        or getattr(model, "feature_columns", None)
    )
    actual_output = _as_columns(getattr(model, "output_columns", None))

    if actual_input is None:
        errors.append("model does not declare input columns")
    elif actual_input != contract.input_columns:
        errors.append(
            "input schema mismatch: "
            f"expected {list(contract.input_columns)}, got {list(actual_input)}"
        )

    if actual_output is None:
        errors.append("model does not declare output columns")
    elif actual_output != contract.output_columns:
        errors.append(
            "output schema mismatch: "
            f"expected {list(contract.output_columns)}, got {list(actual_output)}"
        )

    return ConformanceResult(model=contract.name, ok=not errors, errors=tuple(errors))


def run_conformance_suite(
    contracts: Mapping[str, ModelContract],
    model_loader: Callable[[str], Any],
) -> list[ConformanceResult]:
    """Iterate over all registered models and check each against its contract.

    ``model_loader`` resolves a registered model name to its artifact. A loader
    failure is reported as a per-model error rather than aborting the suite so
    the full report is always produced.
    """
    results: list[ConformanceResult] = []
    for name, contract in contracts.items():
        try:
            model = model_loader(name)
        except Exception as exc:  # noqa: BLE001 - report, don't abort the suite
            results.append(
                ConformanceResult(
                    model=name,
                    ok=False,
                    errors=(f"failed to load model: {exc}",),
                )
            )
            continue
        results.append(check_model_conformance(contract, model))
    return results


def assert_conformance(results: Iterable[ConformanceResult]) -> None:
    """Raise ``ContractMismatchError`` with a per-model report on any failure."""
    results = tuple(results)
    if any(not r.ok for r in results):
        raise ContractMismatchError(results)
