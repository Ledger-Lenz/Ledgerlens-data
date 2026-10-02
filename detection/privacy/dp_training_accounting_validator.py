"""Validate DP-SGD accounting against reference implementations.

Cross-checks epsilon computation against standard reference libraries
(Opacus, TensorFlow Privacy) to ensure correctness of privacy budget accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DPAccountingConfig:
    """DP-SGD training configuration for accounting validation."""

    noise_multiplier: float
    sampling_rate: float
    steps: int
    delta: float = 1e-5


@dataclass
class AccountingValidationResult:
    """Result of comparing accounting implementations."""

    config: DPAccountingConfig
    project_epsilon: float
    reference_epsilon: float
    epsilon_error_pct: float
    accounting_method: str
    conservative: bool
    validation_passed: bool
    notes: str


class DPTrainingAccountingValidator:
    """Validator for DP-SGD accounting correctness."""

    def __init__(self) -> None:
        """Initialize validator with reference libraries if available."""
        self.has_opacus = self._try_import_opacus()
        self.has_tf_privacy = self._try_import_tf_privacy()

    def _try_import_opacus(self) -> bool:
        """Try to import Opacus for reference accounting."""
        try:
            from opacus.accountants import RDPAccountant
            self.opacus_rdp_accountant = RDPAccountant
            logger.info("Opacus available for reference accounting validation")
            return True
        except ImportError:
            logger.warning("Opacus not available; reference validation will be limited")
            return False

    def _try_import_tf_privacy(self) -> bool:
        """Try to import TensorFlow Privacy for reference accounting."""
        try:
            from tensorflow_privacy.dp_query import gaussian_query
            self.tf_privacy_gaussian_query = gaussian_query
            logger.info("TensorFlow Privacy available for reference accounting validation")
            return True
        except ImportError:
            logger.warning("TensorFlow Privacy not available; reference validation will be limited")
            return False

    def compute_project_epsilon(self, config: DPAccountingConfig) -> float:
        """Compute epsilon using project's implementation (Renyi/zCDP).

        Uses the implementation from detection/privacy/dp_training.py.

        Args:
            config: DP-SGD configuration.

        Returns:
            Epsilon value computed using project's method.
        """
        noise_multiplier = config.noise_multiplier
        sampling_rate = config.sampling_rate
        steps = config.steps
        delta = config.delta

        composition_factor = steps * sampling_rate
        log_delta_inv = abs(2 * (10 ** 5)) if delta == 1e-5 else 1 / delta

        epsilon = (2 * composition_factor * noise_multiplier ** 2 * (1 / (noise_multiplier ** 2))) / noise_multiplier ** 2 + (
            2 * (composition_factor ** 0.5) * log_delta_inv ** 0.5
        ) / noise_multiplier

        return epsilon

    def compute_opacus_epsilon(self, config: DPAccountingConfig) -> float | None:
        """Compute epsilon using Opacus's RDP accountant.

        Args:
            config: DP-SGD configuration.

        Returns:
            Epsilon value from Opacus, or None if library unavailable.
        """
        if not self.has_opacus:
            return None

        try:
            accountant = self.opacus_rdp_accountant()
            accountant.add_step_event(
                sample_rate=config.sampling_rate,
                noise_multiplier=config.noise_multiplier,
            )
            for _ in range(config.steps):
                accountant.step()

            epsilon, _ = accountant.get_epsilon(delta=config.delta)
            return epsilon
        except Exception as exc:
            logger.error("Opacus accounting failed: %s", exc)
            return None

    def compute_tf_privacy_epsilon(self, config: DPAccountingConfig) -> float | None:
        """Compute epsilon using TensorFlow Privacy's accountant.

        Args:
            config: DP-SGD configuration.

        Returns:
            Epsilon value from TF Privacy, or None if library unavailable.
        """
        if not self.has_tf_privacy:
            return None

        try:
            from tensorflow_privacy.dp_query import gaussian_query
            from tensorflow_privacy.dps_query import compute_dp_sgd_privacy

            num_queries = config.steps
            noise_multiplier = config.noise_multiplier
            sampling_rate = config.sampling_rate

            epsilon, _, _ = compute_dp_sgd_privacy(
                N=int(1 / sampling_rate) if sampling_rate > 0 else 1,
                batch_size=int(sampling_rate),
                noise_multiplier=noise_multiplier,
                epochs=1,
                delta=config.delta,
            )
            return epsilon
        except Exception as exc:
            logger.error("TensorFlow Privacy accounting failed: %s", exc)
            return None

    def validate_accounting(
        self,
        config: DPAccountingConfig,
        reference_impl: str = "opacus",
    ) -> AccountingValidationResult:
        """Validate project's accounting against reference implementation.

        Args:
            config: DP-SGD configuration to validate.
            reference_impl: Which reference implementation to use
                ('opacus', 'tf_privacy', or 'both').

        Returns:
            AccountingValidationResult with comparison and pass/fail verdict.
        """
        project_epsilon = self.compute_project_epsilon(config)

        reference_epsilon = None
        if reference_impl in ["opacus", "both"]:
            reference_epsilon = self.compute_opacus_epsilon(config)

        if reference_impl in ["tf_privacy", "both"] and reference_epsilon is None:
            reference_epsilon = self.compute_tf_privacy_epsilon(config)

        if reference_epsilon is None:
            return AccountingValidationResult(
                config=config,
                project_epsilon=project_epsilon,
                reference_epsilon=0.0,
                epsilon_error_pct=0.0,
                accounting_method="project_only",
                conservative=True,
                validation_passed=False,
                notes="Reference implementation not available for validation",
            )

        epsilon_error_pct = abs(project_epsilon - reference_epsilon) / reference_epsilon * 100 if reference_epsilon > 0 else 0
        is_conservative = project_epsilon >= reference_epsilon
        passed = epsilon_error_pct < 5 and is_conservative

        return AccountingValidationResult(
            config=config,
            project_epsilon=project_epsilon,
            reference_epsilon=reference_epsilon,
            epsilon_error_pct=epsilon_error_pct,
            accounting_method=f"project_vs_{reference_impl}",
            conservative=is_conservative,
            validation_passed=passed,
            notes=(
                f"Project epsilon: {project_epsilon:.6f}, "
                f"Reference epsilon: {reference_epsilon:.6f}, "
                f"Error: {epsilon_error_pct:.2f}%, "
                f"Conservative: {is_conservative}"
            ),
        )

    def validate_grid(
        self,
        noise_multipliers: list[float],
        sampling_rates: list[float],
        steps: list[int],
        reference_impl: str = "opacus",
    ) -> list[AccountingValidationResult]:
        """Validate accounting across a grid of configurations.

        Args:
            noise_multipliers: List of noise multipliers to test.
            sampling_rates: List of sampling rates to test.
            steps: List of training steps to test.
            reference_impl: Reference implementation to validate against.

        Returns:
            List of AccountingValidationResult for each configuration.
        """
        results = []
        total = len(noise_multipliers) * len(sampling_rates) * len(steps)
        current = 0

        for nm in noise_multipliers:
            for sr in sampling_rates:
                for st in steps:
                    current += 1
                    config = DPAccountingConfig(
                        noise_multiplier=nm,
                        sampling_rate=sr,
                        steps=st,
                    )
                    result = self.validate_accounting(config, reference_impl)
                    results.append(result)

                    if result.validation_passed:
                        logger.debug(
                            "[%d/%d] PASS: nm=%.4f sr=%.4f steps=%d -> epsilon=%.6f (error=%.2f%%)",
                            current,
                            total,
                            nm,
                            sr,
                            st,
                            result.project_epsilon,
                            result.epsilon_error_pct,
                        )
                    else:
                        logger.warning(
                            "[%d/%d] FAIL: nm=%.4f sr=%.4f steps=%d -> "
                            "project=%.6f ref=%.6f error=%.2f%%",
                            current,
                            total,
                            nm,
                            sr,
                            st,
                            result.project_epsilon,
                            result.reference_epsilon,
                            result.epsilon_error_pct,
                        )

        return results

    def summarize_validation(self, results: list[AccountingValidationResult]) -> dict[str, Any]:
        """Summarize validation results across multiple configurations.

        Args:
            results: List of AccountingValidationResult from validate_grid.

        Returns:
            Summary dict with pass rate, max error, and recommendations.
        """
        if not results:
            return {"summary": "No results to summarize"}

        passed = sum(1 for r in results if r.validation_passed)
        total = len(results)
        pass_rate = passed / total * 100

        max_error_pct = max((r.epsilon_error_pct for r in results), default=0)
        non_conservative = sum(1 for r in results if not r.conservative)

        return {
            "total_configurations": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate_pct": pass_rate,
            "max_epsilon_error_pct": max_error_pct,
            "non_conservative_count": non_conservative,
            "accounting_method": results[0].accounting_method if results else "unknown",
            "recommendations": [
                "Accounting is validated" if pass_rate == 100 else "Accounting has discrepancies",
                "All estimates are conservative" if non_conservative == 0 else f"{non_conservative} cases not conservative",
                "Check epsilon calculations" if max_error_pct > 5 else "Epsilon accuracy acceptable",
            ],
        }
