"""Audit of differential-privacy composition in meta-learning training loops.

Verifies that privacy budget accounting correctly captures all gradient computations
across meta-learning inner and outer loop steps, preventing silent under-accounting
of total privacy loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetaLearnerDPAuditConfig:
    """Configuration for meta-learning DP audit."""

    inner_steps: int
    outer_steps: int
    inner_lr: float
    outer_lr: float
    noise_multiplier: float
    sampling_rate: float
    delta: float = 1e-5


@dataclass
class MetaLearnerCompositionAnalysis:
    """Analysis of privacy budget composition in meta-learning."""

    config: MetaLearnerDPAuditConfig
    total_gradient_accesses: int
    inner_loop_gradient_steps: int
    outer_loop_gradient_steps: int
    privacy_accounted_for: bool
    composition_method: str
    total_epsilon: float
    notes: str


class MetaLearnerDPAuditor:
    """Auditor for meta-learning differential privacy composition."""

    def __init__(self) -> None:
        """Initialize meta-learner DP auditor."""
        self.audit_log = []

    def analyze_training_loop(
        self,
        config: MetaLearnerDPAuditConfig,
    ) -> MetaLearnerCompositionAnalysis:
        """Analyze privacy composition across meta-learning training loop.

        Meta-learning (MAML-style) structure:
        - Outer loop: T_outer steps updating model parameters φ
        - Inner loop: For each outer step, take T_inner gradient steps on
          task-specific data to compute adaptation gradients
        - Total gradient accesses: T_outer * T_inner + T_outer (inner + outer)

        All gradient computations that touch private data must be accounted for
        in privacy composition.

        Args:
            config: Meta-learning DP configuration.

        Returns:
            MetaLearnerCompositionAnalysis documenting composition structure.
        """
        inner_steps = config.inner_steps
        outer_steps = config.outer_steps

        inner_loop_gradient_steps = outer_steps * inner_steps
        outer_loop_gradient_steps = outer_steps

        total_gradient_accesses = inner_loop_gradient_steps + outer_loop_gradient_steps

        epsilon_per_access = self._compute_epsilon_per_access(config)
        total_epsilon = epsilon_per_access * total_gradient_accesses

        composition_method = (
            f"Composition over {total_gradient_accesses} gradient accesses: "
            f"{inner_loop_gradient_steps} inner-loop + {outer_loop_gradient_steps} outer-loop"
        )

        notes = (
            f"Inner loop: {inner_steps} steps × {outer_steps} outer iterations = {inner_loop_gradient_steps} gradient accesses. "
            f"Outer loop: {outer_steps} gradient accesses. "
            f"Total: {total_gradient_accesses} accesses to private data. "
            f"Privacy composition must account for all {total_gradient_accesses} steps, not just outer loop. "
            f"Total epsilon: {total_epsilon:.6f}."
        )

        analysis = MetaLearnerCompositionAnalysis(
            config=config,
            total_gradient_accesses=total_gradient_accesses,
            inner_loop_gradient_steps=inner_loop_gradient_steps,
            outer_loop_gradient_steps=outer_loop_gradient_steps,
            privacy_accounted_for=True,
            composition_method=composition_method,
            total_epsilon=total_epsilon,
            notes=notes,
        )

        self.audit_log.append(
            {
                "config": config,
                "analysis": analysis,
                "timestamp": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
            }
        )

        return analysis

    def _compute_epsilon_per_access(self, config: MetaLearnerDPAuditConfig) -> float:
        """Compute epsilon per gradient access using DP-SGD accounting.

        Args:
            config: Meta-learning DP configuration.

        Returns:
            Epsilon per single gradient step.
        """
        noise_multiplier = config.noise_multiplier
        sampling_rate = config.sampling_rate

        log_delta_inv = 1 / config.delta if config.delta > 0 else 1e5

        epsilon_per_step = (2 * (sampling_rate ** 0.5) * (log_delta_inv ** 0.5)) / noise_multiplier

        return epsilon_per_step

    def verify_composition_accounting(
        self,
        implementation_epsilon: float,
        config: MetaLearnerDPAuditConfig,
    ) -> dict[str, Any]:
        """Verify that implementation's epsilon accounting matches composition analysis.

        Args:
            implementation_epsilon: Epsilon value from the actual implementation.
            config: Meta-learning DP configuration.

        Returns:
            Dict with verification results and any discrepancies.
        """
        analysis = self.analyze_training_loop(config)

        expected_epsilon = analysis.total_epsilon
        error_pct = abs(implementation_epsilon - expected_epsilon) / expected_epsilon * 100 if expected_epsilon > 0 else 0

        is_conservative = implementation_epsilon >= expected_epsilon
        passed = error_pct < 5 and is_conservative

        issues = []
        if error_pct >= 5:
            issues.append(
                f"Epsilon discrepancy: implementation={implementation_epsilon:.6f}, "
                f"expected={expected_epsilon:.6f}, error={error_pct:.2f}%"
            )

        if not is_conservative:
            issues.append(
                f"Non-conservative accounting: implementation epsilon {implementation_epsilon:.6f} "
                f"< expected {expected_epsilon:.6f}"
            )

        return {
            "passed": passed,
            "issues": issues,
            "implementation_epsilon": implementation_epsilon,
            "expected_epsilon": expected_epsilon,
            "error_pct": error_pct,
            "is_conservative": is_conservative,
            "analysis": analysis,
        }

    def document_composition(self, config: MetaLearnerDPAuditConfig) -> str:
        """Generate documentation of privacy composition for meta-learning.

        Args:
            config: Meta-learning DP configuration.

        Returns:
            Markdown documentation of the composition structure.
        """
        analysis = self.analyze_training_loop(config)

        doc = f"""# Meta-Learning Differential Privacy Composition

## Training Loop Structure

### Inner Loop
- **Steps per outer iteration**: {config.inner_steps}
- **Total inner-loop gradient accesses**: {analysis.inner_loop_gradient_steps}
  - Calculated as: {config.outer_steps} outer iterations × {config.inner_steps} inner steps

### Outer Loop
- **Outer loop steps**: {config.outer_steps}
- **Outer-loop gradient accesses**: {analysis.outer_loop_gradient_steps}

### Total
- **Total gradient accesses touching private data**: {analysis.total_gradient_accesses}
- **Composition method**: {analysis.composition_method}

## Privacy Accounting

### Per-Access Privacy Budget
- Noise multiplier: {config.noise_multiplier}
- Sampling rate: {config.sampling_rate}
- Per-access epsilon: ≈ {self._compute_epsilon_per_access(config):.6f}

### Total Privacy Budget
- **Total epsilon**: {analysis.total_epsilon:.6f}
- **Delta**: {config.delta}

## Key Points

1. **All gradient accesses must be composed**: Both inner-loop adaptation steps and
   outer-loop model updates touch private data and must be counted in privacy accounting.

2. **Avoid double-counting composition**: Each gradient step contributes to privacy loss
   exactly once. Naive composition can underestimate if it only counts outer-loop steps.

3. **Implementation verification**: Ensure that the implementation's epsilon accounting
   reflects all {analysis.total_gradient_accesses} gradient accesses, not just outer-loop steps.

## Verification

- Inner-loop gradient steps: {analysis.inner_loop_gradient_steps}
- Outer-loop gradient steps: {analysis.outer_loop_gradient_steps}
- All accounted for: {analysis.privacy_accounted_for}
"""
        return doc
