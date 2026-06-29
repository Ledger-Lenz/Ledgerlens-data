"""Adversarial robustness tooling for the wash-trade detection ensemble.

Provides gradient-based evasion attacks (`FGSMAttack`, `PGDAttack`) that
estimate how much an operator would have to perturb their on-chain feature
footprint to push the LedgerLens Risk Score below the alert threshold, plus
the evaluation (`robustness`) and adversarial-augmentation (`augmentation`)
helpers used by `scripts/run_adversarial_eval.py`.

Gradients are estimated by finite differences against
`RiskScorer.score_continuous` because the ensemble is tree-based (no
analytic gradient) and the public `score` is integer-rounded.
"""

from detection.adversarial.attack import FGSMAttack, PGDAttack

__all__ = ["FGSMAttack", "PGDAttack"]
