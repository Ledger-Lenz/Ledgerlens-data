"""Adversarial robustness evaluation for the wash-trade detection ensemble.

Public API:
    EnsembleScoreFunction, feature_scale_from_matrix  -- scoring + scaling
    FGSMAttack, PGDAttack                             -- gradient attacks
    AdversarialEvaluator                              -- success rate + feature analysis
    adversarial_augmentation_gain                     -- robustness via adversarial training
"""

from detection.adversarial.attack import (
    EnsembleScoreFunction,
    FGSMAttack,
    GradientAttack,
    PGDAttack,
    feature_scale_from_matrix,
)
from detection.adversarial.augmentation import (
    adversarial_augmentation_gain,
    generate_adversarial_examples,
)
from detection.adversarial.evaluate import (
    AdversarialEvaluator,
    RobustnessSummary,
    evaluate_attack,
    min_epsilon_per_feature,
    most_vulnerable_features,
)

__all__ = [
    "EnsembleScoreFunction",
    "feature_scale_from_matrix",
    "GradientAttack",
    "FGSMAttack",
    "PGDAttack",
    "AdversarialEvaluator",
    "RobustnessSummary",
    "evaluate_attack",
    "min_epsilon_per_feature",
    "most_vulnerable_features",
    "adversarial_augmentation_gain",
    "generate_adversarial_examples",
]
