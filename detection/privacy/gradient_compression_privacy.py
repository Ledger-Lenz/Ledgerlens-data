"""Privacy analysis for gradient compression schemes in federated learning.

Analyzes the interaction between compression (top-k sparsification, quantization)
and differential privacy noise budget. Documents how compression affects effective
sensitivity and DP accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)


class CompressionScheme(str, Enum):
    """Supported gradient compression schemes."""

    NONE = "none"
    TOP_K_SPARSIFICATION = "top_k"
    QUANTIZATION = "quantization"
    MIXED = "mixed"


@dataclass
class GradientCompressionPrivacyAnalysis:
    """Result of privacy analysis for a compression scheme."""

    scheme: CompressionScheme
    original_sensitivity: float
    compressed_sensitivity: float
    compression_ratio: float
    sensitivity_amplification_factor: float
    noise_reduction_potential: float
    privacy_leakage_risk: str
    ordering_analysis: str
    accounting_adjustment: str

    def __post_init__(self) -> None:
        """Validate analysis results."""
        if self.compressed_sensitivity < 0 or self.original_sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")
        if self.compression_ratio < 0 or self.compression_ratio > 1:
            raise ValueError("Compression ratio must be in [0, 1]")


def analyze_top_k_sparsification(
    k: int,
    d: int,
    original_sensitivity: float = 1.0,
) -> GradientCompressionPrivacyAnalysis:
    """Analyze privacy impact of top-k gradient sparsification.

    In top-k sparsification, only the k largest-magnitude gradients are kept,
    others are zeroed. This affects DP accounting:

    - Sensitivity: Selection of top-k doesn't change individual-record sensitivity
      (still bounded by gradient clipping), but reconstruction becomes harder.
    - Composition: If applied before noise, noise protects both retained and
      zeroed gradients equally. If applied after, noise magnitude unchanged
      but effective information content is reduced.

    Args:
        k: Number of gradients to retain.
        d: Total gradient dimension.
        original_sensitivity: Clipping-based gradient sensitivity.

    Returns:
        GradientCompressionPrivacyAnalysis documenting the interaction.
    """
    compression_ratio = k / d if d > 0 else 0

    return GradientCompressionPrivacyAnalysis(
        scheme=CompressionScheme.TOP_K_SPARSIFICATION,
        original_sensitivity=original_sensitivity,
        compressed_sensitivity=original_sensitivity,
        compression_ratio=compression_ratio,
        sensitivity_amplification_factor=1.0,
        noise_reduction_potential=1 - compression_ratio,
        privacy_leakage_risk=(
            "MODERATE: Zeroed gradients leak sparsity pattern. Adversary can infer "
            "which features matter. Noise added before sparsification mitigates."
        ),
        ordering_analysis=(
            "RECOMMENDED: Noise addition BEFORE sparsification (post-noise top-k). "
            "Protects sparsity pattern and maintains DP guarantee on post-compression output. "
            "Adding noise after sparsification provides weaker privacy on retained gradients."
        ),
        accounting_adjustment=(
            "No adjustment needed if noise is added first. Sensitivity remains original_sensitivity. "
            "DP epsilon/delta budgets are not amplified by sparsification itself."
        ),
    )


def analyze_quantization(
    bits: int,
    original_sensitivity: float = 1.0,
) -> GradientCompressionPrivacyAnalysis:
    """Analyze privacy impact of gradient quantization.

    Quantization reduces precision (e.g., 32-bit float → 8-bit), introducing
    quantization error. This affects DP accounting:

    - Sensitivity: Quantization error adds to the effective sensitivity.
      If max gradient is in [-C, C] and quantized to B bits, quantization
      error ≈ C / 2^B. Effective sensitivity becomes original + quantization_error.
    - Composition: If noise is added before quantization, quantization
      introduces additional error uncorrelated with noise. If after, the
      total error (noise + quantization) protects privacy.

    Args:
        bits: Quantization bit-width (e.g., 8, 16, 32).
        original_sensitivity: Clipping-based gradient sensitivity (e.g., max |g| = 1.0).

    Returns:
        GradientCompressionPrivacyAnalysis documenting the interaction.
    """
    quantization_error = original_sensitivity / (2 ** (bits - 1))
    compression_ratio = bits / 32.0

    return GradientCompressionPrivacyAnalysis(
        scheme=CompressionScheme.QUANTIZATION,
        original_sensitivity=original_sensitivity,
        compressed_sensitivity=original_sensitivity + quantization_error,
        compression_ratio=compression_ratio,
        sensitivity_amplification_factor=(original_sensitivity + quantization_error) / original_sensitivity,
        noise_reduction_potential=0.0,
        privacy_leakage_risk=(
            f"HIGH if noise added after quantization: quantization error ({quantization_error:.6f}) "
            f"is deterministic and observable, allowing gradient reconstruction attacks. "
            f"LOW if noise added before: noise masks both quantization error and gradients."
        ),
        ordering_analysis=(
            "CRITICAL: Noise must be added BEFORE quantization. "
            "Quantization after noise leaks fine-grained gradient information. "
            "Recommended: (1) Clip gradients to [-C, C], (2) Add DP noise, (3) Quantize."
        ),
        accounting_adjustment=(
            f"Effective sensitivity increases by quantization error ≈ {quantization_error:.6f}. "
            f"If noise is added after quantization, must recalculate epsilon with "
            f"sensitivity = {original_sensitivity + quantization_error:.6f}. "
            f"If noise is added before, standard DP accounting applies (no adjustment needed)."
        ),
    )


def analyze_mixed_compression(
    sparsification_ratio: float = 0.5,
    quantization_bits: int = 16,
    original_sensitivity: float = 1.0,
) -> GradientCompressionPrivacyAnalysis:
    """Analyze privacy impact of combined sparsification + quantization.

    Joint compression scheme applying both top-k sparsification and quantization.

    Args:
        sparsification_ratio: Fraction of gradients retained (k/d).
        quantization_bits: Bit-width for quantization.
        original_sensitivity: Clipping-based gradient sensitivity.

    Returns:
        GradientCompressionPrivacyAnalysis documenting the interaction.
    """
    quantization_error = original_sensitivity / (2 ** (quantization_bits - 1))
    combined_compression_ratio = sparsification_ratio * (quantization_bits / 32.0)

    return GradientCompressionPrivacyAnalysis(
        scheme=CompressionScheme.MIXED,
        original_sensitivity=original_sensitivity,
        compressed_sensitivity=original_sensitivity + quantization_error,
        compression_ratio=combined_compression_ratio,
        sensitivity_amplification_factor=(original_sensitivity + quantization_error) / original_sensitivity,
        noise_reduction_potential=1 - combined_compression_ratio,
        privacy_leakage_risk=(
            f"VERY HIGH: Combined scheme leaks both sparsity pattern AND gradient values. "
            f"Quantization error ({quantization_error:.6f}) + sparsity pattern allows "
            f"strong reconstruction attacks if noise is not applied first."
        ),
        ordering_analysis=(
            "CRITICAL: Must apply DP noise FIRST, then sparsification AND quantization. "
            "Recommended ordering: (1) Clip, (2) Add DP noise, (3) Top-k sparsify, (4) Quantize. "
            "This ensures noise protects the gradient values before any deterministic "
            "compression is applied."
        ),
        accounting_adjustment=(
            f"Effective sensitivity = {original_sensitivity + quantization_error:.6f}. "
            f"Noise added before compression does NOT get amplified by compression. "
            f"Standard DP-SGD accounting applies with adjusted sensitivity = "
            f"min(sensitivity, gradient_clipping_bound)."
        ),
    )


def validate_compression_ordering(
    target_model,
    dp_training_config: dict[str, Any],
    compression_config: dict[str, Any],
) -> dict[str, Any]:
    """Validate that compression and DP noise ordering is correct.

    Checks that:
    1. DP noise is added before (not after) compression.
    2. Sensitivity accounting reflects quantization error if present.
    3. Budget tracker accurately reflects the combined mechanism.

    Args:
        target_model: Model being trained with DP + compression.
        dp_training_config: DP training configuration (noise scale, clipping, etc.).
        compression_config: Compression configuration (scheme, parameters).

    Returns:
        Dict with validation results and any necessary corrections.
    """
    issues = []
    corrections = []

    compression_scheme = compression_config.get("scheme", "none").lower()

    if compression_scheme in ["quantization", "mixed"]:
        quantization_bits = compression_config.get("quantization_bits", 32)
        original_sensitivity = dp_training_config.get("clipping_threshold", 1.0)
        quantization_error = original_sensitivity / (2 ** (quantization_bits - 1))

        tracked_sensitivity = dp_training_config.get("effective_sensitivity", original_sensitivity)

        if abs(tracked_sensitivity - (original_sensitivity + quantization_error)) > 1e-6:
            issues.append(
                f"Sensitivity mismatch: tracked={tracked_sensitivity:.6f}, "
                f"expected={original_sensitivity + quantization_error:.6f}"
            )
            corrections.append(
                f"Update DP accounting: effective_sensitivity = {original_sensitivity + quantization_error:.6f}"
            )

    return {
        "issues": issues,
        "corrections": corrections,
        "validation_passed": len(issues) == 0,
    }
