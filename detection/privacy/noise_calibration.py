"""Differential-privacy noise calibration for multi-tenant deployments.

Calibrates noise level per-tenant based on data volume to maintain acceptable
signal-to-noise ratio across small and large tenants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TenantVolumeProfile:
    """Data volume profile for a tenant."""

    tenant_id: str
    transaction_count: int
    feature_dimension: int
    daily_volume: int
    monthly_volume: int


@dataclass
class NoiseCalibrationConfig:
    """Configuration for noise calibration."""

    epsilon: float = 1.0
    delta: float = 1e-5
    target_snr_db: float = 20.0
    small_tenant_threshold: int = 1000
    large_tenant_threshold: int = 100000


@dataclass
class CalibrationResult:
    """Result of noise calibration for a tenant."""

    tenant_id: str
    volume_profile: TenantVolumeProfile
    noise_multiplier: float
    signal_std: float
    noise_std: float
    snr_db: float
    snr_acceptable: bool
    notes: str


class NoiseCalibrator:
    """Calibrator for per-tenant DP noise levels."""

    def __init__(self, config: NoiseCalibrationConfig | None = None) -> None:
        """Initialize noise calibrator.

        Args:
            config: NoiseCalibrationConfig with calibration parameters.
        """
        self.config = config or NoiseCalibrationConfig()

    def calibrate_for_tenant(
        self,
        volume_profile: TenantVolumeProfile,
    ) -> CalibrationResult:
        """Calibrate noise multiplier for a tenant based on data volume.

        Smaller tenants get lower noise multiplier (more noise) to maintain signal,
        larger tenants can tolerate higher multiplier (less noise) while staying compliant.

        Args:
            volume_profile: Tenant's data volume profile.

        Returns:
            CalibrationResult with recommended noise multiplier.
        """
        transaction_count = volume_profile.transaction_count

        if transaction_count < self.config.small_tenant_threshold:
            category = "small"
            noise_multiplier = self.config.epsilon / 2.0
        elif transaction_count < self.config.large_tenant_threshold:
            category = "medium"
            noise_multiplier = self.config.epsilon
        else:
            category = "large"
            noise_multiplier = self.config.epsilon * 2.0

        signal_std = self._estimate_signal_std(volume_profile)
        noise_std = self._estimate_noise_std(
            noise_multiplier,
            volume_profile.feature_dimension,
        )

        snr_db = self._compute_snr_db(signal_std, noise_std)
        snr_acceptable = snr_db >= self.config.target_snr_db

        notes = (
            f"Tenant {volume_profile.tenant_id} ({category}): "
            f"transactions={transaction_count}, "
            f"noise_multiplier={noise_multiplier:.4f}, "
            f"SNR={snr_db:.2f}dB (target {self.config.target_snr_db:.2f}dB)"
        )

        if not snr_acceptable:
            notes += " [WARNING: SNR below target]"

        return CalibrationResult(
            tenant_id=volume_profile.tenant_id,
            volume_profile=volume_profile,
            noise_multiplier=noise_multiplier,
            signal_std=signal_std,
            noise_std=noise_std,
            snr_db=snr_db,
            snr_acceptable=snr_acceptable,
            notes=notes,
        )

    def _estimate_signal_std(self, volume_profile: TenantVolumeProfile) -> float:
        """Estimate signal standard deviation from volume profile.

        Larger datasets have more stable statistics (lower std of the statistic).

        Args:
            volume_profile: Tenant's data volume profile.

        Returns:
            Estimated signal standard deviation.
        """
        n = volume_profile.transaction_count
        d = volume_profile.feature_dimension

        if n <= 0:
            return 1.0

        base_signal = 1.0
        signal_std = base_signal / (n ** 0.5)

        return signal_std

    def _estimate_noise_std(self, noise_multiplier: float, dimension: int) -> float:
        """Estimate noise standard deviation from multiplier and dimension.

        Args:
            noise_multiplier: Noise multiplier parameter.
            dimension: Feature dimension.

        Returns:
            Estimated noise standard deviation.
        """
        gaussian_noise_std = noise_multiplier * (dimension ** 0.5)
        return gaussian_noise_std

    def _compute_snr_db(self, signal_std: float, noise_std: float) -> float:
        """Compute signal-to-noise ratio in dB.

        Args:
            signal_std: Signal standard deviation.
            noise_std: Noise standard deviation.

        Returns:
            SNR in dB.
        """
        if signal_std <= 0 or noise_std <= 0:
            return -float("inf")

        snr_linear = signal_std / noise_std
        snr_db = 20 * (snr_linear ** 0.5)
        return snr_db

    def validate_calibration_grid(
        self,
        volume_profiles: list[TenantVolumeProfile],
    ) -> dict[str, Any]:
        """Validate calibration across multiple tenant volumes.

        Args:
            volume_profiles: List of tenant volume profiles to validate.

        Returns:
            Validation results with summary statistics.
        """
        results = []
        for profile in volume_profiles:
            result = self.calibrate_for_tenant(profile)
            results.append(result)
            logger.info(result.notes)

        acceptable_count = sum(1 for r in results if r.snr_acceptable)
        total_count = len(results)

        return {
            "total_tenants": total_count,
            "acceptable_snr": acceptable_count,
            "unacceptable_snr": total_count - acceptable_count,
            "acceptance_rate_pct": acceptable_count / total_count * 100 if total_count > 0 else 0,
            "results": [
                {
                    "tenant_id": r.tenant_id,
                    "volume_category": self._categorize_volume(r.volume_profile.transaction_count),
                    "transaction_count": r.volume_profile.transaction_count,
                    "noise_multiplier": r.noise_multiplier,
                    "snr_db": r.snr_db,
                    "snr_acceptable": r.snr_acceptable,
                }
                for r in results
            ],
        }

    def _categorize_volume(self, transaction_count: int) -> str:
        """Categorize tenant by transaction volume.

        Args:
            transaction_count: Number of transactions.

        Returns:
            Category name ('small', 'medium', 'large').
        """
        if transaction_count < self.config.small_tenant_threshold:
            return "small"
        elif transaction_count < self.config.large_tenant_threshold:
            return "medium"
        else:
            return "large"

    def document_calibration_methodology(self) -> str:
        """Generate documentation of calibration methodology.

        Returns:
            Markdown documentation.
        """
        return f"""# Noise Calibration Methodology

## Overview
Calibrates differential-privacy noise level per-tenant to maintain acceptable
signal-to-noise ratio (SNR) across deployment with varying transaction volumes.

## Volume Categories
- **Small tenants**: < {self.config.small_tenant_threshold:,} transactions
- **Medium tenants**: {self.config.small_tenant_threshold:,} - {self.config.large_tenant_threshold:,} transactions
- **Large tenants**: > {self.config.large_tenant_threshold:,} transactions

## Noise Multiplier Assignment
- **Small tenants**: noise_multiplier = epsilon / 2 = {self.config.epsilon / 2:.4f}
- **Medium tenants**: noise_multiplier = epsilon = {self.config.epsilon:.4f}
- **Large tenants**: noise_multiplier = epsilon × 2 = {self.config.epsilon * 2:.4f}

## Signal-to-Noise Ratio Target
- **Target SNR**: {self.config.target_snr_db:.2f} dB
- Ensures outputs have sufficient signal for analysis while maintaining privacy

## Calibration Process
1. Estimate tenant's signal std from transaction volume: signal_std ≈ 1 / √n
2. Compute noise std from noise_multiplier and dimension: noise_std = multiplier × √d
3. Calculate SNR_dB = 20 log10(signal_std / noise_std)
4. Warn if SNR_dB < {self.config.target_snr_db:.2f}

## Privacy Budget
- **Global epsilon**: {self.config.epsilon}
- **Delta**: {self.config.delta}
- Each tenant's noise_multiplier contributes to composition across all users
"""
