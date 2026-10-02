"""K-anonymity and l-diversity checks for exported reports.

Ensures exported statistics cannot be used for re-identification attacks by
verifying minimum cohort sizes and redacting/aggregating small-sample figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AnonymityCheckConfig:
    """Configuration for anonymity checks on exports."""

    k_anonymity_threshold: int = 5
    l_diversity_threshold: int = 2
    enable_redaction: bool = True
    audit_log_path: str | None = None


@dataclass
class RedactionAuditEntry:
    """Audit log entry for a redaction."""

    timestamp: str
    report_id: str
    statistic_name: str
    original_cohort_size: int
    k_anonymity_violation: bool
    l_diversity_violation: bool
    action_taken: str
    user: str


class AnonymityChecker:
    """Checker for k-anonymity and l-diversity in exported reports."""

    def __init__(self, config: AnonymityCheckConfig | None = None) -> None:
        """Initialize anonymity checker.

        Args:
            config: AnonymityCheckConfig with thresholds.
        """
        self.config = config or AnonymityCheckConfig()
        self.redaction_log: list[RedactionAuditEntry] = []

    def check_k_anonymity(self, cohort_size: int) -> bool:
        """Check if cohort size satisfies k-anonymity threshold.

        K-anonymity requires that any quasi-identifier appears in at least k records,
        preventing linking to fewer than k individuals.

        Args:
            cohort_size: Number of underlying entities in the aggregate.

        Returns:
            True if cohort_size >= k_anonymity_threshold, False otherwise.
        """
        return cohort_size >= self.config.k_anonymity_threshold

    def check_l_diversity(self, distinct_values: int) -> bool:
        """Check if attribute has sufficient l-diversity.

        L-diversity requires at least l distinct values for sensitive attributes,
        preventing inference of individual attribute values.

        Args:
            distinct_values: Number of distinct values in the sensitive attribute.

        Returns:
            True if distinct_values >= l_diversity_threshold, False otherwise.
        """
        return distinct_values >= self.config.l_diversity_threshold

    def validate_statistic(
        self,
        statistic_name: str,
        cohort_size: int,
        distinct_values: int | None = None,
        redact_if_violated: bool = True,
    ) -> tuple[bool, str]:
        """Validate a single statistic for anonymity compliance.

        Args:
            statistic_name: Name of the statistic being validated.
            cohort_size: Number of underlying entities.
            distinct_values: Number of distinct values (for l-diversity check).
            redact_if_violated: Whether to redact if checks fail.

        Returns:
            Tuple of (is_compliant, action_message).
        """
        k_passed = self.check_k_anonymity(cohort_size)
        l_passed = True if distinct_values is None else self.check_l_diversity(distinct_values)

        if k_passed and l_passed:
            return True, f"{statistic_name}: PASS (cohort_size={cohort_size}, distinct={distinct_values})"

        if not redact_if_violated:
            return False, f"{statistic_name}: FAIL (k={not k_passed}, l={not l_passed})"

        action_message = f"{statistic_name}: REDACTED due to "
        if not k_passed:
            action_message += f"k-anonymity violation (cohort_size={cohort_size} < {self.config.k_anonymity_threshold})"
        if not l_passed:
            if not k_passed:
                action_message += " and "
            action_message += f"l-diversity violation (distinct={distinct_values} < {self.config.l_diversity_threshold})"

        return False, action_message

    def validate_report(
        self,
        report_data: dict[str, Any],
        report_id: str,
        user: str = "system",
    ) -> dict[str, Any]:
        """Validate entire report for anonymity compliance.

        Checks all statistics in report_data for k-anonymity and l-diversity.

        Args:
            report_data: Dictionary with statistics and metadata.
                Expected to contain 'statistics' key with list of
                {'name': str, 'cohort_size': int, 'distinct_values': int, 'value': float}.
            report_id: Unique identifier for the report.
            user: User performing the export.

        Returns:
            Dict with validation results and redacted report if needed.
        """
        statistics = report_data.get("statistics", [])
        redactions = []
        compliance_issues = []

        for stat in statistics:
            stat_name = stat.get("name", "unknown")
            cohort_size = stat.get("cohort_size", 0)
            distinct_values = stat.get("distinct_values")

            is_compliant, message = self.validate_statistic(
                stat_name,
                cohort_size,
                distinct_values,
                redact_if_violated=self.config.enable_redaction,
            )

            logger.info(message)

            if not is_compliant:
                compliance_issues.append(message)
                if self.config.enable_redaction:
                    k_violated = not self.check_k_anonymity(cohort_size)
                    l_violated = distinct_values is not None and not self.check_l_diversity(distinct_values)

                    audit_entry = RedactionAuditEntry(
                        timestamp=datetime.now(UTC).isoformat(),
                        report_id=report_id,
                        statistic_name=stat_name,
                        original_cohort_size=cohort_size,
                        k_anonymity_violation=k_violated,
                        l_diversity_violation=l_violated,
                        action_taken="redaction",
                        user=user,
                    )
                    self.redaction_log.append(audit_entry)
                    redactions.append(stat_name)

        report_compliant = len(compliance_issues) == 0

        return {
            "report_id": report_id,
            "compliant": report_compliant,
            "issues": compliance_issues,
            "redacted_statistics": redactions,
            "audit_entries": len(self.redaction_log),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def get_redaction_audit_log(self) -> list[dict[str, Any]]:
        """Get audit log of all redactions performed.

        Returns:
            List of redaction audit entries as dictionaries.
        """
        return [
            {
                "timestamp": entry.timestamp,
                "report_id": entry.report_id,
                "statistic_name": entry.statistic_name,
                "original_cohort_size": entry.original_cohort_size,
                "k_anonymity_violation": entry.k_anonymity_violation,
                "l_diversity_violation": entry.l_diversity_violation,
                "action_taken": entry.action_taken,
                "user": entry.user,
            }
            for entry in self.redaction_log
        ]

    def redact_statistic(self, value: float, cohort_size: int) -> str:
        """Redact a statistic value based on cohort size.

        Args:
            value: Original statistic value.
            cohort_size: Number of underlying entities.

        Returns:
            Redacted representation of the statistic.
        """
        if not self.check_k_anonymity(cohort_size):
            return f"REDACTED (n<{self.config.k_anonymity_threshold})"
        return str(value)

    def aggregate_further(
        self,
        statistics: list[dict[str, Any]],
        min_cohort_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Aggregate statistics further to meet anonymity thresholds.

        Combines statistics with small cohorts into larger aggregates.

        Args:
            statistics: List of statistics with cohort_size metadata.
            min_cohort_size: Minimum cohort size (defaults to k_anonymity_threshold).

        Returns:
            List of aggregated statistics meeting anonymity threshold.
        """
        min_cohort = min_cohort_size or self.config.k_anonymity_threshold
        aggregated = []
        small_cohort_buffer = []

        for stat in statistics:
            cohort_size = stat.get("cohort_size", 0)

            if cohort_size >= min_cohort:
                if small_cohort_buffer:
                    combined = self._combine_statistics(small_cohort_buffer)
                    if combined.get("cohort_size", 0) >= min_cohort:
                        aggregated.append(combined)
                    small_cohort_buffer = []

                aggregated.append(stat)
            else:
                small_cohort_buffer.append(stat)

        if small_cohort_buffer:
            combined = self._combine_statistics(small_cohort_buffer)
            if combined.get("cohort_size", 0) >= min_cohort:
                aggregated.append(combined)

        return aggregated

    def _combine_statistics(self, statistics: list[dict[str, Any]]) -> dict[str, Any]:
        """Combine multiple statistics into a single aggregate.

        Args:
            statistics: List of statistics to combine.

        Returns:
            Combined statistic with summed cohort size and aggregated name.
        """
        if not statistics:
            return {}

        total_cohort = sum(s.get("cohort_size", 0) for s in statistics)
        combined_name = "AGGREGATED_" + "_".join(s.get("name", "unknown") for s in statistics)

        return {
            "name": combined_name,
            "cohort_size": total_cohort,
            "value": "aggregated",
            "original_count": len(statistics),
        }
