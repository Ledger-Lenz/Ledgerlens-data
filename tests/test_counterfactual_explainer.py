"""Tests for Issue #193: Counterfactual Explanation Generator.

Covers:
  - All returned counterfactuals produce predicted scores below the flag threshold.
  - Counterfactuals differ from each other by at least 1 feature value (diversity).
  - Counterfactual output does not contain model weights or training data.
  - Feature constraints: non-negative features stay >= 0 in all CFs.
  - Immutable features are not modified in any counterfactual.
  - Generation completes in < 10 seconds per wallet.
  - CounterfactualResult.to_dict() is JSON-serialisable.
  - Wallet below threshold returns empty CF list immediately.
  - interpret_action produces non-empty strings for known features.
  - ForensicReport.counterfactual_result field is serialised correctly.
"""

import json
import time

import numpy as np
import pandas as pd
import pytest

from detection.counterfactual_explainer import (
    IMMUTABLE_FEATURES,
    NON_NEGATIVE_FEATURES,
    CounterfactualExplainer,
    CounterfactualResult,
    _interpret_action,
)
from detection.model_inference import RiskScorer
from detection.model_training import save_models, train_models
from scripts.generate_synthetic_dataset import generate_synthetic_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scorer_and_data(tmp_path_factory):
    df = generate_synthetic_dataset(n_wallets=150, seed=42)
    results = train_models(df, test_size=0.2, random_state=42)
    model_dir = str(tmp_path_factory.mktemp("cf_models"))
    save_models(results, model_dir)
    scorer = RiskScorer(model_dir=model_dir)
    return scorer, df


@pytest.fixture(scope="module")
def explainer(scorer_and_data):
    scorer, df = scorer_and_data
    X_train = df.drop(columns=["label", "wallet"], errors="ignore")
    return CounterfactualExplainer(
        scorer,
        X_train,
        flag_threshold=70.0,
        n_cfs=5,
        timeout_seconds=10.0,
        random_state=42,
    )


def _high_risk_row(scorer, df):
    """Return the feature row with the highest risk score."""
    feature_df = df.drop(columns=["label"])
    scores = feature_df.apply(scorer.score_continuous, axis=1)
    return feature_df.loc[scores.idxmax()]


# ---------------------------------------------------------------------------
# Core acceptance criteria (Issue #193)
# ---------------------------------------------------------------------------


class TestCounterfactualScoresBelowThreshold:
    def test_all_cfs_score_below_flag_threshold(self, scorer_and_data, explainer):
        """All returned counterfactuals must produce scores below the flag threshold."""
        scorer, df = scorer_and_data
        row = _high_risk_row(scorer, df)
        result = explainer.explain(row, wallet="test_wallet")

        assert result.original_score >= 70.0, (
            "Test row should have a high risk score; got "
            f"{result.original_score:.1f}. Pick a more suspicious wallet."
        )

        for cf in result.counterfactuals:
            actual_score = scorer.score_continuous(
                pd.Series(cf.feature_values)
            )
            assert actual_score < explainer.flag_threshold, (
                f"CF {cf.cf_index} scores {actual_score:.1f} which is above "
                f"the flag threshold {explainer.flag_threshold}"
            )

    def test_diversity_at_least_one_feature_differs(self, scorer_and_data, explainer):
        """Each counterfactual must differ from every other CF by at least 1 feature."""
        scorer, df = scorer_and_data
        row = _high_risk_row(scorer, df)
        result = explainer.explain(row)

        if len(result.counterfactuals) < 2:
            pytest.skip("Need at least 2 CFs for diversity test")

        for i, cf_a in enumerate(result.counterfactuals):
            for cf_b in result.counterfactuals[i + 1 :]:
                differing = sum(
                    1
                    for k in cf_a.feature_values
                    if k in cf_b.feature_values
                    and abs(cf_a.feature_values[k] - cf_b.feature_values[k]) > 1e-6
                )
                assert differing >= 1, (
                    f"CF {cf_a.cf_index} and CF {cf_b.cf_index} are identical — "
                    "diversity constraint violated"
                )


class TestFeatureConstraints:
    def test_non_negative_features_stay_non_negative(self, scorer_and_data, explainer):
        """Non-negative features must remain >= 0 in all counterfactual vectors."""
        scorer, df = scorer_and_data
        row = _high_risk_row(scorer, df)
        result = explainer.explain(row)

        for cf in result.counterfactuals:
            for col, val in cf.feature_values.items():
                if col in NON_NEGATIVE_FEATURES:
                    assert val >= 0.0, (
                        f"CF {cf.cf_index}: feature '{col}' = {val} must be >= 0"
                    )

    def test_immutable_features_not_modified(self, scorer_and_data, explainer):
        """Immutable features must not appear in the action list of any CF."""
        scorer, df = scorer_and_data
        row = _high_risk_row(scorer, df)
        result = explainer.explain(row)

        for cf in result.counterfactuals:
            for action in cf.actions:
                assert action.feature not in IMMUTABLE_FEATURES, (
                    f"CF {cf.cf_index} modifies immutable feature '{action.feature}'"
                )

    def test_immutable_features_values_unchanged(self, scorer_and_data, explainer):
        """Immutable feature values must be identical to the original row in every CF."""
        scorer, df = scorer_and_data
        row = _high_risk_row(scorer, df)
        result = explainer.explain(row)

        for cf in result.counterfactuals:
            for col in IMMUTABLE_FEATURES:
                if col in row.index and col in cf.feature_values:
                    assert abs(cf.feature_values[col] - float(row[col])) < 1e-9, (
                        f"CF {cf.cf_index}: immutable feature '{col}' changed from "
                        f"{float(row[col])} to {cf.feature_values[col]}"
                    )


class TestPerformance:
    def test_generation_completes_within_10_seconds(self, scorer_and_data, explainer):
        """CF generation must complete in < 10 seconds per wallet (Issue #193)."""
        scorer, df = scorer_and_data
        row = _high_risk_row(scorer, df)

        t0 = time.monotonic()
        result = explainer.explain(row)
        elapsed = time.monotonic() - t0

        assert elapsed < 10.0, (
            f"Counterfactual generation took {elapsed:.2f}s — exceeds 10s limit"
        )
        assert result.generation_time_seconds < 10.0


class TestSecurity:
    def test_output_contains_no_model_weights(self, scorer_and_data, explainer):
        """CF output must not expose model weights or training data."""
        scorer, df = scorer_and_data
        row = _high_risk_row(scorer, df)
        result = explainer.explain(row)
        result_dict = result.to_dict()

        serialised = json.dumps(result_dict)
        # Model weights would appear as large lists of floats or sklearn internals
        assert "feature_importances_" not in serialised
        assert "estimators_" not in serialised
        assert "n_estimators" not in serialised
        # Training data rows should not be in the output
        assert "X_train" not in serialised

    def test_output_is_json_serialisable(self, scorer_and_data, explainer):
        """CounterfactualResult.to_dict() must be JSON-serialisable."""
        scorer, df = scorer_and_data
        row = _high_risk_row(scorer, df)
        result = explainer.explain(row)
        # Should not raise
        serialised = json.dumps(result.to_dict())
        assert len(serialised) > 0


class TestEdgeCases:
    def test_below_threshold_wallet_returns_empty(self, scorer_and_data, explainer):
        """A wallet already scoring below the threshold should return 0 CFs immediately."""
        scorer, df = scorer_and_data
        # Find the row with the lowest score
        feature_df = df.drop(columns=["label"])
        scores = feature_df.apply(scorer.score_continuous, axis=1)
        low_row = feature_df.loc[scores.idxmin()]

        if scorer.score_continuous(low_row) >= 70.0:
            pytest.skip("No wallet below threshold in this dataset")

        result = explainer.explain(low_row, wallet="low_risk_wallet")
        assert result.n_found == 0
        assert result.counterfactuals == []

    def test_result_has_correct_n_requested(self, scorer_and_data, explainer):
        """n_requested must match the explainer's n_cfs setting."""
        scorer, df = scorer_and_data
        row = _high_risk_row(scorer, df)
        result = explainer.explain(row)
        assert result.n_requested == explainer.n_cfs

    def test_n_found_matches_counterfactuals_length(self, scorer_and_data, explainer):
        """n_found must equal len(counterfactuals)."""
        scorer, df = scorer_and_data
        row = _high_risk_row(scorer, df)
        result = explainer.explain(row)
        assert result.n_found == len(result.counterfactuals)

    def test_cf_actions_not_empty(self, scorer_and_data, explainer):
        """Each counterfactual must have at least one action."""
        scorer, df = scorer_and_data
        row = _high_risk_row(scorer, df)
        result = explainer.explain(row)
        for cf in result.counterfactuals:
            assert len(cf.actions) >= 1, (
                f"CF {cf.cf_index} has no actions — must specify at least one change"
            )


# ---------------------------------------------------------------------------
# Interpretation tests
# ---------------------------------------------------------------------------


class TestInterpretAction:
    def test_counterparty_concentration_interpretation(self):
        msg = _interpret_action("counterparty_concentration_ratio", 0.87, 0.45)
        assert "counterpart" in msg.lower()
        assert "0.87" in msg or "87" in msg

    def test_benford_mad_interpretation(self):
        msg = _interpret_action("benford_mad_24h", 0.025, 0.010)
        assert "benford" in msg.lower() or "mad" in msg.lower() or "amount" in msg.lower()

    def test_round_trip_interpretation(self):
        msg = _interpret_action("round_trip_frequency", 0.6, 0.1)
        assert "round" in msg.lower() or "trip" in msg.lower()

    def test_unknown_feature_returns_generic_message(self):
        msg = _interpret_action("totally_unknown_feature_xyz", 1.0, 0.5)
        assert "totally_unknown_feature_xyz" in msg
        assert "1" in msg and "0.5" in msg

    def test_interpretation_is_non_empty_string(self):
        for feature in [
            "counterparty_concentration_ratio",
            "round_trip_frequency",
            "self_matching_rate",
            "benford_mad_24h",
            "benford_chi_square_1h",
            "order_cancellation_rate",
            "off_hours_activity_ratio",
            "volume_spike_frequency",
            "pair_diversity_score",
            "intra_minute_clustering",
            "net_asset_flow_deviation",
            "net_roundtrip_ratio",
            "volume_per_counterparty_ratio",
            "funding_source_similarity",
        ]:
            msg = _interpret_action(feature, 0.8, 0.3)
            assert isinstance(msg, str) and len(msg) > 0, (
                f"Empty interpretation for feature '{feature}'"
            )


# ---------------------------------------------------------------------------
# ForensicReport integration
# ---------------------------------------------------------------------------


class TestForensicReportIntegration:
    def test_forensic_report_accepts_counterfactual_result(self, scorer_and_data, explainer):
        """ForensicReport can hold and serialise a CounterfactualResult."""
        from detection.forensic_report import ForensicReport, TradeEvidence

        scorer, df = scorer_and_data
        row = _high_risk_row(scorer, df)
        cf_result = explainer.explain(row, wallet="GTEST")

        report = ForensicReport(
            report_id="test-id-001",
            generated_at="2026-06-29T00:00:00Z",
            wallet="GTEST",
            asset_pair="XLM:native/USDC:GA5Z",
            risk_score=85,
            score_lower=80,
            score_upper=92,
            verdict="wash_trade",
            top_shap_features=[],
            benford_analysis={},
            trade_evidence=[],
            model_metadata={"name": "test", "version": "0.2.0"},
            counterfactual_result=cf_result,
        )

        report_dict = report.to_dict()
        assert "counterfactual_result" in report_dict
        cf_dict = report_dict["counterfactual_result"]
        assert cf_dict["wallet"] == "GTEST"
        assert isinstance(cf_dict["counterfactuals"], list)

        # Ensure the whole report is JSON-serialisable
        serialised = json.dumps(report_dict)
        assert len(serialised) > 0

    def test_forensic_report_without_cf_result_unchanged(self):
        """ForensicReport without counterfactual_result still serialises correctly."""
        from detection.forensic_report import ForensicReport

        report = ForensicReport(
            report_id="test-id-002",
            generated_at="2026-06-29T00:00:00Z",
            wallet="GTEST2",
            asset_pair="XLM:native/USDC:GA5Z",
            risk_score=50,
            score_lower=45,
            score_upper=58,
            verdict="clean",
            top_shap_features=[],
            benford_analysis={},
            trade_evidence=[],
            model_metadata={"name": "test", "version": "0.2.0"},
        )
        report_dict = report.to_dict()
        assert "counterfactual_result" not in report_dict
