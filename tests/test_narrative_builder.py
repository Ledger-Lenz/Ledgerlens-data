"""Tests for reporting.narrative_builder.build_narrative (Issue #285)."""

from reporting.narrative_builder import build_narrative

WALLET = "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF"
PAIR = "XLM:USDC"


def _base_report(**overrides):
    report = {
        "wallet": WALLET,
        "asset_pair": PAIR,
        "risk_score": 88,
        "verdict": "wash_trade",
        "top_shap_features": [
            {"feature": "benford_mad_1h", "contribution": 0.41, "value": 0.12},
            {"feature": "counterparty_concentration_ratio", "contribution": 0.33, "value": 0.9},
            {"feature": "round_trip_frequency", "contribution": 0.1, "value": 0.5},
        ],
    }
    report.update(overrides)
    return report


def test_ring_and_benford_violation_renders_both_templates():
    report = _base_report(
        ring_detection={"ring_id": "ring-7", "ring_size": 5},
        benford_analysis={"chi_square": 18.4, "p_value": 0.001},
    )

    narrative = build_narrative(report)

    assert "wash-trading ring" in narrative
    assert "ring-7" in narrative
    assert "Benford's Law violation" in narrative
    assert "18.4" in narrative
    assert "0.001" in narrative


def test_missing_shap_values_still_produces_valid_paragraph():
    report = _base_report(top_shap_features=[], benford_analysis={"chi_square": 12.0, "p_value": 0.02})

    narrative = build_narrative(report)

    assert "Benford's Law violation" in narrative
    assert WALLET in narrative
    assert "None" not in narrative
    assert len(narrative.split()) > 0


def test_no_signal_falls_back_to_low_confidence():
    report = _base_report()

    narrative = build_narrative(report)

    assert "no single detector signal" in narrative


def test_output_never_exceeds_300_words():
    huge_shap = [
        {"feature": f"feature_{i}", "contribution": float(i), "value": float(i)} for i in range(50)
    ]
    report = _base_report(
        top_shap_features=huge_shap,
        ring_detection={"ring_id": "ring-1", "ring_size": 3},
        benford_analysis={"chi_square": 99.9, "p_value": 0.0001},
        velocity_anomaly={"multiple_of_baseline": 12.5},
    )

    narrative = build_narrative(report)

    assert len(narrative.split()) <= 300


def test_markdown_format_bolds_feature_labels():
    report = _base_report(narrative_format="markdown")

    narrative = build_narrative(report)

    assert "**" in narrative
