"""Tests for label-distribution summary in generate_synthetic_dataset."""

import io

import pandas as pd

from scripts.generate_synthetic_dataset import (
    generate_synthetic_dataset,
    print_dataset_summary,
)


def test_print_dataset_summary_shows_label_counts_and_features():
    df = generate_synthetic_dataset(n_wallets=100, seed=42)
    buf = io.StringIO()

    print_dataset_summary(df, profile="NaiveAttacker", file=buf)
    text = buf.getvalue()

    assert "Label distribution:" in text
    assert "wash_trade  (label=1):" in text
    assert "legitimate  (label=0):" in text
    assert "50.0%" in text
    assert "Feature summary (wash_trade rows):" in text
    assert "benford_chi_square_24h:" in text
    assert "counterparty_concentration_ratio:" in text
    assert "Profile breakdown:" not in text


def test_print_dataset_summary_shows_profile_breakdown_for_simulator_profile():
    df = pd.DataFrame(
        {
            "wallet": ["G1", "G2", "G3", "G4"],
            "label": [1, 1, 0, 0],
            "profile": ["RingAttacker"] * 4,
            "benford_chi_square_24h": [60.0, 55.0, 5.0, 4.0],
            "counterparty_concentration_ratio": [0.9, 0.8, 0.2, 0.3],
        }
    )
    buf = io.StringIO()

    print_dataset_summary(df, profile="RingAttacker", file=buf)
    text = buf.getvalue()

    assert "Profile breakdown:" in text
    assert "RingAttacker: 4 rows  (wash=2, legitimate=2)" in text


def test_print_dataset_summary_handles_empty_dataframe():
    buf = io.StringIO()
    print_dataset_summary(pd.DataFrame(), profile="NaiveAttacker", file=buf)
    assert buf.getvalue() == ""
