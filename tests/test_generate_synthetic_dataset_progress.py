"""Tests for the tqdm progress bar in generate_synthetic_dataset."""

from scripts.generate_synthetic_dataset import generate_synthetic_dataset


def test_generate_synthetic_dataset_completes_with_small_n_wallets():
    """The tqdm-wrapped wallet loop still completes without error for a
    small n_wallets and returns exactly that many rows."""
    df = generate_synthetic_dataset(n_wallets=10, seed=42)
    assert len(df) == 10
