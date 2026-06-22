"""Adversarial attack strategies for probing the LedgerLens ensemble.

Each attack implements ``AttackStrategy.perturb(trades_df) -> pd.DataFrame``
that modifies a trades DataFrame while preserving its schema.

``PGDAttack`` operates on the *feature vector* directly (not raw trades) and
is kept in this subpackage, excluded from ``detection/__init__.py``, so it
cannot be accidentally imported by production code paths.

    >>> from detection.adversarial.attacks import AmountJitter, PGDAttack
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass  # guard clause — keeps PGDAttack out of runtime production imports


class AttackStrategy(abc.ABC):
    """Base class for all trade-sequence perturbation strategies."""

    @abc.abstractmethod
    def perturb(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Return a *copy* of ``trades_df`` with the attack applied."""


# ---------------------------------------------------------------------------
# 1. AmountRounding
# ---------------------------------------------------------------------------


class AmountRounding(AttackStrategy):
    """Round all trade amounts to ``sig_figs`` significant figures.

    Suppresses Benford deviation by forcing leading digits to cluster around
    a small set of values (e.g. 500, 5000, 50000).
    """

    def __init__(self, sig_figs: int = 2) -> None:
        self.sig_figs = sig_figs

    def perturb(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        df = trades_df.copy()
        amounts = df["amount"].clip(lower=1e-12)
        magnitude = np.floor(np.log10(amounts))
        factor = 10.0 ** (magnitude - self.sig_figs + 1)
        df["amount"] = (np.round(amounts / factor) * factor).clip(lower=1e-12)
        return df


# ---------------------------------------------------------------------------
# 2. AmountJitter
# ---------------------------------------------------------------------------


class AmountJitter(AttackStrategy):
    """Add Gaussian noise (σ = ``noise_std`` * amount) to each trade amount.

    Tests the minimum noise level needed to neutralise chi-square without
    making the bot-generated pattern obviously noisy.
    """

    def __init__(self, noise_std: float = 0.005) -> None:
        self.noise_std = noise_std

    def perturb(
        self, trades_df: pd.DataFrame, rng: np.random.Generator | None = None
    ) -> pd.DataFrame:
        rng = rng or np.random.default_rng(0)
        df = trades_df.copy()
        noise = 1.0 + rng.normal(0, self.noise_std, size=len(df))
        df["amount"] = (df["amount"] * noise).clip(lower=1e-12)
        return df


# ---------------------------------------------------------------------------
# 3. TemporalSpreading
# ---------------------------------------------------------------------------


class TemporalSpreading(AttackStrategy):
    """Redistribute trades across a longer time window.

    Scales all ``ledger_close_time`` timestamps so the trade sequence spans
    ``spread_factor`` times the original window, reducing burst-rate features.
    """

    def __init__(self, spread_factor: float = 3.0) -> None:
        self.spread_factor = spread_factor

    def perturb(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        df = trades_df.copy()
        times = pd.to_datetime(df["ledger_close_time"])
        t0 = times.min()
        shifted = t0 + (times - t0) * self.spread_factor
        df["ledger_close_time"] = shifted
        return df


# ---------------------------------------------------------------------------
# 4. DecoyWallets
# ---------------------------------------------------------------------------


class DecoyWallets(AttackStrategy):
    """Inject synthetic intermediate wallets to dilute Jaccard similarity.

    For each existing trade, a decoy wallet is added as an additional
    counterparty in ``n_decoys`` duplicate trades routed through a fresh
    account, fragmenting the counterparty concentration signal.
    """

    def __init__(self, n_decoys: int = 5) -> None:
        self.n_decoys = n_decoys

    def perturb(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        if trades_df.empty:
            return trades_df.copy()

        rng = np.random.default_rng(1)
        decoy_rows = []
        sample_size = min(self.n_decoys, len(trades_df))
        sample = trades_df.sample(n=sample_size, random_state=1)

        for _idx, row in sample.iterrows():
            decoy_wallet = f"GDECOY{rng.integers(1_000_000, 9_999_999)}"
            new_row = row.copy()
            new_row["counter_account"] = decoy_wallet
            new_row["amount"] = row["amount"] * rng.uniform(0.1, 0.5)
            decoy_rows.append(new_row)

        return pd.concat([trades_df, pd.DataFrame(decoy_rows)], ignore_index=True)


# ---------------------------------------------------------------------------
# 5. SybilInjection
# ---------------------------------------------------------------------------


class SybilInjection(AttackStrategy):
    """Spawn N sock-puppet wallets that each do one round-trip trade.

    Dilutes network-centrality signal by adding many low-volume wallets that
    each interact exactly once with the target wallet.
    """

    def __init__(self, n_sybils: int = 20) -> None:
        self.n_sybils = n_sybils

    def perturb(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        if trades_df.empty:
            return trades_df.copy()

        rng = np.random.default_rng(2)
        base = trades_df.iloc[0].copy()
        sybil_rows = []

        for i in range(self.n_sybils):
            sybil = f"GSYBIL{i:06d}"
            row = base.copy()
            row["counter_account"] = sybil
            row["amount"] = rng.uniform(1, 100)
            sybil_rows.append(row)

        return pd.concat([trades_df, pd.DataFrame(sybil_rows)], ignore_index=True)


# ---------------------------------------------------------------------------
# 6. VolumeClipping
# ---------------------------------------------------------------------------


class VolumeClipping(AttackStrategy):
    """Cap individual trade sizes to stay within the top ``percentile``% of
    the legitimate volume distribution.

    Prevents large-amount outliers that trigger volume-spike features.
    """

    def __init__(self, percentile: float = 95.0) -> None:
        self.percentile = percentile

    def perturb(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        df = trades_df.copy()
        cap = float(np.percentile(df["amount"], self.percentile))
        df["amount"] = df["amount"].clip(upper=cap)
        return df


# ---------------------------------------------------------------------------
# 7. CrossPairDispersion
# ---------------------------------------------------------------------------


class CrossPairDispersion(AttackStrategy):
    """Spread wash volume across ``n_pairs`` synthetic asset pairs.

    Reassigns trades to one of ``n_pairs`` fabricated base-asset values so
    per-pair feature density is reduced below detection thresholds.
    """

    def __init__(self, n_pairs: int = 10) -> None:
        self.n_pairs = n_pairs

    def perturb(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        df = trades_df.copy()
        pair_labels = [f"FAKE{i}:GISSUER{i:06d}" for i in range(self.n_pairs)]
        df["base_asset"] = [pair_labels[i % self.n_pairs] for i in range(len(df))]
        return df


# ---------------------------------------------------------------------------
# 8. PGDAttack  (white-box, feature-space; sandboxed in this subpackage)
# ---------------------------------------------------------------------------


class PGDAttack:
    """Projected Gradient Descent on the continuous feature vector.

    White-box L∞-bounded attack (ε = ``epsilon``) that directly perturbs the
    feature row to minimise the ensemble's predicted wash-trade probability.

    **Sandboxed** — this class operates on feature vectors, not raw trades,
    and is intentionally excluded from the ``detection`` package's public
    ``__init__.py``.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        n_steps: int = 20,
        step_size: float | None = None,
    ) -> None:
        self.epsilon = epsilon
        self.n_steps = n_steps
        self.step_size = step_size or (epsilon / n_steps * 2)

    def perturb(
        self,
        feature_vector: np.ndarray,
        models: dict,
        feature_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Return a perturbed copy of ``feature_vector`` within L∞ ε.

        Parameters
        ----------
        feature_vector:
            1-D float array of model features.
        models:
            Dict of ``{name: fitted_sklearn_estimator}``.
        feature_bounds:
            Optional ``(lower, upper)`` arrays to project into valid
            feature space after each step.  Defaults to ``[0, 1]``.
        """
        lo = feature_bounds[0] if feature_bounds is not None else np.zeros_like(feature_vector)
        hi = feature_bounds[1] if feature_bounds is not None else np.ones_like(feature_vector)

        x = feature_vector.copy().astype(float)
        original = x.copy()

        for _ in range(self.n_steps):
            # Estimate gradient via finite differences (model-agnostic)
            grad = np.zeros_like(x)
            for j in range(len(x)):
                delta = np.zeros_like(x)
                delta[j] = 1e-4
                x_plus = np.clip(x + delta, lo, hi)
                x_minus = np.clip(x - delta, lo, hi)

                score_plus = self._ensemble_score(x_plus, models)
                score_minus = self._ensemble_score(x_minus, models)
                grad[j] = (score_plus - score_minus) / 2e-4

            # Gradient descent step (minimise score)
            x = x - self.step_size * np.sign(grad)

            # Project back into L∞ ball and feature bounds
            x = np.clip(x, original - self.epsilon, original + self.epsilon)
            x = np.clip(x, lo, hi)

        return x

    @staticmethod
    def _ensemble_score(feature_vector: np.ndarray, models: dict) -> float:
        X = feature_vector.reshape(1, -1)
        probs = [m.predict_proba(X)[0][1] for m in models.values()]
        return float(sum(probs) / len(probs))
