"""Active learning query strategies for LedgerLens.

Each strategy selects the most informative wallets from an unlabelled pool
for human annotation, maximising model improvement per labelling hour.

All strategies implement:
    select(pool: pd.DataFrame, n_query: int, model=None) -> list[str]

returning wallet IDs (``wallet`` column values).
"""

from __future__ import annotations

import abc

import numpy as np
import pandas as pd

from detection.model_training import FEATURE_COLUMNS_EXCLUDE


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_COLUMNS_EXCLUDE]


def _proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return (n_samples, 2) probability array."""
    return model.predict_proba(X)


class BaseQueryStrategy(abc.ABC):
    @abc.abstractmethod
    def select(self, pool: pd.DataFrame, n_query: int, model=None) -> list[str]:
        """Return up to *n_query* wallet IDs from *pool*."""


# ---------------------------------------------------------------------------
# Uncertainty sampling
# ---------------------------------------------------------------------------


class LeastConfidence(BaseQueryStrategy):
    """Select wallets with the lowest max predicted probability."""

    def select(self, pool: pd.DataFrame, n_query: int, model=None) -> list[str]:
        if model is None:
            raise ValueError("LeastConfidence requires a model")
        X = pool[_feature_cols(pool)].astype(float)
        probs = _proba(model, X)
        scores = probs.max(axis=1)  # lower = more uncertain
        idx = np.argsort(scores)[: min(n_query, len(pool))]
        return pool.iloc[idx]["wallet"].tolist()


class MarginSampling(BaseQueryStrategy):
    """Select wallets with the smallest margin between top-2 probabilities."""

    def select(self, pool: pd.DataFrame, n_query: int, model=None) -> list[str]:
        if model is None:
            raise ValueError("MarginSampling requires a model")
        X = pool[_feature_cols(pool)].astype(float)
        probs = _proba(model, X)
        sorted_probs = np.sort(probs, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        idx = np.argsort(margins)[: min(n_query, len(pool))]
        return pool.iloc[idx]["wallet"].tolist()


class Entropy(BaseQueryStrategy):
    """Select wallets with the highest Shannon entropy over class probabilities."""

    def select(self, pool: pd.DataFrame, n_query: int, model=None) -> list[str]:
        if model is None:
            raise ValueError("Entropy requires a model")
        X = pool[_feature_cols(pool)].astype(float)
        probs = _proba(model, X)
        # Clip to avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log2(probs), axis=1)
        idx = np.argsort(-entropy)[: min(n_query, len(pool))]
        return pool.iloc[idx]["wallet"].tolist()


# ---------------------------------------------------------------------------
# Coverage / diversity
# ---------------------------------------------------------------------------


class CoreSet(BaseQueryStrategy):
    """Greedy k-center: maximises coverage of unlabelled feature space.

    Requires *labelled_pool* kwarg (pd.DataFrame of already-labelled rows)
    passed at select time, or selects greedily from the pool itself.
    """

    def select(
        self,
        pool: pd.DataFrame,
        n_query: int,
        model=None,
        labelled_pool: pd.DataFrame | None = None,
    ) -> list[str]:
        cols = _feature_cols(pool)
        pool_X = pool[cols].astype(float).values

        if labelled_pool is not None and len(labelled_pool) > 0:
            labelled_X = labelled_pool[_feature_cols(labelled_pool)].astype(float).values
            # min dist from each pool point to its nearest labelled point
            dist_to_labelled = _min_dist_to_set(pool_X, labelled_X)
        else:
            dist_to_labelled = np.full(len(pool), np.inf)

        selected_idx: list[int] = []
        remaining = dist_to_labelled.copy()

        for _ in range(min(n_query, len(pool))):
            chosen = int(np.argmax(remaining))
            selected_idx.append(chosen)
            # Update remaining distances
            chosen_X = pool_X[chosen : chosen + 1]
            dist_to_chosen = _min_dist_to_set(pool_X, chosen_X)
            remaining = np.minimum(remaining, dist_to_chosen)
            remaining[chosen] = -np.inf  # don't re-select

        return pool.iloc[selected_idx]["wallet"].tolist()


def _min_dist_to_set(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """For each point, return the distance to its nearest reference point."""
    diffs = points[:, np.newaxis, :] - reference[np.newaxis, :, :]  # (N, M, D)
    dists = np.sqrt((diffs**2).sum(axis=2))  # (N, M)
    return dists.min(axis=1)  # (N,)


class BADGE(BaseQueryStrategy):
    """Batch Active learning by Diverse Gradient Embeddings.

    Uses the gradient of the loss w.r.t. the last-layer embedding as a
    combined uncertainty+diversity signal, approximated by the model's
    predicted class probability for tree ensembles (no true gradient).

    Implementation: k-means++ seeding in (prob * feature) space.
    """

    def select(self, pool: pd.DataFrame, n_query: int, model=None) -> list[str]:
        if model is None:
            raise ValueError("BADGE requires a model")
        cols = _feature_cols(pool)
        X = pool[cols].astype(float).values
        probs = _proba(model, pd.DataFrame(X, columns=cols))[:, 1]

        # Gradient embedding: scale features by uncertainty
        uncertainty = 1.0 - np.abs(2 * probs - 1)  # high near 0.5
        embeddings = X * uncertainty[:, np.newaxis]

        selected_idx = _kmeans_pp_indices(embeddings, min(n_query, len(pool)))
        return pool.iloc[selected_idx]["wallet"].tolist()


def _kmeans_pp_indices(X: np.ndarray, k: int) -> list[int]:
    """k-means++ seeding — returns k indices."""
    rng = np.random.default_rng(42)
    idx = [int(rng.integers(len(X)))]
    for _ in range(k - 1):
        dists = _min_dist_to_set(X, X[idx])
        probs = dists**2 / (dists**2).sum()
        idx.append(int(rng.choice(len(X), p=probs)))
    return idx


# ---------------------------------------------------------------------------
# Committee disagreement
# ---------------------------------------------------------------------------


class CommitteeDisagreement(BaseQueryStrategy):
    """Query by Committee: selects wallets where RF, XGBoost, LightGBM disagree most.

    Disagreement is measured as the variance of the three models' class-1
    probability estimates.  Higher variance = more disagreement.

    Pass ``models`` (dict of name->estimator) via the *models* kwarg.
    Falls back to *model* (single estimator) if *models* is not provided.
    """

    def select(
        self,
        pool: pd.DataFrame,
        n_query: int,
        model=None,
        models: dict | None = None,
    ) -> list[str]:
        cols = _feature_cols(pool)
        X = pool[cols].astype(float)

        estimators = list((models or {}).values()) or ([model] if model is not None else [])
        if not estimators:
            raise ValueError("CommitteeDisagreement requires model or models kwarg")

        all_probs = np.stack([m.predict_proba(X)[:, 1] for m in estimators], axis=1)
        # variance across committee members per sample
        disagreement = all_probs.var(axis=1)
        idx = np.argsort(-disagreement)[: min(n_query, len(pool))]
        return pool.iloc[idx]["wallet"].tolist()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, type[BaseQueryStrategy]] = {
    "least_confidence": LeastConfidence,
    "margin": MarginSampling,
    "entropy": Entropy,
    "coreset": CoreSet,
    "badge": BADGE,
    "committee_disagreement": CommitteeDisagreement,
}


def get_strategy(name: str) -> BaseQueryStrategy:
    """Return an instantiated strategy by registry name."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown query strategy '{name}'. Choose from: {list(STRATEGY_REGISTRY)}")
    return cls()
