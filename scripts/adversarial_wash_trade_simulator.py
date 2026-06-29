"""Adversarial Wash Trade Simulator using a genetic algorithm.

Evolves wash trade strategies that minimise their LedgerLens risk score
while maintaining economic plausibility (volume 1,000–10,000,000 XLM).

Fitness = 1 / (risk_score + 1)  — lower score → higher fitness.

Exposes best-evolved adversarial risk score as Prometheus gauge
``ledgerlens_adversarial_lowest_score``.

Usage::

    python -m scripts.adversarial_wash_trade_simulator \\
        --generations 100 --population 50 \\
        --output data/adversarial_trades.parquet
"""

from __future__ import annotations

import argparse
import os
import uuid
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

try:
    from prometheus_client import Gauge
    adversarial_lowest_score: Any = Gauge(
        "ledgerlens_adversarial_lowest_score",
        "Best (lowest) risk score achieved by the adversarial genetic algorithm",
    )
except Exception:
    adversarial_lowest_score = None

TRADE_COLUMNS = [
    "trade_id", "ledger_close_time", "base_account", "counter_account",
    "base_asset", "counter_asset", "amount", "price",
]

# Parameter bounds: (min, max)
_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "n_trades": (10, 500),
    "amount_mean": (1.0, 10_000.0),
    "amount_std": (0.01, 1_000.0),
    "inter_trade_seconds": (1.0, 3_600.0),
    "n_counterparties": (1, 20),
    "jitter_fraction": (0.0, 1.0),
}


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _random_individual(rng: np.random.Generator) -> dict:
    ind = {
        k: rng.uniform(lo, hi) for k, (lo, hi) in _PARAM_BOUNDS.items()
    }
    ind["use_round_numbers"] = bool(rng.integers(0, 2))
    return ind


def _make_trades(ind: dict, seed: int = 0) -> pd.DataFrame:
    """Generate a trade DataFrame from a strategy individual."""
    rng = np.random.default_rng(seed)
    n = max(1, int(ind["n_trades"]))
    n_cp = max(1, int(ind["n_counterparties"]))

    wallet = "GADV0000"
    counterparties = [f"GADV{i:04d}" for i in range(1, n_cp + 1)]

    start = pd.Timestamp("2024-01-01", tz="UTC")
    intervals = rng.exponential(ind["inter_trade_seconds"], size=n)
    jitter = rng.uniform(-1, 1, size=n) * ind["inter_trade_seconds"] * ind["jitter_fraction"]
    intervals = np.abs(intervals + jitter)
    times = start + pd.to_timedelta(np.cumsum(intervals), unit="s")

    amounts = rng.normal(ind["amount_mean"], max(ind["amount_std"], 0.01), size=n)
    if ind["use_round_numbers"]:
        amounts = np.round(amounts, 0)
    amounts = np.abs(amounts).clip(0.01)

    # Enforce volume constraint: 1,000 – 10,000,000 XLM
    total_volume = float(amounts.sum())
    if total_volume < 1_000:
        amounts = amounts * (1_000 / (total_volume + 1e-9))
    elif total_volume > 10_000_000:
        amounts = amounts * (10_000_000 / total_volume)

    cp_idx = rng.integers(0, n_cp, size=n)
    rows = {
        "trade_id": [str(uuid.UUID(int=i)) for i in range(n)],
        "ledger_close_time": times.astype(str),
        "base_account": [wallet] * n,
        "counter_account": [counterparties[i] for i in cp_idx],
        "base_asset": ["XLM:native"] * n,
        "counter_asset": ["USDC:GA5Z"] * n,
        "amount": amounts,
        "price": rng.uniform(0.09, 0.11, size=n),
    }
    return pd.DataFrame(rows)


def _score_individual(ind: dict, scorer, seed: int = 0) -> float:
    """Return risk score for the individual's generated trades."""
    trades = _make_trades(ind, seed=seed)
    try:
        score = scorer(trades)
    except Exception:
        score = 50.0  # neutral fallback
    return float(np.clip(score, 0.0, 100.0))


def _tournament_select(population: list[dict], fitnesses: list[float], k: int, rng) -> dict:
    idx = rng.integers(0, len(population), size=k)
    best = idx[int(np.argmax([fitnesses[i] for i in idx]))]
    return deepcopy(population[best])


def _crossover(a: dict, b: dict, rng) -> dict:
    child = {}
    for key in _PARAM_BOUNDS:
        child[key] = a[key] if rng.random() < 0.5 else b[key]
    child["use_round_numbers"] = a["use_round_numbers"] if rng.random() < 0.5 else b["use_round_numbers"]
    return child


def _mutate(ind: dict, rng, sigma: float = 0.1) -> dict:
    mutated = deepcopy(ind)
    for key, (lo, hi) in _PARAM_BOUNDS.items():
        if rng.random() < 0.3:  # per-gene mutation probability
            noise = rng.normal(0, sigma * (hi - lo))
            mutated[key] = _clip(mutated[key] + noise, lo, hi)
    if rng.random() < 0.1:
        mutated["use_round_numbers"] = not mutated["use_round_numbers"]
    return mutated


def _heuristic_score(trades: pd.DataFrame) -> float:
    """Heuristic scorer used when ML models are unavailable."""
    if trades.empty:
        return 50.0
    n_cp = trades["counter_account"].nunique()
    cv = float(trades["amount"].std() / (trades["amount"].mean() + 1e-9))
    score = 80.0 - min(n_cp * 5, 40) - min(cv * 10, 30)
    return float(np.clip(score, 0.0, 100.0))


def _build_scorer(model_dir: str):
    """Return a callable trades→score. Falls back to a heuristic if no models."""
    try:
        import os
        if not os.path.exists(os.path.join(model_dir, "random_forest.joblib")):
            return _heuristic_score

        from detection.model_inference import RiskScorer  # lazy import
        from detection.feature_engineering import build_feature_matrix  # lazy import

        scorer_obj = RiskScorer(model_dir=model_dir)

        def _ml_score(trades: pd.DataFrame) -> float:
            fm = build_feature_matrix(trades)
            if fm.empty:
                return 50.0
            row = fm.iloc[0]
            return float(scorer_obj.score_continuous(row))

        return _ml_score
    except Exception:
        return _heuristic_score


class AdversarialWashTradeSimulator:
    """Genetic algorithm that evolves wash trade strategies to evade detection."""

    def __init__(self, model_dir: str = "./models", random_seed: int = 42):
        self.model_dir = model_dir
        self.rng = np.random.default_rng(random_seed)
        self._scorer = _build_scorer(model_dir)
        self._best_individual: dict | None = None
        self._best_score: float = 100.0
        self._generation_history: list[dict] = []

    def run(self, n_generations: int = 100, population_size: int = 50) -> dict:
        """Run the genetic algorithm.

        Returns
        -------
        dict with keys: best_strategy, best_score, generation_history
        """
        population = [_random_individual(self.rng) for _ in range(population_size)]
        elite_count = max(1, population_size // 20)

        for gen in range(n_generations):
            fitnesses = [
                1.0 / (_score_individual(ind, self._scorer, seed=gen) + 1.0)
                for ind in population
            ]
            scores = [1.0 / f - 1.0 for f in fitnesses]

            best_idx = int(np.argmax(fitnesses))
            if scores[best_idx] < self._best_score:
                self._best_score = scores[best_idx]
                self._best_individual = deepcopy(population[best_idx])
                if adversarial_lowest_score is not None:
                    adversarial_lowest_score.set(self._best_score)

            self._generation_history.append({
                "generation": gen,
                "best_score": min(scores),
                "mean_score": float(np.mean(scores)),
            })

            # Sort by fitness (descending) for elitism
            ranked = sorted(range(population_size), key=lambda i: fitnesses[i], reverse=True)
            elite = [deepcopy(population[i]) for i in ranked[:elite_count]]

            # Breed next generation
            new_pop = elite[:]
            while len(new_pop) < population_size:
                parent_a = _tournament_select(population, fitnesses, k=3, rng=self.rng)
                parent_b = _tournament_select(population, fitnesses, k=3, rng=self.rng)
                child = _crossover(parent_a, parent_b, self.rng)
                child = _mutate(child, self.rng)
                new_pop.append(child)
            population = new_pop

        return {
            "best_strategy": self._best_individual,
            "best_score": self._best_score,
            "generation_history": self._generation_history,
        }

    def get_adversarial_trades(self) -> pd.DataFrame:
        """Return trades from the best evolved strategy with label=1."""
        if self._best_individual is None:
            return pd.DataFrame(columns=TRADE_COLUMNS + ["label"])
        trades = _make_trades(self._best_individual)
        trades["label"] = 1
        return trades

    def save_to_dataset(self, path: str) -> None:
        """Append adversarial trades (labelled wash=1) to an existing parquet dataset."""
        trades = self.get_adversarial_trades()
        if trades.empty:
            return
        if os.path.exists(path):
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, trades], ignore_index=True)
        else:
            combined = trades
        combined.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial wash trade simulator")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--output", default="data/adversarial_trades.parquet")
    parser.add_argument("--model-dir", default="./models")
    args = parser.parse_args()

    sim = AdversarialWashTradeSimulator(model_dir=args.model_dir)
    result = sim.run(n_generations=args.generations, population_size=args.population)
    print(f"Best adversarial risk score: {result['best_score']:.2f}")
    print(f"Best strategy: {result['best_strategy']}")
    sim.save_to_dataset(args.output)
    print(f"Adversarial trades saved to {args.output}")


if __name__ == "__main__":
    main()
