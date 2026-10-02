"""Coreset vs random vs full-pool benchmark (#889).

Compares ``CoresetSelector`` against random sampling and full-pool training
across labelling budgets, reporting AUC, accuracy and wall-clock (selection +
training). Deterministic for a given ``--seed``.

Run::

    python -m benchmarks.coreset_benchmark --seed 42 --out reports/coreset_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from detection.active_learning.coreset_selector import CoresetSelector

DEFAULT_BUDGETS = (50, 100, 200, 400, 800)


def _fit_eval(X, y, X_test, y_test) -> tuple[float, float, float]:
    t0 = time.perf_counter()
    if len(np.unique(y)) < 2:
        return 0.5, float((y_test == y[0]).mean()), time.perf_counter() - t0
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    elapsed = time.perf_counter() - t0
    proba = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, proba), accuracy_score(y_test, proba > 0.5), elapsed


def run(
    seed: int = 42, n_pool: int = 10_000, budgets=DEFAULT_BUDGETS, seed_size: int = 20
) -> dict:
    X, y = make_classification(
        n_samples=n_pool + 2000, n_features=32, n_informative=12, n_clusters_per_class=4,
        weights=[0.9, 0.1], random_state=seed,
    )
    X = X.astype("float32")
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=2000, stratify=y, random_state=seed
    )
    rng = np.random.default_rng(seed)
    seed_idx = rng.choice(len(X_pool), seed_size, replace=False)
    rest = np.setdiff1d(np.arange(len(X_pool)), seed_idx)

    auc, acc, fit_s = _fit_eval(X_pool, y_pool, X_test, y_test)
    rows = [{"strategy": "full_pool", "budget": len(X_pool), "auc": auc, "accuracy": acc,
             "select_s": 0.0, "train_s": fit_s}]

    for budget in budgets:
        n_sel = budget - seed_size
        t0 = time.perf_counter()
        chosen = rest[CoresetSelector().select(X_pool[rest], n_sel, X_pool[seed_idx])]
        sel_s = time.perf_counter() - t0
        idx = np.concatenate([seed_idx, chosen])
        auc, acc, fit_s = _fit_eval(X_pool[idx], y_pool[idx], X_test, y_test)
        rows.append({"strategy": "coreset", "budget": budget, "auc": auc, "accuracy": acc,
                     "select_s": sel_s, "train_s": fit_s})

        idx = np.concatenate([seed_idx, rng.choice(rest, n_sel, replace=False)])
        auc, acc, fit_s = _fit_eval(X_pool[idx], y_pool[idx], X_test, y_test)
        rows.append({"strategy": "random", "budget": budget, "auc": auc, "accuracy": acc,
                     "select_s": 0.0, "train_s": fit_s})

    return {"seed": seed, "n_pool": len(X_pool), "results": rows}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-pool", type=int, default=10_000)
    p.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    p.add_argument("--out", type=Path, default=Path("reports/coreset_benchmark.json"))
    a = p.parse_args(argv)
    report = run(a.seed, a.n_pool, a.budgets)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    for r in report["results"]:
        print(f"{r['strategy']:>9} budget={r['budget']:>6} auc={r['auc']:.3f} "
              f"acc={r['accuracy']:.3f} select={r['select_s']:.2f}s train={r['train_s']:.2f}s")


if __name__ == "__main__":
    main()
