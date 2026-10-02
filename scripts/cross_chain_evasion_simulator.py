"""Cross-chain timing-jitter evasion simulator (#882).

Generates synthetic Stellar -> external-chain bridge transfers where an
adversary inserts a random delay (uniform in ``[0, jitter_seconds]``) between
the source and destination legs, mixed with background traffic on both chains
and "collision" transfers of popular round amounts. Compares the naive
fixed-window matcher (``BehavioralMatcher.match_amount_fingerprints``) against
the jitter-robust matcher (``BehavioralMatcher.match_jitter_robust``).

Usage::

    python scripts/cross_chain_evasion_simulator.py            # markdown table
    python scripts/cross_chain_evasion_simulator.py --seeds 10 --json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from detection.cross_chain.behavioral_matcher import BehavioralMatcher

ROUND_AMOUNTS = (100.0, 250.0, 500.0, 1000.0, 5000.0)


@dataclass(frozen=True)
class Scenario:
    stellar_txs: list[dict[str, Any]]
    external_txs: list[dict[str, Any]]
    truth: dict[str, str]  # stellar tx id -> external tx id


def simulate(
    n_transfers: int = 150,
    n_background: int = 300,
    jitter_seconds: float = 0.0,
    fee_rate: float = 0.003,
    round_amount_fraction: float = 0.15,
    honest_latency_seconds: float = 20.0,
    horizon_seconds: float = 3 * 86400.0,
    seed: int = 0,
) -> Scenario:
    """Build one synthetic scenario.

    * Bridged transfers: amount ~ lognormal (distinctive decimals), or a
      popular round amount with probability ``round_amount_fraction``. The
      destination receives ``amount * (1 - fee_rate)`` after
      ``Exp(honest_latency) + Uniform(0, jitter_seconds)``.
    * Background: independent Stellar outflows and external inflows with the
      same amount distribution and uniform timing — plausible decoys.
    """
    rng = np.random.default_rng(seed)

    def amount() -> float:
        if rng.random() < round_amount_fraction:
            return float(rng.choice(ROUND_AMOUNTS))
        return round(float(rng.lognormal(6.0, 1.2)), 7)

    stellar, external, truth = [], [], {}
    for i in range(n_transfers):
        t0 = float(rng.uniform(0, horizon_seconds))
        amt = amount()
        delay = float(rng.exponential(honest_latency_seconds)) + float(
            rng.uniform(0, jitter_seconds)
        )
        s_id, e_id = f"s{i}", f"e{i}"
        stellar.append({"id": s_id, "wallet": f"GUSER{i}", "timestamp": t0, "amount": amt})
        external.append(
            {
                "id": e_id,
                "wallet": f"0xuser{i}",
                "chain": "ethereum",
                "timestamp": t0 + delay,
                "amount": round(amt * (1 - fee_rate), 7),
            }
        )
        truth[s_id] = e_id
    for j in range(n_background):
        stellar.append(
            {
                "id": f"sb{j}",
                "wallet": f"GBG{j}",
                "timestamp": float(rng.uniform(0, horizon_seconds)),
                "amount": amount(),
            }
        )
        external.append(
            {
                "id": f"eb{j}",
                "wallet": f"0xbg{j}",
                "chain": "ethereum",
                "timestamp": float(rng.uniform(0, horizon_seconds)),
                "amount": round(amount() * (1 - fee_rate), 7),
            }
        )
    return Scenario(stellar, external, truth)


def evaluate(links: list[dict[str, Any]], truth: dict[str, str]) -> dict[str, float]:
    """Transfer-level precision / recall of emitted links against ``truth``."""
    predicted = {
        (lk["metadata"]["stellar_tx_id"], lk["metadata"]["external_tx_id"]) for lk in links
    }
    correct = sum(1 for s, e in predicted if truth.get(s) == e)
    precision = correct / len(predicted) if predicted else 1.0
    recall = correct / len(truth) if truth else 0.0
    return {"precision": precision, "recall": recall, "links": len(predicted)}


def run_naive(scenario: Scenario, fee_rate: float) -> list[dict[str, Any]]:
    # Give the naive matcher enough amount tolerance to absorb the bridge fee
    # so the comparison isolates timing robustness.
    return BehavioralMatcher.match_amount_fingerprints(
        scenario.stellar_txs,
        scenario.external_txs,
        tolerance=fee_rate + 0.0005,
        window_seconds=60.0,
    )


def run_robust(
    scenario: Scenario, fee_rate: float, match_prior: float | None = None
) -> list[dict[str, Any]]:
    # Default the prior to the scenario's base rate: the share of Stellar
    # legs whose destination leg is present. In production this is estimated
    # from ingestion coverage, not known exactly.
    if match_prior is None:
        match_prior = len(scenario.truth) / len(scenario.stellar_txs)
    return BehavioralMatcher.match_jitter_robust(
        scenario.stellar_txs,
        scenario.external_txs,
        fee_models=[(0.0, fee_rate)],
        match_prior=match_prior,
    )


def compare(
    jitters: tuple[float, ...] = (0.0, 300.0, 1800.0, 3600.0, 4 * 3600.0),
    seeds: int = 5,
    fee_rate: float = 0.003,
    match_prior: float | None = None,
    **scenario_kwargs: Any,
) -> list[dict[str, float]]:
    """Mean precision/recall of both matchers per jitter level over ``seeds`` runs."""
    rows = []
    for jitter in jitters:
        acc = {"naive": [], "robust": []}
        for seed in range(seeds):
            sc = simulate(jitter_seconds=jitter, fee_rate=fee_rate, seed=seed, **scenario_kwargs)
            acc["naive"].append(evaluate(run_naive(sc, fee_rate), sc.truth))
            acc["robust"].append(evaluate(run_robust(sc, fee_rate, match_prior), sc.truth))
        row: dict[str, float] = {"jitter_seconds": jitter}
        for name, results in acc.items():
            row[f"{name}_precision"] = float(np.mean([r["precision"] for r in results]))
            row[f"{name}_recall"] = float(np.mean([r["recall"] for r in results]))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--fee-rate", type=float, default=0.003)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = compare(seeds=args.seeds, fee_rate=args.fee_rate)
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print("| jitter (s) | naive recall | naive precision | robust recall | robust precision |")
    print("|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['jitter_seconds']:.0f} | {r['naive_recall']:.3f} | {r['naive_precision']:.3f}"
            f" | {r['robust_recall']:.3f} | {r['robust_precision']:.3f} |"
        )


if __name__ == "__main__":
    main()
