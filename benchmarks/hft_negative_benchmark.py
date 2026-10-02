"""Hard-negative mining with vs without domain HFT negatives (#890).

Measures (a) mining wall-clock overhead and (b) how often mined negatives are
the legitimate-HFT look-alikes the encoder must separate from wash trading.
For the full downstream comparison, pretrain twice (``hft_negative_fraction``
0.0 vs default) and compare with ``detection/contrastive/finetune.py`` +
``evaluation/backtest.py`` on the HFT-vs-wash slice.

Run::

    python -m benchmarks.hft_negative_benchmark --seed 42
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from detection.contrastive.negative_miner import HardNegativeMiner


def run(seed: int = 42, n_clean: int = 20_000, n_hft: int = 500, dim: int = 128,
        batch: int = 256, k: int = 16, iters: int = 50) -> dict:
    rng = np.random.default_rng(seed)
    clean = rng.normal(size=(n_clean, dim)).astype("float32")
    hft_pos = rng.choice(n_clean, n_hft, replace=False)
    centre = rng.normal(size=dim).astype("float32") * 2
    clean[hft_pos] = centre + 0.5 * rng.normal(size=(n_hft, dim))  # HFT near wash cluster
    anchors = (centre + 0.5 * rng.normal(size=(batch, dim))).astype("float32")
    hft_set = set(hft_pos.tolist())

    out = {}
    for label, frac in (("baseline", 0.0), ("domain_hft", 0.5)):
        m = HardNegativeMiner(dim, curriculum_epochs=0, rng_seed=seed, hft_negative_fraction=frac)
        t0 = time.perf_counter()
        m.build_clean_index(clean)
        if frac > 0:
            m.set_hft_negatives(hft_pos, clean)
        build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(iters):
            neg = m.mine_negatives(anchors, k, epoch=10)
        mine_s = (time.perf_counter() - t0) / iters
        out[label] = {"build_s": build_s, "mine_ms_per_batch": mine_s * 1000,
                      "hft_negative_rate": float(np.isin(neg, list(hft_set)).mean())}
    return {"seed": seed, **out}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=42)
    print(json.dumps(run(p.parse_args(argv).seed), indent=2))


if __name__ == "__main__":
    main()
