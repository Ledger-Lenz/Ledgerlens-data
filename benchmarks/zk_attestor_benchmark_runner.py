#!/usr/bin/env python3
"""Benchmark runner for ZK attestor proof verification (Issue #952).

Run this script to measure proof-verification latency and throughput at
representative production volume and (optionally) save results to a JSON file.

Usage::

    # Print results to stdout
    python -m benchmarks.zk_attestor_benchmark_runner

    # Save results to a JSON file
    python -m benchmarks.zk_attestor_benchmark_runner --output results.json

    # Custom parameters
    python -m benchmarks.zk_attestor_benchmark_runner \\
        --n-verifications 2000 \\
        --n-unique-proofs 100 \\
        --amounts-per-proof 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def run(
    n_verifications: int = 1000,
    n_unique_proofs: int = 50,
    amounts_per_proof: int = 100,
    output: str | None = None,
    quiet: bool = False,
) -> dict:
    """Execute the benchmark and return the structured results dict."""
    from integrations.zk_attestor import benchmark_proof_verification

    results = {}

    for use_cache in (False, True):
        label = "with_cache" if use_cache else "no_cache"
        if not quiet:
            print(
                f"Running {n_verifications:,} verifications "
                f"({n_unique_proofs} unique proofs, {amounts_per_proof} amounts each) "
                f"[{label}]...",
                flush=True,
            )
        r = benchmark_proof_verification(
            n_verifications=n_verifications,
            n_unique_proofs=n_unique_proofs,
            amounts_per_proof=amounts_per_proof,
            use_cache=use_cache,
        )
        results[label] = r.to_dict()
        if not quiet:
            print(
                f"  throughput : {r.throughput_per_second:,.1f} verifications/sec\n"
                f"  p50        : {r.p50_ms:.3f} ms\n"
                f"  p95        : {r.p95_ms:.3f} ms\n"
                f"  p99        : {r.p99_ms:.3f} ms\n"
                f"  cache hits : {r.cache_hit_rate * 100:.1f}%"
            )

    # Speedup summary
    if not quiet and "no_cache" in results and "with_cache" in results:
        nc = results["no_cache"]["throughput_per_second"]
        wc = results["with_cache"]["throughput_per_second"]
        if nc > 0:
            print(f"\nCache speedup: {wc / nc:.1f}× throughput improvement")

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            json.dump(results, fh, indent=2)
        if not quiet:
            print(f"\nResults saved to {out_path}")

    return results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark ZK attestor proof verification latency and throughput."
    )
    p.add_argument(
        "--n-verifications",
        type=int,
        default=1000,
        metavar="N",
        help="Total verification calls per mode (default: 1000)",
    )
    p.add_argument(
        "--n-unique-proofs",
        type=int,
        default=50,
        metavar="N",
        help="Number of distinct proofs generated before the timed loop (default: 50)",
    )
    p.add_argument(
        "--amounts-per-proof",
        type=int,
        default=100,
        metavar="N",
        help="Trade amounts per proof (default: 100)",
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Write JSON results to this path (optional)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout output (useful when piping JSON output only)",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    results = run(
        n_verifications=args.n_verifications,
        n_unique_proofs=args.n_unique_proofs,
        amounts_per_proof=args.amounts_per_proof,
        output=args.output,
        quiet=args.quiet,
    )
    if args.quiet:
        # Print JSON to stdout when quiet so the caller can capture it
        json.dump(results, sys.stdout, indent=2)
        print()
