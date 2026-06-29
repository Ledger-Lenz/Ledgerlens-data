"""Causal discovery on wallet transaction features using the PC algorithm.

Usage:
    python -m scripts.run_causal_discovery --data data/synthetic_dataset.parquet

This module exposes :func:`discover_feature_dag` which wraps the
``causal-learn`` PC implementation and persists the discovered DAG as
both a DOT file and a JSON adjacency list.
"""

from __future__ import annotations

import json
import logging
import os
import signal
from contextlib import contextmanager
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_DOT_PATH = os.path.join(_PROJECT_ROOT, "analysis", "feature_dag.dot")
_DEFAULT_JSON_PATH = os.path.join(_PROJECT_ROOT, "analysis", "feature_dag.json")


# ---------------------------------------------------------------------------
# Timeout context manager
# ---------------------------------------------------------------------------


@contextmanager
def _timeout(seconds: int):
    """Raise TimeoutError after *seconds* seconds (POSIX only)."""

    def _handler(signum, frame):
        raise TimeoutError(f"Causal discovery timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def discover_feature_dag(
    features_df: pd.DataFrame,
    significance_level: float = 0.05,
    max_cond_set_size: int = 3,
    timeout_seconds: int = 600,
    dot_path: str = _DEFAULT_DOT_PATH,
    json_path: str = _DEFAULT_JSON_PATH,
) -> dict:
    """Run the PC algorithm on *features_df* and persist the resulting DAG.

    Missing values are handled via listwise deletion (rows with any NaN are
    dropped).  Columns that are entirely NaN are excluded before deletion.

    Args:
        features_df: Feature matrix (rows = wallets, columns = features).
        significance_level: Independence test significance threshold (α).
        max_cond_set_size: Maximum conditioning set size for PC skeleton search.
        timeout_seconds: Hard wall-clock limit; partial DAG returned on timeout.
        dot_path: Output path for the DOT representation.
        json_path: Output path for the JSON adjacency list.

    Returns:
        Adjacency list ``{node: [list_of_children]}`` reflecting the DAG.
    """
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz, g2
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "causal-learn is required for causal discovery. "
            "Install it with: pip install causal-learn"
        ) from exc

    df = _preprocess(features_df)
    if df.empty:
        logger.warning("No data remaining after preprocessing; returning empty DAG.")
        adj: dict[str, list[str]] = {}
        _persist(adj, df.columns.tolist(), dot_path, json_path)
        return adj

    # Determine independence test per column type
    # Use Fisher's Z for continuous, G² for discrete
    col_types = _infer_column_types(df)
    # PC from causal-learn expects a single test; default to fisherz for mixed
    # (handles continuous majority) and fall back to g2 for fully discrete.
    discrete_ratio = sum(1 for t in col_types.values() if t == "discrete") / max(len(col_types), 1)
    indep_test = g2 if discrete_ratio > 0.9 else fisherz

    data = df.values.astype(float)
    node_names = df.columns.tolist()

    cg = None
    try:
        with _timeout(timeout_seconds):
            cg = pc(
                data,
                alpha=significance_level,
                indep_test=indep_test,
                depth=max_cond_set_size,
                node_names=node_names,
                show_progress=False,
            )
    except TimeoutError:
        logger.warning(
            "Causal discovery timed out after %ds; returning partial DAG.",
            timeout_seconds,
        )

    adj = _extract_adjacency(cg, node_names)
    _persist(adj, node_names, dot_path, json_path)
    _log_summary(adj)
    return adj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all-NaN columns, then apply listwise deletion."""
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        logger.warning(
            "Excluding %d all-NaN column(s) before causal discovery: %s",
            len(all_nan_cols),
            all_nan_cols,
        )
        df = df.drop(columns=all_nan_cols)

    before = len(df)
    df = df.dropna()
    after = len(df)
    if before != after:
        logger.warning(
            "Listwise deletion removed %d row(s) with missing values (%d → %d).",
            before - after,
            before,
            after,
        )
    return df.reset_index(drop=True)


def _infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    types: dict[str, str] = {}
    for col in df.columns:
        n_unique = df[col].nunique()
        types[col] = "discrete" if (df[col].dtype == bool or n_unique <= 10) else "continuous"
    return types


def _extract_adjacency(cg, node_names: list[str]) -> dict[str, list[str]]:
    """Convert causal-learn CausalGraph adjacency matrix to a dict."""
    adj: dict[str, list[str]] = {n: [] for n in node_names}
    if cg is None:
        return adj

    try:
        g = cg.G
        # causal-learn uses an adjacency matrix where entry [i,j] = -1 means i→j
        mat = g.graph
        n = len(node_names)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # mat[i, j] == -1 and mat[j, i] == 1 encodes edge i → j
                if mat[i, j] == -1 and mat[j, i] == 1:
                    adj[node_names[i]].append(node_names[j])
    except Exception as exc:
        logger.warning("Could not extract adjacency from CausalGraph: %s", exc)

    return adj


def _persist(adj: dict[str, list[str]], node_names: list[str], dot_path: str, json_path: str) -> None:
    os.makedirs(os.path.dirname(dot_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # DOT
    lines = ["digraph feature_dag {"]
    for node, children in adj.items():
        for child in children:
            lines.append(f'    "{node}" -> "{child}";')
    lines.append("}")
    with open(dot_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    # JSON
    with open(json_path, "w") as f:
        json.dump(adj, f, indent=2)

    logger.info("DAG persisted: %s  %s", dot_path, json_path)


def _log_summary(adj: dict[str, list[str]]) -> None:
    edges = [(src, tgt) for src, tgts in adj.items() for tgt in tgts]
    if not edges:
        logger.info("Causal discovery: no statistically supported edges found.")
        return
    logger.info("Causal discovery: found %d edge(s):", len(edges))
    for src, tgt in edges:
        logger.info("  %s → %s", src, tgt)
