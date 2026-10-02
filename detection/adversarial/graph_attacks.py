"""Graph-structural evasion attacks for transaction DataFrames.

These attacks operate on the graph encoded by ``base_account`` and
``counter_account`` columns while preserving the input trade schema. They are
intended for offline robustness evaluation of motif and community detectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from detection.adversarial.attacks import AttackStrategy

_REQUIRED_COLUMNS = {"base_account", "counter_account"}


def _validate_graph_columns(trades_df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS - set(trades_df.columns)
    if missing:
        raise ValueError(f"transaction graph requires columns: {sorted(missing)}")


@dataclass
class EdgeInsertionEvasion(AttackStrategy):
    """Insert benign-looking intermediary edges to dilute graph motifs.

    For each selected source/counterparty edge ``source -> target``, the direct
    edge is retained and a deterministic two-hop path ``source -> intermediary
    -> target`` is added. Keeping the original row makes the attack suitable
    for counterfactual evaluation: only structural decoys are introduced.
    """

    n_edges: int = 1
    amount_fraction: float = 0.01

    def __post_init__(self) -> None:
        if self.n_edges < 0:
            raise ValueError("n_edges must be non-negative")
        if not 0.0 < self.amount_fraction <= 1.0:
            raise ValueError("amount_fraction must be in (0, 1]")

    def perturb(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        _validate_graph_columns(trades_df)
        if trades_df.empty or self.n_edges == 0:
            return trades_df.copy()

        selected = trades_df.head(min(self.n_edges, len(trades_df)))
        additions: list[pd.Series] = []
        for ordinal, (_, row) in enumerate(selected.iterrows()):
            intermediary = f"GEDGE_DECOY_{ordinal:06d}"
            first = row.copy()
            second = row.copy()
            first["counter_account"] = intermediary
            second["base_account"] = intermediary
            if "trade_id" in first.index:
                first["trade_id"] = f"{row['trade_id']}-edge-a-{ordinal}"
                second["trade_id"] = f"{row['trade_id']}-edge-b-{ordinal}"
            if "amount" in first.index:
                first["amount"] = float(row["amount"]) * self.amount_fraction
                second["amount"] = float(row["amount"]) * self.amount_fraction
            additions.extend((first, second))
        return pd.concat([trades_df, pd.DataFrame(additions)], ignore_index=True)


@dataclass
class NodeSplittingEvasion(AttackStrategy):
    """Split one high-degree wallet across deterministic Sybil identities.

    Rows involving ``target_wallet`` are assigned round-robin to fresh aliases.
    This preserves trade volume and schema while changing degree, community, and
    motif structure. If no target is supplied, the most frequent graph endpoint
    is selected.
    """

    n_sybils: int = 2
    target_wallet: str | None = None

    def __post_init__(self) -> None:
        if self.n_sybils < 1:
            raise ValueError("n_sybils must be at least 1")

    def perturb(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        _validate_graph_columns(trades_df)
        df = trades_df.copy()
        if df.empty or self.n_sybils == 1:
            return df

        if self.target_wallet is None:
            endpoints = pd.concat([df["base_account"], df["counter_account"]], ignore_index=True)
            target = str(endpoints.value_counts().index[0])
        else:
            target = self.target_wallet

        aliases = [f"GSPLIT_{i:06d}" for i in range(self.n_sybils)]
        occurrence = 0
        for index in df.index:
            for column in ("base_account", "counter_account"):
                if df.at[index, column] == target:
                    df.at[index, column] = aliases[occurrence % self.n_sybils]
                    occurrence += 1
        return df
