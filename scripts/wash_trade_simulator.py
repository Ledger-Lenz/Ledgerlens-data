"""Wash Trade Simulation Engine (WTSE) — high-fidelity attacker strategy profiles.

Each profile generates realistic trade DataFrames that mimic sophisticated
wash-trading behaviours on the Stellar SDEX.  The output schema matches
``ingestion.historical_loader.trades_to_dataframe`` so that features can be
computed with ``detection.feature_engineering.build_feature_matrix``.
"""

from __future__ import annotations

import abc
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import config

RANDOM_SEED = 42

TRADE_COLUMNS = [
    "trade_id",
    "ledger_close_time",
    "base_account",
    "counter_account",
    "base_asset",
    "counter_asset",
    "amount",
    "price",
]


def _make_wallet(idx: int) -> str:
    return f"GSIM{idx:06d}"


def _make_trade_id(wallet_idx: int, trade_idx: int) -> str:
    return f"SIM-{wallet_idx:06d}-{trade_idx:06d}"


def _default_asset_fn(_wallet_idx: int, _trade_idx: int) -> tuple[str, str]:
    return "USDC:GA5Z.../XLM:native", "native"


# ---------------------------------------------------------------------------
# Base profile
# ---------------------------------------------------------------------------


@dataclass
class BaseAttackerProfile(abc.ABC):
    """Abstract base for all attacker strategy profiles.

    Subclasses must implement :meth:`generate_trades` which returns a
    DataFrame conforming to the schema of
    ``ingestion.historical_loader.trades_to_dataframe``.
    """

    n_wallets: int = 50
    trades_per_wallet: int = 100
    seed: int = RANDOM_SEED
    name: str = "base"

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    @abc.abstractmethod
    def generate_trades(self) -> pd.DataFrame:
        """Produce a trade DataFrame.  See ``trades_to_dataframe`` for the schema."""

    def _make_trade_rows(
        self,
        wallet_indices: list[int],
        amount_fn,
        time_fn,
        counterparty_fn,
        asset_fn=_default_asset_fn,
    ) -> list[dict]:
        rows = []
        rng = self._rng()
        for wi in wallet_indices:
            wallet = _make_wallet(wi)
            cp = counterparty_fn(wi)
            for ti in range(self.trades_per_wallet):
                ba, ca = asset_fn(wi, ti)
                rows.append(
                    {
                        "trade_id": _make_trade_id(wi, ti),
                        "ledger_close_time": time_fn(wi, ti),
                        "base_account": wallet,
                        "counter_account": cp,
                        "base_asset": ba,
                        "counter_asset": ca,
                        "amount": amount_fn(wi, ti),
                        "price": 1.0,
                    }
                )
        return rows


# ---------------------------------------------------------------------------
# Profile 1: NaiveAttacker
# ---------------------------------------------------------------------------


@dataclass
class NaiveAttacker(BaseAttackerProfile):
    """Fixed amounts, regular intervals — baseline wash trader.

    Each wallet trades a fixed amount (500.0) at perfectly regular 1-minute
    intervals with a single counterparty.  This is the simplest profile and
    serves as the baseline comparison.
    """

    name: str = "naive"
    fixed_amount: float = 500.0
    interval_seconds: int = 60

    def generate_trades(self) -> pd.DataFrame:
        rng = self._rng()
        base_time = pd.Timestamp("2024-01-01", tz="UTC")
        rows = self._make_trade_rows(
            wallet_indices=list(range(self.n_wallets)),
            amount_fn=lambda _wi, _ti: self.fixed_amount,
            time_fn=lambda _wi, ti: base_time + pd.Timedelta(seconds=ti * self.interval_seconds),
            counterparty_fn=lambda wi: _make_wallet(self.n_wallets + wi),
        )
        return pd.DataFrame(rows, columns=TRADE_COLUMNS)


# ---------------------------------------------------------------------------
# Profile 2: TimingJitterAttacker
# ---------------------------------------------------------------------------


@dataclass
class TimingJitterAttacker(BaseAttackerProfile):
    """Poisson-distributed trade intervals with configurable lambda.

    Instead of fixed intervals, the gap between consecutive trades follows
    an exponential distribution (equivalent to a Poisson process), making
    the timing pattern more realistic.
    """

    name: str = "timing_jitter"
    base_amount: float = 500.0
    lambda_seconds: float = 60.0

    def generate_trades(self) -> pd.DataFrame:
        rng = self._rng()
        base_time = pd.Timestamp("2024-01-01", tz="UTC")

        def time_fn(_wi: int, ti: int) -> pd.Timestamp:
            if ti == 0:
                return base_time
            gaps = rng.exponential(scale=self.lambda_seconds, size=ti)
            return base_time + pd.Timedelta(seconds=float(gaps.sum()))

        rows = self._make_trade_rows(
            wallet_indices=list(range(self.n_wallets)),
            amount_fn=lambda _wi, _ti: self.base_amount,
            time_fn=time_fn,
            counterparty_fn=lambda wi: _make_wallet(self.n_wallets + wi),
        )
        return pd.DataFrame(rows, columns=TRADE_COLUMNS)


# ---------------------------------------------------------------------------
# Profile 3: AmountConformanceAttacker
# ---------------------------------------------------------------------------


@dataclass
class AmountConformanceAttacker(BaseAttackerProfile):
    """Draws amounts from a Benford-conforming distribution via inverse-transform sampling.

    The log-uniform distribution over a two-decade range produces leading
    digits that naturally conform to Benford's Law (by scale invariance).
    """

    name: str = "amount_conformance"
    min_amount: float = 50.0
    max_amount: float = 5000.0

    def generate_trades(self) -> pd.DataFrame:
        rng = self._rng()
        log_min = math.log10(self.min_amount)
        log_max = math.log10(self.max_amount)

        def amount_fn(_wi: int, _ti: int) -> float:
            return float(10.0 ** rng.uniform(log_min, log_max))

        base_time = pd.Timestamp("2024-01-01", tz="UTC")

        rows = self._make_trade_rows(
            wallet_indices=list(range(self.n_wallets)),
            amount_fn=amount_fn,
            time_fn=lambda _wi, ti: base_time + pd.Timedelta(minutes=ti),
            counterparty_fn=lambda wi: _make_wallet(self.n_wallets + wi),
        )
        return pd.DataFrame(rows, columns=TRADE_COLUMNS)


# ---------------------------------------------------------------------------
# Profile 4: RingAttacker
# ---------------------------------------------------------------------------


@dataclass
class RingAttacker(BaseAttackerProfile):
    """N-wallet ring where each wallet trades only with its neighbours.

    Wallet i sends trades to wallet (i+1) % n_wallets, creating a cycle.
    This produces high network centrality and low funding_source_similarity
    when n_wallets is sufficiently large.
    """

    name: str = "ring"
    fixed_amount: float = 500.0
    interval_seconds: int = 60

    def generate_trades(self) -> pd.DataFrame:
        base_time = pd.Timestamp("2024-01-01", tz="UTC")

        def counterparty_fn(wi: int) -> str:
            return _make_wallet((wi + 1) % self.n_wallets)

        rows = self._make_trade_rows(
            wallet_indices=list(range(self.n_wallets)),
            amount_fn=lambda _wi, _ti: self.fixed_amount,
            time_fn=lambda _wi, ti: base_time + pd.Timedelta(seconds=ti * self.interval_seconds),
            counterparty_fn=counterparty_fn,
        )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Profile 5: LayeringAttacker
# ---------------------------------------------------------------------------


@dataclass
class LayeringAttacker(BaseAttackerProfile):
    """Interleaves wash trades with legitimate-looking noise trades at 3:1 ratio.

    For every wash trade (is_wash=True), three noise trades (is_wash=False)
    are generated with random amounts, random counterparties, and realistic
    timing to mimic legitimate activity.

    The returned DataFrame includes an ``is_wash`` column indicating which
    rows represent wash trades.
    """

    name: str = "layering"
    wash_amount: float = 500.0
    noise_min_amount: float = 10.0
    noise_max_amount: float = 1000.0
    wash_to_noise_ratio: int = 3  # 3 noise trades per wash trade

    def generate_trades(self) -> pd.DataFrame:
        rng = self._rng()
        base_time = pd.Timestamp("2024-01-01", tz="UTC")
        rows = []
        trade_idx = 0

        for wi in range(self.n_wallets):
            wallet = _make_wallet(wi)
            cp_wash = _make_wallet(self.n_wallets + wi)
            for _ in range(self.trades_per_wallet // (self.wash_to_noise_ratio + 1)):
                wash_time = base_time + pd.Timedelta(minutes=trade_idx)
                rows.append(
                    {
                        "trade_id": _make_trade_id(wi, trade_idx),
                        "ledger_close_time": wash_time,
                        "base_account": wallet,
                        "counter_account": cp_wash,
                        "base_asset": "USDC:GA5Z.../XLM:native",
                        "counter_asset": "native",
                        "amount": self.wash_amount,
                        "price": 1.0,
                        "is_wash": True,
                    }
                )
                trade_idx += 1

                for _ in range(self.wash_to_noise_ratio):
                    noise_cp = _make_wallet(
                        rng.integers(0, self.n_wallets * 10)
                    )
                    noise_time = base_time + pd.Timedelta(minutes=trade_idx)
                    rows.append(
                        {
                            "trade_id": _make_trade_id(wi, trade_idx),
                            "ledger_close_time": noise_time,
                            "base_account": wallet,
                            "counter_account": noise_cp,
                            "base_asset": "USDC:GA5Z.../XLM:native",
                            "counter_asset": "native",
                            "amount": float(
                                rng.uniform(self.noise_min_amount, self.noise_max_amount)
                            ),
                            "price": 1.0,
                            "is_wash": False,
                        }
                    )
                    trade_idx += 1

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Profile 6: CrossPairAttacker
# ---------------------------------------------------------------------------


@dataclass
class CrossPairAttacker(BaseAttackerProfile):
    """Rotates wash volume across K asset pairs to dilute per-pair signal.

    Each wallet distributes its trades across ``n_pairs`` different asset
    pairs, rotating through them in round-robin order.  This lowers the
    per-pair Benford anomaly and reduces cross-pair coordination signals.
    """

    name: str = "cross_pair"
    fixed_amount: float = 500.0
    interval_seconds: int = 60
    n_pairs: int = 3
    pair_assets: list[tuple[str, str]] = field(
        default_factory=lambda: [
            ("USDC:GA5Z.../XLM:native", "native"),
            ("BTC:GA5Z.../XLM:native", "native"),
            ("AQUA:GA5Z.../XLM:native", "native"),
        ]
    )

    def generate_trades(self) -> pd.DataFrame:
        base_time = pd.Timestamp("2024-01-01", tz="UTC")

        def asset_fn(wi: int, ti: int) -> tuple[str, str]:
            pair_idx = ti % self.n_pairs
            if pair_idx < len(self.pair_assets):
                return self.pair_assets[pair_idx]
            return self.pair_assets[0]

        rows = self._make_trade_rows(
            wallet_indices=list(range(self.n_wallets)),
            amount_fn=lambda _wi, _ti: self.fixed_amount,
            time_fn=lambda _wi, ti: base_time + pd.Timedelta(seconds=ti * self.interval_seconds),
            counterparty_fn=lambda wi: _make_wallet(self.n_wallets + wi),
            asset_fn=asset_fn,
        )
        return pd.DataFrame(rows, columns=TRADE_COLUMNS)


# ---------------------------------------------------------------------------
# Profile 7: AdaptiveAttacker
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveAttacker(BaseAttackerProfile):
    """Reads the current model's feature importances and down-weights
    the highest-importance features.

    The attacker generates trades like :class:`NaiveAttacker`, loads a
    trained model from ``model_path``, extracts feature importances, and
    then adjusts trade parameters (amounts, timing, counterparty selection)
    to reduce the top-K most discriminative features.
    """

    name: str = "adaptive"
    model_path: str = ""
    fixed_amount: float = 500.0
    interval_seconds: int = 60
    top_k: int = 3
    reduction_factor: float = 0.3

    def _load_feature_importances(self) -> dict[str, float]:
        if not self.model_path:
            return {}
        try:
            import joblib

            model = joblib.load(self.model_path)
            if hasattr(model, "feature_importances_") and hasattr(model, "feature_names_in_"):
                names = model.feature_names_in_
                importances = model.feature_importances_
                return dict(zip(names, importances))
            if isinstance(model, dict):
                for _name, est in model.items():
                    if hasattr(est, "feature_importances_"):
                        names = est.feature_names_in_
                        importances = est.feature_importances_
                        return dict(zip(names, importances))
        except Exception:
            return {}
        return {}

    def generate_trades(self) -> pd.DataFrame:
        importances = self._load_feature_importances()
        rng = self._rng()
        base_time = pd.Timestamp("2024-01-01", tz="UTC")
        rows = []

        top_features = (
            sorted(importances, key=lambda k: importances[k], reverse=True)[: self.top_k]
            if importances
            else []
        )
        uses_benford = any("benford" in f for f in top_features)
        uses_concentration = any("concentration" in f for f in top_features)
        uses_timing = any("inter_arrival" in f or "clustering" in f for f in top_features)

        for wi in range(self.n_wallets):
            wallet = _make_wallet(wi)
            cp = _make_wallet(self.n_wallets + wi)
            for ti in range(self.trades_per_wallet):
                amount = self.fixed_amount
                if uses_benford:
                    log_min = math.log10(amount / 10.0)
                    log_max = math.log10(amount * 10.0)
                    amount = 10.0 ** rng.uniform(log_min, log_max)
                elif uses_concentration:
                    amount = float(rng.uniform(amount * 0.5, amount * 1.5))

                interval: float = float(self.interval_seconds)
                if uses_timing:
                    interval = float(rng.uniform(interval * 0.5, interval * 2.0))

                t = base_time + pd.Timedelta(seconds=ti * interval)
                rows.append(
                    {
                        "trade_id": _make_trade_id(wi, ti),
                        "ledger_close_time": t,
                        "base_account": wallet,
                        "counter_account": cp,
                        "base_asset": "USDC:GA5Z.../XLM:native",
                        "counter_asset": "native",
                        "amount": float(amount),
                        "price": 1.0,
                    }
                )

        return pd.DataFrame(rows, columns=TRADE_COLUMNS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_wallets": self.n_wallets,
            "trades_per_wallet": self.trades_per_wallet,
            "model_path": self.model_path,
            "top_k": self.top_k,
            "reduction_factor": self.reduction_factor,
        }


# ---------------------------------------------------------------------------
# Simulator factory
# ---------------------------------------------------------------------------

PROFILE_REGISTRY: dict[str, type[BaseAttackerProfile]] = {
    "NaiveAttacker": NaiveAttacker,
    "TimingJitterAttacker": TimingJitterAttacker,
    "AmountConformanceAttacker": AmountConformanceAttacker,
    "RingAttacker": RingAttacker,
    "LayeringAttacker": LayeringAttacker,
    "CrossPairAttacker": CrossPairAttacker,
    "AdaptiveAttacker": AdaptiveAttacker,
}


def create_profile(name: str, **kwargs) -> BaseAttackerProfile:
    """Instantiate an attacker profile by name.

    ``kwargs`` override the dataclass defaults.
    """
    cls = PROFILE_REGISTRY.get(name)
    if cls is None:
        msg = f"Unknown profile {name!r}. Choices: {list(PROFILE_REGISTRY)}"
        raise ValueError(msg)
    return cls(**kwargs)


def trades_to_feature_matrix(trades: pd.DataFrame) -> pd.DataFrame:
    """Convert a simulated trade DataFrame to a feature matrix.

    Uses ``detection.feature_engineering.build_feature_matrix`` to compute
    the full 30+ feature vector per wallet.

    If ``trades`` contains an ``is_wash`` column, it is preserved (but
    renamed to ``label``) in the output.
    """
    from detection.feature_engineering import build_feature_matrix

    if trades.empty:
        return pd.DataFrame()

    has_is_wash = "is_wash" in trades.columns
    df = build_feature_matrix(trades)
    if has_is_wash and not df.empty:
        labels = {}
        for wallet in df["wallet"]:
            mask = (trades["base_account"] == wallet) & (trades.get("is_wash", pd.Series([True])))
            is_wash_trades = trades.loc[
                trades["base_account"] == wallet, "is_wash"
            ]
            labels[wallet] = int(is_wash_trades.any()) if len(is_wash_trades) > 0 else 1
        df["label"] = df["wallet"].map(labels)
    else:
        df["label"] = 1
    return df
