"""Pydantic schemas for raw data pulled from the Stellar Horizon API.

These models are the shared contract between the ingestion layer and the
detection layer (Benford engine + feature engineering). Keep field names
stable — downstream code and the `ledgerlens-core` shared types mirror them.

Precision Handling
------------------
Financial fields (amounts, prices) use Decimal type with validation to prevent
float precision errors. See utils.decimal_guards for the precision guard system.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol

from pydantic import BaseModel, Field, field_validator


class NormalizedAmount(Protocol):
    """Structural return type for currency-normalized values."""

    value: Decimal


class NormalizationStrategy(Protocol):
    """Minimal normalization interface used by ingestion records."""

    def normalize(self, amount: Decimal, asset: "Asset", at: datetime) -> Any: ...


class Asset(BaseModel):
    code: str
    issuer: str | None = None  # None / "native" for XLM

    def pair_id(self, other: "Asset") -> str:
        return f"{self.code}:{self.issuer or 'native'}/{other.code}:{other.issuer or 'native'}"


class Trade(BaseModel):
    """A single executed trade on the SDEX.

    Precision Note
    --------------
    Amount and price fields use Decimal type to prevent floating-point
    precision errors. Stellar amounts have 7-decimal precision (stroops).
    Values are validated on initialization to ensure they meet Stellar
    constraints (7 decimals max, int64 bounds).
    """

    trade_id: str
    ledger_close_time: datetime
    base_account: str
    counter_account: str
    base_asset: Asset
    counter_asset: Asset
    base_amount: Decimal
    counter_amount: Decimal
    price: Decimal

    @field_validator("base_amount", "counter_amount", "price", mode="before")
    @classmethod
    def validate_stellar_amounts(cls, v):
        """Validate amounts meet Stellar precision requirements."""
        from utils.decimal_guards import validate_stellar_amount

        return validate_stellar_amount(v)

    def __repr__(self) -> str:
        # Compact summary for readable log/debugger output, not exhaustive field dump.
        pair = f"{self.base_asset.code}:{self.counter_asset.code}"
        return (
            f"Trade(id={self.trade_id!r}, pair={pair}, "
            f"amount={self.base_amount}, time={self.ledger_close_time.isoformat()})"
        )

    @property
    def amount(self) -> Decimal:
        """Primary amount used for Benford digit analysis."""
        return self.base_amount

    def normalize_base_amount(
        self,
        strategy: "NormalizationStrategy",
    ) -> "NormalizedAmount":
        """Normalize base amount to strategy's base currency.

        Parameters
        ----------
        strategy : NormalizationStrategy
            Normalization strategy (e.g., XLMNormalization, USDNormalization)

        Returns
        -------
        NormalizedAmount
            Normalized base amount

        Examples
        --------
        >>> from utils.currency_normalization import create_xlm_strategy
        >>>
        >>> trade = Trade(...)
        >>> strategy = create_xlm_strategy()
        >>> normalized = trade.normalize_base_amount(strategy)
        >>> print(f"Base: {normalized.value} XLM")
        """
        return strategy.normalize(
            self.base_amount,
            self.base_asset,
            self.ledger_close_time,
        )

    def normalize_counter_amount(
        self,
        strategy: "NormalizationStrategy",
    ) -> "NormalizedAmount":
        """Normalize counter amount to strategy's base currency.

        Parameters
        ----------
        strategy : NormalizationStrategy
            Normalization strategy

        Returns
        -------
        NormalizedAmount
            Normalized counter amount

        Examples
        --------
        >>> trade = Trade(...)
        >>> strategy = create_xlm_strategy()
        >>> normalized = trade.normalize_counter_amount(strategy)
        >>> print(f"Counter: {normalized.value} XLM")
        """
        return strategy.normalize(
            self.counter_amount,
            self.counter_asset,
            self.ledger_close_time,
        )

    def normalize_both_amounts(
        self,
        strategy: "NormalizationStrategy",
    ) -> tuple["NormalizedAmount", "NormalizedAmount"]:
        """Normalize both base and counter amounts.

        This is the most common use case for fraud detection: normalize both
        sides of a trade to a common currency for comparison.

        Parameters
        ----------
        strategy : NormalizationStrategy
            Normalization strategy

        Returns
        -------
        tuple[NormalizedAmount, NormalizedAmount]
            (normalized_base, normalized_counter)

        Examples
        --------
        >>> trade = Trade(...)
        >>> strategy = create_xlm_strategy()
        >>> norm_base, norm_counter = trade.normalize_both_amounts(strategy)
        >>>
        >>> # Now comparable!
        >>> if abs(norm_base.value - norm_counter.value) > Decimal("0.01"):
        ...     print("Trade amounts don't match after normalization")
        """
        return (
            self.normalize_base_amount(strategy),
            self.normalize_counter_amount(strategy),
        )

    def get_normalized_trade_value(
        self,
        strategy: "NormalizationStrategy",
    ) -> "NormalizedAmount":
        """Get total trade value in base currency.

        Returns the base amount normalized (typically used as the "trade value").

        Parameters
        ----------
        strategy : NormalizationStrategy
            Normalization strategy

        Returns
        -------
        NormalizedAmount
            Trade value in base currency

        Examples
        --------
        >>> trade = Trade(...)
        >>> strategy = create_xlm_strategy()
        >>> value = trade.get_normalized_trade_value(strategy)
        >>> print(f"Trade value: {value.value} XLM")
        """
        return self.normalize_base_amount(strategy)


class OrderBookEvent(BaseModel):
    """Order placement / cancellation event.

    Precision Note
    --------------
    Amount and price fields use Decimal type to prevent floating-point
    precision errors. Values are validated on initialization.
    """

    event_id: str
    account: str
    ledger_close_time: datetime
    selling: Asset
    buying: Asset
    amount: Decimal
    price: Decimal
    action: str = Field(description="one of: created, cancelled, updated")

    @field_validator("amount", "price", mode="before")
    @classmethod
    def validate_stellar_amounts(cls, v):
        """Validate amounts meet Stellar precision requirements."""
        from utils.decimal_guards import validate_stellar_amount

        return validate_stellar_amount(v)

    def normalize_amount(
        self,
        strategy: "NormalizationStrategy",
    ) -> "NormalizedAmount":
        """Normalize order amount to strategy's base currency.

        Parameters
        ----------
        strategy : NormalizationStrategy
            Normalization strategy

        Returns
        -------
        NormalizedAmount
            Normalized amount

        Examples
        --------
        >>> from utils.currency_normalization import create_xlm_strategy
        >>>
        >>> order = OrderBookEvent(...)
        >>> strategy = create_xlm_strategy()
        >>> normalized = order.normalize_amount(strategy)
        >>> print(f"Order size: {normalized.value} XLM")
        """
        return strategy.normalize(
            self.amount,
            self.selling,
            self.ledger_close_time,
        )

    def get_normalized_order_value(
        self,
        strategy: "NormalizationStrategy",
    ) -> "NormalizedAmount":
        """Get total order value (amount * price) in base currency.

        Calculates the order value and normalizes it to base currency.

        Parameters
        ----------
        strategy : NormalizationStrategy
            Normalization strategy

        Returns
        -------
        NormalizedAmount
            Order value in base currency

        Examples
        --------
        >>> order = OrderBookEvent(...)
        >>> strategy = create_xlm_strategy()
        >>> value = order.get_normalized_order_value(strategy)
        >>> print(f"Order value: {value.value} XLM")
        """
        # Calculate order value in selling asset
        order_value = self.amount * self.price

        return strategy.normalize(
            order_value,
            self.selling,
            self.ledger_close_time,
        )


class AccountActivity(BaseModel):
    """Lightweight summary of an account, used for wallet graph features."""

    account_id: str
    account_created_at: datetime
    funding_account: str | None = None
    home_domain: str | None = None


class BotFingerprint(BaseModel):
    """Bot detection fingerprint extracted from Horizon event patterns."""

    account_id: str
    fingerprint_version: str = Field(
        default="v1", description="Fingerprint feature schema version for compatibility tracking"
    )
    trust_line_creation_latency_seconds: float | None = Field(
        default=None, description="Time in seconds from account creation to first trust line"
    )
    inter_trade_interval_cv: float | None = Field(
        default=None, description="Coefficient of variation of inter-trade intervals (low=robotic)"
    )
    account_management_cluster_score: float = Field(
        default=0.0, description="Entropy of operation type distribution (low=clustered=bot)"
    )
    is_valid: bool = Field(default=True, description="False if insufficient data (< 5 trades)")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for bot fingerprint validity [0, 1]",
    )
