"""Pydantic schemas for raw data pulled from the Stellar Horizon API.

These models are the shared contract between the ingestion layer and the
detection layer (Benford engine + feature engineering). Keep field names
stable — downstream code and the `ledgerlens-core` shared types mirror them.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class Asset(BaseModel):
    code: str
    issuer: str | None = None  # None / "native" for XLM

    def pair_id(self, other: "Asset") -> str:
        return f"{self.code}:{self.issuer or 'native'}/{other.code}:{other.issuer or 'native'}"


class Trade(BaseModel):
    """A single executed trade on the SDEX."""

    trade_id: str
    ledger_close_time: datetime
    base_account: str
    counter_account: str
    base_asset: Asset
    counter_asset: Asset
    base_amount: float
    counter_amount: float
    price: float

    @property
    def amount(self) -> float:
        """Primary amount used for Benford digit analysis."""
        return self.base_amount


class OrderBookEvent(BaseModel):
    """Order placement / cancellation event."""

    event_id: str
    account: str
    ledger_close_time: datetime
    selling: Asset
    buying: Asset
    amount: float
    price: float
    action: str = Field(description="one of: created, cancelled, updated")


class AccountActivity(BaseModel):
    """Lightweight summary of an account, used for wallet graph features."""

    account_id: str
    account_created_at: datetime
    funding_account: str | None = None
    home_domain: str | None = None
