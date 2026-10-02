"""Registry of all LedgerLens data importers with capability metadata.

This module registers all existing importers with the capability discovery system.
Import this module to populate the global importer registry.

The registration happens at import time via decorators, so simply importing this
module ensures all importers are discoverable:

    >>> from ingestion.registered_importers import *  # Registers all importers
    >>> from ingestion.importer_registry import get_registry
    >>>
    >>> registry = get_registry()
    >>> print(registry.list_all())
    ['account_activity_loader', 'amm_pool_loader', 'asset_metadata_fetcher', ...]

Design Notes
------------
Each registered importer is a thin wrapper around the actual implementation that:
1. Declares capabilities via the @register_importer decorator
2. Delegates all actual work to the underlying implementation
3. Preserves the original API for backward compatibility

This approach allows us to add capability discovery without modifying existing code.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

import pandas as pd
from stellar_sdk import Asset as SdkAsset

# Import actual implementations
from ingestion import (
    account_activity_loader,
    amm_pool_loader,
    asset_metadata_fetcher,
    historical_loader,
    horizon_streamer,
    orderbook_loader,
    payment_path_analyzer,
)
from ingestion.data_models import (
    AccountActivity,
    OrderBookEvent,
    Trade,
)
from ingestion.importer_registry import (
    DataSource,
    DataType,
    ImporterCapability,
    PerformanceCharacteristics,
    register_importer,
)
from utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "HorizonStreamerRegistry",
    "HistoricalLoaderRegistry",
    "OrderbookLoaderRegistry",
    "AccountActivityLoaderRegistry",
    "AMMPoolLoaderRegistry",
    "AssetMetadataFetcherRegistry",
    "PaymentPathAnalyzerRegistry",
]


# ============================================================================
# 1. Horizon Streamer (Real-time SSE)
# ============================================================================


@register_importer(
    strict=True,
    name="horizon_streamer",
    description="""Real-time trade streaming via Horizon Server-Sent Events (SSE).
    
    Provides continuous trade data with sub-5-second latency. Supports multi-region
    failover when HORIZON_FAILOVER_URLS is configured. Automatically reconnects on
    connection loss and preserves cursor to avoid missing trades.
    
    Key features:
    - Real-time: ~3-5 second latency from ledger close
    - Failover: Transparent multi-region endpoint switching
    - Cursor management: Never loses position across reconnects
    - Retry: Exponential backoff up to 5 attempts
    """,
    capabilities=(
        ImporterCapability.STREAMING
        | ImporterCapability.REAL_TIME
        | ImporterCapability.FAILOVER
        | ImporterCapability.RETRY
        | ImporterCapability.CURSOR_MANAGEMENT
        | ImporterCapability.VALIDATION
        | ImporterCapability.ASSET_FILTER
    ),
    data_types={DataType.TRADE},
    sources={DataSource.HORIZON_SSE},
    performance=PerformanceCharacteristics(
        typical_latency_ms=3000,  # 3 seconds
        throughput_records_per_sec=100,  # Depends on DEX activity
        memory_overhead_mb=50,  # Per stream
        supports_batching=False,
    ),
    supports_failover=True,
    requires_authentication=False,
    supports_rate_limiting=False,
    version="1.0.0",
    dependencies=frozenset({"stellar-sdk", "urllib3"}),
)
class HorizonStreamerRegistry:
    """Registered wrapper for horizon_streamer module."""

    @staticmethod
    def stream_trades(
        base_asset: SdkAsset,
        counter_asset: SdkAsset,
        cursor: str = "now",
        max_reconnect_attempts: int = 5,
    ) -> Iterator[Trade]:
        """Stream trades in real-time via Horizon SSE.

        See horizon_streamer.stream_trades() for full documentation.
        """
        return horizon_streamer.stream_trades(
            base_asset=base_asset,
            counter_asset=counter_asset,
            cursor=cursor,
            max_reconnect_attempts=max_reconnect_attempts,
        )

    @staticmethod
    def stream_all_watched_pairs() -> Iterator[Trade]:
        """Stream trades for all configured asset pairs.

        See horizon_streamer.stream_all_watched_pairs() for full documentation.
        """
        return horizon_streamer.stream_all_watched_pairs()


# ============================================================================
# 2. Historical Loader (Bulk paginated)
# ============================================================================


@register_importer(
    strict=True,
    name="historical_loader",
    description="""Bulk historical trade loading via Horizon's paginated REST API.
    
    Loads historical trades in batches with configurable page size. Supports time
    range filtering and automatic pagination. Best for backfilling, backtesting,
    and offline analysis.
    
    Key features:
    - Bulk loading: Efficient batch retrieval
    - Pagination: Automatic cursor-based paging
    - Time filtering: Start time boundary
    - DataFrame output: Direct pandas integration
    - Retry: Exponential backoff on failure
    """,
    capabilities=(
        ImporterCapability.BULK
        | ImporterCapability.PAGINATION
        | ImporterCapability.RETRY
        | ImporterCapability.CURSOR_MANAGEMENT
        | ImporterCapability.TIME_RANGE_FILTER
        | ImporterCapability.ASSET_FILTER
        | ImporterCapability.DATAFRAME_OUTPUT
        | ImporterCapability.VALIDATION
    ),
    data_types={DataType.TRADE},
    sources={DataSource.HORIZON_REST},
    performance=PerformanceCharacteristics(
        typical_latency_ms=500,  # Per page
        throughput_records_per_sec=400,  # 200 records per 0.5s page
        memory_overhead_mb=100,  # Depends on batch size
        supports_batching=True,
    ),
    supports_failover=False,  # Could be added via endpoint pool
    requires_authentication=False,
    supports_rate_limiting=True,
    version="1.0.0",
    dependencies=frozenset({"stellar-sdk", "pandas"}),
)
class HistoricalLoaderRegistry:
    """Registered wrapper for historical_loader module."""

    @staticmethod
    def load_trades(
        base_asset: SdkAsset,
        counter_asset: SdkAsset,
        start_time: datetime | None = None,
        limit_per_page: int = 200,
    ) -> Iterator[Trade]:
        """Load historical trades with pagination.

        See historical_loader.load_trades() for full documentation.
        """
        return historical_loader.load_trades(
            base_asset=base_asset,
            counter_asset=counter_asset,
            start_time=start_time,
            limit_per_page=limit_per_page,
        )

    @staticmethod
    def load_pair_to_dataframe(
        base_asset: SdkAsset,
        counter_asset: SdkAsset,
        start_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Load trades as DataFrame.

        See historical_loader.load_pair_to_dataframe() for full documentation.
        """
        return historical_loader.load_pair_to_dataframe(
            base_asset=base_asset,
            counter_asset=counter_asset,
            start_time=start_time,
        )

    @staticmethod
    def load_watched_pairs_to_dataframe(
        start_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Load all configured pairs as DataFrame.

        See historical_loader.load_watched_pairs_to_dataframe() for full documentation.
        """
        return historical_loader.load_watched_pairs_to_dataframe(start_time=start_time)


# ============================================================================
# 3. Orderbook Loader (Order placement/cancellation events)
# ============================================================================


@register_importer(
    strict=True,
    name="orderbook_loader",
    description="""Order-book event ingestion via Horizon's operations endpoint.
    
    Loads order placement, cancellation, and update events from manage_offer
    operations. Essential for computing order_cancellation_rate feature.
    
    Key features:
    - Bulk loading: Paginated operation history
    - Event classification: Created/cancelled/updated detection
    - DataFrame output: Direct pandas integration
    - Retry: Exponential backoff on failure
    - Validation: Filters non-offer operations
    """,
    capabilities=(
        ImporterCapability.BULK
        | ImporterCapability.PAGINATION
        | ImporterCapability.RETRY
        | ImporterCapability.CURSOR_MANAGEMENT
        | ImporterCapability.ACCOUNT_FILTER
        | ImporterCapability.DATAFRAME_OUTPUT
        | ImporterCapability.VALIDATION
    ),
    data_types={DataType.ORDERBOOK_EVENT},
    sources={DataSource.HORIZON_REST},
    performance=PerformanceCharacteristics(
        typical_latency_ms=500,  # Per page
        throughput_records_per_sec=400,
        memory_overhead_mb=50,
        supports_batching=True,
    ),
    supports_failover=False,
    requires_authentication=False,
    supports_rate_limiting=True,
    version="1.0.0",
    dependencies=frozenset({"stellar-sdk", "pandas"}),
)
class OrderbookLoaderRegistry:
    """Registered wrapper for orderbook_loader module."""

    @staticmethod
    def load_orderbook_events(
        account_id: str,
        limit_per_page: int = 200,
    ) -> Iterator[OrderBookEvent]:
        """Load order-book events for an account.

        See orderbook_loader.load_orderbook_events() for full documentation.
        """
        return orderbook_loader.load_orderbook_events(
            account_id=account_id,
            limit_per_page=limit_per_page,
        )

    @staticmethod
    def load_accounts_orderbook_events(account_ids: list[str]) -> pd.DataFrame:
        """Load order-book events for multiple accounts as DataFrame.

        See orderbook_loader.load_accounts_orderbook_events() for full documentation.
        """
        return orderbook_loader.load_accounts_orderbook_events(account_ids=account_ids)


# ============================================================================
# 4. Account Activity Loader (Funding graph data)
# ============================================================================


@register_importer(
    strict=True,
    name="account_activity_loader",
    description="""Account creation and funding data via Horizon's effects endpoint.
    
    Fetches account_created effects to discover funding relationships for wallet
    graph features (funding_source_similarity, network_centrality).
    
    Key features:
    - Bulk loading: Batch account queries
    - Funding graph: Discovers account creator relationships
    - Retry: Exponential backoff on failure
    - Graceful degradation: Logs warnings, continues on individual failures
    """,
    capabilities=(
        ImporterCapability.BULK
        | ImporterCapability.PAGINATION
        | ImporterCapability.RETRY
        | ImporterCapability.ACCOUNT_FILTER
        | ImporterCapability.VALIDATION
    ),
    data_types={DataType.ACCOUNT_ACTIVITY},
    sources={DataSource.HORIZON_REST},
    performance=PerformanceCharacteristics(
        typical_latency_ms=300,  # Per account
        throughput_records_per_sec=10,  # Rate-limited by Horizon
        memory_overhead_mb=10,
        supports_batching=True,
    ),
    supports_failover=False,
    requires_authentication=False,
    supports_rate_limiting=True,
    version="1.0.0",
    dependencies=frozenset({"stellar-sdk"}),
)
class AccountActivityLoaderRegistry:
    """Registered wrapper for account_activity_loader module."""

    @staticmethod
    def load_account_activity(account_id: str) -> AccountActivity | None:
        """Load account creation/funding data for a single account.

        See account_activity_loader.load_account_activity() for full documentation.
        """
        return account_activity_loader.load_account_activity(account_id=account_id)

    @staticmethod
    def load_accounts_activity(account_ids: list[str]) -> list[AccountActivity]:
        """Load account activity for multiple accounts.

        See account_activity_loader.load_accounts_activity() for full documentation.
        """
        return account_activity_loader.load_accounts_activity(account_ids=account_ids)


# ============================================================================
# 5. AMM Pool Loader (Liquidity pool trades)
# ============================================================================


@register_importer(
    strict=True,
    name="amm_pool_loader",
    description="""AMM liquidity pool trade ingestion via Horizon's pool endpoints.
    
    Supports both bulk historical loading and real-time streaming of AMM pool trades.
    Includes pool discovery by asset and pool ID validation.
    
    Key features:
    - Dual mode: Bulk historical + real-time streaming
    - Pool discovery: Find pools by asset
    - Validation: 64-character hex pool ID format
    - Time filtering: Date range boundary
    - DataFrame output: Direct pandas integration
    - Retry: Exponential backoff on failure
    """,
    capabilities=(
        ImporterCapability.BULK
        | ImporterCapability.STREAMING
        | ImporterCapability.PAGINATION
        | ImporterCapability.RETRY
        | ImporterCapability.CURSOR_MANAGEMENT
        | ImporterCapability.TIME_RANGE_FILTER
        | ImporterCapability.ASSET_FILTER
        | ImporterCapability.DATAFRAME_OUTPUT
        | ImporterCapability.VALIDATION
        | ImporterCapability.DEDUPLICATION
        | ImporterCapability.POOL_DISCOVERY
    ),
    data_types={DataType.TRADE},
    sources={DataSource.HORIZON_LIQUIDITY_POOLS, DataSource.HORIZON_SSE},
    performance=PerformanceCharacteristics(
        typical_latency_ms=500,  # Bulk mode
        throughput_records_per_sec=400,
        memory_overhead_mb=100,
        supports_batching=True,
    ),
    supports_failover=False,
    requires_authentication=False,
    supports_rate_limiting=True,
    version="1.0.0",
    dependencies=frozenset({"stellar-sdk", "pandas", "requests"}),
)
class AMMPoolLoaderRegistry:
    """Registered wrapper for amm_pool_loader module."""

    @staticmethod
    def load_amm_pool_trades(
        pool_id: str,
        since: datetime,
        until: datetime,
        limit_per_page: int = 200,
    ) -> pd.DataFrame:
        """Load historical AMM pool trades as DataFrame.

        See amm_pool_loader.load_amm_pool_trades() for full documentation.
        """
        return amm_pool_loader.load_amm_pool_trades(
            pool_id=pool_id,
            since=since,
            until=until,
            limit_per_page=limit_per_page,
        )

    @staticmethod
    def stream_amm_pool_trades(pool_id: str) -> Iterator[Trade]:
        """Stream real-time AMM pool trades.

        See amm_pool_loader.stream_amm_pool_trades() for full documentation.
        """
        return amm_pool_loader.stream_amm_pool_trades(pool_id=pool_id)

    @staticmethod
    def list_active_pools(asset_code: str, asset_issuer: str) -> list[str]:
        """Discover active liquidity pools for an asset.

        See amm_pool_loader.list_active_pools() for full documentation.
        """
        return amm_pool_loader.list_active_pools(
            asset_code=asset_code,
            asset_issuer=asset_issuer,
        )


# ============================================================================
# 6. Asset Metadata Fetcher (Circulating supply)
# ============================================================================


@register_importer(
    strict=True,
    name="asset_metadata_fetcher",
    description="""Asset metadata fetcher for circulating supply from Horizon.
    
    Fetches and caches asset circulating supply with 1-hour TTL. Supports both
    Redis distributed cache and in-process fallback. Every record carries a
    trust tier and staleness (primary -> cache -> alternate fallback chain).
    
    Key features:
    - Metadata enrichment: Circulating supply for liquidity scoring
    - Caching: 1-hour TTL via Redis or in-process cache
    - Graceful degradation: Falls back to local cache on Redis failure
    - Asset filtering: Query by asset code and issuer
    """,
    capabilities=(
        ImporterCapability.BULK
        | ImporterCapability.METADATA_ENRICHMENT
        | ImporterCapability.ASSET_FILTER
    ),
    data_types={DataType.ASSET_METADATA},
    sources={DataSource.HORIZON_REST, DataSource.CACHED},
    performance=PerformanceCharacteristics(
        typical_latency_ms=100,  # Cached hit
        throughput_records_per_sec=100,  # Cache throughput
        memory_overhead_mb=5,  # Local cache
        supports_batching=False,
    ),
    supports_failover=False,
    requires_authentication=False,
    supports_rate_limiting=True,
    version="1.0.0",
    dependencies=frozenset({"urllib3"}),
)
class AssetMetadataFetcherRegistry:
    """Registered wrapper for asset_metadata_fetcher module."""

    @staticmethod
    def get_asset_supply(
        asset_code: str,
        asset_issuer: str,
        horizon_url: str,
        redis_client=None,
    ) -> float | None:
        """Fetch circulating supply for an asset (cached).

        See asset_metadata_fetcher.get_asset_supply() for full documentation.
        """
        return asset_metadata_fetcher.get_asset_supply(
            asset_code=asset_code,
            asset_issuer=asset_issuer,
            horizon_url=horizon_url,
            redis_client=redis_client,
        )

    @staticmethod
    def get_asset_metadata(
        asset_code: str,
        asset_issuer: str,
        horizon_url: str,
        redis_client=None,
        alternate_horizon_url: str | None = None,
    ) -> asset_metadata_fetcher.AssetMetadataRecord:
        """Fetch asset metadata labelled with its trust tier and staleness.

        See asset_metadata_fetcher.get_asset_metadata() for the fallback chain.
        """
        return asset_metadata_fetcher.get_asset_metadata(
            asset_code=asset_code,
            asset_issuer=asset_issuer,
            horizon_url=horizon_url,
            redis_client=redis_client,
            alternate_horizon_url=alternate_horizon_url,
        )


# ============================================================================
# 7. Payment Path Analyzer (Multi-hop wash trade detection)
# ============================================================================


@register_importer(
    strict=True,
    name="payment_path_analyzer",
    description="""Payment path analysis for multi-hop wash trade routing detection.
    
    Reconstructs multi-hop payment flows from Stellar path payment operations to
    detect sophisticated wash traders obfuscating connections via intermediaries.
    
    Key features:
    - Multi-hop analysis: Reconstruct flows up to 6 hops (Stellar's max)
    - Round-trip detection: Identify closed-loop wash trades
    - Path validation: Enforce Stellar schema constraints
    - Volume attribution: Compute round-trip frequency metric
    """,
    capabilities=(
        ImporterCapability.BULK
        | ImporterCapability.MULTI_HOP_ANALYSIS
        | ImporterCapability.VALIDATION
        | ImporterCapability.TIME_RANGE_FILTER
        | ImporterCapability.ACCOUNT_FILTER
    ),
    data_types={DataType.PAYMENT_PATH},
    sources={DataSource.DERIVED},  # Computed from Horizon operation data
    performance=PerformanceCharacteristics(
        typical_latency_ms=50,  # Per path reconstruction
        throughput_records_per_sec=200,
        memory_overhead_mb=20,
        supports_batching=True,
    ),
    supports_failover=False,
    requires_authentication=False,
    supports_rate_limiting=False,
    version="1.0.0",
    dependencies=frozenset({"pandas"}),
)
class PaymentPathAnalyzerRegistry:
    """Registered wrapper for payment_path_analyzer module."""

    @staticmethod
    def reconstruct_path_flow(
        path_payment_op: dict,
        all_operations: pd.DataFrame | None = None,
    ) -> payment_path_analyzer.ReconstructedPathFlow | None:
        """Reconstruct effective source/destination from a path payment operation.

        See payment_path_analyzer.reconstruct_path_flow() for full documentation.
        """
        return payment_path_analyzer.reconstruct_path_flow(
            path_payment_op=path_payment_op,
            all_operations=all_operations,
        )

    @staticmethod
    def compute_path_payment_round_trip_frequency(
        wallet: str,
        path_flows: list[payment_path_analyzer.ReconstructedPathFlow],
        time_window_hours: int = payment_path_analyzer.ROUND_TRIP_WINDOW_HOURS,
    ) -> float:
        """Compute fraction of wallet's volume returning within time window.

        See payment_path_analyzer.compute_path_payment_round_trip_frequency()
        for full documentation.
        """
        return payment_path_analyzer.compute_path_payment_round_trip_frequency(
            wallet=wallet,
            path_flows=path_flows,
            time_window_hours=time_window_hours,
        )

    @staticmethod
    def validate_path_schema(path_payment_op: dict) -> bool:
        """Validate path payment operation against Stellar schema.

        See payment_path_analyzer.validate_path_schema() for full documentation.
        """
        return payment_path_analyzer.validate_path_schema(path_payment_op=path_payment_op)


# ============================================================================
# Auto-registration verification
# ============================================================================


def verify_registration() -> dict[str, bool]:
    """Verify that all importers were registered successfully.

    Returns
    -------
    dict[str, bool]
        Map of importer names to registration status

    Examples
    --------
    >>> status = verify_registration()
    >>> assert all(status.values()), "Some importers failed to register"
    """
    from ingestion.importer_registry import get_registry

    registry = get_registry()
    expected_importers = [
        "horizon_streamer",
        "historical_loader",
        "orderbook_loader",
        "account_activity_loader",
        "amm_pool_loader",
        "asset_metadata_fetcher",
        "payment_path_analyzer",
    ]

    return {name: name in registry.list_all() for name in expected_importers}


# Verify on import
_registration_status = verify_registration()
if not all(_registration_status.values()):
    failed = [name for name, ok in _registration_status.items() if not ok]
    logger.warning(
        "Some importers failed to register: %s. "
        "This may indicate a circular import or decorator issue.",
        ", ".join(failed),
    )
else:
    logger.info("Successfully registered %d importers", len(_registration_status))
