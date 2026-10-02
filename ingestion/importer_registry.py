"""Importer capability discovery system for supported data sources.

This module provides a durable, reusable system for discovering, validating,
and querying the capabilities of all data source importers in LedgerLens-data.

Architecture
------------
The importer registry follows a capability-based design where each importer
declares its supported features through typed metadata. This enables:

1. **Runtime discovery** - Automatically detect all available importers
2. **Capability queries** - Find importers by required features
3. **Validation** - Ensure importers meet minimum requirements
4. **Diagnostics** - Clear error messages when capabilities are missing
5. **Extensibility** - Easy to add new importers without modifying core code

Design Principles
-----------------
- **Protocol-based**: Uses Python Protocols for clear contracts (PEP 544)
- **Immutable metadata**: Capability descriptors are frozen dataclasses
- **Zero-config**: Importers self-register via decorators
- **Type-safe**: Full mypy compatibility with runtime type checking
- **Performance**: Registry is built once at import time, O(1) lookups

Example Usage
-------------
Register an importer::

    from ingestion.importer_registry import register_importer, ImporterCapability

    @register_importer(
        name="horizon_streamer",
        capabilities=ImporterCapability.STREAMING | ImporterCapability.REAL_TIME,
        data_types={DataType.TRADE},
        sources={DataSource.HORIZON_SSE},
    )
    class HorizonStreamer:
        ...

Query by capability::

    from ingestion.importer_registry import get_registry

    registry = get_registry()

    # Find all streaming importers
    streamers = registry.find_by_capability(ImporterCapability.STREAMING)

    # Find importers for specific data type
    trade_loaders = registry.find_by_data_type(DataType.TRADE)

    # Get detailed info
    info = registry.get_importer_info("horizon_streamer")
    print(f"Supports failover: {info.supports_failover}")

Validate before use::

    from ingestion.importer_registry import validate_importer_requirements

    # Check if required capabilities exist
    result = validate_importer_requirements(
        required_capabilities=ImporterCapability.STREAMING | ImporterCapability.PAGINATION,
        required_data_types={DataType.TRADE, DataType.ORDERBOOK},
    )

    if not result.is_valid:
        print(f"Missing capabilities: {result.missing_capabilities}")
        print(f"Suggestions: {result.suggestions}")
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, TypedDict, runtime_checkable

from utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Enums: Capability flags, data types, and sources
# ============================================================================


class ImporterCapability(enum.IntFlag):
    """Capability flags for importers (bitwise combinable).

    These flags describe what an importer can do. Multiple capabilities
    can be combined using bitwise OR (|).

    Examples
    --------
    >>> caps = ImporterCapability.STREAMING | ImporterCapability.REAL_TIME
    >>> ImporterCapability.STREAMING in caps
    True
    """

    NONE = 0

    # Data access patterns
    STREAMING = 1 << 0  # Yields data continuously (SSE, WebSocket)
    BULK = 1 << 1  # Loads historical data in batches
    PAGINATION = 1 << 2  # Supports cursor-based pagination
    REAL_TIME = 1 << 3  # Sub-minute latency for live data

    # Reliability features
    RETRY = 1 << 4  # Automatic retry with exponential backoff
    FAILOVER = 1 << 5  # Multi-region endpoint failover
    DEDUPLICATION = 1 << 6  # Handles duplicate records automatically
    CURSOR_MANAGEMENT = 1 << 7  # Preserves cursor across restarts

    # Data transformation
    DATAFRAME_OUTPUT = 1 << 8  # Can output pandas DataFrame
    VALIDATION = 1 << 9  # Validates data against schema
    NORMALIZATION = 1 << 10  # Normalizes amounts across currencies

    # Query features
    TIME_RANGE_FILTER = 1 << 11  # Supports filtering by time range
    ACCOUNT_FILTER = 1 << 12  # Supports filtering by account ID
    ASSET_FILTER = 1 << 13  # Supports filtering by asset pair

    # Advanced features
    MULTI_HOP_ANALYSIS = 1 << 14  # Reconstructs multi-hop payment paths
    METADATA_ENRICHMENT = 1 << 15  # Adds metadata (supply, liquidity)
    POOL_DISCOVERY = 1 << 16  # Discovers AMM liquidity pools


class DataType(enum.Enum):
    """Types of data that importers can provide."""

    TRADE = "trade"  # Executed trades (SDEX or AMM)
    ORDERBOOK_EVENT = "orderbook_event"  # Order placements/cancellations
    ACCOUNT_ACTIVITY = "account_activity"  # Account creation/funding
    PAYMENT_PATH = "payment_path"  # Multi-hop payment flows
    ASSET_METADATA = "asset_metadata"  # Asset supply/liquidity
    BOT_FINGERPRINT = "bot_fingerprint"  # Bot detection signals

    def __str__(self) -> str:
        return self.value


class DataSource(enum.Enum):
    """Data sources that importers connect to."""

    HORIZON_SSE = "horizon_sse"  # Horizon Server-Sent Events stream
    HORIZON_REST = "horizon_rest"  # Horizon REST API (paginated)
    HORIZON_LIQUIDITY_POOLS = "horizon_liquidity_pools"  # AMM pool endpoint
    DERIVED = "derived"  # Computed from other data
    CACHED = "cached"  # Cached/memoized data

    def __str__(self) -> str:
        return self.value


# ============================================================================
# Metadata dataclasses
# ============================================================================


@dataclass(frozen=True)
class PerformanceCharacteristics:
    """Performance profile of an importer."""

    typical_latency_ms: int | None = None  # Typical data latency (milliseconds)
    throughput_records_per_sec: int | None = None  # Sustained throughput
    memory_overhead_mb: int | None = None  # Typical memory usage
    supports_batching: bool = False  # Can process multiple requests together

    def __str__(self) -> str:
        parts = []
        if self.typical_latency_ms:
            parts.append(f"latency ~{self.typical_latency_ms}ms")
        if self.throughput_records_per_sec:
            parts.append(f"throughput ~{self.throughput_records_per_sec} rec/s")
        if self.memory_overhead_mb:
            parts.append(f"memory ~{self.memory_overhead_mb}MB")
        return ", ".join(parts) if parts else "no performance data"


@dataclass(frozen=True)
class ImporterMetadata:
    """Complete metadata descriptor for an importer.

    This is the immutable capability declaration for an importer. It should
    be constructed once at registration time and never modified.
    """

    name: str
    description: str
    capabilities: ImporterCapability
    data_types: frozenset[DataType]
    sources: frozenset[DataSource]

    # Optional metadata
    module_path: str | None = None
    version: str = "1.0.0"
    performance: PerformanceCharacteristics = field(default_factory=PerformanceCharacteristics)

    # Feature flags
    supports_failover: bool = False
    requires_authentication: bool = False
    supports_rate_limiting: bool = False

    # Validation
    min_python_version: tuple[int, int] = (3, 11)
    dependencies: frozenset[str] = field(default_factory=frozenset)

    # Registration timestamp
    registered_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate metadata consistency."""
        if not self.name:
            raise ValueError("Importer name cannot be empty")
        if not self.data_types:
            raise ValueError(f"Importer {self.name} must declare at least one data type")
        if not self.sources:
            raise ValueError(f"Importer {self.name} must declare at least one data source")

    def has_capability(self, capability: ImporterCapability) -> bool:
        """Check if importer has a specific capability."""
        return bool(self.capabilities & capability)

    def has_all_capabilities(self, capabilities: ImporterCapability) -> bool:
        """Check if importer has all specified capabilities."""
        return (self.capabilities & capabilities) == capabilities

    def supports_data_type(self, data_type: DataType) -> bool:
        """Check if importer supports a data type."""
        return data_type in self.data_types

    def uses_source(self, source: DataSource) -> bool:
        """Check if importer uses a data source."""
        return source in self.sources


# ============================================================================
# Registry and query results
# ============================================================================


@dataclass(frozen=True)
class ValidationResult:
    """Result of capability validation."""

    is_valid: bool
    missing_capabilities: frozenset[ImporterCapability] = field(default_factory=frozenset)
    missing_data_types: frozenset[DataType] = field(default_factory=frozenset)
    available_importers: frozenset[str] = field(default_factory=frozenset)
    suggestions: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.is_valid:
            return "✓ All requirements satisfied"

        parts = ["✗ Validation failed:"]
        if self.missing_capabilities:
            caps = ", ".join(cap.name for cap in self.missing_capabilities)
            parts.append(f"  Missing capabilities: {caps}")
        if self.missing_data_types:
            types = ", ".join(str(dt) for dt in self.missing_data_types)
            parts.append(f"  Missing data types: {types}")
        if self.suggestions:
            parts.append("  Suggestions:")
            for suggestion in self.suggestions:
                parts.append(f"    - {suggestion}")
        return "\n".join(parts)


class QueryResult(TypedDict):
    """Result of a registry query."""

    importer_name: str
    metadata: ImporterMetadata
    match_score: float  # Relevance score [0.0, 1.0]


# ============================================================================
# Registration-time conformance checks
# ============================================================================


class ImporterConformanceError(ValueError):
    """Raised when an importer does not implement a capability it declares."""


def _public_methods(importer_class: type) -> dict[str, Callable]:
    return {
        n: getattr(importer_class, n)
        for n in dir(importer_class)
        if not n.startswith("_") and callable(getattr(importer_class, n))
    }


def _has_method(prefixes: tuple[str, ...]) -> Callable[[type], str | None]:
    def check(importer_class: type) -> str | None:
        if any(n.startswith(prefixes) for n in _public_methods(importer_class)):
            return None
        return f"no public method starting with {prefixes}"

    return check


def _has_dataframe_output(importer_class: type) -> str | None:
    for n, fn in _public_methods(importer_class).items():
        if "dataframe" in n or "DataFrame" in str(getattr(fn, "__annotations__", {}).get("return")):
            return None
    return "no method returning a pandas DataFrame"


# Minimal contract per capability. Capabilities without an entry (e.g. RETRY,
# VALIDATION) describe runtime behaviour that can't be checked statically.
CAPABILITY_CONTRACTS: dict[ImporterCapability, Callable[[type], str | None]] = {
    ImporterCapability.STREAMING: _has_method(("stream",)),
    ImporterCapability.BULK: _has_method(
        ("load", "get", "fetch", "list", "compute", "reconstruct")
    ),
    ImporterCapability.DATAFRAME_OUTPUT: _has_dataframe_output,
    ImporterCapability.POOL_DISCOVERY: _has_method(("list_active_pools", "discover")),
    ImporterCapability.MULTI_HOP_ANALYSIS: _has_method(("reconstruct",)),
}


def check_conformance(importer_class: type, metadata: ImporterMetadata) -> None:
    """Raise ``ImporterConformanceError`` if a declared capability is unimplemented."""
    failures = []
    for capability, contract in CAPABILITY_CONTRACTS.items():
        if metadata.has_capability(capability):
            problem = contract(importer_class)
            if problem:
                failures.append(f"{capability.name}: {problem}")
    if failures:
        raise ImporterConformanceError(
            f"Importer {metadata.name!r} fails conformance: " + "; ".join(failures)
        )


@runtime_checkable
class Importer(Protocol):
    """Protocol that all registered importers should implement.

    This is a structural type (duck typing) - importers don't need to
    explicitly inherit from this, they just need to have these attributes.
    """

    __importer_metadata__: ImporterMetadata


# ============================================================================
# Global registry
# ============================================================================


class ImporterRegistry:
    """Central registry for all data source importers.

    This is a singleton that stores metadata for all registered importers
    and provides query methods to find importers by capability.

    The registry is built at module import time and should be treated as
    read-only after initialization.
    """

    def __init__(self) -> None:
        self._importers: dict[str, ImporterMetadata] = {}
        self._by_capability: dict[ImporterCapability, set[str]] = {}
        self._by_data_type: dict[DataType, set[str]] = {}
        self._by_source: dict[DataSource, set[str]] = {}
        self._importer_classes: dict[str, type] = {}

    def register(
        self,
        importer_class: type,
        metadata: ImporterMetadata,
        *,
        strict: bool = False,
    ) -> None:
        """Register an importer with its metadata.

        Parameters
        ----------
        importer_class : type
            The importer class to register
        metadata : ImporterMetadata
            Complete metadata descriptor

        Raises
        ------
        ValueError
            If an importer with the same name is already registered
        ImporterConformanceError
            If ``strict`` and a declared capability isn't implemented
        """
        if strict:
            check_conformance(importer_class, metadata)
        if metadata.name in self._importers:
            raise ValueError(
                f"Importer {metadata.name!r} is already registered. "
                f"Use a unique name or unregister the existing importer first."
            )

        # Store metadata and class
        self._importers[metadata.name] = metadata
        self._importer_classes[metadata.name] = importer_class

        # Index by capability
        for capability in ImporterCapability:
            if capability == ImporterCapability.NONE:
                continue
            if metadata.has_capability(capability):
                self._by_capability.setdefault(capability, set()).add(metadata.name)

        # Index by data type
        for data_type in metadata.data_types:
            self._by_data_type.setdefault(data_type, set()).add(metadata.name)

        # Index by source
        for source in metadata.sources:
            self._by_source.setdefault(source, set()).add(metadata.name)

        logger.debug(
            "Registered importer %r with capabilities: %s, data types: %s",
            metadata.name,
            metadata.capabilities,
            ", ".join(str(dt) for dt in metadata.data_types),
        )

    def verify_conformance(self) -> list[str]:
        """Run conformance checks on every registered importer; return failures."""
        failures = []
        for name, metadata in self._importers.items():
            try:
                check_conformance(self._importer_classes[name], metadata)
            except ImporterConformanceError as exc:
                failures.append(str(exc))
        return failures

    def unregister(self, name: str) -> None:
        """Remove an importer from the registry (mainly for testing)."""
        if name not in self._importers:
            return

        self._importers.pop(name)
        self._importer_classes.pop(name, None)

        # Remove from indexes
        for capability_set in self._by_capability.values():
            capability_set.discard(name)
        for data_type_set in self._by_data_type.values():
            data_type_set.discard(name)
        for source_set in self._by_source.values():
            source_set.discard(name)

    def get_importer_info(self, name: str) -> ImporterMetadata:
        """Get metadata for a specific importer.

        Parameters
        ----------
        name : str
            Importer name

        Returns
        -------
        ImporterMetadata
            Complete metadata descriptor

        Raises
        ------
        KeyError
            If importer is not registered
        """
        if name not in self._importers:
            available = ", ".join(self._importers.keys())
            raise KeyError(
                f"Importer {name!r} not found in registry. " f"Available importers: {available}"
            )
        return self._importers[name]

    def get_importer_class(self, name: str) -> type:
        """Get the actual class for an importer.

        Parameters
        ----------
        name : str
            Importer name

        Returns
        -------
        type
            The registered importer class

        Raises
        ------
        KeyError
            If importer is not registered
        """
        if name not in self._importer_classes:
            raise KeyError(f"Importer {name!r} not found in registry")
        return self._importer_classes[name]

    def list_all(self) -> list[str]:
        """Return sorted list of all registered importer names."""
        return sorted(self._importers.keys())

    def find_by_capability(
        self,
        capability: ImporterCapability,
        require_all: bool = False,
    ) -> list[QueryResult]:
        """Find importers with specific capabilities.

        Parameters
        ----------
        capability : ImporterCapability
            Capability or combination of capabilities (use | to combine)
        require_all : bool, default False
            If True, importer must have ALL specified capabilities.
            If False, importer must have AT LEAST ONE capability.

        Returns
        -------
        list[QueryResult]
            Matching importers sorted by relevance
        """
        results: list[QueryResult] = []

        for name, metadata in self._importers.items():
            if require_all:
                match = metadata.has_all_capabilities(capability)
            else:
                match = bool(metadata.capabilities & capability)

            if match:
                # Calculate match score (fraction of requested capabilities present)
                requested_bits = bin(capability).count("1")
                matched_bits = bin(metadata.capabilities & capability).count("1")
                score = matched_bits / requested_bits if requested_bits > 0 else 0.0

                results.append(
                    {
                        "importer_name": name,
                        "metadata": metadata,
                        "match_score": score,
                    }
                )

        # Sort by match score descending
        results.sort(key=lambda r: r["match_score"], reverse=True)
        return results

    def find_by_data_type(self, data_type: DataType) -> list[QueryResult]:
        """Find importers that support a data type.

        Parameters
        ----------
        data_type : DataType
            Data type to search for

        Returns
        -------
        list[QueryResult]
            Matching importers with score 1.0
        """
        importer_names = self._by_data_type.get(data_type, set())
        results: list[QueryResult] = []

        for name in importer_names:
            metadata = self._importers[name]
            results.append(
                {
                    "importer_name": name,
                    "metadata": metadata,
                    "match_score": 1.0,
                }
            )

        return sorted(results, key=lambda r: r["importer_name"])

    def find_by_source(self, source: DataSource) -> list[QueryResult]:
        """Find importers that use a data source.

        Parameters
        ----------
        source : DataSource
            Data source to search for

        Returns
        -------
        list[QueryResult]
            Matching importers with score 1.0
        """
        importer_names = self._by_source.get(source, set())
        results: list[QueryResult] = []

        for name in importer_names:
            metadata = self._importers[name]
            results.append(
                {
                    "importer_name": name,
                    "metadata": metadata,
                    "match_score": 1.0,
                }
            )

        return sorted(results, key=lambda r: r["importer_name"])

    def find_best_match(
        self,
        required_capabilities: ImporterCapability | None = None,
        required_data_types: Iterable[DataType] | None = None,
        required_sources: Iterable[DataSource] | None = None,
        prefer_capabilities: ImporterCapability | None = None,
    ) -> QueryResult | None:
        """Find the best matching importer for requirements.

        This is a convenience method that combines multiple filters and
        returns the single best match based on a weighted scoring system.

        Parameters
        ----------
        required_capabilities : ImporterCapability, optional
            Capabilities that MUST be present
        required_data_types : Iterable[DataType], optional
            Data types that MUST be supported
        required_sources : Iterable[DataSource], optional
            Data sources that MUST be used
        prefer_capabilities : ImporterCapability, optional
            Capabilities that are preferred but not required

        Returns
        -------
        QueryResult | None
            Best matching importer, or None if no match found
        """
        candidates = self._importers.copy()

        # Filter by required capabilities
        if required_capabilities:
            candidates = {
                name: meta
                for name, meta in candidates.items()
                if meta.has_all_capabilities(required_capabilities)
            }

        # Filter by required data types
        if required_data_types:
            req_types = set(required_data_types)
            candidates = {
                name: meta
                for name, meta in candidates.items()
                if req_types.issubset(meta.data_types)
            }

        # Filter by required sources
        if required_sources:
            req_sources = set(required_sources)
            candidates = {
                name: meta
                for name, meta in candidates.items()
                if req_sources.issubset(meta.sources)
            }

        if not candidates:
            return None

        # Score remaining candidates
        scored: list[tuple[str, ImporterMetadata, float]] = []
        for name, metadata in candidates.items():
            score = 1.0

            # Boost score for preferred capabilities
            if prefer_capabilities:
                preferred_bits = bin(prefer_capabilities).count("1")
                matched_bits = bin(metadata.capabilities & prefer_capabilities).count("1")
                if preferred_bits > 0:
                    score += 0.5 * (matched_bits / preferred_bits)

            # Boost score for more capabilities (tie-breaker)
            total_caps = bin(metadata.capabilities).count("1")
            score += 0.01 * total_caps

            scored.append((name, metadata, score))

        # Return highest score
        scored.sort(key=lambda x: x[2], reverse=True)
        name, metadata, score = scored[0]

        return {
            "importer_name": name,
            "metadata": metadata,
            "match_score": min(score / 1.5, 1.0),  # Normalize to [0, 1]
        }

    def validate_requirements(
        self,
        required_capabilities: ImporterCapability | None = None,
        required_data_types: Iterable[DataType] | None = None,
    ) -> ValidationResult:
        """Validate that required capabilities are available in the registry.

        Parameters
        ----------
        required_capabilities : ImporterCapability, optional
            Required capability flags
        required_data_types : Iterable[DataType], optional
            Required data types

        Returns
        -------
        ValidationResult
            Validation result with missing items and suggestions
        """
        missing_caps: set[ImporterCapability] = set()
        missing_types: set[DataType] = set()
        available: set[str] = set()
        suggestions: list[str] = []

        # Check capabilities
        if required_capabilities:
            for capability in ImporterCapability:
                if capability == ImporterCapability.NONE:
                    continue
                if bool(required_capabilities & capability):
                    matching = self._by_capability.get(capability, set())
                    if not matching:
                        missing_caps.add(capability)
                    else:
                        available.update(matching)

        # Check data types
        if required_data_types:
            for data_type in required_data_types:
                matching = self._by_data_type.get(data_type, set())
                if not matching:
                    missing_types.add(data_type)
                else:
                    available.update(matching)

        # Generate suggestions
        if missing_caps:
            caps_str = ", ".join(cap.name for cap in missing_caps)
            suggestions.append(
                f"No importers found with capabilities: {caps_str}. "
                "Consider adding a new importer or using a combination of existing ones."
            )

        if missing_types:
            types_str = ", ".join(str(dt) for dt in missing_types)
            suggestions.append(
                f"No importers found for data types: {types_str}. "
                "These may need to be derived from other data sources."
            )

        if available:
            importers_str = ", ".join(sorted(available))
            suggestions.append(f"Available importers that partially match: {importers_str}")

        is_valid = not missing_caps and not missing_types

        return ValidationResult(
            is_valid=is_valid,
            missing_capabilities=frozenset(missing_caps),
            missing_data_types=frozenset(missing_types),
            available_importers=frozenset(available),
            suggestions=suggestions,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Return statistics about the registry."""
        return {
            "total_importers": len(self._importers),
            "importers_by_data_type": {
                str(dt): len(importers) for dt, importers in self._by_data_type.items()
            },
            "importers_by_source": {
                str(src): len(importers) for src, importers in self._by_source.items()
            },
            "streaming_importers": len(
                self._by_capability.get(ImporterCapability.STREAMING, set())
            ),
            "bulk_importers": len(self._by_capability.get(ImporterCapability.BULK, set())),
            "failover_capable": sum(
                1 for meta in self._importers.values() if meta.supports_failover
            ),
        }


# ============================================================================
# Module-level singleton registry
# ============================================================================


_global_registry: ImporterRegistry | None = None


def get_registry() -> ImporterRegistry:
    """Get the global importer registry (singleton).

    Returns
    -------
    ImporterRegistry
        The global registry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ImporterRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (mainly for testing)."""
    global _global_registry
    _global_registry = ImporterRegistry()


# ============================================================================
# Decorator for registration
# ============================================================================


def register_importer(
    name: str,
    description: str = "",
    capabilities: ImporterCapability = ImporterCapability.NONE,
    data_types: Iterable[DataType] | None = None,
    sources: Iterable[DataSource] | None = None,
    strict: bool = False,
    **kwargs: Any,
) -> Callable[[type], type]:
    """Decorator to register an importer with the global registry.

    Parameters
    ----------
    name : str
        Unique importer name
    description : str, optional
        Human-readable description
    capabilities : ImporterCapability, default NONE
        Capability flags (combine with |)
    data_types : Iterable[DataType], optional
        Data types this importer provides
    sources : Iterable[DataSource], optional
        Data sources this importer uses
    strict : bool, default False
        Reject registration if a declared capability isn't implemented
        (see ``CAPABILITY_CONTRACTS``).
    **kwargs
        Additional metadata fields (performance, version, etc.)

    Returns
    -------
    Callable
        Decorator function

    Examples
    --------
    >>> @register_importer(
    ...     name="my_importer",
    ...     description="My custom importer",
    ...     capabilities=ImporterCapability.STREAMING | ImporterCapability.REAL_TIME,
    ...     data_types={DataType.TRADE},
    ...     sources={DataSource.HORIZON_SSE},
    ... )
    ... class MyImporter:
    ...     pass
    """

    def decorator(cls: type) -> type:
        # Build metadata
        metadata = ImporterMetadata(
            name=name,
            description=description or cls.__doc__ or "",
            capabilities=capabilities,
            data_types=frozenset(data_types or []),
            sources=frozenset(sources or []),
            module_path=f"{cls.__module__}.{cls.__name__}",
            **kwargs,
        )

        # Attach metadata to class
        cls.__importer_metadata__ = metadata  # type: ignore[attr-defined]

        # Register with global registry
        registry = get_registry()
        registry.register(cls, metadata, strict=strict)

        return cls

    return decorator


# ============================================================================
# Convenience functions
# ============================================================================


def list_all_importers() -> list[str]:
    """List all registered importer names."""
    return get_registry().list_all()


def get_importer_info(name: str) -> ImporterMetadata:
    """Get metadata for a specific importer."""
    return get_registry().get_importer_info(name)


def find_importers_by_capability(
    capability: ImporterCapability,
    require_all: bool = False,
) -> list[QueryResult]:
    """Find importers with specific capabilities."""
    return get_registry().find_by_capability(capability, require_all=require_all)


def find_importers_by_data_type(data_type: DataType) -> list[QueryResult]:
    """Find importers that support a data type."""
    return get_registry().find_by_data_type(data_type)


def validate_importer_requirements(
    required_capabilities: ImporterCapability | None = None,
    required_data_types: Iterable[DataType] | None = None,
) -> ValidationResult:
    """Validate that required capabilities are available."""
    return get_registry().validate_requirements(
        required_capabilities=required_capabilities,
        required_data_types=required_data_types,
    )


def get_registry_statistics() -> dict[str, Any]:
    """Return statistics about the importer registry."""
    return get_registry().get_statistics()
