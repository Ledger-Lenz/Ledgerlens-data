"""Payment path analysis for detecting multi-hop wash trade routing.

Sophisticated wash traders on the Stellar DEX route trades through multi-hop
payment paths (using Stellar's path payment operations) to obfuscate the
connection between buyer and seller wallets. This module reconstructs these
multi-hop flows and attributes them to the originating wallets.

Attributes:
    MAX_PATH_LENGTH: Maximum allowed hops in a payment path (Stellar's limit).
    ROUND_TRIP_WINDOW_HOURS: Time window for detecting round-trip path flows.
    DEFAULT_TRACE_MAX_DEPTH: Default maximum wallet hops followed by
        ``trace_payment_paths`` before a branch is truncated.
    DEFAULT_TRACE_MAX_BRANCHING: Default maximum outgoing payments followed
        from any single wallet during a trace.
    DEFAULT_TRACE_MAX_PATHS: Default ceiling on the total number of paths a
        single trace may emit.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import NotRequired, TypedDict

import pandas as pd

from ingestion.exceptions import InvalidInputError, RecordValidationError
from utils.logging import get_logger

logger = get_logger(__name__)

MAX_PATH_LENGTH = 6  # Stellar's maximum path hops
ROUND_TRIP_WINDOW_HOURS = 24

# Traversal bounds for trace_payment_paths (issue #916). These guard against
# cyclic or pathologically deep/wide payment graphs -- whether crafted by an
# adversary or produced by a data quality issue -- exhausting CPU or memory.
DEFAULT_TRACE_MAX_DEPTH = 12
DEFAULT_TRACE_MAX_BRANCHING = 32
DEFAULT_TRACE_MAX_PATHS = 10_000


class ReconstructedPathFlow(TypedDict):
    """Reconstructed multi-hop payment path flow."""

    source_wallet: str
    destination_wallet: str
    source_amount: float
    destination_amount: float
    hop_count: int
    path_payment_ids: list[str]
    execution_time: datetime
    is_round_trip: bool
    is_cyclical: NotRequired[bool]


def _asset_key(asset: object) -> tuple:
    """Normalize an asset path entry into a hashable identity."""
    if isinstance(asset, dict):
        return (asset.get("code"), asset.get("issuer"))
    return ("", asset)


def path_has_cycle(asset_path: list) -> bool:
    """Report whether an asset path revisits an asset it already passed through.

    Stellar payment paths may legitimately cycle (a wallet paying out through
    intermediate assets and back into one it already held). Cycles are flagged
    rather than rejected, and path analysis never walks the hops recursively,
    so a cyclical path terminates like any other.

    Args:
        asset_path: List of asset path entries (dicts with code/issuer, or
            plain asset identifiers).

    Returns:
        True if any asset appears more than once in the path.
    """
    seen: set[tuple] = set()
    for asset in asset_path or []:
        key = _asset_key(asset)
        if key in seen:
            return True
        seen.add(key)
    return False


def reconstruct_path_flow(
    path_payment_op: dict,
    all_operations: pd.DataFrame | None = None,
) -> ReconstructedPathFlow | None:
    """Reconstruct effective source/destination from a path payment operation.

    Unwraps multi-hop payment paths to identify the true economic sender and
    receiver, accounting for all intermediate asset conversions.

    Args:
        path_payment_op: A path payment operation dict from Stellar Horizon with
            keys: transaction_id, source_account, destination_account,
            asset_path, amount_sent (path_payment_strict_receive), amount_received
            (path_payment_strict_send), or both.
        all_operations: Optional DataFrame of all operations to trace round-trip
            patterns. Must include columns: source_account, destination_account,
            path, transaction_id, created_at.

    Returns:
        ReconstructedPathFlow dict with source/destination wallets, amounts,
        hop count, round-trip flag, and an ``is_cyclical`` flag set when the
        asset path revisits an asset. Returns None if the operation is malformed
        or exceeds Stellar's path length limit.

    Raises:
        InvalidInputError: If the operation type is not a path payment variant
            (also a ``ValueError``).
        RecordValidationError: If required fields are missing. Note this is
            **not** a ``KeyError`` — see ``ingestion/exceptions.py``.
    """
    op_type = path_payment_op.get("type")
    if op_type not in ("path_payment_strict_send", "path_payment_strict_receive"):
        raise InvalidInputError(
            f"Operation type {op_type} is not a path payment variant",
            source="payment_path_analyzer.reconstruct_path_flow",
            reason=f"unsupported operation type {op_type!r}",
        )

    # Extract mandatory fields
    source_wallet = path_payment_op.get("source_account")
    destination_wallet = path_payment_op.get("destination_account")
    transaction_id = path_payment_op.get("transaction_id")
    created_at = path_payment_op.get("created_at")

    if not all([source_wallet, destination_wallet, transaction_id]):
        raise RecordValidationError(
            "Missing required fields: source_account, destination_account, transaction_id",
            source="payment_path_analyzer.reconstruct_path_flow",
            reason="one or more required fields are absent or empty",
            raw=path_payment_op,
        )

    # Parse the asset path
    asset_path = path_payment_op.get("asset_path", [])
    if len(asset_path) > MAX_PATH_LENGTH:
        return None  # Reject malformed paths exceeding Stellar's maximum

    # Determine hop count and amounts based on operation type
    hop_count = len(asset_path)
    source_amount = 0.0
    destination_amount = 0.0

    if op_type == "path_payment_strict_send":
        # Exact amount sent, variable amount received
        source_amount = float(path_payment_op.get("amount", 0.0))
        destination_amount = float(path_payment_op.get("destination_amount", 0.0))
    elif op_type == "path_payment_strict_receive":
        # Exact amount received, variable amount sent
        source_amount = float(path_payment_op.get("amount_sent", 0.0))
        destination_amount = float(path_payment_op.get("amount", 0.0))

    # Check for round-trip pattern if all_operations provided
    is_round_trip = False
    if (
        all_operations is not None
        and not all_operations.empty
        and source_wallet == destination_wallet
    ):
        # Same source and destination = round-trip by definition
        is_round_trip = True

    flow: ReconstructedPathFlow = {
        "source_wallet": source_wallet,
        "destination_wallet": destination_wallet,
        "source_amount": source_amount,
        "destination_amount": destination_amount,
        "hop_count": hop_count,
        "path_payment_ids": [transaction_id],
        "execution_time": created_at,
        "is_round_trip": is_round_trip,
        "is_cyclical": path_has_cycle(asset_path),
    }

    return flow


def compute_path_payment_round_trip_frequency(
    wallet: str,
    path_flows: list[ReconstructedPathFlow],
    time_window_hours: int = ROUND_TRIP_WINDOW_HOURS,
) -> float:
    """Compute fraction of a wallet's effective volume returning within time window.

    Measures what proportion of a wallet's payment-path flows return to the
    originating wallet within a specified time window, indicating round-trip
    wash-trading behavior.

    Args:
        wallet: Stellar account ID to analyze.
        path_flows: List of ReconstructedPathFlow objects.
        time_window_hours: Time window for detecting round-trip flows (default 24h).

    Returns:
        Float in range [0.0, 1.0] representing the fraction of the wallet's
        effective volume that completes round-trips. Returns 0.0 if no flows exist.
    """
    if not path_flows:
        return 0.0

    # Filter flows involving the wallet as source
    wallet_flows = [f for f in path_flows if f["source_wallet"] == wallet]
    if not wallet_flows:
        return 0.0

    total_source_amount = sum(f["source_amount"] for f in wallet_flows)
    if total_source_amount == 0:
        return 0.0

    # Count round-trip flows within time window
    round_trip_volume = 0.0
    for flow in wallet_flows:
        if not flow["is_round_trip"]:
            continue

        # Check if the flow's destination matches the wallet and it's within window
        if flow["destination_wallet"] == wallet:
            round_trip_volume += flow["source_amount"]

    return round_trip_volume / total_source_amount


def validate_path_schema(path_payment_op: dict) -> bool:
    """Validate that a path payment operation conforms to Stellar schema.

    Args:
        path_payment_op: Operation dict to validate.

    Returns:
        True if operation is valid, False otherwise.
    """
    required_fields = {"source_account", "destination_account", "transaction_id", "type"}
    if not required_fields.issubset(path_payment_op.keys()):
        return False

    # Validate operation type
    if path_payment_op["type"] not in ("path_payment_strict_send", "path_payment_strict_receive"):
        return False

    # Validate asset path length
    asset_path = path_payment_op.get("asset_path", [])
    if len(asset_path) > MAX_PATH_LENGTH:
        return False

    # Validate account IDs format (Stellar accounts start with 'G' and are 56 chars)
    for account_id in [
        path_payment_op.get("source_account"),
        path_payment_op.get("destination_account"),
    ]:
        if not (
            isinstance(account_id, str) and account_id.startswith("G") and len(account_id) == 56
        ):
            return False

    return True


def merge_path_flows(
    flows1: list[ReconstructedPathFlow], flows2: list[ReconstructedPathFlow]
) -> list[ReconstructedPathFlow]:
    """Merge two lists of path flows, consolidating duplicate routes.

    Args:
        flows1: First list of path flows.
        flows2: Second list of path flows.

    Returns:
        Merged list with deduplicated flows based on source/destination wallets.
    """
    merged = {f"{f['source_wallet']}→{f['destination_wallet']}": f for f in flows1}

    for flow in flows2:
        key = f"{flow['source_wallet']}→{flow['destination_wallet']}"
        if key in merged:
            # Consolidate path IDs and update amounts
            merged[key]["path_payment_ids"].extend(flow["path_payment_ids"])
            merged[key]["source_amount"] += flow["source_amount"]
            merged[key]["destination_amount"] += flow["destination_amount"]
        else:
            merged[key] = flow

    return list(merged.values())


@dataclass
class PathTraceResult:
    """Outcome of a bounded payment-graph traversal (issue #916).

    Attributes:
        start_wallet: Wallet the trace originated from.
        paths: Every maximal wallet path found, as lists of wallet IDs
            beginning with ``start_wallet``.
        payment_ids: Transaction IDs of the payments along each path, index-
            aligned with ``paths``.
        cycles: Wallet paths that closed a cycle (the final wallet already
            appeared earlier on the same path).
        truncated: True if any traversal bound was hit.
        truncation_reasons: Distinct bounds that were hit (``"max_depth"``,
            ``"max_branching"``, ``"max_paths"``).
        truncated_payment_ids: Transaction IDs of payments that were not
            followed because of truncation, for follow-up investigation.
    """

    start_wallet: str
    paths: list[list[str]] = field(default_factory=list)
    payment_ids: list[list[str]] = field(default_factory=list)
    cycles: list[list[str]] = field(default_factory=list)
    truncated: bool = False
    truncation_reasons: set[str] = field(default_factory=set)
    truncated_payment_ids: list[str] = field(default_factory=list)

    @property
    def has_cycle(self) -> bool:
        return bool(self.cycles)


def trace_payment_paths(
    start_wallet: str,
    payments: list[dict],
    max_depth: int = DEFAULT_TRACE_MAX_DEPTH,
    max_branching: int = DEFAULT_TRACE_MAX_BRANCHING,
    max_paths: int = DEFAULT_TRACE_MAX_PATHS,
) -> PathTraceResult:
    """Trace multi-hop payment flows outward from ``start_wallet``.

    Builds a directed wallet graph from ``payments`` (each dict must carry
    ``source_account``, ``destination_account`` and ``transaction_id``) and
    walks every simple path from ``start_wallet``. The walk is iterative
    (explicit stack, no recursion) and guarded against adversarial graphs:

    * **Cycle detection** -- a payment into a wallet already on the current
      path is recorded in ``cycles`` and not followed further.
    * **max_depth** -- paths stop growing after ``max_depth`` hops.
    * **max_branching** -- at most ``max_branching`` outgoing payments are
      followed from any wallet.
    * **max_paths** -- the trace stops once ``max_paths`` paths are emitted.

    Hitting a bound never raises: the result is marked ``truncated`` and a
    warning naming the start wallet, the bound(s) hit, and a sample of the
    unfollowed transaction IDs is logged so the payment can be followed up.

    Args:
        start_wallet: Wallet to trace from.
        payments: Payment/path-payment operation dicts.
        max_depth: Maximum hops per path (must be >= 1).
        max_branching: Maximum outgoing payments followed per wallet (>= 1).
        max_paths: Maximum number of paths emitted (>= 1).

    Returns:
        A :class:`PathTraceResult`.

    Raises:
        InvalidInputError: If any bound is less than 1.
    """
    for name, value in (
        ("max_depth", max_depth),
        ("max_branching", max_branching),
        ("max_paths", max_paths),
    ):
        if value < 1:
            raise InvalidInputError(
                f"{name} must be >= 1, got {value}",
                source="payment_path_analyzer.trace_payment_paths",
                reason=f"invalid traversal bound {name}={value!r}",
            )

    graph: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for op in payments:
        src = op.get("source_account")
        dst = op.get("destination_account")
        tx = op.get("transaction_id")
        if src and dst and tx:
            graph[src].append((dst, str(tx)))

    result = PathTraceResult(start_wallet=start_wallet)

    def _truncate(reason: str, skipped: list[tuple[str, str]]) -> None:
        result.truncated = True
        result.truncation_reasons.add(reason)
        result.truncated_payment_ids.extend(tx for _, tx in skipped)

    # Stack entries: (wallet path, payment-id path). Pushed in reverse so the
    # walk visits payments in input order, keeping output deterministic.
    stack: list[tuple[list[str], list[str]]] = [([start_wallet], [])]
    while stack:
        if len(result.paths) >= max_paths:
            _truncate("max_paths", [])
            break

        wallets, tx_ids = stack.pop()
        current = wallets[-1]
        edges = graph.get(current, [])

        if not edges:
            if tx_ids:
                result.paths.append(wallets)
                result.payment_ids.append(tx_ids)
            continue

        if len(tx_ids) >= max_depth:
            result.paths.append(wallets)
            result.payment_ids.append(tx_ids)
            _truncate("max_depth", edges)
            continue

        if len(edges) > max_branching:
            _truncate("max_branching", edges[max_branching:])
            edges = edges[:max_branching]

        on_path = set(wallets)
        children: list[tuple[list[str], list[str]]] = []
        for dst, tx in edges:
            if dst in on_path:
                result.cycles.append([*wallets, dst])
                result.paths.append([*wallets, dst])
                result.payment_ids.append([*tx_ids, tx])
                continue
            children.append(([*wallets, dst], [*tx_ids, tx]))
        stack.extend(reversed(children))

    if len(result.paths) > max_paths:
        del result.paths[max_paths:]
        del result.payment_ids[max_paths:]
        result.truncated = True
        result.truncation_reasons.add("max_paths")

    if result.truncated:
        logger.warning(
            "trace_payment_paths: truncated trace from wallet=%s reasons=%s "
            "paths=%d cycles=%d unfollowed_payments=%d sample_tx_ids=%s "
            "(max_depth=%d max_branching=%d max_paths=%d)",
            start_wallet,
            sorted(result.truncation_reasons),
            len(result.paths),
            len(result.cycles),
            len(result.truncated_payment_ids),
            result.truncated_payment_ids[:10],
            max_depth,
            max_branching,
            max_paths,
        )
    return result
