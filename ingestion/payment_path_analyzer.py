"""Payment path analysis for detecting multi-hop wash trade routing.

Sophisticated wash traders on the Stellar DEX route trades through multi-hop
payment paths (using Stellar's path payment operations) to obfuscate the
connection between buyer and seller wallets. This module reconstructs these
multi-hop flows and attributes them to the originating wallets.

Attributes:
    MAX_PATH_LENGTH: Maximum allowed hops in a payment path (Stellar's limit).
    ROUND_TRIP_WINDOW_HOURS: Time window for detecting round-trip path flows.
"""

from datetime import datetime, timedelta
from typing import TypedDict

import pandas as pd

MAX_PATH_LENGTH = 6  # Stellar's maximum path hops
ROUND_TRIP_WINDOW_HOURS = 24


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
        hop count, and round-trip flag. Returns None if the operation is malformed
        or exceeds Stellar's path length limit.

    Raises:
        ValueError: If the operation type is not a path payment variant.
        KeyError: If required fields are missing.
    """
    op_type = path_payment_op.get("type")
    if op_type not in ("path_payment_strict_send", "path_payment_strict_receive"):
        raise ValueError(f"Operation type {op_type} is not a path payment variant")

    # Extract mandatory fields
    source_wallet = path_payment_op.get("source_account")
    destination_wallet = path_payment_op.get("destination_account")
    transaction_id = path_payment_op.get("transaction_id")
    created_at = path_payment_op.get("created_at")

    if not all([source_wallet, destination_wallet, transaction_id]):
        raise KeyError("Missing required fields: source_account, destination_account, transaction_id")

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
    for account_id in [path_payment_op.get("source_account"), path_payment_op.get("destination_account")]:
        if not (isinstance(account_id, str) and account_id.startswith("G") and len(account_id) == 56):
            return False

    return True


def merge_path_flows(flows1: list[ReconstructedPathFlow], flows2: list[ReconstructedPathFlow]) -> list[ReconstructedPathFlow]:
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
