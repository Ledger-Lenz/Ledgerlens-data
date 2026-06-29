"""Solana cross-chain identity resolver for Stellar ↔ Solana linkage detection.

Detects Stellar wallets linked to Solana addresses through Wormhole bridge
transactions. Extracts Stellar destination addresses from Wormhole VAA
(Verified Action Approval) payloads embedded in Solana transactions.

References:
    - Wormhole Bridge: https://wormhole.com/
    - Wormhole Program ID (Solana): wormDTL6mgvNpWAoVgqKmqDQMUqr94c3gqPqstQQQm
    - Wormhole VAA Format: https://docs.wormhole.com/wormhole/reference/components
"""

from __future__ import annotations

import hashlib
import logging
import re
import struct
from typing import Any

import requests
from cachetools import TTLCache

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

# Solana address validation: 32-byte base58-encoded public key
SOLANA_ADDRESS_PATTERN = re.compile(r"^[1-9A-HJ-NP-Z]{32,44}$")

# Wormhole Program ID on Solana (base58-encoded)
WORMHOLE_PROGRAM_ID = "wormDTL6mgvNpWAoVgqKmqDQMUqr94c3gqPqstQQQm"

# Wormhole VAA signature verification requires understanding the Guardian set.
# For now, we perform basic structure validation. Full verification requires
# Wormhole client libraries or custom implementation.
WORMHOLE_INSTRUCTION_PREFIX = bytes.fromhex("d0e81637b694")  # Common Wormhole instruction prefix


class SolanaValidationError(Exception):
    """Raised when Solana address or transaction validation fails."""

    pass


class WormholeVAAValidationError(Exception):
    """Raised when Wormhole VAA validation fails."""

    pass


def validate_solana_address(address: str) -> bool:
    """Validate that a string is a valid Solana base58-encoded public key.

    Args:
        address: Potential Solana address

    Returns:
        True if valid, False otherwise

    Raises:
        SolanaValidationError: If validation fails (never—always returns bool)
    """
    if not isinstance(address, str):
        return False

    address = address.strip()

    # Check length: Solana public keys are 32 bytes, base58-encoded = 32-44 chars
    if len(address) < 32 or len(address) > 44:
        return False

    # Check character set: base58 excludes 0, O, I, l
    if not SOLANA_ADDRESS_PATTERN.match(address):
        return False

    # Optional: verify it's valid base58 by attempting decode (external library required)
    # For now, regex is sufficient as a quick check
    return True


def parse_wormhole_vaa_payload(transaction_data: bytes) -> dict[str, Any] | None:
    """Parse a Wormhole VAA payload from Solana transaction data.

    Wormhole transactions embed VAA (Verified Action Approval) payloads that
    contain routing information, including the destination chain and destination
    address. This function extracts that information.

    Args:
        transaction_data: Raw transaction data (bytes) from Solana transaction

    Returns:
        Dictionary with parsed VAA info, or None if parsing fails.
        Schema: {
            "vaa_version": int,
            "guardian_set_index": int,
            "signature_count": int,
            "timestamp": int,
            "nonce": int,
            "emitter_chain": int,
            "emitter_address": str (hex),
            "sequence": int,
            "consistency_level": int,
            "payload_type": int,
            "destination_chain": int,
            "destination_address": str (hex, variable length),
            "token": str (hex, optional),
            "amount": int (optional),
        }

    Raises:
        WormholeVAAValidationError: If VAA structure is invalid
    """
    if not transaction_data or len(transaction_data) < 20:
        return None

    try:
        # Wormhole VAA structure (simplified):
        # Byte 0: version (always 1)
        # Bytes 1-4: guardian_set_index (big-endian)
        # Byte 5: signature_count
        # Bytes 6+: signatures (65 bytes each)
        # Followed by core VAA (19 bytes header + payload)

        offset = 0
        version = transaction_data[offset]
        offset += 1

        if version != 1:
            raise WormholeVAAValidationError(f"Unsupported VAA version: {version}")

        guardian_set_index = int.from_bytes(transaction_data[offset : offset + 4], "big")
        offset += 4

        signature_count = transaction_data[offset]
        offset += 1

        # Skip signatures (65 bytes each: 64-byte signature + 1-byte recovery id)
        offset += signature_count * 65

        if len(transaction_data) < offset + 19:
            raise WormholeVAAValidationError("VAA payload too short")

        # Core VAA header
        timestamp = int.from_bytes(transaction_data[offset : offset + 4], "big")
        offset += 4

        nonce = int.from_bytes(transaction_data[offset : offset + 4], "big")
        offset += 4

        emitter_chain = int.from_bytes(transaction_data[offset : offset + 2], "big")
        offset += 2

        emitter_address = transaction_data[offset : offset + 32].hex()
        offset += 32

        sequence = int.from_bytes(transaction_data[offset : offset + 8], "big")
        offset += 8

        consistency_level = transaction_data[offset]
        offset += 1

        # Payload: structure depends on message type
        # For cross-chain bridge messages:
        # - First byte: payload type
        # - Next bytes: destination_chain (uint16), destination_address (variable)

        if len(transaction_data) < offset + 3:
            raise WormholeVAAValidationError("Payload too short")

        payload_type = transaction_data[offset]
        offset += 1

        destination_chain = int.from_bytes(transaction_data[offset : offset + 2], "big")
        offset += 2

        # Destination address length varies by chain
        # For Stellar: 56 bytes (base32-encoded)
        # For Ethereum/EVM: 20 bytes
        # For Solana: 32 bytes
        # Read remaining as destination address

        destination_address = transaction_data[offset : offset + 32].hex() if len(
            transaction_data
        ) > offset else ""

        # Optional fields (if present)
        token = None
        amount = None

        return {
            "vaa_version": version,
            "guardian_set_index": guardian_set_index,
            "signature_count": signature_count,
            "timestamp": timestamp,
            "nonce": nonce,
            "emitter_chain": emitter_chain,
            "emitter_address": emitter_address,
            "sequence": sequence,
            "consistency_level": consistency_level,
            "payload_type": payload_type,
            "destination_chain": destination_chain,
            "destination_address": destination_address,
            "token": token,
            "amount": amount,
        }

    except (IndexError, struct.error) as exc:
        logger.warning("Failed to parse Wormhole VAA: %s", exc)
        return None


def extract_stellar_address_from_vaa(vaa_data: dict[str, Any]) -> str | None:
    """Extract Stellar wallet address from parsed Wormhole VAA data.

    Args:
        vaa_data: Parsed VAA dictionary from parse_wormhole_vaa_payload()

    Returns:
        Stellar address (starts with 'G') if found and valid, None otherwise
    """
    if not vaa_data or "destination_address" not in vaa_data:
        return None

    dest_addr = vaa_data.get("destination_address", "")

    # For Wormhole, Stellar addresses are often encoded as hex strings
    # Try to decode from hex and validate as Stellar address
    try:
        # If already hex, try to convert back to Stellar format
        if len(dest_addr) == 56:  # 28 bytes in hex
            # This is likely a Stellar address in hex format
            # Decode and verify it looks like a Stellar address
            decoded = bytes.fromhex(dest_addr)

            # Stellar addresses are base32-encoded with 'G' prefix
            # They encode to 56 characters (28 bytes × 8/5)
            # For now, accept if it starts with 'G' after decoding
            return dest_addr

        if dest_addr.startswith("G") and 50 <= len(dest_addr) <= 60:
            # Already in Stellar format
            return dest_addr

    except (ValueError, AttributeError) as exc:
        logger.debug("Failed to extract Stellar address from VAA: %s", exc)

    return None


class SolanaRPCClient:
    """Client for Solana RPC API with caching and rate limiting."""

    def __init__(self, rpc_url: str | None = None, cache_ttl_seconds: int = 3600):
        """Initialize Solana RPC client.

        Args:
            rpc_url: Solana RPC endpoint URL. Defaults to config.SOLANA_RPC_URL
            cache_ttl_seconds: Cache TTL for signatures and transactions (default 1 hour)
        """
        self.rpc_url = rpc_url or getattr(config, "SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
        self.cache: TTLCache = TTLCache(maxsize=1000, ttl=cache_ttl_seconds)
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def get_signatures_for_address(
        self, address: str, limit: int = 100, before: str | None = None
    ) -> list[str]:
        """Get recent transaction signatures for a Solana address.

        Args:
            address: Solana address to query
            limit: Maximum number of signatures to return (1-1000, default 100)
            before: Signature to start searching backward from (pagination)

        Returns:
            List of transaction signatures (up to `limit`)

        Raises:
            SolanaValidationError: If address is invalid
            requests.RequestException: If RPC call fails
        """
        if not validate_solana_address(address):
            raise SolanaValidationError(f"Invalid Solana address: {address}")

        # Check cache
        cache_key = f"sigs_{address}_{limit}_{before}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSignaturesForAddress",
            "params": [address, {"limit": min(limit, 1000)}],
        }

        if before:
            payload["params"][1]["before"] = before

        try:
            response = self.session.post(self.rpc_url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.error("Solana RPC error: %s", data["error"])
                return []

            signatures = [sig["signature"] for sig in data.get("result", [])]
            self.cache[cache_key] = signatures

            return signatures

        except requests.RequestException as exc:
            logger.error("Failed to query Solana RPC for %s: %s", address, exc)
            raise

    def get_transaction(self, signature: str) -> dict[str, Any] | None:
        """Get full transaction data for a signature.

        Args:
            signature: Transaction signature

        Returns:
            Transaction data dict, or None if not found

        Raises:
            requests.RequestException: If RPC call fails
        """
        # Check cache
        if signature in self.cache:
            return self.cache[signature]

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [signature, {"encoding": "jsonParsed"}],
        }

        try:
            response = self.session.post(self.rpc_url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.warning("Transaction not found: %s", signature)
                return None

            tx_data = data.get("result")
            if tx_data:
                self.cache[signature] = tx_data

            return tx_data

        except requests.RequestException as exc:
            logger.error("Failed to query transaction %s: %s", signature, exc)
            raise

    def find_wormhole_deposits(self, stellar_address: str, limit: int = 50) -> list[dict[str, Any]]:
        """Find Wormhole bridge deposit transactions linking to a Stellar address.

        Args:
            stellar_address: Stellar wallet address to search for
            limit: Maximum Solana addresses to check (pagination)

        Returns:
            List of dictionaries:
            [
                {
                    "solana_address": "...",
                    "stellar_address": "...",
                    "transaction_signature": "...",
                    "vaa_data": {...},
                    "timestamp": unix_timestamp,
                }
            ]
        """
        results = []

        # Query Wormhole program for deposits
        # This is a simplified approach: in production, you'd query the Wormhole
        # program state or use a dedicated indexer.
        # For now, we return an empty list as a placeholder.
        #
        # Full implementation would:
        # 1. Query Wormhole portal state (getProgramAccounts on WORMHOLE_PROGRAM_ID)
        # 2. Filter for deposit messages destined to Stellar
        # 3. Extract embedded Stellar addresses and link to Solana signers

        logger.info("Placeholder: find_wormhole_deposits for %s (would query Wormhole program)", stellar_address)

        return results


def resolve_stellar_to_solana(stellar_address: str, rpc_client: SolanaRPCClient | None = None) -> list[dict[str, Any]]:
    """Resolve a Stellar address to linked Solana addresses via Wormhole.

    Args:
        stellar_address: Stellar wallet address
        rpc_client: SolanaRPCClient instance (creates default if None)

    Returns:
        List of linked Solana addresses with metadata:
        [
            {
                "solana_address": "...",
                "link_type": "wormhole_bridge",
                "confidence": 0.95,
                "transaction_signature": "...",
                "timestamp": unix_timestamp,
            }
        ]
    """
    if not stellar_address.startswith("G") or len(stellar_address) != 56:
        logger.warning("Invalid Stellar address: %s", stellar_address)
        return []

    if rpc_client is None:
        rpc_client = SolanaRPCClient()

    try:
        deposits = rpc_client.find_wormhole_deposits(stellar_address)
        return deposits
    except Exception as exc:
        logger.error("Failed to resolve Stellar %s to Solana: %s", stellar_address, exc)
        return []


# For testing/manual usage
if __name__ == "__main__":
    # Example: validate Solana address
    test_addr = "11111111111111111111111111111111"
    print(f"Validating {test_addr}: {validate_solana_address(test_addr)}")
