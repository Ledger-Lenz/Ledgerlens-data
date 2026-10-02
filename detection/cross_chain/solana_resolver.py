"""Solana cross-chain identity resolver for Stellar ↔ Solana linkage detection.

Detects Stellar wallets linked to Solana addresses through Wormhole bridge
transactions. Extracts Stellar destination addresses from Wormhole VAA
(Verified Action Approval) payloads embedded in Solana transactions.

Also classifies Solana addresses as user-controlled wallets vs program-derived
addresses (PDAs) so program-internal accounting accounts (token vaults,
associated token accounts, DeFi pool authorities, bridge custody accounts) are
not misattributed as user wallets in the cross-chain identity graph (#881).
See ``docs/solana_pda_detection.md``.

References:
    - Wormhole Bridge: https://wormhole.com/
    - Wormhole Program ID (Solana): wormDTL6mgvNpWAoVgqKmqDQMUqr94c3gqPqstQQQm
    - Wormhole VAA Format: https://docs.wormhole.com/wormhole/reference/components
"""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass, field
from typing import Any

import requests
from cachetools import TTLCache

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

# Solana address validation: 32-byte base58-encoded public key
SOLANA_ADDRESS_PATTERN = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")

# Wormhole Program ID on Solana (base58-encoded)
WORMHOLE_PROGRAM_ID = "wormDTL6mgvNpWAoVgqKmqDQMUqr94c3gqPqstQQQm"

# Wormhole VAA signature verification requires understanding the Guardian set.
# For now, we perform basic structure validation. Full verification requires
# Wormhole client libraries or custom implementation.
WORMHOLE_INSTRUCTION_PREFIX = bytes.fromhex("d0e81637b694")  # Common Wormhole instruction prefix


# ---------------------------------------------------------------------------
# Program-derived address (PDA) detection (#881)
# ---------------------------------------------------------------------------

_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_BASE58_INDEX = {ch: i for i, ch in enumerate(_BASE58_ALPHABET)}

# Ed25519 curve parameters (RFC 8032). A PDA is by construction a 32-byte
# value that does NOT decode to a point on this curve, so no private key can
# exist for it and it can only be "signed for" by its owning program.
_ED25519_P = 2**255 - 19
_ED25519_D = (-121665 * pow(121666, _ED25519_P - 2, _ED25519_P)) % _ED25519_P

_PDA_MARKER = b"ProgramDerivedAddress"
_MAX_SEED_LEN = 32
_MAX_SEEDS = 16

SYSTEM_PROGRAM_ID = "11111111111111111111111111111111"
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EAFqZ1ALBG8PS3Kd1pJG"
ASSOCIATED_TOKEN_PROGRAM_ID = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
WORMHOLE_CORE_PROGRAM_ID = "worm2ZoG2kUd4vFXhvjh93UUH596ayRfgQ2MgjNMTth"
WORMHOLE_TOKEN_BRIDGE_PROGRAM_ID = "wormDTUJ6AWPNvk59vGQbDvGJmqbDTdgWgAqcLBCgUb"
RAYDIUM_AMM_V4_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
ORCA_WHIRLPOOL_PROGRAM_ID = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"
MARINADE_PROGRAM_ID = "MarBmsSgKXdrN1egZf5sqe1TMai9K1rChYNDJgjq7aD"

# Program IDs whose owned accounts are program-internal accounting state, not
# user wallets. To cover a new program, add its ID here (and, if its PDAs use a
# fixed seed layout, a derivation helper in ``KNOWN_PDA_SEED_PATTERNS``).
KNOWN_PROGRAM_IDS: dict[str, str] = {
    TOKEN_PROGRAM_ID: "spl_token",
    TOKEN_2022_PROGRAM_ID: "spl_token_2022",
    ASSOCIATED_TOKEN_PROGRAM_ID: "associated_token_account",
    WORMHOLE_CORE_PROGRAM_ID: "wormhole_core",
    WORMHOLE_TOKEN_BRIDGE_PROGRAM_ID: "wormhole_token_bridge",
    RAYDIUM_AMM_V4_PROGRAM_ID: "raydium_amm_v4",
    ORCA_WHIRLPOOL_PROGRAM_ID: "orca_whirlpool",
    MARINADE_PROGRAM_ID: "marinade",
}

# Documented fixed-seed PDA layouts. Each entry maps a pattern name to
# (program_id, seed layout description). Seeds written as <name> are
# pubkeys/values supplied by the caller in ``derivation_hints``.
KNOWN_PDA_SEED_PATTERNS: dict[str, tuple[str, str]] = {
    "associated_token_account": (
        ASSOCIATED_TOKEN_PROGRAM_ID,
        "[<wallet>, <token_program>, <mint>]",
    ),
    "wormhole_token_bridge_custody": (WORMHOLE_TOKEN_BRIDGE_PROGRAM_ID, "[<mint>]"),
    "wormhole_token_bridge_authority_signer": (
        WORMHOLE_TOKEN_BRIDGE_PROGRAM_ID,
        '["authority_signer"]',
    ),
    "wormhole_token_bridge_custody_signer": (
        WORMHOLE_TOKEN_BRIDGE_PROGRAM_ID,
        '["custody_signer"]',
    ),
    "raydium_amm_authority": (RAYDIUM_AMM_V4_PROGRAM_ID, '["amm authority"]'),
    "orca_whirlpool": (
        ORCA_WHIRLPOOL_PROGRAM_ID,
        '["whirlpool", <config>, <mint_a>, <mint_b>, <tick_spacing u16 LE>]',
    ),
    "marinade_reserve": (MARINADE_PROGRAM_ID, '[<state>, "reserve"]'),
}


def base58_decode(value: str) -> bytes:
    """Decode a base58 string into raw bytes (leading '1's become zero bytes)."""
    n = 0
    for ch in value:
        try:
            n = n * 58 + _BASE58_INDEX[ch]
        except KeyError as exc:
            raise SolanaValidationError(f"Invalid base58 character {ch!r}") from exc
    body = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    pad = len(value) - len(value.lstrip("1"))
    return b"\x00" * pad + body


def base58_encode(raw: bytes) -> str:
    """Encode raw bytes as base58 (leading zero bytes become '1's)."""
    n = int.from_bytes(raw, "big")
    out = []
    while n > 0:
        n, r = divmod(n, 58)
        out.append(_BASE58_ALPHABET[r])
    pad = len(raw) - len(raw.lstrip(b"\x00"))
    return "1" * pad + "".join(reversed(out))


def _pubkey_bytes(value: str | bytes) -> bytes:
    raw = value if isinstance(value, bytes) else base58_decode(value)
    if len(raw) != 32:
        raise SolanaValidationError(f"Solana public key must be 32 bytes, got {len(raw)}")
    return raw


def is_on_ed25519_curve(pubkey: str | bytes) -> bool:
    """Return True if the 32-byte key decompresses to a valid Ed25519 point.

    Mirrors ``curve25519_dalek::CompressedEdwardsY::decompress`` as used by
    the Solana runtime: y must be canonical (< p) and x^2 = (y^2 - 1) /
    (d*y^2 + 1) must be a quadratic residue mod p.
    """
    raw = _pubkey_bytes(pubkey)
    y = int.from_bytes(raw, "little") & ((1 << 255) - 1)
    if y >= _ED25519_P:
        return False
    y2 = y * y % _ED25519_P
    u = (y2 - 1) % _ED25519_P
    v = (_ED25519_D * y2 + 1) % _ED25519_P
    x2 = u * pow(v, _ED25519_P - 2, _ED25519_P) % _ED25519_P
    if x2 == 0:
        return True
    return pow(x2, (_ED25519_P - 1) // 2, _ED25519_P) == 1


def _seed_bytes(seed: str | bytes | int) -> bytes:
    if isinstance(seed, bytes):
        return seed
    if isinstance(seed, int):
        return bytes([seed])
    # Treat strings that decode to a 32-byte pubkey as pubkeys, else UTF-8 text.
    if validate_solana_address(seed):
        try:
            raw = base58_decode(seed)
            if len(raw) == 32:
                return raw
        except SolanaValidationError:
            pass
    return seed.encode("utf-8")


def create_program_address(seeds: list[str | bytes | int], program_id: str) -> str | None:
    """Solana ``Pubkey::create_program_address``; None if the result is on-curve."""
    if len(seeds) > _MAX_SEEDS:
        raise SolanaValidationError(f"At most {_MAX_SEEDS} seeds are allowed")
    h = hashlib.sha256()
    for seed in seeds:
        raw = _seed_bytes(seed)
        if len(raw) > _MAX_SEED_LEN:
            raise SolanaValidationError(f"Seed longer than {_MAX_SEED_LEN} bytes")
        h.update(raw)
    h.update(_pubkey_bytes(program_id))
    h.update(_PDA_MARKER)
    digest = h.digest()
    if is_on_ed25519_curve(digest):
        return None
    return base58_encode(digest)


def find_program_address(seeds: list[str | bytes | int], program_id: str) -> tuple[str, int]:
    """Solana ``Pubkey::find_program_address``: canonical PDA and bump seed."""
    for bump in range(255, -1, -1):
        address = create_program_address([*seeds, bytes([bump])], program_id)
        if address is not None:
            return address, bump
    raise SolanaValidationError("Unable to find a viable program address bump seed")


def get_associated_token_address(
    wallet: str, mint: str, token_program_id: str = TOKEN_PROGRAM_ID
) -> str:
    """Derive the associated token account (ATA) PDA for ``wallet`` and ``mint``."""
    address, _ = find_program_address([wallet, token_program_id, mint], ASSOCIATED_TOKEN_PROGRAM_ID)
    return address


@dataclass(frozen=True)
class PDADerivationHint:
    """A candidate derivation to verify an address against."""

    program_id: str
    seeds: list[str | bytes | int] = field(default_factory=list)
    pattern: str = "custom"


@dataclass(frozen=True)
class SolanaAddressClassification:
    """Result of classifying a Solana address.

    ``kind`` is one of:
      - ``"wallet"``: on-curve key, not owned by a known program (user-controlled).
      - ``"pda"``: off-curve and verified against a known derivation.
      - ``"pda_unverified"``: off-curve, so it cannot be a keypair, but no
        supplied derivation reproduced it.
      - ``"program"``: the address is itself a known program ID.
      - ``"program_owned"``: on-curve keypair account owned by a known program
        (e.g. a non-associated SPL token account).
      - ``"invalid"``: not a 32-byte base58 public key.
    """

    address: str
    kind: str
    program_id: str | None = None
    program_name: str | None = None
    pattern: str | None = None
    bump: int | None = None

    @property
    def is_user_wallet(self) -> bool:
        return self.kind == "wallet"


def classify_solana_address(
    address: str,
    derivation_hints: list[PDADerivationHint] | None = None,
    owner_program_id: str | None = None,
) -> SolanaAddressClassification:
    """Classify ``address`` as a user wallet or a program-derived/program-owned account.

    Args:
        address: base58 Solana address.
        derivation_hints: candidate (program_id, seeds) pairs. When one of them
            re-derives ``address`` the PDA is attributed to that program.
        owner_program_id: account owner as reported by ``getAccountInfo``,
            when known. Accounts owned by a program in ``KNOWN_PROGRAM_IDS``
            are program-internal even when on-curve.
    """
    if not validate_solana_address(address):
        return SolanaAddressClassification(address=address, kind="invalid")
    if address in KNOWN_PROGRAM_IDS:
        return SolanaAddressClassification(
            address=address,
            kind="program",
            program_id=address,
            program_name=KNOWN_PROGRAM_IDS[address],
        )
    try:
        on_curve = is_on_ed25519_curve(address)
    except SolanaValidationError:
        return SolanaAddressClassification(address=address, kind="invalid")

    for hint in derivation_hints or []:
        try:
            derived, bump = find_program_address(hint.seeds, hint.program_id)
        except SolanaValidationError:
            continue
        if derived == address:
            return SolanaAddressClassification(
                address=address,
                kind="pda",
                program_id=hint.program_id,
                program_name=KNOWN_PROGRAM_IDS.get(hint.program_id),
                pattern=hint.pattern,
                bump=bump,
            )

    if not on_curve:
        return SolanaAddressClassification(
            address=address,
            kind="pda_unverified",
            program_id=owner_program_id,
            program_name=KNOWN_PROGRAM_IDS.get(owner_program_id or ""),
        )

    if owner_program_id and owner_program_id in KNOWN_PROGRAM_IDS:
        return SolanaAddressClassification(
            address=address,
            kind="program_owned",
            program_id=owner_program_id,
            program_name=KNOWN_PROGRAM_IDS[owner_program_id],
        )

    return SolanaAddressClassification(address=address, kind="wallet")


NON_USER_ADDRESS_KINDS = frozenset({"pda", "pda_unverified", "program", "program_owned"})


def filter_pda_links(
    links: list[dict[str, Any]],
    include_pdas: bool | None = None,
    address_key: str = "solana_address",
) -> list[dict[str, Any]]:
    """Drop (or tag) links whose Solana side is a PDA / program-owned account.

    By default (``config.SOLANA_INCLUDE_PDA_EDGES`` false) such links are
    excluded from the user-identity graph. When included, each surviving link
    is tagged with ``solana_address_kind`` so downstream consumers can tell
    them apart.
    """
    if include_pdas is None:
        include_pdas = bool(getattr(config, "SOLANA_INCLUDE_PDA_EDGES", False))

    kept: list[dict[str, Any]] = []
    for link in links:
        address = link.get(address_key)
        if not address:
            kept.append(link)
            continue
        classification = classify_solana_address(
            address,
            derivation_hints=link.get("derivation_hints"),
            owner_program_id=link.get("owner_program_id"),
        )
        tagged = {**link, "solana_address_kind": classification.kind}
        if classification.kind in NON_USER_ADDRESS_KINDS:
            if not include_pdas:
                logger.info(
                    "Excluding %s Solana address %s from identity graph",
                    classification.kind,
                    address,
                )
                continue
            tagged["program_id"] = classification.program_id
        kept.append(tagged)
    return kept


class SolanaValidationError(Exception):
    """Raised when Solana address or transaction validation fails."""

    pass


class SolanaRPCResponseError(SolanaValidationError):
    """Raised when a Solana RPC response is malformed or not shaped as expected."""

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
    if not transaction_data:
        return None

    try:
        version = transaction_data[0]
        if version != 1:
            raise WormholeVAAValidationError(f"Unsupported VAA version: {version}")

        if len(transaction_data) < 20:
            return None

        # Wormhole VAA structure (simplified):
        # Byte 0: version (always 1)
        # Bytes 1-4: guardian_set_index (big-endian)
        # Byte 5: signature_count
        # Bytes 6+: signatures (65 bytes each)
        # Followed by core VAA (19 bytes header + payload)

        offset = 0
        offset += 1

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

        destination_address = (
            transaction_data[offset : offset + 32].hex() if len(transaction_data) > offset else ""
        )

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
            # Decode (raises ValueError below if not valid hex) and verify it
            # looks like a Stellar address
            bytes.fromhex(dest_addr)

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
        self.rpc_url = rpc_url or getattr(
            config, "SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"
        )
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

            if not isinstance(data, dict):
                raise SolanaRPCResponseError(
                    f"Malformed Solana RPC response for {address}: expected a JSON object, "
                    f"got {type(data).__name__}"
                )

            if "error" in data:
                logger.error("Solana RPC error: %s", data["error"])
                return []

            result = data.get("result")
            if not isinstance(result, list):
                raise SolanaRPCResponseError(
                    f"Malformed Solana RPC response for {address}: expected 'result' to be a list, "
                    f"got {type(result).__name__}"
                )

            signatures = []
            for sig in result:
                if not isinstance(sig, dict) or "signature" not in sig:
                    raise SolanaRPCResponseError(
                        f"Malformed Solana RPC response for {address}: each result item must include "
                        "a string 'signature'"
                    )
                signature = sig["signature"]
                if not isinstance(signature, str):
                    raise SolanaRPCResponseError(
                        f"Malformed Solana RPC response for {address}: 'signature' must be a string"
                    )
                signatures.append(signature)

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

            if not isinstance(data, dict):
                raise SolanaRPCResponseError(
                    f"Malformed Solana RPC response for transaction {signature}: expected a JSON object, "
                    f"got {type(data).__name__}"
                )

            if "error" in data:
                logger.warning("Transaction not found: %s", signature)
                return None

            tx_data = data.get("result")
            if not isinstance(tx_data, dict):
                raise SolanaRPCResponseError(
                    f"Malformed Solana RPC response for transaction {signature}: expected 'result' to be a "
                    f"dict, got {type(tx_data).__name__}"
                )

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

        logger.info(
            "Placeholder: find_wormhole_deposits for %s (would query Wormhole program)",
            stellar_address,
        )

        return results


def resolve_stellar_to_solana(
    stellar_address: str, rpc_client: SolanaRPCClient | None = None
) -> list[dict[str, Any]]:
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
        return filter_pda_links(deposits)
    except Exception as exc:
        logger.error("Failed to resolve Stellar %s to Solana: %s", stellar_address, exc)
        return []


# For testing/manual usage
if __name__ == "__main__":
    # Example: validate Solana address
    test_addr = "11111111111111111111111111111111"
    print(f"Validating {test_addr}: {validate_solana_address(test_addr)}")
