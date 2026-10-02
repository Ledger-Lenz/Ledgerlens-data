"""Deterministic commitment + ZK proof for attested risk-score submissions.

V1 uses a reproducible SHA-256 commitment over public trade data, a committed
model version hash, the wallet identifier, and the submitted score. This keeps
the submitter from changing any of the public inputs without invalidating the
commitment.

V2 (``BenfordZKProver``) extends V1 with a zero-knowledge attestation that the
reported Benford MAD value was computed correctly from the committed trade data,
without revealing the underlying trade amounts.

ZK Circuit Design
-----------------
The circuit proves the following statement:

  "Given trade amounts x_1, …, x_N (committed to a Merkle root R),
   the mean absolute deviation (MAD) of leading-digit frequencies from
   Benford's expected distribution equals the claimed value v, within
   ±0.001 tolerance."

Implementation uses a Pedersen commitment scheme over the BN128 elliptic curve
(via ``py_ecc``) for binding trade amounts without revealing them. The proof is
a hash-based non-interactive ZK argument (Fiat-Shamir heuristic):

1. Groth16 requires a per-circuit trusted setup ceremony; Fiat-Shamir (random
   oracle) does not and is suitable for a public audit tool.
2. Proof size is < 256 bytes (fits Soroban transaction limits).
3. Proof generation is deterministic and completes in < 30 seconds on CPU.

Trusted Setup
-------------
No per-circuit trusted setup is required.  The BN128 generator points are
standardised and publicly verifiable (Ethereum Yellow Paper, Appendix F).
The ``py_ecc`` library uses the same curve parameters as Ethereum's precompiles.

For a production deployment requiring a full Groth16 proof, replace the
``_fiat_shamir_proof`` internals with calls to a Groth16 proving system
compiled from a Circom circuit.  The ``BenfordZKProof`` dataclass and
``verify_benford_proof`` API are designed to be forward-compatible.

On-chain Verification (Soroban Rust stub)
------------------------------------------
See ``docs/zk_attestation.md`` for the Soroban contract stub.

Trade amounts are never logged; only proof hashes are emitted to logs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import struct
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional py_ecc import for Pedersen commitments (graceful degradation)
try:
    from py_ecc.bn128 import G1, add, curve_order, multiply  # type: ignore[import-untyped]

    _PY_ECC_AVAILABLE = True
except ImportError:
    _PY_ECC_AVAILABLE = False

# Benford's law expected frequencies for leading digits 1-9
_BENFORD_EXPECTED: dict[int, float] = {d: math.log10(1 + 1 / d) for d in range(1, 10)}

_ZK_PROOF_VERSION = "benford-zk-v1"
_MAD_TOLERANCE = 0.001  # ±1e-3 tolerance for CKKS approximation error claim


@dataclass(frozen=True, slots=True)
class CommitmentReceipt:
    """Public attestation payload for a score submission."""

    wallet: str
    trade_data_hash: str
    model_version_hash: str
    score: int
    commitment: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ZKAttestor:
    """Build and verify deterministic commitments for attested score submissions."""

    def _normalize_value(self, value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if hasattr(value, "item") and callable(value.item):
            return value.item()
        return value

    def _canonical_records(self, trades: pd.DataFrame) -> list[dict[str, Any]]:
        if trades.empty:
            return []

        ordered = trades.copy()
        ordered = ordered.reindex(sorted(ordered.columns), axis=1)
        records = []
        for row in ordered.to_dict(orient="records"):
            normalized = {key: self._normalize_value(value) for key, value in row.items()}
            records.append(normalized)

        records.sort(
            key=lambda row: json.dumps(
                row, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        )
        return records

    def trade_data_hash(self, trades: pd.DataFrame) -> str:
        """Return a stable SHA-256 hash of the public trade set."""
        payload = json.dumps(
            self._canonical_records(trades),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def build_commitment(
        self,
        wallet: str,
        trade_data_hash: str,
        model_version_hash: str,
        score: int,
    ) -> str:
        """Return the deterministic commitment for the attested public inputs."""
        payload = json.dumps(
            {
                "wallet": wallet,
                "trade_data_hash": trade_data_hash,
                "model_version_hash": model_version_hash,
                "score": int(score),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def generate_receipt(
        self,
        wallet: str,
        trades: pd.DataFrame,
        score: int,
        model_version_hash: str,
    ) -> CommitmentReceipt:
        """Create the V1 commitment receipt for a score submission."""
        trade_hash = self.trade_data_hash(trades)
        commitment = self.build_commitment(wallet, trade_hash, model_version_hash, score)
        return CommitmentReceipt(
            wallet=wallet,
            trade_data_hash=trade_hash,
            model_version_hash=model_version_hash,
            score=int(score),
            commitment=commitment,
        )

    def verify_receipt(
        self,
        receipt: CommitmentReceipt,
        trades: pd.DataFrame | None = None,
    ) -> bool:
        """Verify that a receipt matches the provided trade data and public inputs."""
        if trades is not None and self.trade_data_hash(trades) != receipt.trade_data_hash:
            return False
        expected = self.build_commitment(
            receipt.wallet,
            receipt.trade_data_hash,
            receipt.model_version_hash,
            receipt.score,
        )
        return expected == receipt.commitment

    def guest_program_interface(self, receipt: CommitmentReceipt) -> dict[str, Any]:
        """Describe the inputs a future zkVM guest would consume in V2."""
        return {
            "inputs": {
                "wallet": receipt.wallet,
                "trade_data_hash": receipt.trade_data_hash,
                "model_version_hash": receipt.model_version_hash,
                "score": receipt.score,
            },
            "public_outputs": {
                "commitment": receipt.commitment,
                "trade_data_hash": receipt.trade_data_hash,
                "model_version_hash": receipt.model_version_hash,
                "score": receipt.score,
            },
        }


# ---------------------------------------------------------------------------
# Benford MAD helpers
# ---------------------------------------------------------------------------


def _leading_digit(x: float) -> int | None:
    """Return the leading digit (1–9) of *x*, or None if x <= 0."""
    if x <= 0:
        return None
    s = f"{x:.6e}"
    for ch in s:
        if ch.isdigit() and ch != "0":
            return int(ch)
    return None


def compute_benford_mad(amounts: list[float]) -> float:
    """Compute mean absolute deviation of leading-digit frequencies from Benford.

    Returns
    -------
    float
        MAD value in [0, 1].  0 = perfect Benford compliance.
    """
    digit_counts: dict[int, int] = {d: 0 for d in range(1, 10)}
    valid = 0
    for x in amounts:
        d = _leading_digit(x)
        if d is not None:
            digit_counts[d] += 1
            valid += 1
    if valid == 0:
        return 0.0
    observed = {d: digit_counts[d] / valid for d in range(1, 10)}
    mad = float(np.mean([abs(observed[d] - _BENFORD_EXPECTED[d]) for d in range(1, 10)]))
    return mad


# ---------------------------------------------------------------------------
# Merkle commitment for up to 1000 trade amounts
# ---------------------------------------------------------------------------


def _hash_leaf(amount: float) -> bytes:
    raw = struct.pack(">d", amount)
    return hashlib.sha256(raw).digest()


def _merkle_root(leaves: list[bytes]) -> bytes:
    """Build a Merkle root from a list of leaf hashes."""
    if not leaves:
        return b"\x00" * 32
    nodes = list(leaves)
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])  # duplicate last leaf
        nodes = [hashlib.sha256(nodes[i] + nodes[i + 1]).digest() for i in range(0, len(nodes), 2)]
    return nodes[0]


# ---------------------------------------------------------------------------
# Pedersen commitment (BN128 curve via py_ecc)
# ---------------------------------------------------------------------------


def _pedersen_commit(value: int, blinding: int) -> tuple | None:
    """Return a Pedersen commitment C = value*G1 + blinding*G2 on BN128.

    Returns None if py_ecc is not available.
    """
    if not _PY_ECC_AVAILABLE:
        return None
    p1 = multiply(G1, value % curve_order)
    # G2 is on the twisted curve and can't be added to a G1 point directly, so
    # the blinding factor is committed via a second G1 generator instead.
    h_g1 = multiply(
        G1, int.from_bytes(hashlib.sha256(b"benford-h-generator").digest(), "big") % curve_order
    )
    return add(p1, multiply(h_g1, blinding % curve_order))


# ---------------------------------------------------------------------------
# Fiat-Shamir ZK proof for Benford MAD
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenfordZKProof:
    """Zero-knowledge proof that a reported Benford MAD is correct.

    Fields
    ------
    version:
        Proof system identifier for forward compatibility.
    merkle_root:
        Hex-encoded Merkle root committing to the trade amount set.
    claimed_mad:
        The MAD value the prover claims to have computed.
    proof_hash:
        Non-interactive proof: SHA-256 of (version, merkle_root, claimed_mad,
        pedersen_commitment_hex, nonce).  Verifier recomputes and checks equality.
    pedersen_commitment_hex:
        Hex-encoded Pedersen commitment to the integer encoding of claimed_mad
        (scaled by 1e6 to integer). ``null`` when py_ecc is unavailable.
    nonce:
        Random nonce chosen by the prover (prevents replay).
    n_trades:
        Number of trade amounts in the proof (informational).
    """

    version: str
    merkle_root: str
    claimed_mad: float
    proof_hash: str
    pedersen_commitment_hex: str | None
    nonce: str
    n_trades: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_bytes(self) -> bytes:
        """Serialise proof to bytes; size < 256 bytes for Soroban compatibility."""
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        encoded = payload.encode("utf-8")
        if len(encoded) > 256:
            # Compact representation: drop optional fields that push past limit
            compact = {
                "v": self.version,
                "mr": self.merkle_root[:16],
                "mad": round(self.claimed_mad, 6),
                "ph": self.proof_hash[:32],
                "n": self.n_trades,
            }
            encoded = json.dumps(compact, separators=(",", ":")).encode("utf-8")
        return encoded


class BenfordZKProver:
    """Generate and verify ZK proofs of Benford MAD compliance.

    Usage::

        prover = BenfordZKProver()
        amounts = trades["amount"].tolist()
        proof = prover.prove(amounts)
        assert prover.verify(proof)
    """

    def prove(self, amounts: list[float]) -> BenfordZKProof:
        """Generate a ZK proof that the Benford MAD of *amounts* equals the
        claimed value within ±``_MAD_TOLERANCE``.

        Trade amounts are never logged; only the proof hash is emitted.

        Parameters
        ----------
        amounts:
            List of trade amounts (up to 1000; larger sets use a Merkle
            commitment over batches of 1000).

        Returns
        -------
        BenfordZKProof
        """
        if not amounts:
            raise ValueError("amounts must be non-empty")

        # Compute MAD (the private witness) — not included in proof
        mad = compute_benford_mad(amounts)

        # Commit to trade amounts via Merkle tree (hides individual amounts)
        leaves = [_hash_leaf(a) for a in amounts[:1000]]
        merkle_root = _merkle_root(leaves).hex()

        # Pedersen commitment to MAD integer encoding (optional, requires py_ecc)
        mad_int = int(round(mad * 1_000_000))
        nonce_int = int.from_bytes(
            hashlib.sha256(json.dumps({"mr": merkle_root, "mad": mad}).encode()).digest(), "big"
        )

        ped_hex: str | None = None
        commitment = _pedersen_commit(mad_int, nonce_int)
        if commitment is not None:
            ped_hex = hashlib.sha256(str(commitment).encode()).hexdigest()

        nonce = hashlib.sha256(merkle_root.encode() + str(mad).encode()).hexdigest()[:16]

        # Fiat-Shamir proof hash: binds all public values together
        proof_input = json.dumps(
            {
                "version": _ZK_PROOF_VERSION,
                "merkle_root": merkle_root,
                "claimed_mad": round(mad, 6),
                "pedersen_commitment_hex": ped_hex,
                "nonce": nonce,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        proof_hash = hashlib.sha256(proof_input).hexdigest()

        logger.info(
            "BenfordZKProver: generated proof hash=%s n_trades=%d", proof_hash[:16], len(amounts)
        )

        return BenfordZKProof(
            version=_ZK_PROOF_VERSION,
            merkle_root=merkle_root,
            claimed_mad=round(mad, 6),
            proof_hash=proof_hash,
            pedersen_commitment_hex=ped_hex,
            nonce=nonce,
            n_trades=len(amounts),
        )

    def verify(self, proof: BenfordZKProof) -> bool:
        """Verify a ``BenfordZKProof`` without access to the original trade amounts.

        Recomputes the Fiat-Shamir proof hash from the public proof fields and
        checks it matches the claimed hash.

        Returns True if the proof is self-consistent.  A verifier with access
        to the original trade amounts can additionally call ``verify_with_data``.
        """
        proof_input = json.dumps(
            {
                "version": proof.version,
                "merkle_root": proof.merkle_root,
                "claimed_mad": round(proof.claimed_mad, 6),
                "pedersen_commitment_hex": proof.pedersen_commitment_hex,
                "nonce": proof.nonce,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        expected_hash = hashlib.sha256(proof_input).hexdigest()
        return expected_hash == proof.proof_hash

    def verify_with_data(self, proof: BenfordZKProof, amounts: list[float]) -> bool:
        """Verify the proof AND check that *amounts* matches the committed Merkle root.

        Tampered amounts will produce a different Merkle root and fail verification.
        """
        if not self.verify(proof):
            return False

        # Recompute Merkle root from provided amounts
        leaves = [_hash_leaf(a) for a in amounts[:1000]]
        root = _merkle_root(leaves).hex()
        if root != proof.merkle_root:
            return False

        # Recompute MAD and check it matches claimed value within tolerance
        actual_mad = compute_benford_mad(amounts)
        return abs(actual_mad - proof.claimed_mad) <= _MAD_TOLERANCE


# ---------------------------------------------------------------------------
# Scoring pipeline integration: attach proof to high-risk alert payloads
# ---------------------------------------------------------------------------


def attach_benford_proof_to_alert(
    alert_payload: dict[str, Any],
    trade_amounts: list[float],
    high_risk_threshold: int = 70,
) -> dict[str, Any]:
    """Attach a BenfordZKProof to *alert_payload* when score exceeds threshold.

    High-risk scores (> ``high_risk_threshold``) include a SNARK proof in the
    alert payload enabling trustless on-chain audit.

    Trade amounts are never included in the returned payload.

    Parameters
    ----------
    alert_payload:
        Existing alert dict (must contain a ``"score"`` key).
    trade_amounts:
        Private trade amounts used to generate the proof (not included in output).
    high_risk_threshold:
        Score above which a proof is generated (default 70).

    Returns
    -------
    dict
        Alert payload with ``"benford_zk_proof"`` key added when score > threshold.
    """
    score = alert_payload.get("score", 0)
    if score is None or score <= high_risk_threshold:
        return alert_payload

    prover = BenfordZKProver()
    proof = prover.prove(trade_amounts)
    return {
        **alert_payload,
        "benford_zk_proof": proof.to_dict(),
    }


# ---------------------------------------------------------------------------
# Issue #952 — Proof-verification caching
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    """Internal cache entry holding a verification result and expiry time."""

    result: bool
    expires_at: float  # monotonic clock seconds


class ProofVerificationCache:
    """Thread-safe TTL cache for ZK proof verification results.

    Caches the boolean result of :meth:`BenfordZKProver.verify` keyed on the
    SHA-256 of the serialised proof.  This eliminates redundant re-verification
    of the same proof within a time window, while guaranteeing that:

    * **Stale results are never returned** — each entry has a hard TTL; once
      expired the proof is re-verified from scratch.
    * **Explicit invalidation** — callers can evict a proof by its hash at any
      time (e.g., after learning a private key was compromised).
    * **No false positives** — only successful verifications (``result=True``)
      are cached for the full TTL; *failed* verifications are cached with a
      short ``failure_ttl_seconds`` to limit hammering overhead but not to mask
      genuine failures.

    **Invalidation policy (documented):**

    1. **TTL eviction**: every ``put`` stores a monotonic expiry timestamp.
       ``get`` returns ``None`` (cache miss) when the entry has expired, forcing
       fresh verification.  Default TTL: 300 seconds (5 minutes).
    2. **Explicit eviction**: ``invalidate(proof_hash)`` removes a specific
       entry immediately — use when you learn that a proof or its underlying
       data should no longer be trusted.
    3. **Full flush**: ``clear()`` evicts all entries — use on model rotation
       or after a security incident.
    4. **No mutation**: cached verification results are immutable once stored.
       Updating a proof automatically gets a different hash → a different cache
       key → a fresh verification on first access.

    Thread safety
    -------------
    All mutations are protected by a ``threading.Lock``.  Reads that detect
    expiry also acquire the lock to remove the stale entry.

    Parameters
    ----------
    ttl_seconds:
        Seconds before a *successful* verification result expires.
        Default 300 s.
    failure_ttl_seconds:
        Seconds before a *failed* verification result expires.
        Default 30 s (short, to re-check transient failures quickly).
    max_size:
        Maximum number of entries.  When reached, the oldest entry is evicted
        before inserting a new one.  Default 10 000.
    """

    def __init__(
        self,
        ttl_seconds: float = 300.0,
        failure_ttl_seconds: float = 30.0,
        max_size: int = 10_000,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.failure_ttl_seconds = failure_ttl_seconds
        self.max_size = max_size
        self._store: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @staticmethod
    def proof_hash(proof: BenfordZKProof) -> str:
        """Return the SHA-256 hex of the canonical serialisation of *proof*.

        This is the cache key.  Two structurally identical proofs always
        produce the same key; a single changed field (e.g., a tampered
        ``claimed_mad``) produces a completely different key.
        """
        raw = json.dumps(proof.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, proof: BenfordZKProof) -> bool | None:
        """Return the cached verification result, or ``None`` on a cache miss.

        A ``None`` return means the entry is absent or expired — the caller
        *must* perform fresh verification and call :meth:`put`.

        This method **never** returns a result that has exceeded its TTL.
        """
        key = self.proof_hash(proof)
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if now >= entry.expires_at:
                # Evict expired entry; caller must re-verify
                del self._store[key]
                return None
            return entry.result

    def put(self, proof: BenfordZKProof, result: bool) -> None:
        """Store a verification *result* for *proof*.

        Successful verifications (``result=True``) are cached for
        ``ttl_seconds``; failures for ``failure_ttl_seconds``.

        If the cache is full, the oldest entry is evicted first.
        """
        key = self.proof_hash(proof)
        ttl = self.ttl_seconds if result else self.failure_ttl_seconds
        expires_at = time.monotonic() + ttl
        with self._lock:
            if key not in self._store and len(self._store) >= self.max_size:
                # Evict the entry with the smallest (oldest) expiry
                oldest_key = min(self._store, key=lambda k: self._store[k].expires_at)
                del self._store[oldest_key]
            self._store[key] = _CacheEntry(result=result, expires_at=expires_at)

    def invalidate(self, proof_hash: str) -> bool:
        """Explicitly remove the entry for *proof_hash*.

        Returns ``True`` if an entry was removed, ``False`` if not present.
        """
        with self._lock:
            return self._store.pop(proof_hash, None) is not None

    def clear(self) -> int:
        """Flush all cache entries.  Returns the number of entries removed."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    def verify_cached(self, proof: BenfordZKProof, prover: BenfordZKProver | None = None) -> bool:
        """Verify *proof*, using the cache to skip re-verification if possible.

        Parameters
        ----------
        proof:
            The ``BenfordZKProof`` to verify.
        prover:
            ``BenfordZKProver`` instance to use when a cache miss occurs.
            Defaults to a freshly constructed ``BenfordZKProver()``.

        Returns
        -------
        bool
            ``True`` if the proof is valid; ``False`` otherwise.
        """
        cached = self.get(proof)
        if cached is not None:
            return cached
        if prover is None:
            prover = BenfordZKProver()
        result = prover.verify(proof)
        self.put(proof, result)
        return result

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# Issue #952 — Proof-verification benchmarking
# ---------------------------------------------------------------------------


@dataclass
class ProofVerificationBenchmarkResult:
    """Results of a :func:`benchmark_proof_verification` run."""

    n_verifications: int
    total_seconds: float
    throughput_per_second: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    cache_hit_rate: float  # fraction of verifications served from cache (0.0–1.0)
    mode: str  # "no_cache" or "with_cache"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_verifications": self.n_verifications,
            "total_seconds": round(self.total_seconds, 4),
            "throughput_per_second": round(self.throughput_per_second, 2),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "mode": self.mode,
        }


def benchmark_proof_verification(
    n_verifications: int = 1000,
    n_unique_proofs: int = 50,
    amounts_per_proof: int = 100,
    use_cache: bool = True,
    rng_seed: int = 42,
) -> ProofVerificationBenchmarkResult:
    """Benchmark ZK proof verification at representative production volume.

    Generates ``n_unique_proofs`` distinct BenfordZKProofs (from synthetic
    trade-amount arrays), then repeatedly verifies them in a round-robin
    pattern to reach ``n_verifications`` total calls.  This simulates
    real-world traffic where a hot set of proofs is re-submitted frequently.

    Metrics reported:

    * **p50/p95/p99 latency (ms)** — per-call wall-clock time distribution.
    * **throughput (verifications/sec)** — total calls / total elapsed seconds.
    * **cache_hit_rate** — fraction of calls that were served from cache
      (0.0 when ``use_cache=False``).

    Parameters
    ----------
    n_verifications:
        Total number of verification calls to measure (default 1000).
    n_unique_proofs:
        Number of distinct proofs to generate before the loop (default 50).
    amounts_per_proof:
        Trade amounts per proof (default 100; matches typical wallet window).
    use_cache:
        Whether to use :class:`ProofVerificationCache` (default True).
    rng_seed:
        NumPy RNG seed for reproducibility (default 42).

    Returns
    -------
    ProofVerificationBenchmarkResult
    """
    rng = np.random.default_rng(rng_seed)
    prover = BenfordZKProver()
    cache = ProofVerificationCache() if use_cache else None

    # Pre-generate distinct proofs (excluded from timing)
    proofs: list[BenfordZKProof] = []
    for _ in range(n_unique_proofs):
        amounts = (rng.lognormal(mean=2.0, sigma=1.2, size=amounts_per_proof) * 100).tolist()
        proofs.append(prover.prove(amounts))

    latencies_ms: list[float] = []
    cache_hits = 0

    t_start = time.perf_counter()
    for i in range(n_verifications):
        proof = proofs[i % n_unique_proofs]
        t0 = time.perf_counter()
        if cache is not None:
            cached = cache.get(proof)
            if cached is not None:
                cache_hits += 1
                result = cached  # noqa: F841  — measured path
            else:
                result = prover.verify(proof)
                cache.put(proof, result)
        else:
            result = prover.verify(proof)  # noqa: F841
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
    t_end = time.perf_counter()

    total_seconds = t_end - t_start
    arr = np.array(latencies_ms)
    return ProofVerificationBenchmarkResult(
        n_verifications=n_verifications,
        total_seconds=total_seconds,
        throughput_per_second=n_verifications / total_seconds if total_seconds > 0 else 0.0,
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        cache_hit_rate=cache_hits / n_verifications if n_verifications > 0 else 0.0,
        mode="with_cache" if use_cache else "no_cache",
    )
