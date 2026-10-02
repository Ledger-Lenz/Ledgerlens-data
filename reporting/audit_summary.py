"""Audit-ready summaries for anomaly investigation outputs — Issue #943.

Produces structured, tamper-evident summary documents from anomaly
investigation results that are suitable for compliance review, regulatory
submission, and internal audit trails.

Each ``AuditSummary`` includes:
- A unique summary ID and generation timestamp.
- Investigation metadata (wallet, asset pair, verdict, risk score).
- Key evidence items extracted from the forensic report.
- A SHA-256 integrity hash over all fields (excluding the hash itself).
- An optional chain-of-custody record linking to prior summaries.

Issue #943 additions — cryptographic signing
--------------------------------------------
Exported audit summaries can now be **Ed25519-signed** so that any downstream
recipient can verify the report has not been altered since generation.

Signing::

    from reporting.audit_summary import AuditSummaryBuilder, sign_summary, load_signing_key

    private_key = load_signing_key()           # reads AUDIT_SIGNING_KEY_PATH env var
    summary = AuditSummaryBuilder().build(report)
    signed = sign_summary(summary, private_key)
    # signed["ed25519_signature"] is a hex-encoded detached signature

Verification (recipient side)::

    from reporting.audit_summary import verify_summary_signature, load_verify_key

    verify_key = load_verify_key()             # reads AUDIT_VERIFY_KEY_PATH env var
    ok = verify_summary_signature(signed_doc, verify_key)

CLI wrapper::

    python -m reporting.audit_summary verify report.json

Key management and rotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Keys are Ed25519 key pairs (32-byte seed / 32-byte public key).
- The signing key path defaults to ``AUDIT_SIGNING_KEY_PATH`` (env var) and
  must be protected with mode ``0o600``.
- The verification key path defaults to ``AUDIT_VERIFY_KEY_PATH`` (env var).
- Generate a new key pair with :func:`generate_signing_key_pair`.
- Key rotation: generate a new pair, update ``AUDIT_SIGNING_KEY_PATH`` and
  ``AUDIT_VERIFY_KEY_PATH``.  Previously issued summaries signed with the old
  key remain verifiable if the old verification key is retained — rotate the
  verification key only once all old summaries have been re-signed or archived.
- Store private keys in a secrets manager (Vault, AWS Secrets Manager, etc.);
  never commit them to VCS.

Security invariants
-------------------
- ``summary_sha256`` is computed over all other fields in ``__post_init__``.
- ``verify_integrity()`` recomputes and compares the SHA-256 hash.
- ``ed25519_signature`` covers the canonical JSON of the full summary dict
  (including ``summary_sha256``) so both integrity and provenance are verified
  in one step.
- All timestamps are UTC ISO-8601.
- No user-supplied URLs are included; Horizon links are constructed from
  ``config.HORIZON_URL`` only.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Cryptographic helpers (Ed25519 via cryptography library)
# ---------------------------------------------------------------------------

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
        load_pem_private_key,
        load_pem_public_key,
    )

    _CRYPTO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CRYPTO_AVAILABLE = False


class SigningError(Exception):
    """Raised when signing or verification fails."""


class SignatureVerificationError(SigningError):
    """Raised when a signature does not match the document."""


def _require_crypto() -> None:
    if not _CRYPTO_AVAILABLE:
        raise SigningError(
            "The 'cryptography' package is required for Ed25519 signing. "
            "Install it with: pip install cryptography"
        )


def generate_signing_key_pair(
    private_key_path: str | Path,
    public_key_path: str | Path,
) -> None:
    """Generate a new Ed25519 key pair and write PEM files.

    The private key is written with mode ``0o600`` (owner-read-write only).

    Parameters
    ----------
    private_key_path:
        Destination path for the PEM-encoded private key.
    public_key_path:
        Destination path for the PEM-encoded public key.
    """
    _require_crypto()
    priv_path = Path(private_key_path)
    pub_path = Path(public_key_path)

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    priv_pem = private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    pub_pem = public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

    priv_path.parent.mkdir(parents=True, exist_ok=True)
    priv_path.write_bytes(priv_pem)
    priv_path.chmod(0o600)

    pub_path.parent.mkdir(parents=True, exist_ok=True)
    pub_path.write_bytes(pub_pem)

    logger.info("Generated Ed25519 key pair: private=%s public=%s", priv_path, pub_path)


def load_signing_key(path: str | None = None) -> "Ed25519PrivateKey":
    """Load an Ed25519 private key from a PEM file.

    The path defaults to the ``AUDIT_SIGNING_KEY_PATH`` environment variable.
    """
    _require_crypto()
    import os

    key_path = path or os.getenv("AUDIT_SIGNING_KEY_PATH")
    if not key_path:
        raise SigningError(
            "No signing key path provided. Set AUDIT_SIGNING_KEY_PATH "
            "or pass the path explicitly."
        )
    pem_data = Path(key_path).read_bytes()
    return load_pem_private_key(pem_data, password=None)  # type: ignore[return-value]


def load_verify_key(path: str | None = None) -> "Ed25519PublicKey":
    """Load an Ed25519 public key from a PEM file.

    The path defaults to the ``AUDIT_VERIFY_KEY_PATH`` environment variable.
    """
    _require_crypto()
    import os

    key_path = path or os.getenv("AUDIT_VERIFY_KEY_PATH")
    if not key_path:
        raise SigningError(
            "No verification key path provided. Set AUDIT_VERIFY_KEY_PATH "
            "or pass the path explicitly."
        )
    pem_data = Path(key_path).read_bytes()
    return load_pem_public_key(pem_data)  # type: ignore[return-value]


def _canonical_bytes(doc: dict[str, Any]) -> bytes:
    """Return a deterministic UTF-8 encoding of a dict for signing/verification."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def sign_summary(summary: "AuditSummary", private_key: "Ed25519PrivateKey") -> dict[str, Any]:
    """Sign an :class:`AuditSummary` and return a dict with an ``ed25519_signature`` field.

    The signature covers the canonical JSON of the full summary document
    (including ``summary_sha256``) so both integrity and provenance are
    verified together.

    Parameters
    ----------
    summary:
        The audit summary to sign.
    private_key:
        An Ed25519 private key (e.g. from :func:`load_signing_key`).

    Returns
    -------
    dict
        The full summary dict (as from :meth:`AuditSummary.to_dict`) with an
        additional ``ed25519_signature`` field containing the hex-encoded
        detached signature.
    """
    _require_crypto()
    doc = summary.to_dict()
    payload = _canonical_bytes(doc)
    signature_bytes = private_key.sign(payload)
    doc["ed25519_signature"] = signature_bytes.hex()
    return doc


def verify_summary_signature(
    signed_doc: dict[str, Any],
    public_key: "Ed25519PublicKey",
) -> bool:
    """Verify the Ed25519 signature on a signed audit summary document.

    Parameters
    ----------
    signed_doc:
        A dict produced by :func:`sign_summary` — must contain
        ``ed25519_signature``.
    public_key:
        An Ed25519 public key (e.g. from :func:`load_verify_key`).

    Returns
    -------
    bool
        ``True`` when the signature is valid and the document is untampered.

    Raises
    ------
    SignatureVerificationError
        When the signature is missing, malformed, or invalid.
    """
    _require_crypto()
    from cryptography.exceptions import InvalidSignature

    sig_hex = signed_doc.get("ed25519_signature")
    if not sig_hex:
        raise SignatureVerificationError(
            "Document does not contain an 'ed25519_signature' field."
        )

    # Reconstruct the payload that was signed: full doc minus the signature field
    doc_without_sig = {k: v for k, v in signed_doc.items() if k != "ed25519_signature"}
    payload = _canonical_bytes(doc_without_sig)

    try:
        signature_bytes = bytes.fromhex(sig_hex)
    except ValueError as exc:
        raise SignatureVerificationError(
            f"Invalid signature encoding: {exc}"
        ) from exc

    try:
        public_key.verify(signature_bytes, payload)
        return True
    except InvalidSignature as exc:
        raise SignatureVerificationError(
            "Signature verification failed — the document may have been tampered with."
        ) from exc


# ---------------------------------------------------------------------------
# Verdict classification
# ---------------------------------------------------------------------------

_SEVERITY_MAP: dict[str, str] = {
    "clean": "low",
    "suspicious": "medium",
    "wash_trade": "high",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EvidenceItem:
    """A single piece of evidence supporting the investigation conclusion."""

    category: str  # e.g. "benford_violation", "shap_feature", "trade_anomaly"
    description: str
    value: str
    source_reference: str = ""  # e.g. Horizon URL or feature name


@dataclass
class AuditSummary:
    """Tamper-evident audit summary of an anomaly investigation."""

    summary_id: str
    generated_at: str
    report_id: str
    wallet: str
    asset_pair: str
    risk_score: int
    score_lower: int
    score_upper: int
    verdict: str
    severity: str
    evidence_items: list[EvidenceItem]
    investigation_notes: str
    model_version: str
    prior_summary_id: str | None = None
    summary_sha256: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.summary_sha256 = self._compute_sha256()

    def _to_dict_without_hash(self) -> dict[str, Any]:
        return {
            "summary_id": self.summary_id,
            "generated_at": self.generated_at,
            "report_id": self.report_id,
            "wallet": self.wallet,
            "asset_pair": self.asset_pair,
            "risk_score": self.risk_score,
            "score_lower": self.score_lower,
            "score_upper": self.score_upper,
            "verdict": self.verdict,
            "severity": self.severity,
            "evidence_items": [
                {
                    "category": e.category,
                    "description": e.description,
                    "value": e.value,
                    "source_reference": e.source_reference,
                }
                for e in self.evidence_items
            ],
            "investigation_notes": self.investigation_notes,
            "model_version": self.model_version,
            "prior_summary_id": self.prior_summary_id,
        }

    def _compute_sha256(self) -> str:
        payload = json.dumps(self._to_dict_without_hash(), sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()

    def verify_integrity(self) -> bool:
        """Recompute the SHA-256 hash and verify it matches the stored value."""
        return self._compute_sha256() == self.summary_sha256

    def to_dict(self) -> dict[str, Any]:
        """Serialise the summary to a dictionary."""
        d = self._to_dict_without_hash()
        d["summary_sha256"] = self.summary_sha256
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialise the summary to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------


def _extract_benford_evidence(report: dict[str, Any]) -> list[EvidenceItem]:
    """Extract Benford analysis violations as evidence items."""
    items: list[EvidenceItem] = []
    benford = report.get("benford_analysis")
    if not benford or not isinstance(benford, dict):
        return items

    for window, metrics in benford.items():
        if not isinstance(metrics, dict):
            continue
        if metrics.get("mad_nonconforming"):
            items.append(
                EvidenceItem(
                    category="benford_violation",
                    description=(
                        f"Benford's Law MAD non-conforming in {window}h window "
                        f"(chi2={metrics.get('chi_square', 'N/A')}, "
                        f"MAD={metrics.get('mad', 'N/A')})"
                    ),
                    value=f"chi2={metrics.get('chi_square')}, mad={metrics.get('mad')}",
                    source_reference=f"benford_analysis.{window}h",
                )
            )

    return items


def _extract_shap_evidence(report: dict[str, Any], top_n: int = 5) -> list[EvidenceItem]:
    """Extract top SHAP feature contributions as evidence items."""
    items: list[EvidenceItem] = []
    shap_features = report.get("top_shap_features") or []

    sorted_features = sorted(
        shap_features,
        key=lambda f: abs(f.get("contribution", 0) or 0),
        reverse=True,
    )

    for feat in sorted_features[:top_n]:
        name = feat.get("feature", "unknown")
        contribution = feat.get("contribution", 0)
        value = feat.get("value", "N/A")
        description = feat.get("description", name)
        items.append(
            EvidenceItem(
                category="shap_feature",
                description=f"{description} (contribution={contribution})",
                value=str(value),
                source_reference=name,
            )
        )

    return items


def _extract_trade_evidence(report: dict[str, Any]) -> list[EvidenceItem]:
    """Extract anomalous trade evidence items."""
    items: list[EvidenceItem] = []
    trades = report.get("trade_evidence") or []

    for trade in trades[:10]:  # Cap at 10 trades for summary
        trade_id = trade.get("trade_id", "unknown")
        horizon_base = config.HORIZON_URL.rstrip("/")
        items.append(
            EvidenceItem(
                category="trade_anomaly",
                description=(
                    f"Anomalous trade {trade_id}: "
                    f"base_amount={trade.get('base_amount', 'N/A')}, "
                    f"counter_amount={trade.get('counter_amount', 'N/A')}"
                ),
                value=trade_id,
                source_reference=f"{horizon_base}/trades/{trade_id}",
            )
        )

    return items


def _build_investigation_notes(report: dict[str, Any]) -> str:
    """Build a concise investigation summary from the report data."""
    wallet = report.get("wallet", "unknown")
    verdict = report.get("verdict", "unknown")
    score = report.get("risk_score", 0)
    asset_pair = report.get("asset_pair", "unknown")

    parts = [
        f"Wallet {wallet} scored {score}/100 for asset pair {asset_pair}.",
        f"Verdict: {verdict}.",
    ]

    # Add causal attribution note if present
    causal = report.get("causal_attribution")
    if causal and isinstance(causal, dict):
        root_cause = causal.get("root_cause_wallet")
        if root_cause:
            parts.append(f"Root cause traced to wallet {root_cause}.")
        cf_score = causal.get("counterfactual_score")
        if cf_score is not None:
            parts.append(f"Counterfactual score (without flagged trades): {cf_score}.")

    # Add propagation note if present
    propagation = report.get("propagation_path")
    if propagation and isinstance(propagation, dict):
        prop_risk = propagation.get("propagated_risk")
        if prop_risk is not None:
            parts.append(f"Propagated risk from network: {prop_risk}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class AuditSummaryBuilder:
    """Builds audit-ready summaries from forensic report dictionaries.

    Parameters
    ----------
    top_shap_features : int
        Number of top SHAP features to include in evidence (default 5).
    """

    def __init__(self, *, top_shap_features: int = 5) -> None:
        self._top_shap = top_shap_features

    def build(
        self,
        report: dict[str, Any],
        *,
        prior_summary_id: str | None = None,
    ) -> AuditSummary:
        """Build an audit summary from a forensic report dict.

        Parameters
        ----------
        report : dict
            A dict produced by ``ForensicReport.to_dict()``.
        prior_summary_id : str | None
            If this is a follow-up investigation, link to the prior summary.

        Returns
        -------
        AuditSummary
            A tamper-evident, serialisable audit summary.
        """
        verdict = report.get("verdict", "unknown")
        severity = _SEVERITY_MAP.get(verdict, "unknown")

        evidence: list[EvidenceItem] = []
        evidence.extend(_extract_benford_evidence(report))
        evidence.extend(_extract_shap_evidence(report, top_n=self._top_shap))
        evidence.extend(_extract_trade_evidence(report))

        model_meta = report.get("model_metadata") or {}
        model_version = model_meta.get("version", "unknown")

        return AuditSummary(
            summary_id=uuid.uuid4().hex,
            generated_at=datetime.now(UTC).isoformat(),
            report_id=report.get("report_id", ""),
            wallet=report.get("wallet", ""),
            asset_pair=report.get("asset_pair", ""),
            risk_score=int(report.get("risk_score", 0)),
            score_lower=int(report.get("score_lower", 0)),
            score_upper=int(report.get("score_upper", 0)),
            verdict=verdict,
            severity=severity,
            evidence_items=evidence,
            investigation_notes=_build_investigation_notes(report),
            model_version=model_version,
            prior_summary_id=prior_summary_id,
        )

    def build_batch(
        self,
        reports: list[dict[str, Any]],
    ) -> list[AuditSummary]:
        """Build audit summaries for a batch of forensic reports.

        Parameters
        ----------
        reports : list[dict]
            List of forensic report dicts.

        Returns
        -------
        list[AuditSummary]
            One summary per report.
        """
        return [self.build(r) for r in reports]
