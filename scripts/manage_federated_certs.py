"""CLI for managing federated learning participant certificates.

Commands
--------
    python -m scripts.manage_federated_certs init-ca
        Generate CA key + cert and write to FEDERATED_CA_CERT_PATH.
        The CA private key is printed to stdout ONCE — store it in your HSM.

    python -m scripts.manage_federated_certs issue --cn PARTICIPANT --models MODEL,...
        Issue a new certificate for PARTICIPANT.

    python -m scripts.manage_federated_certs revoke --cn PARTICIPANT
        Revoke PARTICIPANT's certificate (takes effect ≤60 s on coordinator).

    python -m scripts.manage_federated_certs rotate --cn PARTICIPANT --models MODEL,...
        Revoke the old cert and issue a fresh one.

    python -m scripts.manage_federated_certs list
        List all participants and their certificate status.

    python -m scripts.manage_federated_certs check-expiry [--days N]
        List participants whose cert expires within N days (default 30).

Environment variables
---------------------
    FEDERATED_CA_KEY_PEM      — PEM-encoded CA private key (required for issue/revoke/rotate)
    FEDERATED_CA_CERT_PATH    — Path to CA certificate PEM file
    RISK_SCORE_DB_URL         — SQLAlchemy DB URL (default sqlite:///ledgerlens.db)
    PARTICIPANT_CERT_OUT_DIR  — Directory to write issued participant cert PEMs
                                (default ./certs/participants)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from detection.federated.cert_authority import (
    generate_ca_keypair,
    issue_certificate,
    list_all_participants,
    list_expiring_soon,
    revoke_certificate,
    rotate_certificate,
)

_DB_URL = os.getenv("RISK_SCORE_DB_URL", "sqlite:///ledgerlens.db")
_CA_CERT_PATH = os.getenv("FEDERATED_CA_CERT_PATH", "certs/ca.crt")
_OUT_DIR = Path(os.getenv("PARTICIPANT_CERT_OUT_DIR", "certs/participants"))


def _load_ca() -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    """Load CA key from env and cert from file.  Exits on error."""
    key_pem = os.getenv("FEDERATED_CA_KEY_PEM")
    if not key_pem:
        print("ERROR: FEDERATED_CA_KEY_PEM environment variable not set", file=sys.stderr)
        print(
            "       The CA private key must be stored in your HSM/secrets manager and"
            " injected at runtime.",
            file=sys.stderr,
        )
        sys.exit(1)

    ca_key = serialization.load_pem_private_key(key_pem.encode(), password=None)

    cert_path = Path(_CA_CERT_PATH)
    if not cert_path.exists():
        print(f"ERROR: CA certificate not found at {cert_path}", file=sys.stderr)
        sys.exit(1)

    ca_cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    return ca_key, ca_cert  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def cmd_init_ca(args: argparse.Namespace) -> None:
    ca_key, ca_cert = generate_ca_keypair()

    key_pem = ca_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ).decode()

    cert_pem = ca_cert.public_bytes(serialization.Encoding.PEM).decode()

    cert_path = Path(_CA_CERT_PATH)
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    cert_path.write_text(cert_pem)

    print("CA certificate written to:", cert_path)
    print()
    print("=" * 72)
    print("CA PRIVATE KEY — store this in your HSM; it will NOT be saved to disk:")
    print("=" * 72)
    print(key_pem)
    print(
        "WARNING: Set FEDERATED_CA_KEY_PEM to the above key before running any"
        " issue/revoke/rotate commands."
    )


def cmd_issue(args: argparse.Namespace) -> None:
    ca_key, ca_cert = _load_ca()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("ERROR: --models must specify at least one model ID", file=sys.stderr)
        sys.exit(1)

    part_key, part_cert = issue_certificate(
        cn=args.cn,
        allowed_models=models,
        ca_key=ca_key,
        ca_cert=ca_cert,
        validity_days=args.validity_days,
        db_url=_DB_URL,
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    cert_file = _OUT_DIR / f"{args.cn}.crt"
    key_file = _OUT_DIR / f"{args.cn}.key"

    cert_file.write_bytes(part_cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        part_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    key_file.chmod(0o600)

    print(f"Certificate : {cert_file}")
    print(f"Private key : {key_file}  (permissions: 600)")
    print(
        "IMPORTANT: Transmit the private key to the participant over a secure"
        " channel, then delete the local copy."
    )


def cmd_revoke(args: argparse.Namespace) -> None:
    try:
        revoke_certificate(cn=args.cn, db_url=_DB_URL)
        print(f"Certificate for CN={args.cn!r} revoked.")
        print("The coordinator will reload the revocation list within 60 seconds.")
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_rotate(args: argparse.Namespace) -> None:
    ca_key, ca_cert = _load_ca()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("ERROR: --models must specify at least one model ID", file=sys.stderr)
        sys.exit(1)

    part_key, part_cert = rotate_certificate(
        cn=args.cn,
        allowed_models=models,
        ca_key=ca_key,
        ca_cert=ca_cert,
        validity_days=args.validity_days,
        db_url=_DB_URL,
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    cert_file = _OUT_DIR / f"{args.cn}.crt"
    key_file = _OUT_DIR / f"{args.cn}.key"
    cert_file.write_bytes(part_cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        part_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    key_file.chmod(0o600)
    print(f"Rotated certificate for CN={args.cn!r} → {cert_file}")


def cmd_list(args: argparse.Namespace) -> None:
    records = list_all_participants(db_url=_DB_URL)
    if not records:
        print("No participants registered.")
        return
    fmt = "{:<24} {:<8} {:<30} {:<26} {}"
    print(fmt.format("CN", "STATUS", "MODELS", "EXPIRES", "ISSUED"))
    print("-" * 100)
    for r in records:
        status = "REVOKED" if r.revoked else "active"
        print(fmt.format(r.cn, status, r.allowed_models[:28], str(r.expires_at)[:25], str(r.issued_at)[:25]))


def cmd_check_expiry(args: argparse.Namespace) -> None:
    records = list_expiring_soon(within_days=args.days, db_url=_DB_URL)
    if not records:
        print(f"No certificates expiring within {args.days} days.")
        return
    print(f"WARNING: {len(records)} certificate(s) expiring within {args.days} days:")
    for r in records:
        print(f"  CN={r.cn!r}  expires={r.expires_at.date()}  models={r.allowed_models}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.manage_federated_certs",
        description="Manage LedgerLens federated learning participant certificates",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-ca", help="Generate a new CA key and self-signed certificate")

    p_issue = sub.add_parser("issue", help="Issue a new participant certificate")
    p_issue.add_argument("--cn", required=True, help="Participant common name")
    p_issue.add_argument("--models", required=True, help="Comma-separated allowed model IDs")
    p_issue.add_argument("--validity-days", type=int, default=365)

    p_revoke = sub.add_parser("revoke", help="Revoke a participant certificate")
    p_revoke.add_argument("--cn", required=True, help="Participant common name")

    p_rotate = sub.add_parser("rotate", help="Revoke and re-issue a participant certificate")
    p_rotate.add_argument("--cn", required=True)
    p_rotate.add_argument("--models", required=True)
    p_rotate.add_argument("--validity-days", type=int, default=365)

    sub.add_parser("list", help="List all participants and certificate status")

    p_expiry = sub.add_parser("check-expiry", help="List certificates expiring soon")
    p_expiry.add_argument("--days", type=int, default=30)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "init-ca": cmd_init_ca,
        "issue": cmd_issue,
        "revoke": cmd_revoke,
        "rotate": cmd_rotate,
        "list": cmd_list,
        "check-expiry": cmd_check_expiry,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
