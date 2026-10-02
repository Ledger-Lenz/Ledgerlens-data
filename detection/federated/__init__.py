"""Federated learning utilities for privacy-preserving detection.

This package exposes the secure aggregation primitives used by the
federated detection coordinator as well as the certificate authority
that manages participant credentials.
"""

from .cert_authority import (
    CertificateAuthority,
    CertificateError,
    CertificateRevokedError,
    CertificateExpiredError,
    ParticipantCertificate,
)
from .secure_aggregation import SecureAggregator

__all__ = [
    "CertificateAuthority",
    "CertificateError",
    "CertificateRevokedError",
    "CertificateExpiredError",
    "ParticipantCertificate",
    "SecureAggregator",
]
