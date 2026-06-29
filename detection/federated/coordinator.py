"""FastAPI coordinator for federated learning with mTLS participant authentication.

Production deployment
---------------------
Run behind uvicorn with mutual TLS:

    uvicorn detection.federated.coordinator:app \\
        --ssl-certfile certs/server.crt \\
        --ssl-keyfile  certs/server.key \\
        --ssl-ca-certs certs/ca.crt

The TLS layer (OpenSSL via uvicorn) validates the client certificate chain
against the CA before the request ever reaches this application.  This module
adds two additional application-layer checks on top:

  1. **Revocation** — the CN is looked up in a revocation list that refreshes
     from the database every ≤60 seconds.
  2. **Authorisation** — the model_id in the gradient update must be within the
     set of allowed model IDs recorded for the participant's CN.

Dependency injection
--------------------
``CertExtractor`` is injected via FastAPI's dependency-override mechanism so
tests can supply a fake cert without real TLS.
"""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel

from detection.federated.cert_authority import RevocationCache
from utils.logging import get_logger

logger = get_logger(__name__)

_DB_URL = os.getenv("RISK_SCORE_DB_URL", "sqlite:///ledgerlens.db")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class GradientUpdate(BaseModel):
    model_id: str
    round_id: int
    gradients: list[float]
    participant_id: str


class SubmitResponse(BaseModel):
    accepted: bool
    message: str


# ---------------------------------------------------------------------------
# Certificate extraction (dependency)
# ---------------------------------------------------------------------------


class CertExtractor:
    """Extracts the validated client certificate from the underlying TLS transport.

    With uvicorn + ``ssl.CERT_REQUIRED`` the client certificate is already
    verified against the CA by OpenSSL.  This dependency makes the cert
    available to route handlers as a parsed subject dict.

    In tests, override this dependency to return a synthetic cert dict.
    """

    def extract(self, request: Request) -> dict | None:
        transport = request.scope.get("transport")
        if transport is None:
            return None
        get_extra = getattr(transport, "get_extra_info", None)
        if get_extra is None:
            return None
        ssl_obj = get_extra("ssl_object")
        if ssl_obj is None:
            return None
        return ssl_obj.getpeercert()


_extractor = CertExtractor()


def get_cert_extractor() -> CertExtractor:
    return _extractor


# ---------------------------------------------------------------------------
# Shared revocation cache (singleton per process)
# ---------------------------------------------------------------------------

_revocation_cache: RevocationCache | None = None


def _get_revocation_cache() -> RevocationCache:
    global _revocation_cache
    if _revocation_cache is None:
        _revocation_cache = RevocationCache(db_url=_DB_URL)
    return _revocation_cache


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def _cn_from_cert(cert: dict) -> str | None:
    """Extract the Common Name from a parsed ``ssl.getpeercert()`` dict."""
    for rdn in cert.get("subject", ()):
        for key, value in rdn:
            if key == "commonName":
                return value
    return None


def _ou_from_cert(cert: dict) -> list[str]:
    """Extract the OU field (comma-separated model IDs) from the cert subject."""
    for rdn in cert.get("subject", ()):
        for key, value in rdn:
            if key == "organizationalUnitName":
                return [m.strip() for m in value.split(",") if m.strip()]
    return []


async def require_participant(
    request: Request,
    extractor: Annotated[CertExtractor, Depends(get_cert_extractor)],
) -> dict:
    """FastAPI dependency that authenticates and authorises the caller.

    Returns a dict with keys ``cn`` and ``allowed_models``.
    Raises 401/403 on any auth failure — error messages never expose internals.
    """
    cert = extractor.extract(request)
    if cert is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Client certificate required",
        )

    cn = _cn_from_cert(cert)
    if not cn:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Certificate is missing a Common Name",
        )

    cache = _get_revocation_cache()
    if cache.is_revoked(cn):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Certificate has been revoked",
        )

    # Allowed models come from the OU field in the cert subject
    allowed_models = _ou_from_cert(cert)
    if not allowed_models:
        # Fall back to DB lookup (for certs issued without OU encoding)
        allowed_models = cache.get_allowed_models(cn) or []

    return {"cn": cn, "allowed_models": allowed_models}


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(revocation_cache: RevocationCache | None = None) -> FastAPI:
    """Return a configured FastAPI application.

    Pass a *revocation_cache* to inject a pre-configured cache (used in tests).
    """
    global _revocation_cache
    if revocation_cache is not None:
        _revocation_cache = revocation_cache

    application = FastAPI(title="LedgerLens Federated Coordinator")

    @application.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @application.post(
        "/gradient_update",
        response_model=SubmitResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def submit_gradient_update(
        update: GradientUpdate,
        participant: Annotated[dict, Depends(require_participant)],
    ) -> SubmitResponse:
        cn = participant["cn"]
        allowed = participant["allowed_models"]

        if update.model_id not in allowed:
            logger.warning(
                "Participant %r attempted update for model %r (allowed: %s)",
                cn,
                update.model_id,
                allowed,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Participant is not authorised to submit updates for model '{update.model_id}'",
            )

        logger.info(
            "Accepted gradient update from participant=%r model=%r round=%d gradients=%d",
            cn,
            update.model_id,
            update.round_id,
            len(update.gradients),
        )
        return SubmitResponse(
            accepted=True,
            message=f"Gradient update for model '{update.model_id}' accepted (round {update.round_id})",
        )

    return application


# Module-level app instance for uvicorn: ``uvicorn detection.federated.coordinator:app``
app = create_app()
