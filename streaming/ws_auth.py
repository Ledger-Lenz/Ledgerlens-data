"""JWT authentication for WebSocket connections.

Validates bearer tokens using RS256 (asymmetric). Public key is loaded from
config.JWT_PUBLIC_KEY_PATH. Validates claims: exp (expiry), iss (issuer),
sub (subject/client_id), and scope (must contain "scores:read").
"""

import os
<<<<<<< HEAD
from datetime import datetime, timezone
=======
>>>>>>> 14aa8ce (fix: resolve ruff linting errors in ws_auth and ws_server)
from typing import Any

from jose import JWTError, jwt
from jose.exceptions import JWTClaimsError

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)


class JWTAuthenticator:
    """Validates JWT bearer tokens for WebSocket connections."""

    def __init__(self, public_key_path: str | None = None):
        self._public_key_path = public_key_path or config.JWT_PUBLIC_KEY_PATH
        self._public_key: str | None = None
        self._load_public_key()

    def _load_public_key(self) -> None:
        """Load the public key from file. Raises FileNotFoundError if missing."""
        if not os.path.exists(self._public_key_path):
            raise FileNotFoundError(
                f"JWT public key not found at {self._public_key_path}. "
                "Generate one or set JWT_PUBLIC_KEY_PATH in .env"
            )
<<<<<<< HEAD
        with open(self._public_key_path, "r") as f:
=======
        with open(self._public_key_path) as f:
>>>>>>> 14aa8ce (fix: resolve ruff linting errors in ws_auth and ws_server)
            self._public_key = f.read()

    def verify(self, token: str) -> dict[str, Any] | None:
        """Verify JWT token and return claims dict, or None if invalid.

        Validates:
        - Signature using RS256 and the public key
        - exp (expiry time must be in the future)
        - iss (issuer must be "ledgerlens-api")
        - sub (client ID must be present)
        - scope (must contain "scores:read")

        Args:
            token: JWT bearer token (without "Bearer " prefix)

        Returns:
            Claims dict with keys: exp, iss, sub, scope, etc.
            None if token is invalid or claims check fails.
        """
        if not token or not isinstance(token, str):
            logger.warning("JWT verification failed: token is empty or not a string")
            return None

        try:
            # Decode and verify signature, exp, and standard claims
            claims = jwt.decode(
                token,
                self._public_key,
                algorithms=["RS256"],
                issuer="ledgerlens-api",
                options={"verify_exp": True},
            )

            # Check for required 'sub' (subject/client_id)
            if "sub" not in claims:
                logger.warning("JWT verification failed: missing 'sub' claim")
                return None

            # Check for 'scope' claim and that it includes "scores:read"
            scope = claims.get("scope", "")
            if not self._has_scores_read_scope(scope):
                logger.warning(
                    "JWT verification failed: missing 'scores:read' in scope (client_id=%s)",
                    claims.get("sub"),
                )
                return None

            logger.debug("JWT verified successfully (client_id=%s)", claims.get("sub"))
            return claims

        except JWTClaimsError as exc:
            logger.warning("JWT verification failed: invalid claims (%s)", str(exc))
            return None
        except JWTError as exc:
            logger.warning("JWT verification failed: invalid signature or format (%s)", str(exc))
            return None
        except Exception as exc:
            logger.error("Unexpected error during JWT verification: %s", str(exc))
            return None

    def extract_permissions(self, claims: dict[str, Any]) -> set[str]:
        """Extract allowed channel prefixes from scope claim.

        Scope format: "scores:read" (all channels) or
        "scores:read:wallet/GXXX" (specific wallet) or
        "scores:read:pair/XLM:native/USDC:..." (specific pair).

        Args:
            claims: Claims dict from verify()

        Returns:
            Set of allowed channel prefixes. If scope is "scores:read", returns
            {"scores:read:all"}. If scope is "scores:read:wallet/GXXX",
            returns {"scores:read:wallet/GXXX"}.
        """
        scope = claims.get("scope", "")

        if not scope:
            return set()

        # Split by space in case multiple scopes are allowed
        scopes = scope.split()
        permissions = set()

        for s in scopes:
            if s.startswith("scores:read"):
                permissions.add(s)

        return permissions

    @staticmethod
    def _has_scores_read_scope(scope: str) -> bool:
        """Check if scope includes any form of scores:read permission."""
        if not scope:
            return False
        # Scope can be "scores:read", "scores:read:wallet/...", etc.
        scopes = scope.split()
        return any(s.startswith("scores:read") for s in scopes)

    @staticmethod
    def is_permitted_channel(permissions: set[str], channel: str) -> bool:
        """Check if permissions allow subscription to channel.

        Rules:
        - If "scores:read:all" in permissions, all channels allowed
        - If "scores:read:wallet/GXXX" in permissions, only wallet/GXXX allowed
        - If "scores:read:pair/..." in permissions, only pair/... allowed
        - Otherwise, not permitted

        Args:
            permissions: Set from extract_permissions()
            channel: Requested channel (e.g., "wallet/GXXX" or "pair/...")

        Returns:
            True if channel is permitted, False otherwise.
        """
        if not permissions:
            return False

        # Admin channel "all" requires "scores:read:all"
        if channel == "all":
            return "scores:read:all" in permissions

        # Check for exact match or wildcard
        for perm in permissions:
            if perm == "scores:read:all":
                # Admin can subscribe to anything
                return True
            # For wallet/pair restrictions, check if permission prefix matches channel
            if perm.startswith("scores:read:"):
                allowed_channel = perm[len("scores:read:") :]
                if allowed_channel == channel:
                    return True

        return False
