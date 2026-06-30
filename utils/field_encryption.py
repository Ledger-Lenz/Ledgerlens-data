"""AES-256-GCM field-level encryption for sensitive forensic report fields.

Usage
-----
Encryption is transparent to callers: ``encrypt_field`` / ``decrypt_field``
operate on ``str`` values and return ``bytes`` blobs (encrypted) or ``str``
(decrypted).  Each call generates a fresh random 12-byte IV, so encrypting
the same value twice produces different ciphertexts — this prevents IV-reuse
attacks.

Key management
--------------
Set ``FORENSIC_REPORT_ENCRYPTION_KEY`` to a 32-byte (256-bit) value encoded
as 64 lowercase hex characters.  If the variable is absent or empty, a
startup warning is logged and plaintext storage is used (development only).

Example::

    # Generate a key:
    python -c "import secrets; print(secrets.token_hex(32))"

Export authorisation
--------------------
Before generating a PDF that includes decrypted wallet addresses, callers
must check that the requesting user holds the ``forensic_export`` permission
via ``check_export_permission``.
"""

from __future__ import annotations

import os
import warnings

from utils.logging import get_logger

logger = get_logger(__name__)

_KEY_ENV_VAR = "FORENSIC_REPORT_ENCRYPTION_KEY"
_IV_LENGTH = 12   # 96-bit IV recommended for AES-GCM
_TAG_LENGTH = 16  # 128-bit authentication tag


def _load_key() -> bytes | None:
    raw = os.getenv(_KEY_ENV_VAR, "").strip()
    if not raw:
        warnings.warn(
            f"{_KEY_ENV_VAR} is not set — wallet addresses will be stored in "
            "plaintext. Set this variable in production.",
            stacklevel=3,
        )
        return None
    key_bytes = bytes.fromhex(raw)
    if len(key_bytes) != 32:
        raise ValueError(
            f"{_KEY_ENV_VAR} must be exactly 32 bytes (64 hex chars); "
            f"got {len(key_bytes)} bytes"
        )
    return key_bytes


def encrypt_field(plaintext: str) -> bytes:
    """Encrypt a plaintext string with AES-256-GCM.

    Returns ``iv || ciphertext || tag`` as a raw bytes blob.
    If no encryption key is configured, returns the UTF-8 encoded plaintext
    (development fallback only).
    """
    key = _load_key()
    if key is None:
        return plaintext.encode()

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise RuntimeError(
            "The 'cryptography' package is required for field encryption. "
            "Install it with: pip install cryptography"
        ) from exc

    iv = os.urandom(_IV_LENGTH)
    aesgcm = AESGCM(key)
    ciphertext_and_tag = aesgcm.encrypt(iv, plaintext.encode(), None)
    # AESGCM.encrypt appends the 16-byte tag to the ciphertext
    return iv + ciphertext_and_tag


def decrypt_field(blob: bytes) -> str:
    """Decrypt a blob produced by ``encrypt_field``.

    Raises ``cryptography.exceptions.InvalidTag`` on authentication failure
    (wrong key, tampered ciphertext).  Never silently returns corrupt data.

    If no encryption key is configured, treats the blob as UTF-8 plaintext
    (matches the development fallback in ``encrypt_field``).
    """
    key = _load_key()
    if key is None:
        return blob.decode()

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise RuntimeError(
            "The 'cryptography' package is required for field encryption."
        ) from exc

    if len(blob) < _IV_LENGTH + _TAG_LENGTH:
        raise ValueError("Encrypted blob is too short to be valid")

    iv = blob[:_IV_LENGTH]
    ciphertext_and_tag = blob[_IV_LENGTH:]
    aesgcm = AESGCM(key)
    # Raises InvalidTag automatically if authentication fails
    plaintext_bytes = aesgcm.decrypt(iv, ciphertext_and_tag, None)
    return plaintext_bytes.decode()


# ---------------------------------------------------------------------------
# Export authorisation
# ---------------------------------------------------------------------------

_EXPORT_PERMISSION = "forensic_export"


def check_export_permission(user_permissions: set[str]) -> None:
    """Raise ``PermissionError`` if the user lacks the ``forensic_export`` permission.

    Call this before generating any PDF or plaintext export that includes
    decrypted wallet addresses.
    """
    if _EXPORT_PERMISSION not in user_permissions:
        raise PermissionError(
            f"User does not have the '{_EXPORT_PERMISSION}' permission required "
            "to export forensic reports containing decrypted wallet addresses."
        )
