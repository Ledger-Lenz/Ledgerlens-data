"""Tests for AES-256-GCM field-level encryption (issue #239)."""

import os
import secrets

import pytest


_VALID_KEY_HEX = secrets.token_hex(32)  # 32 bytes = 64 hex chars


def _set_key(monkeypatch, hex_key: str) -> None:
    monkeypatch.setenv("FORENSIC_REPORT_ENCRYPTION_KEY", hex_key)


def test_encrypt_decrypt_roundtrip(monkeypatch):
    _set_key(monkeypatch, _VALID_KEY_HEX)
    from utils.field_encryption import decrypt_field, encrypt_field

    wallet = "GBCFXNZQN2P7YBZFPKG4TMZQNHEFGQJZRSVSXFSEXAMPLEWALLET12345"
    blob = encrypt_field(wallet)
    assert isinstance(blob, bytes)
    assert decrypt_field(blob) == wallet


def test_iv_uniqueness(monkeypatch):
    """Two encryptions of the same plaintext must produce different ciphertexts."""
    _set_key(monkeypatch, _VALID_KEY_HEX)
    from utils.field_encryption import encrypt_field

    wallet = "GBCFXNZQN2P7YBZFPKG4TMZQNHEFGQJZRSVSXFSEXAMPLEWALLET12345"
    blob1 = encrypt_field(wallet)
    blob2 = encrypt_field(wallet)
    assert blob1 != blob2


def test_wrong_key_raises_invalid_tag(monkeypatch):
    """Decrypting with the wrong key must raise InvalidTag, not silently corrupt."""
    _set_key(monkeypatch, _VALID_KEY_HEX)
    from utils.field_encryption import encrypt_field

    blob = encrypt_field("G" + "A" * 55)

    wrong_key_hex = secrets.token_hex(32)
    while wrong_key_hex == _VALID_KEY_HEX:
        wrong_key_hex = secrets.token_hex(32)

    monkeypatch.setenv("FORENSIC_REPORT_ENCRYPTION_KEY", wrong_key_hex)

    # Re-import to reload the env var
    import importlib
    import utils.field_encryption as fe_module
    importlib.reload(fe_module)

    from cryptography.exceptions import InvalidTag

    with pytest.raises(InvalidTag):
        fe_module.decrypt_field(blob)


def test_short_key_raises_value_error(monkeypatch):
    monkeypatch.setenv("FORENSIC_REPORT_ENCRYPTION_KEY", "deadbeef")  # 4 bytes only
    import importlib
    import utils.field_encryption as fe_module
    importlib.reload(fe_module)

    with pytest.raises(ValueError, match="32 bytes"):
        fe_module.encrypt_field("some wallet")


def test_no_key_warns_and_stores_plaintext(monkeypatch):
    monkeypatch.delenv("FORENSIC_REPORT_ENCRYPTION_KEY", raising=False)
    import importlib
    import utils.field_encryption as fe_module
    importlib.reload(fe_module)

    wallet = "GBCFXNZQN2P7YBZFPKG4TMZQNHEFGQJZRSVSXFSEXAMPLEWALLET12345"
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        blob = fe_module.encrypt_field(wallet)
        assert len(w) >= 1

    assert fe_module.decrypt_field(blob) == wallet


def test_check_export_permission_passes():
    from utils.field_encryption import check_export_permission

    check_export_permission({"forensic_export", "read"})  # should not raise


def test_check_export_permission_raises_without_permission():
    from utils.field_encryption import check_export_permission

    with pytest.raises(PermissionError, match="forensic_export"):
        check_export_permission({"read", "write"})
