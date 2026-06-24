import pytest

from streaming.ws_auth import JWTAuthenticator


@pytest.fixture
def auth(tmp_path):
    """Fixture providing a JWTAuthenticator with a fake public key file."""
    key_file = tmp_path / "fake_key.pub"
    key_file.write_text("fake_public_key_content")
    return JWTAuthenticator(public_key_path=str(key_file))


# ---------------------------------------------------------------------------
# extract_permissions()
# ---------------------------------------------------------------------------


def test_extract_permissions_unrestricted_scope(auth):
    """scope = 'scores:read' returns unrestricted set ({scores:read:all})"""
    claims = {"scope": "scores:read"}
    perms = auth.extract_permissions(claims)
    assert perms == {"scores:read:all"}


def test_extract_permissions_wallet_specific(auth):
    """scope = 'scores:read:wallet/GABC...' returns wallet-specific prefix"""
    claims = {"scope": "scores:read:wallet/GABC123"}
    perms = auth.extract_permissions(claims)
    assert perms == {"scores:read:wallet/GABC123"}


def test_extract_permissions_admin_access(auth):
    """scope = 'scores:read:all' returns admin access"""
    claims = {"scope": "scores:read:all"}
    perms = auth.extract_permissions(claims)
    assert perms == {"scores:read:all"}


def test_extract_permissions_multiple_wallets(auth):
    """Multiple wallet scopes returns all prefixes"""
    claims = {"scope": "scores:read:wallet/G123 scores:read:wallet/G456"}
    perms = auth.extract_permissions(claims)
    assert perms == {"scores:read:wallet/G123", "scores:read:wallet/G456"}


def test_extract_permissions_missing_scope(auth):
    """Missing scope returns empty set"""
    perms1 = auth.extract_permissions({})
    assert perms1 == set()

    perms2 = auth.extract_permissions({"scope": ""})
    assert perms2 == set()


def test_extract_permissions_invalid_entries_excluded(auth):
    """Invalid scope entries are excluded, valid ones retained"""
    claims = {"scope": "scores:read:wallet/G123 invalid:scope unrelated"}
    perms = auth.extract_permissions(claims)
    assert perms == {"scores:read:wallet/G123"}


def test_is_permitted_channel_wallet_isolation(auth):
    """Wallet-scoped client cannot subscribe to a different wallet channel"""
    claims = {"scope": "scores:read:wallet/G123"}
    perms = auth.extract_permissions(claims)
    assert auth.is_permitted_channel(perms, "wallet/G123") is True
    assert auth.is_permitted_channel(perms, "wallet/G456") is False


# ---------------------------------------------------------------------------
# verify()
# ---------------------------------------------------------------------------


def test_verify_returns_none_for_empty_string(auth):
    """verify() returns None when token is an empty string"""
    assert auth.verify("") is None


def test_verify_returns_none_for_none(auth):
    """verify() returns None when token is None"""
    assert auth.verify(None) is None  # type: ignore[arg-type]


def test_verify_returns_none_for_invalid_jwt(auth):
    """verify() returns None when token is a malformed/unsigned JWT"""
    assert auth.verify("not.a.real.jwt") is None


def test_verify_returns_none_for_bad_segments(auth):
    """verify() returns None for a plausible-looking but invalid token"""
    bad_token = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.invalidsig"
    assert auth.verify(bad_token) is None


# ---------------------------------------------------------------------------
# _has_scores_read_scope()
# ---------------------------------------------------------------------------


def test_has_scores_read_scope_returns_true():
    """_has_scores_read_scope returns True for valid scope strings"""
    assert JWTAuthenticator._has_scores_read_scope("scores:read") is True
    assert JWTAuthenticator._has_scores_read_scope("scores:read:wallet/G123") is True
    assert JWTAuthenticator._has_scores_read_scope("other scores:read:all") is True


def test_has_scores_read_scope_returns_false():
    """_has_scores_read_scope returns False for missing or invalid scopes"""
    assert JWTAuthenticator._has_scores_read_scope("") is False
    assert JWTAuthenticator._has_scores_read_scope("read:scores") is False
    assert JWTAuthenticator._has_scores_read_scope("unrelated:scope") is False


# ---------------------------------------------------------------------------
# is_permitted_channel()
# ---------------------------------------------------------------------------


def test_is_permitted_channel_empty_permissions():
    """is_permitted_channel returns False when permissions set is empty"""
    assert JWTAuthenticator.is_permitted_channel(set(), "wallet/G123") is False


def test_is_permitted_channel_admin_accesses_all():
    """Admin permission allows access to any channel"""
    perms = {"scores:read:all"}
    assert JWTAuthenticator.is_permitted_channel(perms, "wallet/G123") is True
    assert JWTAuthenticator.is_permitted_channel(perms, "pair/XLM:native") is True


def test_is_permitted_channel_admin_channel_requires_admin():
    """The 'all' channel itself requires scores:read:all"""
    assert JWTAuthenticator.is_permitted_channel({"scores:read:all"}, "all") is True
    assert JWTAuthenticator.is_permitted_channel({"scores:read:wallet/G123"}, "all") is False


def test_is_permitted_channel_no_match_returns_false():
    """is_permitted_channel returns False when no permission matches channel"""
    perms = {"scores:read:wallet/G123"}
    assert JWTAuthenticator.is_permitted_channel(perms, "pair/XLM:native") is False


# ---------------------------------------------------------------------------
# FileNotFoundError on missing public key
# ---------------------------------------------------------------------------


def test_load_public_key_raises_when_file_missing():
    """JWTAuthenticator raises FileNotFoundError if key file does not exist"""
    with pytest.raises(FileNotFoundError, match="JWT public key not found"):
        JWTAuthenticator(public_key_path="/nonexistent/path/fake.pub")
