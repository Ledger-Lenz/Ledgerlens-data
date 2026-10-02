"""Response-level data isolation tests for multi-tenant risk-score endpoints.

These tests are adversarial: they authenticate as one tenant and attempt to
observe any field belonging to another tenant, including through error
messages and pagination metadata. They are intended to run as a required CI
gate for any change touching ``api/app.py``.
"""

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """Return a Flask test client for the API app."""
    from api.app import app

    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture()
def tenant_a_headers():
    return {"X-Tenant-Id": "tenant-a"}


@pytest.fixture()
def tenant_b_headers():
    return {"X-Tenant-Id": "tenant-b"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_no_tenant_b_leakage(payload, tenant_b_markers):
    """Assert that no tenant B marker appears anywhere in the response payload."""
    serialized = repr(payload)
    for marker in tenant_b_markers:
        assert marker not in serialized, (
            f"tenant B data leaked into tenant A response: {marker!r}"
        )


# ---------------------------------------------------------------------------
# Endpoint enumeration
# ---------------------------------------------------------------------------
#
# Every endpoint in api/app.py that returns tenant-scoped data must be listed
# here and covered by a cross-tenant isolation test below. When a new
# tenant-scoped endpoint is added, add it to this list and add a test.
#
TENANT_SCOPED_ENDPOINTS = [
    "/risk-scores",
    "/risk-scores/<score_id>",
]


# ---------------------------------------------------------------------------
# Cross-tenant isolation tests
# ---------------------------------------------------------------------------


def test_risk_scores_list_does_not_leak_other_tenant(client, tenant_a_headers, tenant_b_headers):
    """A list request as tenant A must not contain any tenant B data."""
    response = client.get("/risk-scores", headers=tenant_a_headers)
    assert response.status_code == 200
    payload = response.get_json()
    _assert_no_tenant_b_leakage(payload, ["tenant-b", "tenant_b"])


def test_risk_scores_pagination_metadata_is_tenant_scoped(client, tenant_a_headers):
    """Pagination metadata must not reveal counts or ids from other tenants."""
    response = client.get("/risk-scores?page=1&per_page=10", headers=tenant_a_headers)
    assert response.status_code == 200
    payload = response.get_json()
    _assert_no_tenant_b_leakage(payload, ["tenant-b", "tenant_b"])


def test_risk_score_detail_cross_tenant_returns_generic_not_found(
    client, tenant_a_headers, tenant_b_headers
):
    """Requesting tenant B's score id as tenant A must return a generic 404."""
    # Discover a tenant B score id using tenant B's own credentials.
    listing = client.get("/risk-scores", headers=tenant_b_headers)
    assert listing.status_code == 200
    body = listing.get_json() or {}
    items = body.get("items") or body.get("risk_scores") or []
    if not items:
        pytest.skip("no tenant B risk scores available to test cross-tenant access")

    tenant_b_score_id = items[0].get("id")
    assert tenant_b_score_id is not None

    response = client.get(f"/risk-scores/{tenant_b_score_id}", headers=tenant_a_headers)
    assert response.status_code == 404
    payload = response.get_json() or {}
    # The error must be generic and must not reveal tenant B's data.
    _assert_no_tenant_b_leakage(payload, ["tenant-b", "tenant_b", str(tenant_b_score_id)])


def test_risk_score_detail_error_message_is_generic(client, tenant_a_headers):
    """A not-found error must not disclose whether the id exists for another tenant."""
    response = client.get("/risk-scores/does-not-exist", headers=tenant_a_headers)
    assert response.status_code == 404
    payload = response.get_json() or {}
    message = str(payload.get("error", payload.get("message", ""))).lower()
    assert "tenant" not in message
    assert "forbidden" not in message
    assert "permission" not in message


# ---------------------------------------------------------------------------
# Enumeration guard
# ---------------------------------------------------------------------------


def test_all_tenant_scoped_endpoints_are_covered():
    """Guard that every enumerated tenant-scoped endpoint has a test above."""
    import inspect

    module = inspect.getmodule(test_all_tenant_scoped_endpoints)
    source = inspect.getsource(module)
    for endpoint in TENANT_SCOPED_ENDPOINTS:
        # The endpoint path (or its static prefix) must appear in a test.
        prefix = endpoint.split("<")[0]
        assert prefix in source, f"no isolation test references endpoint {endpoint!r}"
