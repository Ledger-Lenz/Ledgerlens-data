"""Pydantic schema definitions for ``config/tenants.yaml``.

Importing from this module does NOT require the rest of the LedgerLens stack;
it is intentionally dependency-light so it can be used in CI validation
scripts without loading the full pipeline.

Schema contract
---------------
Every tenant entry in ``tenants.yaml`` must satisfy :class:`TenantConfigSchema`.
Unknown keys are **rejected** (``model_config = ConfigDict(extra="forbid")``).
Validation errors are raised as ``pydantic.ValidationError`` with field-level
messages, not as opaque ``KeyError`` exceptions at runtime.

Usage::

    from config.tenants_schema import TenantsFileSchema, validate_tenants_yaml

    # Validate a raw dict loaded from YAML
    schema = TenantsFileSchema.model_validate(raw_dict)

    # Or validate a file directly
    validate_tenants_yaml("config/tenants.yaml")
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Tenant-level schema
# ---------------------------------------------------------------------------


class TenantConfigSchema(BaseModel):
    """Strict schema for a single tenant entry in ``tenants.yaml``.

    All fields are required unless they have an explicit default.
    Unknown fields are rejected to prevent silent config drift.
    """

    model_config = ConfigDict(extra="forbid")

    risk_threshold: int = Field(
        ...,
        ge=0,
        le=100,
        description="Risk score (0–100) above which a wallet is flagged.",
    )
    benford_min_sample: int = Field(
        ...,
        ge=1,
        description="Minimum trade count before Benford metrics are computed.",
    )
    alert_channels: list[str] = Field(
        ...,
        min_length=1,
        description="Ordered list of alert delivery channels (stdout, webhook, websocket).",
    )
    asset_pair_whitelist: list[str] = Field(
        ...,
        description="Asset pairs to monitor for this tenant.  Empty list = all pairs.",
    )
    threshold_strategy: Literal["static", "statistical", "adaptive"] = Field(
        default="static",
        description="Algorithm used to compute the per-wallet alert threshold.",
    )
    threshold_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific hyper-parameters (passed to build_strategy()).",
    )

    @field_validator("alert_channels")
    @classmethod
    def validate_alert_channels(cls, v: list[str]) -> list[str]:
        allowed = {"stdout", "webhook", "websocket"}
        for ch in v:
            if ch not in allowed:
                raise ValueError(
                    f"Unknown alert channel {ch!r}. Allowed values: {sorted(allowed)}"
                )
        return v

    @field_validator("asset_pair_whitelist")
    @classmethod
    def validate_asset_pair_format(cls, v: list[str]) -> list[str]:
        """Each entry must match CODE:ISSUER or CODE:ISSUER/CODE:ISSUER."""
        for pair in v:
            parts = pair.split("/")
            if len(parts) not in (1, 2):
                raise ValueError(
                    f"Invalid asset pair {pair!r}: expected CODE:ISSUER or CODE:ISSUER/CODE:ISSUER"
                )
            for part in parts:
                if ":" not in part:
                    raise ValueError(
                        f"Invalid asset identifier {part!r} in pair {pair!r}: "
                        f"expected CODE:ISSUER (e.g. USDC:GA5Z... or XLM:native)"
                    )
        return v


# ---------------------------------------------------------------------------
# Top-level file schema
# ---------------------------------------------------------------------------


class TenantsFileSchema(BaseModel):
    """Schema for the top-level ``tenants.yaml`` document.

    The file must have exactly one top-level key ``tenants`` mapping tenant
    IDs to their configurations.  Tenant IDs must be non-empty strings
    containing only alphanumeric characters, underscores, or hyphens.
    """

    model_config = ConfigDict(extra="forbid")

    tenants: dict[str, TenantConfigSchema] = Field(
        ...,
        min_length=1,
        description="Map of tenant_id → TenantConfigSchema.",
    )

    @field_validator("tenants")
    @classmethod
    def validate_tenant_ids(cls, v: dict[str, TenantConfigSchema]) -> dict[str, TenantConfigSchema]:
        import re

        pattern = re.compile(r"^[A-Za-z0-9_-]+$")
        for tid in v:
            if not pattern.match(tid):
                raise ValueError(
                    f"Tenant ID {tid!r} contains invalid characters. "
                    f"Use only alphanumeric characters, underscores, or hyphens."
                )
        return v


# ---------------------------------------------------------------------------
# Convenience validator
# ---------------------------------------------------------------------------


def validate_tenants_yaml(path: str = "config/tenants.yaml") -> TenantsFileSchema:
    """Load and validate *path* against :class:`TenantsFileSchema`.

    Parameters
    ----------
    path:
        Path to the YAML file to validate.

    Returns
    -------
    TenantsFileSchema
        The validated schema object.

    Raises
    ------
    pydantic.ValidationError
        If any field is missing, has the wrong type, or fails a validator.
    FileNotFoundError
        If *path* does not exist.
    """
    import yaml

    with open(path) as fh:
        raw = yaml.safe_load(fh)

    return TenantsFileSchema.model_validate(raw)
