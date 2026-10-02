"""Tenant configuration for multi-tenant namespace isolation.

Changes in Issue #954
---------------------
* ``load_tenants_config`` now validates the YAML against
  :class:`~config.tenants_schema.TenantsFileSchema` (Pydantic) before
  populating ``_tenant_configs``.  Unknown or malformed fields are rejected
  immediately with a clear ``ValidationError``.
* :class:`TenantConfigDriftDetector` compares the currently loaded config
  against a *last-known-good snapshot* (``config/tenants.yaml.deployed`` by
  default), flagging unexpected additions, removals, or mutations.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import yaml

from config.tenants_schema import TenantsFileSchema


@dataclass
class TenantConfig:
    risk_threshold: int
    benford_min_sample: int
    alert_channels: list[str]
    asset_pair_whitelist: list[str]
    threshold_strategy: str = "static"
    threshold_config: dict[str, Any] = field(default_factory=dict)
    rate_limit: "RateLimitConfig | None" = None
    cost_quota: "CostQuotaConfig | None" = None


@dataclass
class RateLimitConfig:
    """Per-tenant rate limit settings.

    ``rate`` is the sustained token refill rate in requests per second and
    ``burst`` is the maximum bucket capacity (i.e. the largest burst of
    requests allowed at once). Both are per tenant so that one tenant's
    high-volume usage cannot degrade availability for other tenants.
    """

    rate: float
    burst: int


@dataclass
class CostQuotaConfig:
    """Per-tenant budget for expensive, cost-accounted endpoints.

    ``budget`` is the maximum accumulated request cost allowed within a
    ``window_seconds`` sliding window. Each expensive endpoint declares a
    per-request cost estimate (see ``api/app.py``); the API rejects requests
    that would push a tenant over its budget with a retry-after hint instead
    of queuing indefinitely or degrading other tenants.
    """

    budget: float
    window_seconds: int = 60


class TenantNotFoundError(Exception):
    pass


_tenant_configs: dict[str, TenantConfig] = {}
_allowed_tenant_ids: set[str] = set()


def _parse_rate_limit(cfg: dict[str, Any]) -> RateLimitConfig | None:
    """Parse an optional per-tenant ``rate_limit`` block.

    Accepts either a nested mapping::

        rate_limit:
          rate: 50
          burst: 100

    or a shorthand string such as ``"50/s"`` / ``"50"`` (burst defaults to
    the rate). Returns ``None`` when no override is configured, in which case
    the API falls back to its default per-tenant limit.
    """
    raw = cfg.get("rate_limit")
    if raw is None:
        return None
    if isinstance(raw, dict):
        rate = float(raw["rate"])
        burst = int(raw.get("burst", rate))
        return RateLimitConfig(rate=rate, burst=burst)
    if isinstance(raw, str):
        rate = float(raw.rstrip("/s"))
        return RateLimitConfig(rate=rate, burst=int(rate))
    rate = float(raw)
    return RateLimitConfig(rate=rate, burst=int(rate))


def _parse_cost_quota(cfg: dict[str, Any]) -> CostQuotaConfig | None:
    """Parse an optional per-tenant ``cost_quota`` block.

    Accepts either a nested mapping::

        cost_quota:
          budget: 100
          window_seconds: 60

    or a shorthand number such as ``100`` (window defaults to 60s). Returns
    ``None`` when no override is configured, in which case the API falls back
    to its default per-tenant budget.
    """
    raw = cfg.get("cost_quota")
    if raw is None:
        return None
    if isinstance(raw, dict):
        return CostQuotaConfig(
            budget=float(raw["budget"]),
            window_seconds=int(raw.get("window_seconds", 60)),
        )
    return CostQuotaConfig(budget=float(raw))


def load_tenants_config(path: str = "config/tenants.yaml") -> None:
    """Load and validate ``tenants.yaml``, populating the global config dicts.

    Raises
    ------
    pydantic.ValidationError
        If the YAML does not satisfy :class:`~config.tenants_schema.TenantsFileSchema`.
        The error message identifies every invalid field, so operators can fix
        all issues in one pass rather than discovering them one-by-one.
    FileNotFoundError
        If *path* does not exist.
    """
    global _tenant_configs, _allowed_tenant_ids

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Strict schema validation — raises pydantic.ValidationError on any violation
    validated = TenantsFileSchema.model_validate(raw)

    _tenant_configs = {
        tid: TenantConfig(
            risk_threshold=cfg.risk_threshold,
            benford_min_sample=cfg.benford_min_sample,
            alert_channels=list(cfg.alert_channels),
            asset_pair_whitelist=list(cfg.asset_pair_whitelist),
            threshold_strategy=cfg.threshold_strategy,
            threshold_config=dict(cfg.threshold_config),
        )
        for tid, cfg in validated.tenants.items()
    }
    _allowed_tenant_ids = set(_tenant_configs.keys())


def get_tenant_config(tenant_id: str) -> TenantConfig:
    if tenant_id not in _allowed_tenant_ids:
        raise TenantNotFoundError(f"Unknown tenant ID: {tenant_id}")
    return _tenant_configs[tenant_id]


def get_tenant_rate_limit(tenant_id: str) -> RateLimitConfig | None:
    """Return the configured per-tenant rate limit override, if any."""
    return get_tenant_config(tenant_id).rate_limit


def get_tenant_cost_quota(tenant_id: str) -> CostQuotaConfig | None:
    """Return the configured per-tenant cost quota override, if any."""
    return get_tenant_config(tenant_id).cost_quota


def build_threshold_strategy(tenant_id: str) -> Any:
    """Build a ThresholdStrategy instance for the given tenant.

    Returns the appropriate strategy based on the tenant's
    ``threshold_strategy`` and ``threshold_config`` settings.
    """
    from importlib import import_module

    build_strategy = import_module("detection.threshold_strategy").build_strategy

    tc = get_tenant_config(tenant_id)
    kwargs: dict[str, Any] = dict(tc.threshold_config)
    if tc.threshold_strategy == "static" and "threshold" not in kwargs:
        kwargs["threshold"] = tc.risk_threshold / 100.0
    return build_strategy(tc.threshold_strategy, **kwargs)


class TenantContext:
    def __init__(self, tenant_id: str):
        if tenant_id not in _allowed_tenant_ids:
            raise TenantNotFoundError(f"Unknown tenant ID: {tenant_id}")
        self.tenant_id = tenant_id
        self.config = _tenant_configs[tenant_id]

    def redis_key(self, key: str) -> str:
        return f"{self.tenant_id}:{key}"

    def prometheus_labels(self, labels: dict[str, Any]) -> dict[str, Any]:
        return {"tenant": self.tenant_id, **labels}


# ---------------------------------------------------------------------------
# Issue #954 — Tenant config drift detection
# ---------------------------------------------------------------------------


class TenantConfigDriftError(Exception):
    """Raised by :class:`TenantConfigDriftDetector` when config has drifted.

    The error message includes a structured diff of additions, removals, and
    mutations so operators can identify exactly what changed.
    """


class TenantConfigDriftDetector:
    """Compare the currently loaded tenant config against a deployed snapshot.

    The *deployed snapshot* is a JSON representation of the tenant config
    state at the time of the last known-good deployment.  It is stored as a
    sidecar file alongside ``tenants.yaml`` (default:
    ``config/tenants.yaml.deployed``).

    Workflow
    --------
    1. After a successful deployment, call :meth:`save_snapshot` to persist
       the current config state as the new baseline.
    2. On startup (or in CI), call :meth:`detect_drift` to compare the live
       config against the baseline.  Any unexpected change raises
       :exc:`TenantConfigDriftError`.

    What counts as drift
    --------------------
    * A tenant ID added or removed.
    * Any field value changed for an existing tenant (risk_threshold,
      benford_min_sample, alert_channels, asset_pair_whitelist,
      threshold_strategy, threshold_config).

    What does NOT count as drift
    ----------------------------
    * Ordering changes within ``alert_channels`` or ``asset_pair_whitelist``
      lists — the detector compares sorted representations so reordering
      without content changes is not flagged.  This avoids spurious alerts
      from YAML reformatting.
    """

    _SNAPSHOT_DEFAULT = "config/tenants.yaml.deployed"

    def __init__(self, snapshot_path: str | None = None) -> None:
        self.snapshot_path = snapshot_path or self._SNAPSHOT_DEFAULT

    # ------------------------------------------------------------------
    # Snapshot serialisation
    # ------------------------------------------------------------------

    def _configs_to_snapshot(self, configs: dict[str, TenantConfig]) -> dict[str, Any]:
        """Serialise *configs* to a deterministic JSON-compatible dict."""
        return {
            tid: {
                "risk_threshold": tc.risk_threshold,
                "benford_min_sample": tc.benford_min_sample,
                # Sort lists for stable comparison
                "alert_channels": sorted(tc.alert_channels),
                "asset_pair_whitelist": sorted(tc.asset_pair_whitelist),
                "threshold_strategy": tc.threshold_strategy,
                "threshold_config": tc.threshold_config,
            }
            for tid, tc in sorted(configs.items())
        }

    def save_snapshot(
        self,
        configs: dict[str, TenantConfig] | None = None,
        path: str | None = None,
    ) -> str:
        """Persist the current tenant config as the drift-detection baseline.

        Parameters
        ----------
        configs:
            The tenant configs to snapshot.  Defaults to the currently loaded
            global ``_tenant_configs``.
        path:
            Where to write the snapshot.  Defaults to ``self.snapshot_path``.

        Returns
        -------
        str
            The path where the snapshot was written.
        """
        if configs is None:
            configs = _tenant_configs
        dest = path or self.snapshot_path
        snapshot = self._configs_to_snapshot(configs)
        with open(dest, "w") as fh:
            json.dump(snapshot, fh, indent=2, sort_keys=True)
        return dest

    def load_snapshot(self, path: str | None = None) -> dict[str, Any]:
        """Load the deployed snapshot from *path*.

        Raises
        ------
        FileNotFoundError
            If no snapshot file exists (likely first deployment — call
            :meth:`save_snapshot` to establish the baseline).
        """
        src = path or self.snapshot_path
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"No tenant config snapshot found at {src!r}. "
                f"Run TenantConfigDriftDetector().save_snapshot() after a "
                f"successful deployment to establish the baseline."
            )
        with open(src) as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def detect_drift(
        self,
        configs: dict[str, TenantConfig] | None = None,
        snapshot_path: str | None = None,
    ) -> list[str]:
        """Compare *configs* against the deployed snapshot.

        Parameters
        ----------
        configs:
            Tenant configs to compare.  Defaults to the global
            ``_tenant_configs``.
        snapshot_path:
            Path to the snapshot file.  Defaults to ``self.snapshot_path``.

        Returns
        -------
        list[str]
            List of human-readable drift descriptions.  Empty if no drift.

        Raises
        ------
        FileNotFoundError
            If the snapshot file does not exist.
        """
        if configs is None:
            configs = _tenant_configs

        deployed = self.load_snapshot(snapshot_path)
        current = self._configs_to_snapshot(configs)

        diffs: list[str] = []

        deployed_ids = set(deployed.keys())
        current_ids = set(current.keys())

        for added in sorted(current_ids - deployed_ids):
            diffs.append(f"Tenant {added!r} added (not present in deployed snapshot)")
        for removed in sorted(deployed_ids - current_ids):
            diffs.append(f"Tenant {removed!r} removed (was present in deployed snapshot)")

        for tid in sorted(deployed_ids & current_ids):
            d_cfg = deployed[tid]
            c_cfg = current[tid]
            for key in sorted(set(d_cfg) | set(c_cfg)):
                d_val = d_cfg.get(key)
                c_val = c_cfg.get(key)
                if d_val != c_val:
                    diffs.append(
                        f"Tenant {tid!r}: field {key!r} changed "
                        f"from {d_val!r} to {c_val!r}"
                    )

        return diffs

    def assert_no_drift(
        self,
        configs: dict[str, TenantConfig] | None = None,
        snapshot_path: str | None = None,
    ) -> None:
        """Assert that the current config matches the deployed snapshot.

        Raises
        ------
        TenantConfigDriftError
            If any drift is detected.  The error message lists every diff.
        FileNotFoundError
            If the snapshot file does not exist.
        """
        diffs = self.detect_drift(configs=configs, snapshot_path=snapshot_path)
        if diffs:
            diff_block = "\n  ".join(diffs)
            raise TenantConfigDriftError(
                f"Tenant config has drifted from deployed snapshot "
                f"({len(diffs)} change(s)):\n  {diff_block}\n\n"
                f"If this change is intentional, run "
                f"TenantConfigDriftDetector().save_snapshot() after deploying."
            )
