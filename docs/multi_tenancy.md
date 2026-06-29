# Multi-Tenant Namespace Isolation

LedgerLens supports multi-tenant deployments where each exchange client operates with isolated risk score configurations.

## Architecture

Tenant isolation is enforced at multiple layers:

- **Redis keys**: All Redis keys are prefixed with `tenant_id:` to prevent cross-tenant data leakage
- **Prometheus metrics**: Every metric includes a `tenant` label for per-tenant observability
- **Database records**: Risk scores are scoped by tenant_id
- **Configuration**: Each tenant has custom risk thresholds, Benford parameters, and asset pair whitelists

## YAML Schema

Tenant configuration is loaded from `config/tenants.yaml`:

```yaml
tenants:
  exchange_a:
    risk_threshold: 70
    benford_min_sample: 100
    alert_channels:
      - stdout
    asset_pair_whitelist:
      - USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN
```

## Onboarding a New Tenant

1. Add an entry to `config/tenants.yaml` with the tenant's configuration
2. Restart the pipeline to load the new tenant configuration
3. Use the `TenantContext` class in request handlers to inject tenant-specific behavior

## Security

Tenant IDs are validated against the allowlist in `tenants.yaml`. Arbitrary strings are not accepted as tenant IDs to prevent injection attacks.
