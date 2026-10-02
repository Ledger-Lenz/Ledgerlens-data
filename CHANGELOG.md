# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Cycle detection and configurable traversal bounds for payment-path tracing
  (`ingestion/payment_path_analyzer.py`, issue #916). The new
  `trace_payment_paths` walks the wallet payment graph iteratively. It records
  cycles instead of following them, and enforces `max_depth` /
  `max_branching` / `max_paths`. Hitting a bound truncates the trace cleanly
  and logs the affected wallet and the skipped transaction IDs.
- Post-recovery consistency verification (`pipeline/recovery.py`, issue
  #920). `RecoveryManager.complete_recovery` compares expected vs actual
  per-stage record counts and order-independent checksums across the
  recovered range. It produces a human-readable `RecoveryReport` and blocks
  auto-resumption (`RecoveryBlockedError`) on failure until
  `approve_resume` is called. See `docs/recovery_verification.md`.
- Per-stage latency budgets with SLO-burn alerting
  (`monitoring/latency_budget.py`, issue #921). Stage budgets sum to the 10s
  end-to-end detection-latency target, and every stage reports into
  `ledgerlens_stage_latency_seconds{stage}`. Per-stage burn-rate alerts are
  routed through `alerts/router.py`. Also adds the Grafana dashboard
  `latency_budget.json` and Prometheus multi-window burn-rate rules. See
  `docs/latency_slos.md`.
- TTL eviction and size monitoring for the pipeline idempotency-key store
  (`pipeline/idempotency.py`, issue #919): `CheckpointStore.evict_expired()`
  removes keys older than `IDEMPOTENCY_TTL_HOURS` when the store opens and
  hourly as stages start, keeping the store bounded. Store size and evictions
  are exported as `ledgerlens_idempotency_store_entries` and
  `ledgerlens_idempotency_store_evicted_total` and shown on the new
  "Idempotency Key Store" Grafana dashboard. `IDEMPOTENCY_TTL_HOURS` is now read
  from the environment via `config.py` (default `48`; reasoning in
  `docs/idempotency.md`).
- End-to-end exactly-once audit tooling (`pipeline/exactly_once_audit.py`,
  `scripts/audit_exactly_once.py`, issue #918): traces sample records through
  the ingestion, feature/scoring and alerting dedup boundaries and flags any
  boundary that has degraded to at-least-once or at-most-once (unreachable
  backend, non-durable staging, committed keys not recognised, TTL shorter
  than the redelivery window). Runs daily against staging via
  `.github/workflows/exactly-once-audit.yml`. Invariants and operational
  impact are documented in `docs/exactly_once_audit.md`.
- Staleness-aware asset metadata caching with trust-tier fallback
  (`ingestion/asset_metadata_fetcher.py`, issue #917): `get_asset_metadata()`
  returns an `AssetMetadataRecord` labelled with its trust tier (`primary`,
  `cache`, `alternate`, `unavailable`), `fetched_at`, age and staleness. It
  follows a documented primary -> cache -> alternate chain when the primary
  source is unavailable. Forensic reports (`asset_metadata` field and a
  Markdown provenance section) and `build_extended_feature_vector` surface the
  tier and staleness. See `docs/asset_metadata_trust_tiers.md`.
- Schema registry integration with compatibility-mode enforcement
  (`ingestion/avro_codec.py`, issue #914): `HorizonKafkaProducer` now registers
  its Avro schema before publishing, with a Confluent-compatible Schema
  Registry (`SCHEMA_REGISTRY_URL`) or the in-process `SchemaRegistry`.
  Registration enforces `SCHEMA_COMPATIBILITY_MODE` (`NONE`/`BACKWARD`/
  `FORWARD`/`FULL`, default `BACKWARD`) and raises `SchemaCompatibilityError`
  for a breaking change. See `docs/schema_registry_runbook.md`.
- Stream-level ingestion anomaly detection (`ingestion/data_quality.py`, issue
  #913): `StreamQualityMonitor` keeps rolling per-source baselines of batch
  volume, key-field null rates and field means, and routes spikes and drops
  through `alerts/router.py` with source, metric and magnitude context (new
  `ingestion-stream-quality` rule in `alerts/routing_config.yaml`). Known,
  expected changes can be acknowledged with suppression windows. Adds the
  `ingestion_stream_quality.json` Grafana dashboard.
- Perturbation-strength curriculum, robust-accuracy early stopping, and
  per-epoch experiment tracking for the FGSM adversarial training loop
  (`detection.adversarial.robustness.run_adversarial_training`, issue
  #872). `CurriculumScheduler` (`detection/adversarial/augmentation.py`)
  ramps the training epsilon weak-to-strong across epochs (`"linear"` or
  `"step"`); the adversarial *validation* accuracy used for reporting is
  always measured at the final target epsilon so per-epoch numbers stay
  comparable across a curriculum run. Early stopping triggers on stalled
  *robust* (adversarial) validation accuracy, never clean accuracy, per the
  issue's explicit requirement. Both are opt-in (`ADV_TRAINING_CURRICULUM`,
  `ADV_TRAINING_EARLY_STOP_PATIENCE`) and default to the exact pre-#872
  fixed-epsilon, run-every-epoch behavior. Per-epoch clean/robust AUC is
  logged to `mlops.experiment_tracking.JsonlExperimentTracker` when
  `ADV_TRAINING_EXPERIMENT_LOG_PATH` is set. See
  `docs/adversarial_curriculum.md` for recommended defaults and the
  measured no-divergence-across-seeds result.
- Backdoor trigger-feature localization and model-level auto-quarantine
  (`detection/adversarial/backdoor_detector.py`, issue #871).
  `localize_trigger_features` ranks feature columns by a Cohen's-d-style
  effect size between the activation-clustering detector's flagged samples
  and the rest (a simplified spectral-signature decomposition per Tran, Li
  & Madry, 2018) so a flag is now actionable instead of a bare yes/no;
  `ActivationClusteringDetector.structured_report` adds affected
  sample/wallet identities and wash-trading-ring concentration when
  available. `scan_and_quarantine` wires this into
  `detection.model_governance`: a candidate whose flagged fraction exceeds
  `BACKDOOR_SCAN_FLAGGED_FRACTION_THRESHOLD` (default 50%, set above the
  measured 25-47% clean-model noise ceiling to keep false positives low) is
  recorded as
  `status="quarantined"` (`ModelVersionRecord`, migration `0008`) and
  `promote_candidate` now raises `QuarantinedModelError` for that
  `candidate_dir` on every subsequent attempt, whether or not
  `backdoor_report` is passed again — there is no "un-quarantine" API by
  design. Optional pipeline wiring via `BACKDOOR_SCAN_ENABLED`. See
  `docs/adversarial_robustness.md#trigger-feature-localization--model-level-auto-quarantine--issue-871`
  and `docs/security_threat_model.md` (new Model Training tampering row and
  High-Risk Entry Point #7) for the measured clean-model false-positive
  rate and the updated threat model.
- Single, authenticated, cryptographically-gated model promotion/rollback path
  (`detection/model_governance.py`, issue #671): `RiskScorer` now hard-blocks
  on any model that fails Ed25519 signature or transparency-log verification
  instead of logging and loading it anyway; every write to `config.MODEL_DIR`
  (direct training, incremental warm-start, drift-triggered retraining) goes
  through one gate (`guard_production_write`) enforcing signing, an AUC/F1
  regression check, and compatibility validation before publishing; rollback
  is a single authenticated, audited, trust-chain-verified operation
  (`rollback_production`) backed by a queryable `ModelVersionRecord`
  shadow→production→rolled_back history and an append-only
  `promotion_audit_log`. Fixes the `--check-shadow`/`--no-shadow` flags on
  `scripts/retrain_if_drifted.py`, which previously did not exist and made
  every invocation raise `AttributeError`. Adds a drift-monitor heartbeat
  health check (`DriftMonitorHeartbeatStale` alert) and a CI docs-vs-CLI
  consistency test. See
  `docs/model_artifact_trust_and_promotion_adr.md` and
  `docs/model_rollback_runbook.md`. Migration `0007`.
- Unified exactly-once dedup/idempotency library (`pipeline/exactly_once.py`)
  replacing the Kafka worker's and trade ingestion's independent, fail-open
  Redis dedup caches. Fixes a critical bug where a crash mid-processing (e.g.
  `AlertDispatcher.dispatch` raising for the second wallet in a trade) could
  cause a redelivered message to be misclassified as a duplicate and its
  offset committed without reprocessing, silently dropping a wallet's score.
  The new caches are fail-closed: a Redis outage raises
  `DedupBackendUnavailableError` instead of allowing all events through.
  `FeatureBuffer.update()` is now idempotent per `(wallet, trade_id)`.
  `AuditMerkleChain` now persists and rehydrates leaf content so a process
  restart no longer looks like tampering (`TamperDetectedError`). Adds a
  `finality` marker (`provisional`/`final`) to `RiskScoreRecord`, and an
  `AlertDeliveryLedger` + `validation.reconciliation.reconcile_alert_delivery`
  to trace every alert-eligible score to a delivered/dead-lettered/suppressed
  outcome. See `docs/adr/0001-unified-idempotency-finality.md`.
  Migrations `0005` (audit Merkle leaf content) and `0006` (risk-score
  finality). New config: `WORKER_HEALTH_STALE_THRESHOLD_SECONDS`.
- Typed exceptions for ingestion and validation failures: a `LedgerLensError`
  base (`utils/exceptions.py`) and the ingestion taxonomy
  (`ingestion/exceptions.py`): `IngestionError` with `InvalidInputError`,
  `RecordValidationError`, `SchemaValidationError`, and
  `SourceUnavailableError`. Failures carry `source` / `reason` / `raw`
  context mirroring the Kafka dead-letter envelope. Adopted across
  `ingestion/`; degraded-mode behaviour (rate limiter, batch account loads,
  metadata cache) is unchanged. Documented in `docs/ingestion.md` under
  "Error handling".
- `docs/simulator.md`: documents the wash-trade simulators
  (`scripts/wash_trade_simulator.py`,
  `scripts/adversarial_wash_trade_simulator.py`) and the realism evaluation
  (`scripts/evaluate_simulator_realism.py`) — what each generates, what the FFD
  and discriminator-accuracy metrics mean, how to read realism scores, and exact
  generate/evaluate commands.
- `docs/graph_features.md`: added a graph-theory glossary (funding edge,
  ancestor traversal, community, ring, internal edge density, motif,
  reciprocity), each linked to the function that computes it, with the terms
  cross-linked from their first use in the document.
- Cryptographically committed forensic audit trail (`detection/audit_trail.py`):
  signed NDJSON append-only log for report scores, feature/SHAP hashes, and model
  version; `scripts/verify_audit_trail.py` for regulator verification.
  Config: `AUDIT_LOG_PATH`, `AUDIT_VERIFY_PUBLIC_KEY_PATH`.

### Changed
- `ingestion.payment_path_analyzer.reconstruct_path_flow` raises
  `RecordValidationError` instead of `KeyError` when required fields are
  missing. `RecordValidationError` is deliberately not a `KeyError` subclass,
  so callers relying on `except KeyError` here must be updated.

### Fixed
- `scripts/replay_stream.py --resume` now actually seeks to the replay
  consumer group's committed offset per partition — it previously
  unconditionally seeked to the beginning of the topic, silently discarding
  all prior replay progress on every `--resume` invocation. Offsets now
  commit per-message, scoped to that message's exact offset; a persistence
  failure halts the replay run instead of being silently swallowed.
- `migrations/runner.py::MigrationRunner.upgrade(target=...)` no longer
  drops migrations beyond `target` from its returned status report.

## [0.2.0] - 2026-06-13

### Added
- MIT LICENSE.
- Project tooling: `pyproject.toml` (ruff/black/mypy/pytest config), `Makefile`,
  pre-commit hooks, and CI workflow (lint + test on Python 3.11/3.12).
- `Dockerfile` / `.dockerignore` for containerized runs.
- `CONTRIBUTING.md` with local dev setup and PR guidelines.
- GitHub issue templates (bug report, feature request) and a pull request
  template.
- Structured logging (`utils/logging.py`) and a retry/backoff helper
  (`utils/retry.py`) for Horizon API calls.
- Persistence layer for `RiskScore` records (`detection/persistence.py`,
  `detection/risk_score_store.py`) backed by SQLAlchemy and `RISK_SCORE_DB_URL`.
- Order-book event ingestion (`ingestion/orderbook_loader.py`) and a real
  `order_cancellation_rate` feature.
- Wallet funding-graph features: `funding_source_similarity` and
  `network_centrality` (`detection/wallet_graph.py`).
- Soroban contract client (`integrations/contract_client.py`) for
  `submit_score` / `get_score` against `ledgerlens-score`.
- Synthetic labelled dataset generator (`scripts/generate_synthetic_dataset.py`,
  with usage docs in `scripts/README.md`) and a `model_training.py` CLI for
  local training/demo runs.
- Ensemble SHAP aggregation (`ShapExplainer.explain_ensemble`) and explainer
  caching.
- Test coverage for persistence, order-book ingestion, wallet graph features,
  the contract client, the training CLI, and ensemble inference/SHAP.
- Comprehensive unit tests for `JWTAuthenticator.extract_permissions()` and token verification in `tests/test_ws_auth.py`.

### Changed
- `run_pipeline.py` now loads order-book events, persists scored wallets,
  and supports `--no-orderbook`, `--no-persist`, and `--submit-onchain`
  flags.
- `model_inference.py`'s ensemble combination is now a configurable
  `_combine_probabilities` helper, and `confidence` reflects inter-model
  agreement rather than mirroring `score`.

### Fixed
- `RiskScorer.score` and `ShapExplainer` now coerce feature rows to numeric
  dtypes before calling models/explainers, fixing failures with XGBoost and
  newer SHAP versions.
- `extract_permissions` logic to correctly return `{"scores:read:all"}` when given the unrestricted `"scores:read"` scope.
