# Environment Variable Contract

Auto-generated from `config.py` by `scripts/generate_env_contract_docs.py` (Issue #544). **Do not hand-edit.** Regenerate with `make env-docs` after changing `config.py`; `make env-docs-check` (wired into CI) fails the build if this file has drifted from the source.

## General

| Variable | Env Var | Type | Required | Default | Description |
|---|---|---|---|---|---|
| `HORIZON_URL` | `HORIZON_URL` | `str` | No | `'https://horizon.stellar.org'` | — |
| `STELLAR_NETWORK` | `STELLAR_NETWORK` | `str` | No | `'PUBLIC'` | — |
| `LOG_FORMAT` | `LOG_FORMAT` | `str` | No | `'json'` | — |
| `WATCHED_ASSET_PAIRS` | `WATCHED_ASSET_PAIRS` | `list[tuple[str, str]]` | No | `'USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN'` | — |
| `WATCHED_AMM_POOLS` | `WATCHED_AMM_POOLS` | `list[str]` | No | `''` | — |
| `BENFORD_WINDOWS_HOURS` | `BENFORD_WINDOWS_HOURS` | `list[int]` | No | `'1,4,24,168,720'` | — |
| `ASSET_BENFORD_WINDOWS` | — | `dict[str, list[int]]` | No | `{}` | — |
| `BENFORD_MIN_SAMPLE_SIZE` | `BENFORD_MIN_SAMPLE_SIZE` | `int` | No | `'50'` | Adaptive Benford window selection (Issue #178) Minimum number of trades required for a window to produce statistically valid metrics. Must be >= 10 to prevent trivially small samples. Default 50 is recommended. |
| `BENFORD_DRIFT_DETECTION_ENABLED` | `BENFORD_DRIFT_DETECTION_ENABLED` | `bool` | No | `'true'` | Benford Drift Detection (Issue #180) Enable Benford drift detection to trigger retraining when digit distributions shift. |
| `BENFORD_DRIFT_Z_THRESHOLD` | `BENFORD_DRIFT_Z_THRESHOLD` | `float` | No | `'3.0'` | Z-score threshold for flagging a shift in chi-square or MAD per-pair (default 3.0 = 0.27% tail probability). |
| `BENFORD_DRIFT_NUM_PAIRS_TRIGGER` | `BENFORD_DRIFT_NUM_PAIRS_TRIGGER` | `int` | No | `'0'` | Minimum pairs that must drift before firing a global retraining trigger (default 0 = any single pair can trigger). |
| `CONFORMAL_COVERAGE_LEVEL` | `CONFORMAL_COVERAGE_LEVEL` | `float` | No | `'0.90'` | Conformal prediction (Issue #181) Coverage level for conformal prediction intervals (e.g. 0.90 = 90% coverage guarantee). |
| `CONFORMAL_CALIBRATION_PATH` | `CONFORMAL_CALIBRATION_PATH` | `str` | No | `'models/conformal_calibration.joblib'` | Path to the calibration artifact (computed during training, loaded at inference startup). |
| `CONFORMAL_ENABLED` | `CONFORMAL_ENABLED` | `bool` | No | `'true'` | Enable conformal prediction intervals in the API response (default true). |
| `CROSS_PAIR_SYNCHRONY_WINDOW_SECONDS` | `CROSS_PAIR_SYNCHRONY_WINDOW_SECONDS` | `int` | No | `'30'` | — |
| `ALERT_DEDUP_WINDOW_SECONDS` | `ALERT_DEDUP_WINDOW_SECONDS` | `int` | No | `'60'` | Silence window for correlated alert deduplication (alerts/deduplicator.py). |
| `RISK_SCORE_FLAG_THRESHOLD` | `RISK_SCORE_FLAG_THRESHOLD` | `int` | No | `'70'` | — |
| `MAD_NONCONFORMITY_THRESHOLD` | `MAD_NONCONFORMITY_THRESHOLD` | `float` | No | `'0.015'` | Benford MAD above this sets benford_flag = true (Nigrini, 2012). |
| `THRESHOLD_RL_PINNED` | `THRESHOLD_RL_PINNED` | `int` | No | `'0'` | Set to a non-zero integer to pin the alert threshold and disable the RL agent. E.g. THRESHOLD_RL_PINNED=75 → agent is bypassed, threshold is fixed at 75. |
| `RISK_SCORE_DB_URL` | `RISK_SCORE_DB_URL` | `str` | No | `'sqlite:///ledgerlens.db'` | — |
| `DB_POOL_SIZE` | `DB_POOL_SIZE` | `int` | No | `'5'` | Database connection pooling |
| `DB_MAX_OVERFLOW` | `DB_MAX_OVERFLOW` | `int` | No | `'10'` | — |
| `DB_POOL_TIMEOUT` | `DB_POOL_TIMEOUT` | `int` | No | `'30'` | — |
| `MODEL_DIR` | `MODEL_DIR` | `str` | No | `'./models'` | — |
| `BATCH_SCORER_WORKERS` | `BATCH_SCORER_WORKERS` | `int` | No | `10` | — |
| `SOROBAN_RPC_URL` | `SOROBAN_RPC_URL` | `str` | No | `'https://soroban-testnet.stellar.org'` | ledgerlens-score Soroban contract |
| `LEDGERLENS_CONTRACT_ID` | `LEDGERLENS_CONTRACT_ID` | `str` | No | `''` | — |
| `LEDGERLENS_SUBMITTER_SECRET` | `LEDGERLENS_SUBMITTER_SECRET` | `str` | No | `''` | — |
| `SOLANA_RPC_URL` | `SOLANA_RPC_URL` | `str` | No | `'https://api.mainnet-beta.solana.com'` | Solana RPC endpoint for cross-chain resolution |
| `MIN_TRADES_FOR_SCORING` | `MIN_TRADES_FOR_SCORING` | `int` | No | `'20'` | — |
| `LIST_RELOAD_INTERVAL_SECONDS` | `LIST_RELOAD_INTERVAL_SECONDS` | `int` | No | `'60'` | — |
| `DRIFT_WINDOW_SIZE` | `DRIFT_WINDOW_SIZE` | `int` | No | `'1000'` | Live feature drift monitoring (Population Stability Index) |
| `DRIFT_PSI_THRESHOLD` | `DRIFT_PSI_THRESHOLD` | `float` | No | `'0.2'` | Fire an alert when any feature PSI exceeds this value. |
| `DRIFT_REFERENCE_WINDOW_HOURS` | `DRIFT_REFERENCE_WINDOW_HOURS` | `int` | No | `'168'` | Sliding window covariance shift detection (MMD) |
| `DRIFT_TEST_WINDOW_HOURS` | `DRIFT_TEST_WINDOW_HOURS` | `int` | No | `'1'` | — |
| `DRIFT_CHECK_INTERVAL_MINUTES` | `DRIFT_CHECK_INTERVAL_MINUTES` | `int` | No | `'30'` | — |
| `REPORT_CONCURRENCY` | `REPORT_CONCURRENCY` | `int` | No | `'4'` | Forensic reporting |
| `SHAP_INTERACTIONS_ENABLED` | `SHAP_INTERACTIONS_ENABLED` | `bool` | No | `'false'` | SHAP interaction values are O(n * d^2) — disable by default. |
| `WALLET_GRAPH_MAX_DEPTH` | `WALLET_GRAPH_MAX_DEPTH` | `int` | No | `'4'` | Wallet funding graph — multi-hop traversal + wash-trading ring detection |
| `WASH_RING_MIN_SIZE` | `WASH_RING_MIN_SIZE` | `int` | No | `'3'` | — |
| `WASH_RING_RESOLUTION` | `WASH_RING_RESOLUTION` | `float` | No | `'1.0'` | — |
| `WASH_RING_LOUVAIN_SEED` | `WASH_RING_LOUVAIN_SEED` | `int` | No | `'42'` | Fixed seed keeps Louvain community detection deterministic in CI. |
| `MOTIF_CENSUS_TIMEOUT_SECONDS` | `MOTIF_CENSUS_TIMEOUT_SECONDS` | `float` | No | `'5'` | Motif census timeout — partial results returned with census_truncated=True if exceeded. |
| `HORIZON_MAX_RPS` | `HORIZON_MAX_RPS` | `int` | No | `'80'` | Distributed rate limiting for Horizon REST calls (ingestion/rate_limiter.py) |
| `HORIZON_MAX_RETRIES` | `HORIZON_MAX_RETRIES` | `int` | No | `'5'` | — |
| `REDIS_URL` | `REDIS_URL` | `str` | No | `'redis://localhost:6379/0'` | — |
| `STREAMING_BACKEND` | `STREAMING_BACKEND` | `str` | No | `'sse'` | Real-time streaming / alerting STREAMING_BACKEND selects the ingestion transport: "sse"   — existing thread-per-pair Horizon SSE pipeline (default, no Kafka) "kafka" — Apache Kafka producer/consumer distributed pipeline |
| `KAFKA_BOOTSTRAP_SERVERS` | `KAFKA_BOOTSTRAP_SERVERS` | `str` | No | `'localhost:9092'` | Kafka — credentials are read from env vars only, never committed. |
| `KAFKA_SASL_USERNAME` | `KAFKA_SASL_USERNAME` | `str | None` | Yes | — | — |
| `KAFKA_SASL_PASSWORD` | `KAFKA_SASL_PASSWORD` | `str | None` | Yes | — | — |
| `KAFKA_CONSUMER_GROUP` | `KAFKA_CONSUMER_GROUP` | `str` | No | `'ledgerlens-scorer'` | — |
| `KAFKA_TOPIC_PREFIX` | `KAFKA_TOPIC_PREFIX` | `str` | No | `'ledgerlens.trades'` | — |
| `KAFKA_DLQ_TOPIC` | `KAFKA_DLQ_TOPIC` | `str` | No | `'ledgerlens.trades.dlq'` | — |
| `KAFKA_TOPIC_PATTERN` | `KAFKA_TOPIC_PATTERN` | `str` | No | `'^ledgerlens\\.trades\\..*'` | Regex subscription (librdkafka treats a leading '^' as a pattern). Picks up new per-pair topics without a consumer restart; the DLQ topic is skipped in the worker so failed messages are never auto-replayed. |
| `KAFKA_LAG_ALERT_THRESHOLD` | `KAFKA_LAG_ALERT_THRESHOLD` | `int` | No | `'500'` | — |
| `KAFKA_METRICS_PORT` | `KAFKA_METRICS_PORT` | `int` | No | `'9100'` | — |
| `TRADE_AVRO_SCHEMA_PATH` | `TRADE_AVRO_SCHEMA_PATH` | `str` | No | `'data/trade_avro_schema.json'` | — |
| `SCHEMA_REGISTRY_URL` | `SCHEMA_REGISTRY_URL` | `str | None` | Yes | — | Confluent-compatible Schema Registry base URL (ingestion/avro_codec.py). When unset, schemas are registered in an in-process registry instead. |
| `SCHEMA_COMPATIBILITY_MODE` | `SCHEMA_COMPATIBILITY_MODE` | `str` | No | `'BACKWARD'` | NONE, BACKWARD, FORWARD or FULL; enforced when a producer registers its schema. |
| `WORKER_HEALTH_STALE_THRESHOLD_SECONDS` | `WORKER_HEALTH_STALE_THRESHOLD_SECONDS` | `float` | No | `'120'` | Worker health monitoring (streaming/health.py::WorkerHealthMonitor). A worker is marked UNHEALTHY when it has not heartbeat within this many seconds — should comfortably exceed the poll-loop interval plus the slowest expected per-message processing time. |
| `E2E_LATENCY_BUDGET_MS` | `E2E_LATENCY_BUDGET_MS` | `int` | No | `'2000'` | End-to-end latency budget (Issue #124) |
| `LATENCY_ANOMALY_RATE_THRESHOLD` | `LATENCY_ANOMALY_RATE_THRESHOLD` | `float` | No | `'0.90'` | — |
| `METADATA_TOPIC` | `METADATA_TOPIC` | `str` | No | `'ledgerlens.account_metadata'` | Account metadata streaming join (streaming/account_metadata_stream.py, streaming/pipeline.py MetadataJoinState) METADATA_TOPIC: dedicated Kafka topic for account metadata update events. |
| `METADATA_JOIN_WINDOW_SECONDS` | `METADATA_JOIN_WINDOW_SECONDS` | `int` | No | `'3600'` | METADATA_JOIN_WINDOW_SECONDS: how long (seconds) a metadata update enriches incoming trade events.  After this window the update is considered stale and must be refreshed by a subsequent Horizon effect.  Default: 3600 (1 hour). |
| `METADATA_ACTIVE_WALLET_TTL_SECONDS` | `METADATA_ACTIVE_WALLET_TTL_SECONDS` | `int` | No | `'86400'` | METADATA_ACTIVE_WALLET_TTL_SECONDS: wallets that have had no trade activity for this many seconds are pruned from join state to keep memory bounded. Default: 86400 (24 hours) — matches the requirement spec. |
| `ALERT_CHANNEL` | `ALERT_CHANNEL` | `str` | No | `'stdout'` | — |
| `ALERT_WEBHOOK_URL` | `ALERT_WEBHOOK_URL` | `str | None` | Yes | — | — |
| `ALERT_COOLDOWN_SECONDS` | `ALERT_COOLDOWN_SECONDS` | `int` | No | `'3600'` | — |
| `ALERT_DEAD_LETTER_PATH` | `ALERT_DEAD_LETTER_PATH` | `str` | No | `'alerts_dlq.ndjson'` | — |
| `WS_PORT` | `WS_PORT` | `int` | No | `'8765'` | — |
| `WS_BIND_HOST` | `WS_BIND_HOST` | `str` | No | `'127.0.0.1'` | — |
| `WS_ALLOW_EXTERNAL` | `WS_ALLOW_EXTERNAL` | `bool` | No | `''` | — |
| `JWT_PUBLIC_KEY_PATH` | `JWT_PUBLIC_KEY_PATH` | `str` | No | `'./jwt_public_key.pem'` | WebSocket pub/sub server (streaming/ws_server.py) |
| `WS_MAX_CLIENTS` | `WS_MAX_CLIENTS` | `int` | No | `'200'` | — |
| `WS_CLIENT_QUEUE_DEPTH` | `WS_CLIENT_QUEUE_DEPTH` | `int` | No | `'100'` | — |
| `WS_REPLAY_BUFFER_SIZE` | `WS_REPLAY_BUFFER_SIZE` | `int` | No | `'1000'` | — |
| `WS_RATE_LIMIT_MSGS_PER_SECOND` | `WS_RATE_LIMIT_MSGS_PER_SECOND` | `int` | No | `'100'` | — |
| `WS_HEARTBEAT_INTERVAL_SECONDS` | `WS_HEARTBEAT_INTERVAL_SECONDS` | `float` | No | `'30'` | Seconds between WebSocket ping frames sent to each client. If the client does not respond with a pong within this interval, the connection is closed and the subscriber entry is cleaned up. |
| `WS_SLOW_CONSUMER_POLICY` | `WS_SLOW_CONSUMER_POLICY` | `str` | No | `'drop_oldest'` | Slow-consumer policy when a client's outbound queue is full (issue #893): "drop_oldest" (default) or "disconnect". Under drop_oldest, a client that drops more than WS_SLOW_CONSUMER_MAX_DROPS messages is disconnected (0 = never). |
| `WS_SLOW_CONSUMER_MAX_DROPS` | `WS_SLOW_CONSUMER_MAX_DROPS` | `int` | No | `'0'` | — |
| `WS_TOKEN_EXPIRY_GRACE_SECONDS` | `WS_TOKEN_EXPIRY_GRACE_SECONDS` | `float` | No | `'0'` | In-band token refresh (issue #894): seconds of grace after token expiry before the connection is force-closed with code 4001. |
| `WS_ABUSE_MAX_REQUESTS_PER_MINUTE` | `WS_ABUSE_MAX_REQUESTS_PER_MINUTE` | `int` | No | `'300'` | WebSocket abuse detection (issue #223) |
| `WS_ABUSE_MAX_DISTINCT_WALLETS` | `WS_ABUSE_MAX_DISTINCT_WALLETS` | `int` | No | `'50'` | — |
| `WS_ABUSE_WALLET_WINDOW_SECONDS` | `WS_ABUSE_WALLET_WINDOW_SECONDS` | `int` | No | `'60'` | — |
| `WS_ABUSE_BLOCK_DURATION_SECONDS` | `WS_ABUSE_BLOCK_DURATION_SECONDS` | `int` | No | `'300'` | — |
| `WS_ABUSE_MAX_REPUTATION_MULTIPLIER` | `WS_ABUSE_MAX_REPUTATION_MULTIPLIER` | `float` | No | `'3.0'` | Reputation-based adaptive rate limit (issue #895) |
| `WS_ABUSE_NEW_CLIENT_MULTIPLIER` | `WS_ABUSE_NEW_CLIENT_MULTIPLIER` | `float` | No | `'1.0'` | — |
| `WS_ABUSE_REPUTATION_HALF_LIFE_SECONDS` | `WS_ABUSE_REPUTATION_HALF_LIFE_SECONDS` | `float` | No | `'3600'` | — |
| `DP_TARGET_EPSILON` | `DP_TARGET_EPSILON` | `float` | No | `'8.0'` | Differentially private neural training (DP-SGD via Opacus) |
| `DP_TARGET_DELTA` | `DP_TARGET_DELTA` | `float` | No | `'1e-5'` | — |
| `DP_MAX_GRAD_NORM` | `DP_MAX_GRAD_NORM` | `float` | No | `'1.0'` | — |
| `DP_EPOCHS` | `DP_EPOCHS` | `int` | No | `'50'` | — |
| `DP_BUDGET_TOTAL_EPSILON` | `DP_BUDGET_TOTAL_EPSILON` | `float` | No | `'100.0'` | DP privacy budget tracker (Issue #195) Total epsilon cap across all training rounds and inference queries. |
| `DP_BUDGET_ALERT_THRESHOLD` | `DP_BUDGET_ALERT_THRESHOLD` | `float` | No | `'10.0'` | Alert is fired when remaining epsilon drops below this value. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `OTEL_EXPORTER_OTLP_ENDPOINT` | `str` | No | `'http://localhost:4317'` | OpenTelemetry distributed tracing (Issue #198) |
| `OTEL_SAMPLING_RATE` | `OTEL_SAMPLING_RATE` | `float` | No | `'0.1'` | — |
| `ADVERSARIAL_AUG_RATIO` | `ADVERSARIAL_AUG_RATIO` | `float` | No | `'0.0'` | Adversarial training augmentation |
| `ADV_TRAINING_ENABLED` | `ADV_TRAINING_ENABLED` | `bool` | No | `'false'` | FGSM adversarial training (Issue #191) Set ADV_TRAINING_ENABLED=true to enable the FGSM adversarial training loop. |
| `ADV_TRAINING_EPOCHS` | `ADV_TRAINING_EPOCHS` | `int` | No | `'3'` | — |
| `ADV_TRAINING_EPSILON` | `ADV_TRAINING_EPSILON` | `float` | No | `'0.1'` | — |
| `ADV_TRAINING_RATIO` | `ADV_TRAINING_RATIO` | `float` | No | `'0.5'` | — |
| `MODEL_SIGNING_PRIVATE_KEY_PATH` | `MODEL_SIGNING_PRIVATE_KEY_PATH` | `str` | No | `''` | Model integrity & BFT voting |
| `TRUSTED_SIGNING_KEY_FINGERPRINT` | `TRUSTED_SIGNING_KEY_FINGERPRINT` | `str` | No | `''` | — |
| `TRUSTED_SIGNING_PUBLIC_KEY_PATH` | `TRUSTED_SIGNING_PUBLIC_KEY_PATH` | `str` | No | `''` | Ed25519 PUBLIC key used to verify model artifacts at *load* time (the inference-side counterpart to MODEL_SIGNING_PRIVATE_KEY_PATH). Required for RiskScorer to load any model in strict (default) mode — see detection/model_governance.py and docs/model_artifact_lifecycle.md. |
| `MODEL_INTEGRITY_OVERRIDE_ACTOR` | `MODEL_INTEGRITY_OVERRIDE_ACTOR` | `str` | No | `''` | Emergency override for the hard-block artifact integrity gate in RiskScorer._load_models. Both must be set for the override to apply; every use is written to the promotion_audit_log table. Never set this in a persisted environment file — it is meant to be exported for a single incident-response shell session only. |
| `MODEL_INTEGRITY_OVERRIDE_REASON` | `MODEL_INTEGRITY_OVERRIDE_REASON` | `str` | No | `''` | — |
| `MODEL_PROMOTION_AUTHORIZED_ACTORS` | `MODEL_PROMOTION_AUTHORIZED_ACTORS` | `str` | No | `''` | Model promotion / rollback authorization (detection/model_governance.py). Comma-separated allowlist of actor IDs permitted to promote or roll back a production model. MODEL_PROMOTION_SECRET is the HMAC key used to authenticate the actor-supplied credential; rotate it like any other shared secret (e.g. via a secrets manager), never commit it. |
| `MODEL_PROMOTION_SECRET` | `MODEL_PROMOTION_SECRET` | `str` | No | `''` | — |
| `MODEL_PROMOTION_SYSTEM_ACTOR` | `MODEL_PROMOTION_SYSTEM_ACTOR` | `str` | No | `'retrain-pipeline'` | Actor identity used by the automated drift-triggered retraining pipeline (scripts/retrain_if_drifted.py) to authenticate its own promotions. Must be included in MODEL_PROMOTION_AUTHORIZED_ACTORS for automated promotion to succeed; its credential is derived from MODEL_PROMOTION_SECRET so no interactive secret is needed in CI. |
| `MODEL_PROMOTION_REGRESSION_TOLERANCE` | `MODEL_PROMOTION_REGRESSION_TOLERANCE` | `float` | No | `'0.01'` | — |
| `DRIFT_MONITOR_HEARTBEAT_MAX_AGE_SECONDS` | `DRIFT_MONITOR_HEARTBEAT_MAX_AGE_SECONDS` | `int` | No | `'3600'` | Drift-monitor health (monitoring/drift_detector.py). If no successful CovarianceShiftDetector.detect() call has been recorded within this many seconds, the monitor is considered stale and a distinct "drift-check failed" alert fires — see monitoring/alert_rules.yml. |
| `AUDIT_LOG_PATH` | `AUDIT_LOG_PATH` | `str` | No | `'data/audit_trail.ndjson'` | — |
| `AUDIT_VERIFY_PUBLIC_KEY_PATH` | `AUDIT_VERIFY_PUBLIC_KEY_PATH` | `str` | No | `''` | — |
| `BFT_SCORE_DIVERGENCE_THRESHOLD` | `BFT_SCORE_DIVERGENCE_THRESHOLD` | `int` | No | `'30'` | — |
| `BFT_MIN_CONSENSUS` | `BFT_MIN_CONSENSUS` | `int` | No | `'2'` | — |
| `POISON_LABEL_RATIO_THRESHOLD` | `POISON_LABEL_RATIO_THRESHOLD` | `float` | No | `'0.15'` | — |
| `ZERO_SHOT_WEIGHT` | `ZERO_SHOT_WEIGHT` | `float` | No | `'0.0'` | — |
| `ZERO_SHOT_MIN_LABELLED_EXAMPLES` | `ZERO_SHOT_MIN_LABELLED_EXAMPLES` | `int` | No | `'20'` | — |
| `BENFORD_CI_ENABLED` | `BENFORD_CI_ENABLED` | `bool` | No | `'false'` | — |
| `BRIDGE_ROUNDTRIP_WINDOW_HOURS` | `BRIDGE_ROUNDTRIP_WINDOW_HOURS` | `int` | No | `'72'` | — |
| `CROSS_CHAIN_MIN_CONFIDENCE` | `CROSS_CHAIN_MIN_CONFIDENCE` | `float` | No | `'0.65'` | Minimum combined confidence for a cross-chain identity link to carry risk (Issue #879); see docs/cross_chain_identity.md for the rationale. |
| `DP_EPSILON` | `DP_EPSILON` | `float` | No | `'1.0'` | Differential privacy for SHAP explanations (model inversion defence) |
| `DP_DELTA` | `DP_DELTA` | `float` | No | `'1e-5'` | — |
| `DP_RENYI_QUERY_THRESHOLD` | `DP_RENYI_QUERY_THRESHOLD` | `int` | No | `'100'` | — |
| `DP_RENYI_NOISE_MULTIPLIER` | `DP_RENYI_NOISE_MULTIPLIER` | `float` | No | `'3.0'` | — |
| `DP_DEFAULT_SENSITIVITY` | `DP_DEFAULT_SENSITIVITY` | `float` | No | `'0.05'` | — |
| `SHAP_SENSITIVITY_PATH` | `SHAP_SENSITIVITY_PATH` | `str` | No | `'models/shap_sensitivity.json'` | — |
| `MODEL_INVERSION_QUERY_LIMIT` | `MODEL_INVERSION_QUERY_LIMIT` | `int` | No | `'100'` | Model inversion attack defence |
| `MODEL_INVERSION_DP_EPSILON` | `MODEL_INVERSION_DP_EPSILON` | `float` | No | `'1.0'` | — |
| `SCORE_ROUNDING_GRANULARITY` | `SCORE_ROUNDING_GRANULARITY` | `int` | No | `'1'` | — |
| `FEATURE_SELECTION_ENABLED` | `FEATURE_SELECTION_ENABLED` | `bool` | No | `''` | Feature selection |
| `FEATURE_SELECTION_PATH` | `FEATURE_SELECTION_PATH` | `str` | No | `'models/selected_features.json'` | — |
| `ANNOTATION_HMAC_SECRET` | `ANNOTATION_HMAC_SECRET` | `str` | No | `''` | Annotation integrity |
| `AL_QUERY_STRATEGY` | `AL_QUERY_STRATEGY` | `str` | No | `'committee_disagreement'` | Active learning |
| `AL_BATCH_SIZE` | `AL_BATCH_SIZE` | `int` | No | `'20'` | — |
| `AL_RETRAIN_THRESHOLD` | `AL_RETRAIN_THRESHOLD` | `int` | No | `'50'` | — |
| `AL_ROLLBACK_AUC_DROP` | `AL_ROLLBACK_AUC_DROP` | `float` | No | `'0.01'` | — |
| `AL_QUEUE_PATH` | `AL_QUEUE_PATH` | `str` | No | `'data/annotation_queue.json'` | — |
| `ACTIVE_LEARNING_ALPHA` | `ACTIVE_LEARNING_ALPHA` | `float` | No | `'0.5'` | Core-set selection (Issue #253) |
| `CORESET_MIN_DISTANCE` | `CORESET_MIN_DISTANCE` | `float` | No | `'0.1'` | — |
| `ACTIVE_LEARNING_EER_THRESHOLD` | `ACTIVE_LEARNING_EER_THRESHOLD` | `float` | No | `'0.001'` | Active learning stopping criterion (Issue #256) |
| `ACTIVE_LEARNING_CONVERGENCE_WINDOW` | `ACTIVE_LEARNING_CONVERGENCE_WINDOW` | `int` | No | `'5'` | — |
| `GAN_ROUNDS` | `GAN_ROUNDS` | `int` | No | `'5'` | Wash Trade Simulation Engine |
| `GAN_PLATEAU_THRESHOLD` | `GAN_PLATEAU_THRESHOLD` | `float` | No | `'0.005'` | — |
| `SIMULATOR_N_WALLETS` | `SIMULATOR_N_WALLETS` | `int` | No | `'50'` | — |
| `SIMULATOR_TRADES_PER_WALLET` | `SIMULATOR_TRADES_PER_WALLET` | `int` | No | `'100'` | — |
| `GNN_EMBEDDING_DIM` | `GNN_EMBEDDING_DIM` | `int` | No | `'32'` | Graph neural network encoder |
| `GNN_HIDDEN_DIM` | `GNN_HIDDEN_DIM` | `int` | No | `'64'` | — |
| `GNN_NUM_LAYERS` | `GNN_NUM_LAYERS` | `int` | No | `'2'` | — |
| `ENSEMBLE_WEIGHT_SMOOTHING_ALPHA` | `ENSEMBLE_WEIGHT_SMOOTHING_ALPHA` | `float` | No | `'0.1'` | Dynamic ensemble weight adjustment (#268) |
| `ENSEMBLE_SYSTEMIC_FP_THRESHOLD` | `ENSEMBLE_SYSTEMIC_FP_THRESHOLD` | `float` | No | `'0.5'` | — |
| `GNN_DIFFPOOL_CLUSTERS` | `GNN_DIFFPOOL_CLUSTERS` | `int` | No | `'10'` | GNN DiffPool cluster scoring (#269) |
| `GRAPH_STALE_EDGE_MAX_AGE_HOURS` | `GRAPH_STALE_EDGE_MAX_AGE_HOURS` | `int` | No | `'168'` | Incremental wallet graph cache (#203) |
| `SHADOW_TRAFFIC_PERCENT` | `SHADOW_TRAFFIC_PERCENT` | `int` | No | `'20'` | Shadow deployment / concept drift-aware model versioning (#204) |
| `SHADOW_PERIOD_HOURS` | `SHADOW_PERIOD_HOURS` | `int` | No | `'24'` | — |
| `SHADOW_DRIFT_THRESHOLD_POINTS` | `SHADOW_DRIFT_THRESHOLD_POINTS` | `int` | No | `'15'` | — |
| `SHADOW_DRIFT_MAX_RATE` | `SHADOW_DRIFT_MAX_RATE` | `float` | No | `'0.05'` | — |
| `SHADOW_FP_RATE_MAX_EXCESS` | `SHADOW_FP_RATE_MAX_EXCESS` | `float` | No | `'0.10'` | — |
| `FEDERATED_ASYNC_TRIGGER_N` | `FEDERATED_ASYNC_TRIGGER_N` | `int` | No | `'3'` | Async federated learning (#270) |
| `FEDERATED_ASYNC_TRIGGER_SECONDS` | `FEDERATED_ASYNC_TRIGGER_SECONDS` | `int` | No | `'300'` | — |
| `FEDERATED_MAX_STALENESS` | `FEDERATED_MAX_STALENESS` | `int` | No | `'5'` | — |
| `FEATURE_CACHE_TTL_SECONDS` | `FEATURE_CACHE_TTL_SECONDS` | `int` | No | `'30'` | Feature cache (detection/feature_cache.py) — in-memory TTL+LRU cache for per-wallet feature matrices used by the streaming scorer. |
| `FEATURE_CACHE_MAXSIZE` | `FEATURE_CACHE_MAXSIZE` | `int` | No | `'1000'` | — |
| `LABEL_QUALITY_NOISE_THRESHOLD` | `LABEL_QUALITY_NOISE_THRESHOLD` | `float` | No | `'0.1'` | Label quality estimation (#271) |
| `ANNOTATOR_NOISE_RATE_ALERT_THRESHOLD` | `ANNOTATOR_NOISE_RATE_ALERT_THRESHOLD` | `float` | No | `'0.2'` | — |

## Transformer sequence model (#182)

| Variable | Env Var | Type | Required | Default | Description |
|---|---|---|---|---|---|
| `SEQ_MODEL_NUM_PAIRS` | `SEQ_MODEL_NUM_PAIRS` | `int` | No | `'32'` | Number of distinct asset-pair slots in the one-hot pair encoding. Increase this if the deployment monitors more than the default 32 pairs. |
| `SEQ_MODEL_EMBED_DIM` | `SEQ_MODEL_EMBED_DIM` | `int` | No | `'64'` | Dimension of the token embeddings and sequence-level output embedding. |
| `SEQ_MODEL_NUM_HEADS` | `SEQ_MODEL_NUM_HEADS` | `int` | No | `'4'` | Number of self-attention heads.  Must divide SEQ_MODEL_EMBED_DIM evenly. |
| `SEQ_MODEL_NUM_LAYERS` | `SEQ_MODEL_NUM_LAYERS` | `int` | No | `'2'` | Number of transformer encoder layers (2–4 is the sweet spot for latency). |
| `SEQ_MODEL_FFN_DIM` | `SEQ_MODEL_FFN_DIM` | `int` | No | `'128'` | Feed-forward expansion dimension inside each transformer layer. |
| `SEQ_MODEL_DROPOUT` | `SEQ_MODEL_DROPOUT` | `float` | No | `'0.1'` | Dropout probability applied during training (disabled at eval time). |
| `SEQ_MODEL_MAX_LENGTH` | `SEQ_MODEL_MAX_LENGTH` | `int` | No | `'512'` | Maximum allowed input sequence length.  Inputs longer than this are rejected before reaching the model to prevent memory exhaustion. |
| `SEQ_MODEL_ENABLED` | `SEQ_MODEL_ENABLED` | `bool` | No | `'true'` | Whether to attempt loading the sequence model at inference time. Set to "false" to skip loading (e.g., before the first training run). |

## Redis feature store (#183)

| Variable | Env Var | Type | Required | Default | Description |
|---|---|---|---|---|---|
| `FEATURE_STORE_REDIS_URL` | `FEATURE_STORE_REDIS_URL` | `str` | No | `os.getenv('REDIS_URL', 'redis://localhost:6379/1')` | Redis URL for the feature store.  Overrides the rate-limiter REDIS_URL when set. Format: redis://[:password@]host[:port][/db] or rediss://[:password@]host[:port][/db]  (TLS) |
| `FEATURE_STORE_REDIS_TLS` | `FEATURE_STORE_REDIS_TLS` | `bool` | No | `'false'` | Enable TLS for the feature store Redis connection. When FEATURE_STORE_REDIS_URL starts with rediss:// this is implied. |
| `FEATURE_STORE_REDIS_TLS_CA_CERT` | `FEATURE_STORE_REDIS_TLS_CA_CERT` | `str` | No | `''` | Redis CA certificate path for TLS verification (optional). |
| `FEATURE_STORE_REDIS_POOL_SIZE` | `FEATURE_STORE_REDIS_POOL_SIZE` | `int` | No | `'10'` | Redis connection pool: maximum number of pooled connections. |
| `FEATURE_STORE_WINDOW_TTLS` | `FEATURE_STORE_WINDOW_TTLS` | `str` | No | `'1:3600,4:14400,24:86400,168:604800,720:2592000'` | Per-window TTLs (seconds) for cached feature vectors. Format: comma-separated "hours:seconds" pairs, e.g. "1:3600,4:14400,24:86400" When empty, a fixed 300-second TTL is used for all windows. |
| `FEATURE_STORE_FALLBACK_ENABLED` | `FEATURE_STORE_FALLBACK_ENABLED` | `bool` | No | `'true'` | Fallback to direct feature computation when Redis is unavailable. |
| `DP_AGGREGATOR_EPSILON` | `DP_AGGREGATOR_EPSILON` | `float` | No | `'1.0'` | Differential-privacy aggregation of training statistics (Issue #299) |
| `DP_AGGREGATOR_DELTA` | `DP_AGGREGATOR_DELTA` | `float` | No | `'1e-5'` | — |
| `ACTIVE_LEARNING_ALEATORIC_THRESHOLD` | `ACTIVE_LEARNING_ALEATORIC_THRESHOLD` | `float` | No | `'0.7'` | Active learning Aleatoric uncertainty above this cutoff is treated as label noise, not a useful annotation target (detection/active_learning/annotation_queue.py). |
| `ACTIVE_LEARNING_MC_DROPOUT_PASSES` | `ACTIVE_LEARNING_MC_DROPOUT_PASSES` | `int` | No | `'20'` | Monte Carlo Dropout forward passes for epistemic uncertainty (Gal & Ghahramani, 2016). |
| `API_KEYS` | `API_KEYS` | `list[str]` | No | `''` | REST API auth / rate limiting (api/app.py) Comma-separated list of bcrypt-hashed API keys. |
| `API_RATE_LIMIT_RPM` | `API_RATE_LIMIT_RPM` | `int` | No | `'60'` | — |
| `CALIBRATION_SPLIT` | `CALIBRATION_SPLIT` | `float` | No | `'0.20'` | Model calibration holdout (training/train.py, training/calibration.py) |
| `CALIBRATION_RANDOM_SEED` | `CALIBRATION_RANDOM_SEED` | `int` | No | `'42'` | — |
| `CUSUM_TARGET_MEAN` | `CUSUM_TARGET_MEAN` | `float` | No | `'0.0'` | CUSUM change-point detection (monitoring/cusum_detector.py, Issue #289) |
| `CUSUM_ALLOWABLE_SLACK` | `CUSUM_ALLOWABLE_SLACK` | `float` | No | `'0.5'` | — |
| `CUSUM_DECISION_THRESHOLD` | `CUSUM_DECISION_THRESHOLD` | `float` | No | `'5.0'` | — |
| `HORIZON_FAILOVER_URLS` | `HORIZON_FAILOVER_URLS` | `list[str]` | No | `''` | Horizon multi-endpoint failover (ingestion/horizon_streamer.py) |
| `HORIZON_DEV_MODE` | `HORIZON_DEV_MODE` | `bool` | No | `''` | — |
| `HORIZON_FAILOVER_TIMEOUT_SECONDS` | `HORIZON_FAILOVER_TIMEOUT_SECONDS` | `float` | No | `'5.0'` | — |
| `HORIZON_HEALTH_CHECK_INTERVAL_SECONDS` | `HORIZON_HEALTH_CHECK_INTERVAL_SECONDS` | `float` | No | `'30.0'` | — |
| `KAFKA_BACKPRESSURE_HWM` | `KAFKA_BACKPRESSURE_HWM` | `int` | No | `'1000'` | Kafka consumer back-pressure + dead-letter routing (streaming/kafka_worker.py) |
| `KAFKA_BACKPRESSURE_LWM` | `KAFKA_BACKPRESSURE_LWM` | `int` | No | `'500'` | — |
| `KAFKA_MAX_RETRIES` | `KAFKA_MAX_RETRIES` | `int` | No | `'5'` | — |
| `KAFKA_DEAD_LETTER_TOPIC` | `KAFKA_DEAD_LETTER_TOPIC` | `str` | No | `KAFKA_DLQ_TOPIC` | — |
| `KAFKA_DEDUP_TTL_SECONDS` | `KAFKA_DEDUP_TTL_SECONDS` | `int` | No | `'3600'` | — |
| `KAFKA_TRANSACTIONAL_ID_PREFIX` | `KAFKA_TRANSACTIONAL_ID_PREFIX` | `str` | No | `'ledgerlens-producer'` | Producer transactional-ID prefix — deliberately not the hostname (ingestion/kafka_producer.py). |
| `KAFKA_TRANSACTION_TIMEOUT_MS` | `KAFKA_TRANSACTION_TIMEOUT_MS` | `int` | No | `'60000'` | — |
| `MODEL_WATERMARK_KEY` | `MODEL_WATERMARK_KEY` | `str` | No | `''` | Model watermarking for IP theft detection (detection/model_training.py) |
| `MODEL_WATERMARK_TRIGGER_COUNT` | `MODEL_WATERMARK_TRIGGER_COUNT` | `int` | No | `'10'` | — |
| `MODEL_WATERMARK_TRIGGER_PATH` | `MODEL_WATERMARK_TRIGGER_PATH` | `str` | No | `'data/watermark_triggers.json'` | — |
| `REPORT_NARRATIVE_FORMAT` | `REPORT_NARRATIVE_FORMAT` | `str` | No | `'plain_text'` | Forensic report narrative rendering (reporting/narrative_builder.py) |
| `FATF_EXPORT_THRESHOLD` | `FATF_EXPORT_THRESHOLD` | `float` | No | `'0.85'` | FATF regulatory export filter (reporting/fatf_exporter.py) — confirmed by tests/test_fatf_exporter.py, do not change without updating that test. |
| `RISK_PROP_CONVERGENCE_THRESHOLD` | `RISK_PROP_CONVERGENCE_THRESHOLD` | `float` | No | `'0.01'` | Weighted personalised PageRank convergence (detection/risk_propagation.py) |
| `TRADE_DEDUP_TTL_SECONDS` | `TRADE_DEDUP_TTL_SECONDS` | `int` | No | `str(24 * 3600)` | Trade ingestion dedup cache (ingestion/trade_deduplicator.py) |
| `TRADE_DEDUP_CACHE_KEY_PREFIX` | `TRADE_DEDUP_CACHE_KEY_PREFIX` | `str` | No | `'ledgerlens:trades:'` | — |
| `IDEMPOTENCY_TTL_HOURS` | `IDEMPOTENCY_TTL_HOURS` | `int` | No | `'48'` | Pipeline stage idempotency-key TTL (pipeline/idempotency.py); see docs/idempotency.md |
| `PARALLEL_EXECUTOR_BACKEND` | `PARALLEL_EXECUTOR_BACKEND` | `str` | No | `'process'` | Parallel processing controls — Issue #528 (ingestion/parallel_executor.py) Executor backend: "process" uses ProcessPoolExecutor (bypasses the GIL, best for CPU-heavy Benford / feature engineering work); "thread" uses ThreadPoolExecutor (lower overhead for I/O-bound tasks). |
| `PARALLEL_EXECUTOR_MAX_WORKERS` | `PARALLEL_EXECUTOR_MAX_WORKERS` | `int` | No | `str(max(1, (os.cpu_count() or 2) - 1))` | Maximum number of worker processes/threads.  Defaults to CPU count − 1 (≥ 1). |
| `PARALLEL_EXECUTOR_MAX_PENDING` | `PARALLEL_EXECUTOR_MAX_PENDING` | `int` | No | `'64'` | Maximum number of futures that may be in-flight simultaneously (back-pressure). 0 disables the limit. |
| `PARALLEL_EXECUTOR_CHUNK_SIZE` | `PARALLEL_EXECUTOR_CHUNK_SIZE` | `int` | No | `'16'` | Items per chunk submitted in map_chunks() to reduce process-spawn overhead. |
| `PARALLEL_EXECUTOR_TIMEOUT_SECONDS` | `PARALLEL_EXECUTOR_TIMEOUT_SECONDS` | `float` | No | `'300'` | Per-task timeout in seconds for map() calls.  0 means unlimited. |
| `DATASET_STORE_BACKEND` | `DATASET_STORE_BACKEND` | `str` | No | `'local'` | Dataset storage abstraction — Issue #529 (ingestion/dataset_store.py) Storage backend: "local" (default) writes Parquet/CSV/JSON to the local filesystem; "object" routes through fsspec (S3, GCS, Azure Blob, …). |
| `DATASET_STORE_BASE_PATH` | `DATASET_STORE_BASE_PATH` | `str` | No | `'./data'` | Root directory (local backend) or bucket prefix (object backend). |
| `DATASET_STORE_FORMAT` | `DATASET_STORE_FORMAT` | `str` | No | `'parquet'` | Default serialisation format for generated datasets. |
| `DATASET_STORE_OBJECT_STORE_URL` | `DATASET_STORE_OBJECT_STORE_URL` | `str` | No | `''` | Object store URI for the "object" backend, e.g. s3://my-bucket/ledgerlens. Leave empty to fall back to LocalDatasetStore even when backend="object". |
| `MODEL_INPUT_VALIDATOR_STRICTNESS` | `MODEL_INPUT_VALIDATOR_STRICTNESS` | `str` | No | `'warn'` | Model input validators — Issue #531 (detection/model_input_validator.py) Strictness level for schema and range validation at inference time. "raise"   — raise ValueError on the first issue found. "warn"    — log warnings and return cleaned data (default). "coerce"  — silently drop / fix violating rows, no logging. "ignore"  — pass data through unchanged (validation disabled). |
| `MODEL_INPUT_VALIDATOR_RANGES_PATH` | `MODEL_INPUT_VALIDATOR_RANGES_PATH` | `str` | No | `'data/feature_ranges.json'` | Path to feature_ranges.json, which supplies per-feature [min, max] bounds. |
| `MODEL_INPUT_VALIDATOR_NAN_STRATEGY` | `MODEL_INPUT_VALIDATOR_NAN_STRATEGY` | `str` | No | `'drop'` | Strategy for handling NaN / ±Inf values before model scoring. "drop"           — drop rows containing NaN / Inf (default). "impute_zero"    — replace NaN / Inf with 0. "impute_median"  — replace NaN / Inf with the column median. "raise"          — raise ValueError if any NaN / Inf is present. |
| `WATERMARK_STORE_PATH` | `WATERMARK_STORE_PATH` | `str` | No | `'data/watermarks.json'` | Watermark tracking for incremental ledger ingestion — Issue #526 (ingestion/watermark_tracker.py) Path to the JSON file that persists per-pair ingestion watermarks. The file is written atomically (tmpfile + rename) so it is crash-safe. |
| `WATERMARK_FLUSH_EVERY_N` | `WATERMARK_FLUSH_EVERY_N` | `int` | No | `'1'` | Automatically flush the watermark store to disk every N advance() calls. 1 (default) flushes after every trade page — safest, slight I/O overhead. 0 disables auto-flush; callers must invoke WatermarkTracker.flush() manually. |
