"""Central configuration loaded from environment variables / .env."""

import os

from dotenv import load_dotenv

load_dotenv()


def _parse_pairs(raw: str) -> list[tuple[str, str]]:
    pairs = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        code, _, issuer = entry.partition(":")
        pairs.append((code, issuer or "native"))
    return pairs


def _parse_int_list(raw: str) -> list[int]:
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


def _parse_pool_ids(raw: str) -> list[str]:
    import re

    pool_id_re = re.compile(r"^[0-9a-f]{64}$")
    ids = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if not pool_id_re.match(entry):
            raise ValueError(
                f"WATCHED_AMM_POOLS contains invalid pool ID {entry!r} — "
                "must be a 64-character lowercase hex string"
            )
        ids.append(entry)
    return ids


class Config:
    HORIZON_URL: str = os.getenv("HORIZON_URL", "https://horizon.stellar.org")
    STELLAR_NETWORK: str = os.getenv("STELLAR_NETWORK", "PUBLIC")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json").lower()

    WATCHED_ASSET_PAIRS: list[tuple[str, str]] = _parse_pairs(
        os.getenv(
            "WATCHED_ASSET_PAIRS", "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
        )
    )

    WATCHED_AMM_POOLS: list[str] = _parse_pool_ids(os.getenv("WATCHED_AMM_POOLS", ""))

    BENFORD_WINDOWS_HOURS: list[int] = _parse_int_list(
        os.getenv("BENFORD_WINDOWS_HOURS", "1,4,24,168,720")
    )

    ASSET_BENFORD_WINDOWS: dict[str, list[int]] = {}

    CROSS_PAIR_SYNCHRONY_WINDOW_SECONDS: int = int(
        os.getenv("CROSS_PAIR_SYNCHRONY_WINDOW_SECONDS", "30")
    )

    # Silence window for correlated alert deduplication (alerts/deduplicator.py).
    ALERT_DEDUP_WINDOW_SECONDS: int = min(
        300, max(5, int(os.getenv("ALERT_DEDUP_WINDOW_SECONDS", "60")))
    )

    RISK_SCORE_FLAG_THRESHOLD: int = int(os.getenv("RISK_SCORE_FLAG_THRESHOLD", "70"))
    # Set to a non-zero integer to pin the alert threshold and disable the RL agent.
    # E.g. THRESHOLD_RL_PINNED=75 → agent is bypassed, threshold is fixed at 75.
    THRESHOLD_RL_PINNED: int = int(os.getenv("THRESHOLD_RL_PINNED", "0"))

    RISK_SCORE_DB_URL: str = os.getenv("RISK_SCORE_DB_URL", "sqlite:///ledgerlens.db")

    # Database connection pooling
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))

    MODEL_DIR: str = os.getenv("MODEL_DIR", "./models")
    BATCH_SCORER_WORKERS: int = int(os.getenv("BATCH_SCORER_WORKERS", 10))

    # ledgerlens-score Soroban contract
    SOROBAN_RPC_URL: str = os.getenv("SOROBAN_RPC_URL", "https://soroban-testnet.stellar.org")
    LEDGERLENS_CONTRACT_ID: str = os.getenv("LEDGERLENS_CONTRACT_ID", "")
    LEDGERLENS_SUBMITTER_SECRET: str = os.getenv("LEDGERLENS_SUBMITTER_SECRET", "")

    # Solana RPC endpoint for cross-chain resolution
    SOLANA_RPC_URL: str = os.getenv(
        "SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"
    )

    MIN_TRADES_FOR_SCORING: int = int(os.getenv("MIN_TRADES_FOR_SCORING", "20"))
    LIST_RELOAD_INTERVAL_SECONDS: int = int(os.getenv("LIST_RELOAD_INTERVAL_SECONDS", "60"))

    # Live feature drift monitoring (Population Stability Index)
    DRIFT_WINDOW_SIZE: int = int(os.getenv("DRIFT_WINDOW_SIZE", "1000"))
    # Fire an alert when any feature PSI exceeds this value.
    DRIFT_PSI_THRESHOLD: float = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2"))

    # Sliding window covariance shift detection (MMD)
    DRIFT_REFERENCE_WINDOW_HOURS: int = int(os.getenv("DRIFT_REFERENCE_WINDOW_HOURS", "168"))
    DRIFT_TEST_WINDOW_HOURS: int = int(os.getenv("DRIFT_TEST_WINDOW_HOURS", "1"))
    DRIFT_CHECK_INTERVAL_MINUTES: int = int(os.getenv("DRIFT_CHECK_INTERVAL_MINUTES", "30"))

    # Forensic reporting
    REPORT_CONCURRENCY: int = int(os.getenv("REPORT_CONCURRENCY", "4"))
    # SHAP interaction values are O(n * d^2) — disable by default.
    SHAP_INTERACTIONS_ENABLED: bool = os.getenv("SHAP_INTERACTIONS_ENABLED", "false").lower() == "true"

    # Wallet funding graph — multi-hop traversal + wash-trading ring detection
    WALLET_GRAPH_MAX_DEPTH: int = int(os.getenv("WALLET_GRAPH_MAX_DEPTH", "4"))
    WASH_RING_MIN_SIZE: int = int(os.getenv("WASH_RING_MIN_SIZE", "3"))
    WASH_RING_RESOLUTION: float = float(os.getenv("WASH_RING_RESOLUTION", "1.0"))
    # Fixed seed keeps Louvain community detection deterministic in CI.
    WASH_RING_LOUVAIN_SEED: int = int(os.getenv("WASH_RING_LOUVAIN_SEED", "42"))

    # Distributed rate limiting for Horizon REST calls (ingestion/rate_limiter.py)
    HORIZON_MAX_RPS: int = min(100, int(os.getenv("HORIZON_MAX_RPS", "80")))
    HORIZON_MAX_RETRIES: int = int(os.getenv("HORIZON_MAX_RETRIES", "5"))
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Real-time streaming / alerting
    # STREAMING_BACKEND selects the ingestion transport:
    #   "sse"   — existing thread-per-pair Horizon SSE pipeline (default, no Kafka)
    #   "kafka" — Apache Kafka producer/consumer distributed pipeline
    STREAMING_BACKEND: str = os.getenv("STREAMING_BACKEND", "sse")

    # Kafka — credentials are read from env vars only, never committed.
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_SASL_USERNAME: str | None = os.getenv("KAFKA_SASL_USERNAME")
    KAFKA_SASL_PASSWORD: str | None = os.getenv("KAFKA_SASL_PASSWORD")
    KAFKA_CONSUMER_GROUP: str = os.getenv("KAFKA_CONSUMER_GROUP", "ledgerlens-scorer")
    KAFKA_TOPIC_PREFIX: str = os.getenv("KAFKA_TOPIC_PREFIX", "ledgerlens.trades")
    KAFKA_DLQ_TOPIC: str = os.getenv("KAFKA_DLQ_TOPIC", "ledgerlens.trades.dlq")
    # Regex subscription (librdkafka treats a leading '^' as a pattern). Picks up
    # new per-pair topics without a consumer restart; the DLQ topic is skipped
    # in the worker so failed messages are never auto-replayed.
    KAFKA_TOPIC_PATTERN: str = os.getenv("KAFKA_TOPIC_PATTERN", "^ledgerlens\\.trades\\..*")
    KAFKA_LAG_ALERT_THRESHOLD: int = int(os.getenv("KAFKA_LAG_ALERT_THRESHOLD", "500"))
    KAFKA_METRICS_PORT: int = int(os.getenv("KAFKA_METRICS_PORT", "9100"))
    TRADE_AVRO_SCHEMA_PATH: str = os.getenv("TRADE_AVRO_SCHEMA_PATH", "data/trade_avro_schema.json")

    # Account metadata streaming join (streaming/account_metadata_stream.py,
    # streaming/pipeline.py MetadataJoinState)
    # METADATA_TOPIC: dedicated Kafka topic for account metadata update events.
    METADATA_TOPIC: str = os.getenv("METADATA_TOPIC", "ledgerlens.account_metadata")
    # METADATA_JOIN_WINDOW_SECONDS: how long (seconds) a metadata update enriches
    # incoming trade events.  After this window the update is considered stale and
    # must be refreshed by a subsequent Horizon effect.  Default: 3600 (1 hour).
    METADATA_JOIN_WINDOW_SECONDS: int = int(os.getenv("METADATA_JOIN_WINDOW_SECONDS", "3600"))
    # METADATA_ACTIVE_WALLET_TTL_SECONDS: wallets that have had no trade activity
    # for this many seconds are pruned from join state to keep memory bounded.
    # Default: 86400 (24 hours) — matches the requirement spec.
    METADATA_ACTIVE_WALLET_TTL_SECONDS: int = int(
        os.getenv("METADATA_ACTIVE_WALLET_TTL_SECONDS", "86400")
    )

    ALERT_CHANNEL: str = os.getenv("ALERT_CHANNEL", "stdout")
    ALERT_WEBHOOK_URL: str | None = os.getenv("ALERT_WEBHOOK_URL")
    ALERT_COOLDOWN_SECONDS: int = int(os.getenv("ALERT_COOLDOWN_SECONDS", "3600"))
    ALERT_DEAD_LETTER_PATH: str = os.getenv("ALERT_DEAD_LETTER_PATH", "alerts_dlq.ndjson")
    WS_PORT: int = int(os.getenv("WS_PORT", "8765"))
    WS_BIND_HOST: str = os.getenv("WS_BIND_HOST", "127.0.0.1")
    WS_ALLOW_EXTERNAL: bool = os.getenv("WS_ALLOW_EXTERNAL", "") == "1"

    # WebSocket pub/sub server (streaming/ws_server.py)
    JWT_PUBLIC_KEY_PATH: str = os.getenv("JWT_PUBLIC_KEY_PATH", "./jwt_public_key.pem")
    WS_MAX_CLIENTS: int = int(os.getenv("WS_MAX_CLIENTS", "200"))
    WS_CLIENT_QUEUE_DEPTH: int = int(os.getenv("WS_CLIENT_QUEUE_DEPTH", "100"))
    WS_REPLAY_BUFFER_SIZE: int = int(os.getenv("WS_REPLAY_BUFFER_SIZE", "1000"))
    WS_RATE_LIMIT_MSGS_PER_SECOND: int = int(os.getenv("WS_RATE_LIMIT_MSGS_PER_SECOND", "100"))

    # WebSocket abuse detection (issue #223)
    WS_ABUSE_MAX_REQUESTS_PER_MINUTE: int = int(os.getenv("WS_ABUSE_MAX_REQUESTS_PER_MINUTE", "300"))
    WS_ABUSE_MAX_DISTINCT_WALLETS: int = int(os.getenv("WS_ABUSE_MAX_DISTINCT_WALLETS", "50"))
    WS_ABUSE_WALLET_WINDOW_SECONDS: int = int(os.getenv("WS_ABUSE_WALLET_WINDOW_SECONDS", "60"))
    WS_ABUSE_BLOCK_DURATION_SECONDS: int = int(os.getenv("WS_ABUSE_BLOCK_DURATION_SECONDS", "300"))

    # Differentially private neural training (DP-SGD via Opacus)
    DP_TARGET_EPSILON: float = float(os.getenv("DP_TARGET_EPSILON", "8.0"))
    DP_TARGET_DELTA: float = float(os.getenv("DP_TARGET_DELTA", "1e-5"))
    DP_MAX_GRAD_NORM: float = float(os.getenv("DP_MAX_GRAD_NORM", "1.0"))
    DP_EPOCHS: int = int(os.getenv("DP_EPOCHS", "50"))

    # Adversarial training augmentation
    ADVERSARIAL_AUG_RATIO: float = float(os.getenv("ADVERSARIAL_AUG_RATIO", "0.0"))

    # Model integrity & BFT voting
    MODEL_SIGNING_PRIVATE_KEY_PATH: str = os.getenv("MODEL_SIGNING_PRIVATE_KEY_PATH", "")
    TRUSTED_SIGNING_KEY_FINGERPRINT: str = os.getenv("TRUSTED_SIGNING_KEY_FINGERPRINT", "")
    AUDIT_LOG_PATH: str = os.getenv("AUDIT_LOG_PATH", "data/audit_trail.ndjson")
    AUDIT_VERIFY_PUBLIC_KEY_PATH: str = os.getenv("AUDIT_VERIFY_PUBLIC_KEY_PATH", "")
    BFT_SCORE_DIVERGENCE_THRESHOLD: int = int(os.getenv("BFT_SCORE_DIVERGENCE_THRESHOLD", "30"))
    BFT_MIN_CONSENSUS: int = int(os.getenv("BFT_MIN_CONSENSUS", "2"))
    POISON_LABEL_RATIO_THRESHOLD: float = float(os.getenv("POISON_LABEL_RATIO_THRESHOLD", "0.15"))
    ZERO_SHOT_WEIGHT: float = float(os.getenv("ZERO_SHOT_WEIGHT", "0.0"))
    ZERO_SHOT_MIN_LABELLED_EXAMPLES: int = int(os.getenv("ZERO_SHOT_MIN_LABELLED_EXAMPLES", "20"))
    BENFORD_CI_ENABLED: bool = os.getenv("BENFORD_CI_ENABLED", "false").lower() == "true"
    BRIDGE_ROUNDTRIP_WINDOW_HOURS: int = int(os.getenv("BRIDGE_ROUNDTRIP_WINDOW_HOURS", "72"))

    # Differential privacy for SHAP explanations (model inversion defence)
    DP_EPSILON: float = float(os.getenv("DP_EPSILON", "1.0"))
    DP_DELTA: float = float(os.getenv("DP_DELTA", "1e-5"))
    DP_RENYI_QUERY_THRESHOLD: int = int(os.getenv("DP_RENYI_QUERY_THRESHOLD", "100"))
    DP_RENYI_NOISE_MULTIPLIER: float = float(os.getenv("DP_RENYI_NOISE_MULTIPLIER", "3.0"))
    DP_DEFAULT_SENSITIVITY: float = float(os.getenv("DP_DEFAULT_SENSITIVITY", "0.05"))
    SHAP_SENSITIVITY_PATH: str = os.getenv("SHAP_SENSITIVITY_PATH", "models/shap_sensitivity.json")

    # Model inversion attack defence
    MODEL_INVERSION_QUERY_LIMIT: int = int(os.getenv("MODEL_INVERSION_QUERY_LIMIT", "100"))
    MODEL_INVERSION_DP_EPSILON: float = float(os.getenv("MODEL_INVERSION_DP_EPSILON", "1.0"))
    SCORE_ROUNDING_GRANULARITY: int = int(os.getenv("SCORE_ROUNDING_GRANULARITY", "1"))

    # Graph Neural Network encoder (detection/gnn_encoder.py)
    GNN_EMBEDDING_DIM: int = int(os.getenv("GNN_EMBEDDING_DIM", "32"))
    GNN_HIDDEN_DIM: int = int(os.getenv("GNN_HIDDEN_DIM", "64"))

    # Feature selection
    FEATURE_SELECTION_ENABLED: bool = os.getenv("FEATURE_SELECTION_ENABLED", "").lower() in ("1", "true", "yes")
    FEATURE_SELECTION_PATH: str = os.getenv("FEATURE_SELECTION_PATH", "models/selected_features.json")

    # Annotation integrity
    ANNOTATION_HMAC_SECRET: str = os.getenv("ANNOTATION_HMAC_SECRET", "")

    # Active learning
    AL_QUERY_STRATEGY: str = os.getenv("AL_QUERY_STRATEGY", "committee_disagreement")
    AL_BATCH_SIZE: int = int(os.getenv("AL_BATCH_SIZE", "20"))
    AL_RETRAIN_THRESHOLD: int = int(os.getenv("AL_RETRAIN_THRESHOLD", "50"))
    AL_ROLLBACK_AUC_DROP: float = float(os.getenv("AL_ROLLBACK_AUC_DROP", "0.01"))
    AL_QUEUE_PATH: str = os.getenv("AL_QUEUE_PATH", "data/annotation_queue.json")

    # Wash Trade Simulation Engine
    GAN_ROUNDS: int = int(os.getenv("GAN_ROUNDS", "5"))
    GAN_PLATEAU_THRESHOLD: float = float(os.getenv("GAN_PLATEAU_THRESHOLD", "0.005"))
    SIMULATOR_N_WALLETS: int = int(os.getenv("SIMULATOR_N_WALLETS", "50"))
    SIMULATOR_TRADES_PER_WALLET: int = int(os.getenv("SIMULATOR_TRADES_PER_WALLET", "100"))
    GNN_EMBEDDING_DIM: int = int(os.getenv("GNN_EMBEDDING_DIM", "32"))

    # Graph neural network encoder
    GNN_EMBEDDING_DIM: int = int(os.getenv("GNN_EMBEDDING_DIM", "32"))
    GNN_HIDDEN_DIM: int = int(os.getenv("GNN_HIDDEN_DIM", "64"))
    GNN_NUM_LAYERS: int = int(os.getenv("GNN_NUM_LAYERS", "2"))

    # Dynamic ensemble weight adjustment (#268)
    ENSEMBLE_WEIGHT_SMOOTHING_ALPHA: float = float(os.getenv("ENSEMBLE_WEIGHT_SMOOTHING_ALPHA", "0.1"))
    ENSEMBLE_SYSTEMIC_FP_THRESHOLD: float = float(os.getenv("ENSEMBLE_SYSTEMIC_FP_THRESHOLD", "0.5"))

    # GNN DiffPool cluster scoring (#269)
    GNN_DIFFPOOL_CLUSTERS: int = int(os.getenv("GNN_DIFFPOOL_CLUSTERS", "10"))

    # Async federated learning (#270)
    FEDERATED_ASYNC_TRIGGER_N: int = int(os.getenv("FEDERATED_ASYNC_TRIGGER_N", "3"))
    FEDERATED_ASYNC_TRIGGER_SECONDS: int = int(os.getenv("FEDERATED_ASYNC_TRIGGER_SECONDS", "300"))
    FEDERATED_MAX_STALENESS: int = int(os.getenv("FEDERATED_MAX_STALENESS", "5"))

    # Label quality estimation (#271)
    LABEL_QUALITY_NOISE_THRESHOLD: float = float(os.getenv("LABEL_QUALITY_NOISE_THRESHOLD", "0.1"))
    ANNOTATOR_NOISE_RATE_ALERT_THRESHOLD: float = float(os.getenv("ANNOTATOR_NOISE_RATE_ALERT_THRESHOLD", "0.2"))

    # Incremental LightGBM training
    # Maximum number of labelled events held in the RecentDataBuffer for
    # incremental training.  When the buffer reaches this size, incremental
    # training is triggered automatically.
    INCREMENTAL_BUFFER_SIZE: int = int(os.getenv("INCREMENTAL_BUFFER_SIZE", "10000"))
    # Number of new trees appended per incremental pass.
    INCREMENTAL_N_NEW_TREES: int = int(os.getenv("INCREMENTAL_N_NEW_TREES", "100"))
    # Maximum consecutive incremental training rounds before forcing a full
    # retrain to prevent catastrophic forgetting.
    MAX_INCREMENTAL_ROUNDS: int = int(os.getenv("MAX_INCREMENTAL_ROUNDS", "10"))

    @classmethod
    def validate(cls, require_onchain: bool = False):
        errors = []

        if not cls.WATCHED_ASSET_PAIRS:
            errors.append("WATCHED_ASSET_PAIRS is not set.")

        if not cls.RISK_SCORE_DB_URL.strip():
            errors.append("RISK_SCORE_DB_URL is not set.")

        if not cls.MODEL_DIR.strip():
            errors.append("MODEL_DIR is not set.")

        if cls.DP_AGGREGATOR_EPSILON <= 0:
            errors.append("DP_AGGREGATOR_EPSILON must be > 0.")

        if not (0 < cls.DP_AGGREGATOR_DELTA < 0.5):
            errors.append("DP_AGGREGATOR_DELTA must be in (0, 0.5).")

        if require_onchain:
            if not cls.LEDGERLENS_CONTRACT_ID.strip():
                errors.append("LEDGERLENS_CONTRACT_ID is not set.")

            if not cls.LEDGERLENS_SUBMITTER_SECRET.strip():
                errors.append("LEDGERLENS_SUBMITTER_SECRET is not set.")

        if errors:
            raise OSError("LedgerLens configuration errors:\n- " + "\n- ".join(errors))

    @classmethod
    def load_asset_benford_windows(cls):
        import glob
        import json
        cls.ASSET_BENFORD_WINDOWS = {}
        model_dir = cls.MODEL_DIR or "./models"
        pattern = os.path.join(model_dir, "*_benford_windows.json")
        for filepath in glob.glob(pattern):
            filename = os.path.basename(filepath)
            asset_key = filename[:-len("_benford_windows.json")]
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "asset" in data and "windows" in data:
                        cls.ASSET_BENFORD_WINDOWS[data["asset"]] = [int(w) for w in data["windows"]]
                    elif isinstance(data, list):
                        if "_" in asset_key:
                            parts = asset_key.split("_", 1)
                            asset_name = f"{parts[0]}:{parts[1]}"
                        else:
                            asset_name = asset_key
                        cls.ASSET_BENFORD_WINDOWS[asset_name] = [int(w) for w in data]
                        cls.ASSET_BENFORD_WINDOWS[asset_key] = [int(w) for w in data]
            except Exception:
                pass


config = Config()
Config.load_asset_benford_windows()

# Validate security parameters
if config.MODEL_INVERSION_QUERY_LIMIT <= 0:
    raise ValueError(f"MODEL_INVERSION_QUERY_LIMIT must be > 0, got {config.MODEL_INVERSION_QUERY_LIMIT}")
if config.MODEL_INVERSION_DP_EPSILON <= 0:
    raise ValueError(f"MODEL_INVERSION_DP_EPSILON must be > 0, got {config.MODEL_INVERSION_DP_EPSILON}")
if config.SCORE_ROUNDING_GRANULARITY <= 0:
    raise ValueError(f"SCORE_ROUNDING_GRANULARITY must be > 0, got {config.SCORE_ROUNDING_GRANULARITY}")
