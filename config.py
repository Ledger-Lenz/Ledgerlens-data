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

    WATCHED_ASSET_PAIRS: list[tuple[str, str]] = _parse_pairs(
        os.getenv(
            "WATCHED_ASSET_PAIRS", "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
        )
    )

    WATCHED_AMM_POOLS: list[str] = _parse_pool_ids(os.getenv("WATCHED_AMM_POOLS", ""))

    BENFORD_WINDOWS_HOURS: list[int] = _parse_int_list(
        os.getenv("BENFORD_WINDOWS_HOURS", "1,4,24,168,720")
    )

    CROSS_PAIR_SYNCHRONY_WINDOW_SECONDS: int = int(
        os.getenv("CROSS_PAIR_SYNCHRONY_WINDOW_SECONDS", "30")
    )

    RISK_SCORE_FLAG_THRESHOLD: int = int(os.getenv("RISK_SCORE_FLAG_THRESHOLD", "70"))

    RISK_SCORE_DB_URL: str = os.getenv("RISK_SCORE_DB_URL", "sqlite:///ledgerlens.db")

    MODEL_DIR: str = os.getenv("MODEL_DIR", "./models")

    # ledgerlens-score Soroban contract
    SOROBAN_RPC_URL: str = os.getenv("SOROBAN_RPC_URL", "https://soroban-testnet.stellar.org")
    LEDGERLENS_CONTRACT_ID: str = os.getenv("LEDGERLENS_CONTRACT_ID", "")
    LEDGERLENS_SUBMITTER_SECRET: str = os.getenv("LEDGERLENS_SUBMITTER_SECRET", "")

    MIN_TRADES_FOR_SCORING: int = int(os.getenv("MIN_TRADES_FOR_SCORING", "20"))

    # Forensic reporting
    REPORT_CONCURRENCY: int = int(os.getenv("REPORT_CONCURRENCY", "4"))

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

    ALERT_CHANNEL: str = os.getenv("ALERT_CHANNEL", "stdout")
    ALERT_WEBHOOK_URL: str | None = os.getenv("ALERT_WEBHOOK_URL")
    ALERT_COOLDOWN_SECONDS: int = int(os.getenv("ALERT_COOLDOWN_SECONDS", "3600"))
    WS_PORT: int = int(os.getenv("WS_PORT", "8765"))
    WS_BIND_HOST: str = os.getenv("WS_BIND_HOST", "127.0.0.1")
    WS_ALLOW_EXTERNAL: bool = os.getenv("WS_ALLOW_EXTERNAL", "") == "1"

    # WebSocket pub/sub server (streaming/ws_server.py)
    JWT_PUBLIC_KEY_PATH: str = os.getenv("JWT_PUBLIC_KEY_PATH", "./jwt_public_key.pem")
    WS_MAX_CLIENTS: int = int(os.getenv("WS_MAX_CLIENTS", "200"))
    WS_CLIENT_QUEUE_DEPTH: int = int(os.getenv("WS_CLIENT_QUEUE_DEPTH", "100"))
    WS_REPLAY_BUFFER_SIZE: int = int(os.getenv("WS_REPLAY_BUFFER_SIZE", "1000"))
    WS_RATE_LIMIT_MSGS_PER_SECOND: int = int(os.getenv("WS_RATE_LIMIT_MSGS_PER_SECOND", "100"))

    # Adversarial training augmentation
    ADVERSARIAL_AUG_RATIO: float = float(os.getenv("ADVERSARIAL_AUG_RATIO", "0.0"))

    # Model integrity & BFT voting
    MODEL_SIGNING_PRIVATE_KEY_PATH: str = os.getenv("MODEL_SIGNING_PRIVATE_KEY_PATH", "")
    TRUSTED_SIGNING_KEY_FINGERPRINT: str = os.getenv("TRUSTED_SIGNING_KEY_FINGERPRINT", "")
    BFT_SCORE_DIVERGENCE_THRESHOLD: int = int(os.getenv("BFT_SCORE_DIVERGENCE_THRESHOLD", "30"))
    BFT_MIN_CONSENSUS: int = int(os.getenv("BFT_MIN_CONSENSUS", "2"))
    POISON_LABEL_RATIO_THRESHOLD: float = float(os.getenv("POISON_LABEL_RATIO_THRESHOLD", "0.15"))

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

    @classmethod
    def validate(cls, require_onchain: bool = False):
        errors = []

        if not cls.WATCHED_ASSET_PAIRS:
            errors.append("WATCHED_ASSET_PAIRS is not set.")

        if not cls.RISK_SCORE_DB_URL.strip():
            errors.append("RISK_SCORE_DB_URL is not set.")

        if not cls.MODEL_DIR.strip():
            errors.append("MODEL_DIR is not set.")

        if require_onchain:
            if not cls.LEDGERLENS_CONTRACT_ID.strip():
                errors.append("LEDGERLENS_CONTRACT_ID is not set.")

            if not cls.LEDGERLENS_SUBMITTER_SECRET.strip():
                errors.append("LEDGERLENS_SUBMITTER_SECRET is not set.")

        if errors:
            raise OSError("LedgerLens configuration errors:\n- " + "\n- ".join(errors))


config = Config()
