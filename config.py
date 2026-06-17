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


class Config:
    HORIZON_URL: str = os.getenv("HORIZON_URL", "https://horizon.stellar.org")
    STELLAR_NETWORK: str = os.getenv("STELLAR_NETWORK", "PUBLIC")

    WATCHED_ASSET_PAIRS: list[tuple[str, str]] = _parse_pairs(os.getenv("WATCHED_ASSET_PAIRS", ""))

    BENFORD_WINDOWS_HOURS: list[int] = _parse_int_list(
        os.getenv("BENFORD_WINDOWS_HOURS", "1,4,24,168,720")
    )

    RISK_SCORE_FLAG_THRESHOLD: int = int(os.getenv("RISK_SCORE_FLAG_THRESHOLD", "70"))

    RISK_SCORE_DB_URL: str = os.getenv("RISK_SCORE_DB_URL", "sqlite:///ledgerlens.db")

    MODEL_DIR: str = os.getenv("MODEL_DIR", "./models")

    # ledgerlens-score Soroban contract
    SOROBAN_RPC_URL: str = os.getenv("SOROBAN_RPC_URL", "https://soroban-testnet.stellar.org")
    LEDGERLENS_CONTRACT_ID: str = os.getenv("LEDGERLENS_CONTRACT_ID", "")
    LEDGERLENS_SUBMITTER_SECRET: str = os.getenv("LEDGERLENS_SUBMITTER_SECRET", "")

    # Real-time streaming / alerting
    ALERT_CHANNEL: str = os.getenv("ALERT_CHANNEL", "stdout")
    ALERT_WEBHOOK_URL: str | None = os.getenv("ALERT_WEBHOOK_URL")
    ALERT_COOLDOWN_SECONDS: int = int(os.getenv("ALERT_COOLDOWN_SECONDS", "3600"))
    WS_PORT: int = int(os.getenv("WS_PORT", "8765"))
    WS_BIND_HOST: str = os.getenv("WS_BIND_HOST", "127.0.0.1")
    WS_ALLOW_EXTERNAL: bool = os.getenv("WS_ALLOW_EXTERNAL", "") == "1"

    def validate(self, require_onchain: bool = True) -> None:
        """Raise ValueError if required config is missing."""
        if not self.WATCHED_ASSET_PAIRS:
            raise ValueError("WATCHED_ASSET_PAIRS is not configured")
        if require_onchain and not self.LEDGERLENS_CONTRACT_ID:
            raise ValueError("LEDGERLENS_CONTRACT_ID is not configured")


config = Config()
