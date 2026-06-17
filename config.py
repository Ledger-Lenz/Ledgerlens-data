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
            raise EnvironmentError(
                "LedgerLens configuration errors:\n- "
                + "\n- ".join(errors)
            )
config = Config()
