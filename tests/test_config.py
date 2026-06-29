import pytest

from config import Config


def test_validate_passes_with_valid_minimal_config(monkeypatch):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", [("USDC", "native")])
    monkeypatch.setattr(Config, "RISK_SCORE_DB_URL", "sqlite:///test.db")
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "LEDGERLENS_CONTRACT_ID", "contract-id")
    monkeypatch.setattr(Config, "LEDGERLENS_SUBMITTER_SECRET", "secret")

    Config.validate(require_onchain=False)


def test_validate_raises_when_watched_asset_pairs_empty(monkeypatch):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", [])
    monkeypatch.setattr(Config, "RISK_SCORE_DB_URL", "sqlite:///test.db")
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")

    with pytest.raises(OSError) as exc:
        Config.validate(require_onchain=False)

    assert "WATCHED_ASSET_PAIRS is not set." in str(exc.value)


def test_validate_raises_when_risk_score_db_url_empty(monkeypatch):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", [("USDC", "native")])
    monkeypatch.setattr(Config, "RISK_SCORE_DB_URL", "")
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")

    with pytest.raises(OSError) as exc:
        Config.validate(require_onchain=False)

    assert "RISK_SCORE_DB_URL is not set." in str(exc.value)


def test_validate_raises_when_model_dir_empty(monkeypatch):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", [("USDC", "native")])
    monkeypatch.setattr(Config, "RISK_SCORE_DB_URL", "sqlite:///test.db")
    monkeypatch.setattr(Config, "MODEL_DIR", "")

    with pytest.raises(OSError) as exc:
        Config.validate(require_onchain=False)

    assert "MODEL_DIR is not set." in str(exc.value)


def test_validate_require_onchain_missing_contract_id_raises(monkeypatch):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", [("USDC", "native")])
    monkeypatch.setattr(Config, "RISK_SCORE_DB_URL", "sqlite:///test.db")
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "LEDGERLENS_CONTRACT_ID", "")
    monkeypatch.setattr(Config, "LEDGERLENS_SUBMITTER_SECRET", "secret")

    with pytest.raises(OSError) as exc:
        Config.validate(require_onchain=True)

    assert "LEDGERLENS_CONTRACT_ID is not set." in str(exc.value)


def test_validate_require_onchain_missing_submitter_secret_raises(monkeypatch):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", [("USDC", "native")])
    monkeypatch.setattr(Config, "RISK_SCORE_DB_URL", "sqlite:///test.db")
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "LEDGERLENS_CONTRACT_ID", "contract-id")
    monkeypatch.setattr(Config, "LEDGERLENS_SUBMITTER_SECRET", "")

    with pytest.raises(OSError) as exc:
        Config.validate(require_onchain=True)

    assert "LEDGERLENS_SUBMITTER_SECRET is not set." in str(exc.value)


def test_validate_reports_multiple_errors_in_one_exception(monkeypatch):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", [])
    monkeypatch.setattr(Config, "RISK_SCORE_DB_URL", "")
    monkeypatch.setattr(Config, "MODEL_DIR", "")
    monkeypatch.setattr(Config, "LEDGERLENS_CONTRACT_ID", "")
    monkeypatch.setattr(Config, "LEDGERLENS_SUBMITTER_SECRET", "")

    with pytest.raises(OSError) as exc:
        Config.validate(require_onchain=True)

    msg = str(exc.value)
    # Ensure all errors are present (living documentation of required fields)
    assert "WATCHED_ASSET_PAIRS is not set." in msg
    assert "RISK_SCORE_DB_URL is not set." in msg
    assert "MODEL_DIR is not set." in msg
    assert "LEDGERLENS_CONTRACT_ID is not set." in msg
    assert "LEDGERLENS_SUBMITTER_SECRET is not set." in msg

