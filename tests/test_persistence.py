from detection.persistence import get_engine, get_session_factory
from detection.risk_score_store import RiskScoreStore


def make_store() -> RiskScoreStore:
    engine = get_engine("sqlite:///:memory:")
    return RiskScoreStore(get_session_factory(engine))


def test_upsert_creates_record():
    store = make_store()
    record = store.upsert(
        "GABC",
        "USDC:issuer/XLM:native",
        {"score": 80, "benford_flag": True, "ml_flag": True, "confidence": 80},
    )
    assert record.wallet == "GABC"
    assert record.score == 80


def test_upsert_updates_existing_record():
    store = make_store()
    pair = "USDC:issuer/XLM:native"
    store.upsert(
        "GABC", pair, {"score": 50, "benford_flag": False, "ml_flag": False, "confidence": 50}
    )
    store.upsert(
        "GABC", pair, {"score": 90, "benford_flag": True, "ml_flag": True, "confidence": 90}
    )

    record = store.get("GABC", pair)
    assert record.score == 90
    assert record.benford_flag is True


def test_to_risk_score_shape():
    store = make_store()
    pair = "USDC:issuer/XLM:native"
    store.upsert(
        "GABC", pair, {"score": 75, "benford_flag": True, "ml_flag": False, "confidence": 60}
    )

    risk_score = store.get("GABC", pair).to_risk_score()
    assert set(risk_score) == {"score", "benford_flag", "ml_flag", "timestamp", "confidence"}
    assert risk_score["score"] == 75


def test_list_flagged_filters_by_threshold():
    store = make_store()
    pair = "USDC:issuer/XLM:native"
    store.upsert(
        "GABC", pair, {"score": 80, "benford_flag": True, "ml_flag": True, "confidence": 80}
    )
    store.upsert(
        "GXYZ", pair, {"score": 20, "benford_flag": False, "ml_flag": False, "confidence": 20}
    )

    flagged = store.list_flagged(70)
    assert [r.wallet for r in flagged] == ["GABC"]
