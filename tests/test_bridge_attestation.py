"""Tests for bridge proof attestation (Issue #884).

The Wormhole cases verify *real* mainnet VAAs recorded from wormholescan
(tests/fixtures/wormhole_vaas_recorded.json) against guardian set 7 as pinned
from the Ethereum core contract (data/bridge_attestation/). No network access.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from detection.cross_chain.bridge_detector import BridgeDetector
from detection.cross_chain.identity_graph import IdentityGraph
from detection.cross_chain.resolver import resolve_weighted_risk_scores
from detection.persistence import Base, get_engine, get_session_factory
from integrations import bridge_attestation as ba
from integrations.bridge_attestation import (
    ATTESTATION_CONFIDENCE,
    ATTESTATION_GUARDIAN,
    ATTESTATION_HEURISTIC,
    ATTESTATION_ZK,
    SUPPORTED_BRIDGES,
    BridgeAttestationVerifier,
    keccak256,
    load_guardian_sets,
    parse_token_bridge_transfer,
    parse_vaa,
    register_zk_verifier,
    verify_vaa,
)

VAAS = json.loads((Path(__file__).parent / "fixtures" / "wormhole_vaas_recorded.json").read_text())[
    "vaas"
]
STELLAR = "GBRPYHIL2CI3FNQ4BXLFMNDLFJUNPU2HY3ZMFSHONUCEOASW7QC7OX2H"


@pytest.fixture(scope="module")
def guardian_sets():
    return load_guardian_sets()


@pytest.fixture(scope="module")
def verifier(guardian_sets):
    return BridgeAttestationVerifier(guardian_sets)


def _recipient_evm(vaa_b64: str) -> str:
    transfer = parse_token_bridge_transfer(parse_vaa(vaa_b64).payload)
    return "0x" + transfer.to_address[12:].hex()


def _tamper(vaa_b64: str, offset_from_end: int) -> str:
    raw = bytearray(base64.b64decode(vaa_b64))
    raw[-offset_from_end] ^= 0x01
    return base64.b64encode(bytes(raw)).decode()


# -- primitives ---------------------------------------------------------------


def test_keccak256_known_vectors():
    assert keccak256(b"").hex() == (
        "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    )
    assert keccak256(b"transfer(address,uint256)")[:4].hex() == "a9059cbb"
    # multi-block input (> 136-byte rate)
    assert keccak256(b"a" * 200) != keccak256(b"a" * 201)


def test_pinned_guardian_set_shape(guardian_sets):
    assert len(guardian_sets[7]) == 19
    assert all(len(addr) == 20 for addr in guardian_sets[7])


# -- real recorded VAAs ----------------------------------------------------------


@pytest.mark.parametrize("record", VAAS, ids=[v["id"] for v in VAAS])
def test_recorded_mainnet_vaa_verifies(record, guardian_sets):
    vaa = parse_vaa(record["vaa"])
    assert vaa.id == record["id"]
    # Digest independently computed by wormholescan must match ours.
    assert vaa.digest.hex() == record["digest"]
    check = verify_vaa(vaa, guardian_sets)
    assert check.valid, check.reason
    assert check.valid_signatures >= check.quorum == 13


def test_tampered_payload_is_rejected(verifier):
    result = verifier.verify({"bridge": "wormhole", "vaa": _tamper(VAAS[0]["vaa"], 1)})
    assert result.level == ATTESTATION_HEURISTIC
    assert not result.verified


def test_tampered_signature_is_rejected(guardian_sets):
    vaa = parse_vaa(VAAS[0]["vaa"])
    idx, sig = vaa.signatures[0]
    bad = bytes([sig[0] ^ 1]) + sig[1:]
    forged = ba.WormholeVAA(**{**vaa.__dict__, "signatures": [(idx, bad), *vaa.signatures[1:]]})
    assert not verify_vaa(forged, guardian_sets).valid


def test_below_quorum_and_duplicate_indices_rejected(guardian_sets):
    vaa = parse_vaa(VAAS[0]["vaa"])
    short = ba.WormholeVAA(**{**vaa.__dict__, "signatures": vaa.signatures[:12]})
    assert verify_vaa(short, guardian_sets).reason == "insufficient signatures"
    dup = ba.WormholeVAA(
        **{**vaa.__dict__, "signatures": [vaa.signatures[0], *vaa.signatures[:-1]]}
    )
    assert verify_vaa(dup, guardian_sets).reason == "signature indices not ascending"


def test_unknown_guardian_set_rejected():
    vaa = parse_vaa(VAAS[0]["vaa"])
    assert not verify_vaa(vaa, {3: []}).valid


def test_malformed_vaa_is_heuristic(verifier):
    result = verifier.verify({"bridge": "wormhole", "vaa": base64.b64encode(b"\x01\x00").decode()})
    assert result.level == ATTESTATION_HEURISTIC


# -- bridge detector integration ---------------------------------------------


def _memo_tx(address: str, proof: dict | None = None) -> dict:
    tx = {"id": "tx1", "source_account": STELLAR, "memo_type": "text", "memo": address}
    if proof is not None:
        tx["bridge_proof"] = proof
    return tx


def test_detector_upgrades_link_with_matching_verified_proof(verifier):
    recipient = _recipient_evm(VAAS[0]["vaa"])
    detector = BridgeDetector(attestation_verifier=verifier)
    [link] = detector.detect_bridge_links(
        [_memo_tx(recipient, {"bridge": "wormhole", "vaa": VAAS[0]["vaa"]})]
    )
    assert link["attestation"] == ATTESTATION_GUARDIAN
    assert link["confidence"] == ATTESTATION_CONFIDENCE[ATTESTATION_GUARDIAN]
    assert link["attestation_details"]["vaa_id"] == VAAS[0]["id"]


def test_detector_heuristic_without_proof(verifier):
    detector = BridgeDetector(attestation_verifier=verifier)
    [link] = detector.detect_bridge_links([_memo_tx(_recipient_evm(VAAS[0]["vaa"]))])
    assert link["attestation"] == ATTESTATION_HEURISTIC
    assert link["confidence"] == ATTESTATION_CONFIDENCE[ATTESTATION_HEURISTIC]


def test_detector_ignores_valid_proof_for_a_different_recipient(verifier):
    other = _recipient_evm(VAAS[1]["vaa"])
    detector = BridgeDetector(attestation_verifier=verifier)
    [link] = detector.detect_bridge_links(
        [_memo_tx(other, {"bridge": "wormhole", "vaa": VAAS[0]["vaa"]})]
    )
    assert link["attestation"] == ATTESTATION_HEURISTIC
    assert "does not attest" in link["attestation_details"]["proof_rejected"]


def test_detector_rejects_tampered_proof(verifier):
    recipient = _recipient_evm(VAAS[0]["vaa"])
    detector = BridgeDetector(attestation_verifier=verifier)
    [link] = detector.detect_bridge_links(
        [_memo_tx(recipient, {"bridge": "wormhole", "vaa": _tamper(VAAS[0]["vaa"], 1)})]
    )
    assert link["attestation"] == ATTESTATION_HEURISTIC


# -- zk verifier registry --------------------------------------------------------


@pytest.fixture
def fake_zk_bridge(monkeypatch):
    monkeypatch.setattr(ba, "_ZK_VERIFIERS", {})
    monkeypatch.setattr(ba, "SUPPORTED_BRIDGES", dict(SUPPORTED_BRIDGES))
    register_zk_verifier("testzk", lambda proof: proof.get("ok") is True, "test")
    return "testzk"


def test_registered_zk_verifier_gives_highest_level(fake_zk_bridge):
    v = BridgeAttestationVerifier({})
    assert v.verify({"bridge": fake_zk_bridge, "ok": True}).level == ATTESTATION_ZK
    assert ba.SUPPORTED_BRIDGES[fake_zk_bridge].attestation == ATTESTATION_ZK
    assert v.verify({"bridge": fake_zk_bridge, "ok": False}).level == ATTESTATION_HEURISTIC


def test_raising_zk_verifier_falls_back_to_heuristic(monkeypatch):
    monkeypatch.setattr(ba, "_ZK_VERIFIERS", {})
    monkeypatch.setattr(ba, "SUPPORTED_BRIDGES", dict(SUPPORTED_BRIDGES))

    def boom(proof):
        raise RuntimeError("bad proof encoding")

    register_zk_verifier("boomzk", boom, "test")
    assert BridgeAttestationVerifier({}).verify({"bridge": "boomzk"}).level == (
        ATTESTATION_HEURISTIC
    )


def test_confidence_ordering():
    c = ATTESTATION_CONFIDENCE
    assert c[ATTESTATION_ZK] > c[ATTESTATION_GUARDIAN] > c[ATTESTATION_HEURISTIC]


def test_supported_bridge_registry_documented():
    doc = (Path(__file__).parent.parent / "docs" / "bridge_attestation.md").read_text()
    for name in SUPPORTED_BRIDGES:
        assert f"`{name}`" in doc


# -- downstream confidence weighting -------------------------------------------


def test_verified_link_imports_more_risk_than_heuristic(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'att.db'}"
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)
    graph = IdentityGraph(get_session_factory(engine))
    graph.add_node("0xaaaa", "ethereum", risk_score=80.0)
    graph.add_node("0xbbbb", "ethereum", risk_score=80.0)
    verified = ATTESTATION_CONFIDENCE[ATTESTATION_GUARDIAN]
    heuristic = ATTESTATION_CONFIDENCE[ATTESTATION_HEURISTIC]
    graph.add_edge("GVERIFIED", "0xaaaa", "bridge", confidence=verified)
    graph.add_edge("GHEURISTIC", "0xbbbb", "bridge", confidence=heuristic)

    assert resolve_weighted_risk_scores("GVERIFIED", db_url)["0xaaaa"] == pytest.approx(
        80.0 * verified
    )
    assert resolve_weighted_risk_scores("GHEURISTIC", db_url)["0xbbbb"] == pytest.approx(
        80.0 * heuristic
    )
