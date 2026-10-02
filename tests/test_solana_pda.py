"""Tests for Solana program-derived address (PDA) detection (Issue #881)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from detection.cross_chain.identity_graph import IdentityGraph
from detection.cross_chain.solana_resolver import (
    ASSOCIATED_TOKEN_PROGRAM_ID,
    KNOWN_PDA_SEED_PATTERNS,
    KNOWN_PROGRAM_IDS,
    TOKEN_PROGRAM_ID,
    PDADerivationHint,
    base58_decode,
    base58_encode,
    classify_solana_address,
    filter_pda_links,
    find_program_address,
    get_associated_token_address,
    is_on_ed25519_curve,
)
from detection.persistence import Base, get_engine, get_session_factory

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "solana_pda_addresses.json").read_text()
)
STELLAR = "GBRPYHIL2CI3FNQ4BXLFMNDLFJUNPU2HY3ZMFSHONUCEOASW7QC7OX2H"


@pytest.fixture
def session_factory(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 'pda.db'}")
    Base.metadata.create_all(engine)
    return get_session_factory(engine)


def test_base58_round_trip():
    for addr in FIXTURES["wallets"] + [TOKEN_PROGRAM_ID, "11111111111111111111111111111111"]:
        raw = base58_decode(addr)
        assert len(raw) == 32
        assert base58_encode(raw) == addr


@pytest.mark.parametrize("wallet", FIXTURES["wallets"])
def test_user_wallets_are_on_curve_and_classified_as_wallet(wallet):
    assert is_on_ed25519_curve(wallet)
    result = classify_solana_address(wallet)
    assert result.kind == "wallet"
    assert result.is_user_wallet


@pytest.mark.parametrize("ata", FIXTURES["associated_token_accounts"])
def test_associated_token_account_derivation_matches_reference(ata):
    assert get_associated_token_address(ata["wallet"], ata["mint"]) == ata["address"]
    assert find_program_address(
        [ata["wallet"], TOKEN_PROGRAM_ID, ata["mint"]], ASSOCIATED_TOKEN_PROGRAM_ID
    ) == (ata["address"], ata["bump"])


@pytest.mark.parametrize("ata", FIXTURES["associated_token_accounts"])
def test_associated_token_account_classified_as_verified_pda(ata):
    hint = PDADerivationHint(
        program_id=ASSOCIATED_TOKEN_PROGRAM_ID,
        seeds=[ata["wallet"], TOKEN_PROGRAM_ID, ata["mint"]],
        pattern="associated_token_account",
    )
    result = classify_solana_address(ata["address"], derivation_hints=[hint])
    assert result.kind == "pda"
    assert result.program_name == "associated_token_account"
    assert result.bump == ata["bump"]
    assert not result.is_user_wallet


@pytest.mark.parametrize("vault", FIXTURES["program_vaults"])
def test_defi_and_bridge_vault_pdas(vault):
    assert vault["pattern"] in KNOWN_PDA_SEED_PATTERNS
    hint = PDADerivationHint(vault["program_id"], vault["seeds"], vault["pattern"])
    result = classify_solana_address(vault["address"], derivation_hints=[hint])
    assert result.kind == "pda"
    assert result.program_id == vault["program_id"]
    # Even without a hint, an off-curve address can never be a keypair wallet.
    assert classify_solana_address(vault["address"]).kind == "pda_unverified"


def test_wrong_derivation_hint_does_not_verify():
    vault = FIXTURES["program_vaults"][0]
    hint = PDADerivationHint(vault["program_id"], ["some_other_seed"])
    assert classify_solana_address(vault["address"], [hint]).kind == "pda_unverified"


def test_program_owned_keypair_token_account():
    acct = FIXTURES["program_owned_token_account"]
    result = classify_solana_address(acct["address"], owner_program_id=acct["owner_program_id"])
    assert result.kind == "program_owned"
    assert result.program_name == "spl_token"


def test_known_program_ids_classified_as_programs():
    for program_id, name in KNOWN_PROGRAM_IDS.items():
        result = classify_solana_address(program_id)
        assert result.kind == "program"
        assert result.program_name == name


def test_invalid_addresses():
    assert classify_solana_address("not-a-solana-address").kind == "invalid"


def test_filter_pda_links_excludes_by_default_and_tags_when_included():
    ata = FIXTURES["associated_token_accounts"][0]
    links = [
        {"stellar_address": STELLAR, "solana_address": ata["wallet"]},
        {"stellar_address": STELLAR, "solana_address": ata["address"]},
    ]
    kept = filter_pda_links(links, include_pdas=False)
    assert [link["solana_address"] for link in kept] == [ata["wallet"]]
    assert kept[0]["solana_address_kind"] == "wallet"

    tagged = filter_pda_links(links, include_pdas=True)
    assert [link["solana_address_kind"] for link in tagged] == ["wallet", "pda_unverified"]


def test_identity_graph_before_after_pda_exclusion(session_factory):
    """Before: PDA-as-wallet edges pollute the component. After: only real wallets."""
    wallets = FIXTURES["wallets"]
    pdas = [a["address"] for a in FIXTURES["associated_token_accounts"]] + [
        v["address"] for v in FIXTURES["program_vaults"]
    ]

    before = IdentityGraph(session_factory, include_solana_pdas=True)
    for addr in wallets + pdas:
        before.add_edge(STELLAR, addr, "bridge")
    before_sol = {n["address"] for n in before.get_connected_component(STELLAR)["sol"]}
    assert before_sol == set(wallets) | set(pdas)

    engine = get_engine(f"sqlite:///{Path(session_factory.kw['bind'].url.database).parent}/b.db")
    Base.metadata.create_all(engine)
    after = IdentityGraph(get_session_factory(engine), include_solana_pdas=False)
    results = [after.add_edge(STELLAR, addr, "bridge") for addr in wallets + pdas]
    assert results[len(wallets) :] == [None] * len(pdas)
    after_sol = {n["address"] for n in after.get_connected_component(STELLAR)["sol"]}
    assert after_sol == set(wallets)


def test_identity_graph_uses_owner_metadata(session_factory):
    acct = FIXTURES["program_owned_token_account"]
    graph = IdentityGraph(session_factory, include_solana_pdas=False)
    edge = graph.add_edge(
        STELLAR,
        acct["address"],
        "bridge",
        metadata={"owner_program_ids": {acct["address"]: acct["owner_program_id"]}},
    )
    assert edge is None
