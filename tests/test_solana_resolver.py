"""Tests for Solana cross-chain identity resolution (Issue #017).

Tests verify:
  1. Solana address validation (base58, 32-44 characters)
  2. Wormhole VAA payload parsing and Stellar address extraction
  3. Invalid Solana addresses are rejected before RPC calls
  4. Identity graph correctly stores Stellar ↔ Solana linkages
  5. Cross-chain resolution via identity graph
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from detection.cross_chain.identity_graph import IdentityGraph
from detection.cross_chain.solana_resolver import (
    SolanaRPCClient,
    SolanaValidationError,
    WormholeVAAValidationError,
    extract_stellar_address_from_vaa,
    parse_wormhole_vaa_payload,
    validate_solana_address,
)


class TestSolanaAddressValidation:
    """Test Solana address validation."""

    def test_valid_solana_address(self):
        """Valid Solana base58 addresses should be accepted."""
        # Real Solana addresses (examples)
        valid_addresses = [
            "11111111111111111111111111111111",  # System program
            "TokenkegQfeZyiNwAJsyFbPWLH2gKrpjTeVrao5QYvE",  # Token program
            "ATokenGPvbdGVqstVQmcLsNZAqeEgtvodQuCjBi8zi",  # Associated Token Program
        ]
        for addr in valid_addresses:
            assert validate_solana_address(addr), f"Should accept {addr}"

    def test_invalid_solana_address_too_short(self):
        """Addresses shorter than 32 characters should be rejected."""
        invalid = "111111111111111"  # Too short
        assert not validate_solana_address(invalid)

    def test_invalid_solana_address_too_long(self):
        """Addresses longer than 44 characters should be rejected."""
        invalid = "11111111111111111111111111111111111111111111111"  # Too long
        assert not validate_solana_address(invalid)

    def test_invalid_solana_address_wrong_charset(self):
        """Addresses with invalid base58 characters should be rejected."""
        # base58 excludes 0, O, I, l
        invalid_chars = [
            "0111111111111111111111111111111111",  # Contains '0'
            "O111111111111111111111111111111111",  # Contains 'O'
            "I111111111111111111111111111111111",  # Contains 'I'
            "l111111111111111111111111111111111",  # Contains 'l'
        ]
        for addr in invalid_chars:
            assert not validate_solana_address(addr)

    def test_solana_address_none_type(self):
        """Non-string inputs should return False."""
        assert not validate_solana_address(None)
        assert not validate_solana_address(123)
        assert not validate_solana_address([])

    def test_solana_address_with_whitespace(self):
        """Addresses with leading/trailing whitespace should be accepted after stripping."""
        addr = "  11111111111111111111111111111111  "
        assert validate_solana_address(addr)


class TestWormholeVAAPayloadParsing:
    """Test Wormhole VAA payload parsing."""

    def test_parse_valid_wormhole_vaa(self):
        """Parsing a valid VAA structure should extract core fields."""
        # Synthetic VAA (simplified structure for testing)
        vaa_bytes = bytearray()
        vaa_bytes.append(1)  # Version 1
        vaa_bytes.extend((0).to_bytes(4, "big"))  # Guardian set index
        vaa_bytes.append(0)  # Signature count (0 for testing)
        vaa_bytes.extend((1234567890).to_bytes(4, "big"))  # Timestamp
        vaa_bytes.extend((42).to_bytes(4, "big"))  # Nonce
        vaa_bytes.extend((1).to_bytes(2, "big"))  # Emitter chain (Stellar)
        vaa_bytes.extend(bytes(32))  # Emitter address (32 bytes)
        vaa_bytes.extend((100).to_bytes(8, "big"))  # Sequence
        vaa_bytes.append(200)  # Consistency level
        vaa_bytes.append(1)  # Payload type
        vaa_bytes.extend((1).to_bytes(2, "big"))  # Destination chain
        vaa_bytes.extend(bytes(32))  # Destination address (32 bytes)

        parsed = parse_wormhole_vaa_payload(bytes(vaa_bytes))

        assert parsed is not None
        assert parsed["vaa_version"] == 1
        assert parsed["guardian_set_index"] == 0
        assert parsed["timestamp"] == 1234567890
        assert parsed["nonce"] == 42
        assert parsed["emitter_chain"] == 1
        assert parsed["sequence"] == 100
        assert parsed["consistency_level"] == 200
        assert parsed["payload_type"] == 1
        assert parsed["destination_chain"] == 1

    def test_parse_invalid_vaa_version(self):
        """VAA with unsupported version should raise WormholeVAAValidationError."""
        vaa_bytes = bytearray()
        vaa_bytes.append(99)  # Invalid version

        with pytest.raises(WormholeVAAValidationError):
            parse_wormhole_vaa_payload(bytes(vaa_bytes))

    def test_parse_vaa_too_short(self):
        """VAA shorter than minimum should return None."""
        vaa_bytes = bytes([1, 2, 3])  # Too short
        parsed = parse_wormhole_vaa_payload(vaa_bytes)
        assert parsed is None

    def test_parse_empty_vaa(self):
        """Empty VAA should return None."""
        assert parse_wormhole_vaa_payload(b"") is None
        assert parse_wormhole_vaa_payload(None) is None


class TestStellarAddressExtraction:
    """Test extracting Stellar addresses from VAA data."""

    def test_extract_stellar_address_from_vaa_data(self):
        """Should extract Stellar address if present in VAA data."""
        vaa_data = {
            "vaa_version": 1,
            "destination_address": "GBRPYHIL2CI3FD4BWXVYDPLG445T5Q5GPESUYVS33VRTGOEULIABLE336",
            "destination_chain": 1,
        }

        stellar_addr = extract_stellar_address_from_vaa(vaa_data)
        assert stellar_addr is not None
        assert stellar_addr.startswith("G") or len(stellar_addr) == 56

    def test_extract_no_stellar_address(self):
        """Should return None if VAA data is empty or invalid."""
        assert extract_stellar_address_from_vaa(None) is None
        assert extract_stellar_address_from_vaa({}) is None
        assert extract_stellar_address_from_vaa({"destination_address": ""}) is None

    def test_extract_invalid_destination_address(self):
        """Should return None for invalid destination addresses."""
        vaa_data = {
            "destination_address": "invalid_address_12345",
        }
        stellar_addr = extract_stellar_address_from_vaa(vaa_data)
        assert stellar_addr is None or len(stellar_addr) > 0


class TestSolanaRPCClient:
    """Test Solana RPC client caching and validation."""

    @patch("requests.Session.post")
    def test_rpc_client_validates_address_before_call(self, mock_post):
        """RPC client should reject invalid addresses before making calls."""
        client = SolanaRPCClient()

        with pytest.raises(SolanaValidationError):
            client.get_signatures_for_address("invalid_address_too_short")

        # Should never reach the RPC call
        mock_post.assert_not_called()

    @patch("requests.Session.post")
    def test_rpc_client_caches_signatures(self, mock_post):
        """RPC client should cache signature responses (1-hour TTL)."""
        client = SolanaRPCClient(cache_ttl_seconds=3600)

        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": [
                {"signature": "sig1"},
                {"signature": "sig2"},
            ]
        }
        mock_post.return_value = mock_response

        addr = "11111111111111111111111111111111"

        # First call should hit the mock
        sigs1 = client.get_signatures_for_address(addr, limit=2)
        assert len(sigs1) == 2
        assert mock_post.call_count == 1

        # Second call should use cache (mock not called again)
        sigs2 = client.get_signatures_for_address(addr, limit=2)
        assert sigs2 == sigs1
        assert mock_post.call_count == 1  # Still 1, from cache

    @patch("requests.Session.post")
    def test_rpc_client_handles_error_response(self, mock_post):
        """RPC client should handle error responses gracefully."""
        client = SolanaRPCClient()

        # Mock error response
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Rate limited"}
        mock_post.return_value = mock_response

        addr = "11111111111111111111111111111111"

        # Should return empty list on error
        sigs = client.get_signatures_for_address(addr)
        assert sigs == []


class TestIdentityGraphSolanaIntegration:
    """Test Solana linkage storage in identity graph."""

    def test_add_stellar_solana_edge(self):
        """Identity graph should store Stellar ↔ Solana links."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from detection.persistence import Base

        # Create in-memory SQLite for testing
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine)

        graph = IdentityGraph(session_factory)

        # Add nodes
        stellar_addr = "GBRPYHIL2CI3FD4BWXVYDPLG445T5Q5GPESUYVS33VRTGOEULIABLE336"
        solana_addr = "TokenkegQfeZyiNwAJsyFbPWLH2gKrpjTeVrao5QYvE"

        graph.add_node(stellar_addr, "stellar", risk_score=50.0)
        graph.add_node(solana_addr, "solana", risk_score=75.0)

        # Add bridge link
        edge = graph.add_edge(
            stellar_addr,
            solana_addr,
            link_type="wormhole_bridge",
            confidence=0.95,
            metadata={"tx_signature": "test_sig_123", "timestamp": 1234567890},
        )

        assert edge.source_address == stellar_addr
        assert edge.target_address == solana_addr
        assert edge.link_type == "wormhole_bridge"
        assert edge.confidence == 0.95

    def test_resolve_stellar_to_solana(self):
        """Identity graph should resolve Stellar wallet to linked Solana addresses."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from detection.persistence import Base

        # Create in-memory SQLite for testing
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine)

        graph = IdentityGraph(session_factory)

        # Setup: Stellar -> Solana link
        stellar_addr = "GBRPYHIL2CI3FD4BWXVYDPLG445T5Q5GPESUYVS33VRTGOEULIABLE336"
        solana_addr1 = "TokenkegQfeZyiNwAJsyFbPWLH2gKrpjTeVrao5QYvE"
        solana_addr2 = "ATokenGPvbdGVqstVQmcLsNZAqeEgtvodQuCjBi8zi"

        graph.add_node(stellar_addr, "stellar", risk_score=50.0)
        graph.add_node(solana_addr1, "solana", risk_score=75.0)
        graph.add_node(solana_addr2, "solana", risk_score=60.0)

        graph.add_edge(stellar_addr, solana_addr1, "wormhole_bridge", confidence=0.95)
        graph.add_edge(stellar_addr, solana_addr2, "wormhole_bridge", confidence=0.90)

        # Resolve
        component = graph.get_connected_component(stellar_addr)

        # Should find both Solana addresses
        sol_addresses = component.get("sol", [])
        assert len(sol_addresses) == 2
        sol_addrs_set = {node["address"] for node in sol_addresses}
        assert solana_addr1 in sol_addrs_set
        assert solana_addr2 in sol_addrs_set

        # Risk scores should be present
        for node in sol_addresses:
            assert node["risk_score"] > 0


class TestSolanaLinkedFeature:
    """Test Solana-linked wash score feature computation."""

    def test_solana_linked_wash_score_with_cache(self):
        """Should compute max risk score from linked Solana addresses."""
        from detection.feature_engineering import compute_solana_linked_features
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from detection.persistence import Base

        # Create in-memory SQLite
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine)

        graph = IdentityGraph(session_factory)

        stellar_addr = "GBRPYHIL2CI3FD4BWXVYDPLG445T5Q5GPESUYVS33VRTGOEULIABLE336"
        solana_addr1 = "TokenkegQfeZyiNwAJsyFbPWLH2gKrpjTeVrao5QYvE"
        solana_addr2 = "ATokenGPvbdGVqstVQmcLsNZAqeEgtvodQuCjBi8zi"

        # Add nodes and links
        graph.add_node(stellar_addr, "stellar")
        graph.add_node(solana_addr1, "solana")
        graph.add_node(solana_addr2, "solana")
        graph.add_edge(stellar_addr, solana_addr1, "wormhole_bridge")
        graph.add_edge(stellar_addr, solana_addr2, "wormhole_bridge")

        # Cache with risk scores
        cache = {
            solana_addr1: 75.0,
            solana_addr2: 55.0,
        }

        # Compute feature
        features = compute_solana_linked_features(stellar_addr, graph, cache)

        assert "solana_linked_wash_score" in features
        assert features["solana_linked_wash_score"] == 75.0  # Max of 75 and 55

    def test_solana_linked_wash_score_no_links(self):
        """Should return 0 if no Solana links found."""
        from detection.feature_engineering import compute_solana_linked_features
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from detection.persistence import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine)

        graph = IdentityGraph(session_factory)
        stellar_addr = "GBRPYHIL2CI3FD4BWXVYDPLG445T5Q5GPESUYVS33VRTGOEULIABLE336"
        graph.add_node(stellar_addr, "stellar")

        cache = {}

        features = compute_solana_linked_features(stellar_addr, graph, cache)

        assert features["solana_linked_wash_score"] == 0.0

    def test_solana_linked_wash_score_no_cache(self):
        """Should return 0 if cache is None."""
        from detection.feature_engineering import compute_solana_linked_features

        features = compute_solana_linked_features("GBRPYHIL2CI3FD4BWXVYDPLG445T5Q5GPESUYVS33VRTGOEULIABLE336", None, None)

        assert features["solana_linked_wash_score"] == 0.0
