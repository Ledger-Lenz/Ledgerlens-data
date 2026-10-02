# Bridge proof attestation (#884)

Bridge detection (`detection/cross_chain/bridge_detector.py`) is heuristic by default:
it links a Stellar wallet to an EVM or Solana address found in a bridge memo. When the
bridge publishes a verifiable proof of the lock/mint event, `detect_bridge_links` now
checks that proof via `integrations/bridge_attestation.py` and uses it as a
high-confidence signal.

## Attestation levels and confidence

| level | what was verified | link `confidence` |
|---|---|---|
| `zk_proof` | a succinct validity proof against a pinned verifying key (trust-minimised) | 1.00 |
| `guardian_signatures` | a committee attestation checked offline, e.g. a Wormhole VAA with ≥ 13/19 valid guardian signatures | 0.98 |
| `heuristic` | no proof, an invalid proof, or a proof for a different transfer | 0.90 |

Each link carries `attestation`, `confidence` and `attestation_details` (VAA id, digest,
signature count, attested amount and recipient, or the reason a proof was rejected).
Downstream, `IdentityGraph.get_connected_component` reports each linked node's
`link_confidence`: the strongest path, as the product of edge confidences.
`propagate_risk_scores` imports linked-chain risk as `risk × link_confidence` (via
`resolve_weighted_risk_scores_bulk`), so risk that crosses a verified link counts for
more than risk that crosses a heuristic one.

**Binding rule:** a proof upgrades a link only if it verifies **and** attests a transfer
to the memo's linked address. For EVM addresses any EVM destination chain counts, since
the same key is valid on all of them. A valid VAA attached to an unrelated transfer is
ignored and the rejection is recorded in `attestation_details.proof_rejected`.

## Supported bridges

| bridge | attestation status | proof format | notes |
|---|---|---|---|
| `wormhole` | `guardian_signatures`: **verified end-to-end** | VAA v1, secp256k1 signatures from the 19-member guardian set, quorum 13 | Verified against 5 recorded **mainnet** Token Bridge VAAs (`tests/fixtures/wormhole_vaas_recorded.json`), with guardian set 7 pinned from the Ethereum core contract `getGuardianSet(7)` (`data/bridge_attestation/wormhole_guardian_sets.json`). Our VAA digests match wormholescan's independently computed digests. |
| `allbridge` | `heuristic` | none published | Stellar ↔ EVM/Solana; validators sign off-chain, and no public per-transfer proof exists. |
| `stellar_anchor_sep6` | `heuristic` | none | Custodial SEP-6 anchors; memo correlation only. |

### zk-proof bridges: status

Wormhole is **not** a zk bridge: its VAAs are committee signatures, which the
`guardian_signatures` level reflects. The bridges that do publish per-transfer validity
proofs (e.g. Polyhedra zkBridge, SP1-based light clients) were not found serving the
Stellar/Solana ↔ EVM routes this repository ingests, and no recorded proofs from one
are available here. So **no zk bridge is integrated yet**. Re-check that when adding
routes. The `zk_proof` level is plumbed end-to-end
through `register_zk_verifier(bridge, verifier, proof_format)`. Tests exercise it with a
stub verifier, and it gets confidence 1.0.

To add a zk bridge:

1. Implement `verifier(proof: dict) -> bool` that checks the proof against a **pinned**
   verifying key and returns False on any malformed input. Exceptions are also treated
   as failures.
2. Make sure the verified proof exposes the recipient, then extend
   `attested_recipient_matches` for its format.
3. Call `register_zk_verifier(...)` at import time. Add a row above, and add recorded
   testnet or mainnet proofs as fixtures with a test that verifies them.

## Operations

- **Guardian set rotation:** Wormhole VAAs name their guardian set index. When a new set
  is activated, add it to `wormhole_guardian_sets.json` by calling `getGuardianSet(i)` on
  the core contract (`0x98f3c9e6E3fAce36bAAd05FE09d375Ef1464288B`). Until then, VAAs
  signed by the new set are rejected as `unknown guardian set` and the links fall back
  to heuristic. The system fails closed.
- **Supplying proofs:** attach `bridge_proof = {"bridge": "wormhole", "vaa": "<base64>"}`
  to the transaction record passed to `detect_bridge_links`. The VAA can be fetched from
  the wormholescan API by `chain/emitter/sequence`.
- No network access happens at detection time; verification is fully offline.

## Limitations

- For `TransferWithPayload` (payload 3), the attested recipient is usually a relayer or
  contract, not the end user, so such links rarely bind and stay heuristic.
- For Solana destinations, the attested recipient is the destination **token account**,
  usually an ATA PDA (see `docs/solana_pda_detection.md`), not the wallet. Binding to the
  wallet requires deriving the ATA from the wallet and mint.
- Verification is pure Python (`keccak256` here plus `ecdsa` public-key recovery): about
  150 ms per VAA, dominated by 13 signature recoveries. That is fine for per-link
  verification but not for bulk re-verification. Swap in `coincurve` for that.
