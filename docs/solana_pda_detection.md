# Solana PDA detection in identity resolution (#881)

## Problem

A Solana program-derived address (PDA) is a 32-byte value derived from seeds and a
program ID that deliberately lies **off** the Ed25519 curve, so no private key exists
for it. Only its owning program can sign for it. Token vaults, associated token
accounts (ATAs), AMM authorities and bridge custody accounts are all PDAs or
program-owned accounts. If the resolver treats them as user wallets, one
bridge-custody or pool-vault address links thousands of unrelated users into one
"identity".

## Classification

`detection/cross_chain/solana_resolver.py::classify_solana_address` returns one of:

| kind | meaning | user wallet? |
|---|---|---|
| `wallet` | on-curve key, not owned by a known program | yes |
| `pda` | off-curve **and** re-derived from a supplied `(program_id, seeds)` hint | no |
| `pda_unverified` | off-curve, no hint reproduced it. Still cannot be a keypair | no |
| `program` | the address is itself a known program ID | no |
| `program_owned` | on-curve keypair account whose `owner` (from `getAccountInfo`) is a known program, e.g. a non-associated SPL token account | no |
| `invalid` | not a base58 32-byte key | — |

Derivation verification is a byte-exact port of `Pubkey::create_program_address` /
`find_program_address`: `sha256(seeds ‖ bump ‖ program_id ‖ "ProgramDerivedAddress")`,
with the on-curve check mirroring `curve25519-dalek` decompression. The fixture
addresses were cross-checked against `solders.Pubkey.find_program_address`.

## Effect on the identity graph

- `IdentityGraph.add_edge` returns `None` and writes nothing when either endpoint is a
  non-user Solana address. Pass `metadata={"owner_program_ids": {addr: owner}}` so it
  also catches on-curve program-owned accounts.
- `resolve_stellar_to_solana` runs deposits through `filter_pda_links`.
- To include PDAs anyway (e.g. for forensic views), set `SOLANA_INCLUDE_PDA_EDGES=true`
  or pass `IdentityGraph(include_solana_pdas=True)`. `filter_pda_links(include_pdas=True)`
  keeps them but tags each link with `solana_address_kind` and `program_id`.

**Before/after** (`tests/test_solana_pda.py::test_identity_graph_before_after_pda_exclusion`):
with the curated fixtures, one Stellar wallet bridged to 3 real wallets, 3 ATAs and 4
program vaults resolves to **10** Solana "wallets" with PDA edges enabled and to the
**3** real wallets with them disabled. The 7 PDA-as-wallet edges are excluded.

## Covered programs and PDA patterns

Program IDs (`KNOWN_PROGRAM_IDS`):

| program | ID |
|---|---|
| SPL Token | `TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA` |
| SPL Token-2022 | `TokenzQdBNbLqP5VEhdkAS6EAFqZ1ALBG8PS3Kd1pJG` |
| Associated Token Account | `ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL` |
| Wormhole Core Bridge | `worm2ZoG2kUd4vFXhvjh93UUH596ayRfgQ2MgjNMTth` |
| Wormhole Token Bridge | `wormDTUJ6AWPNvk59vGQbDvGJmqbDTdgWgAqcLBCgUb` |
| Raydium AMM v4 | `675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8` |
| Orca Whirlpool | `whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc` |
| Marinade | `MarBmsSgKXdrN1egZf5sqe1TMai9K1rChYNDJgjq7aD` |

Seed layouts (`KNOWN_PDA_SEED_PATTERNS`):

| pattern | program | seeds |
|---|---|---|
| `associated_token_account` | ATA | `[wallet, token_program, mint]` |
| `wormhole_token_bridge_custody` | Wormhole Token Bridge | `[mint]` |
| `wormhole_token_bridge_authority_signer` | Wormhole Token Bridge | `["authority_signer"]` |
| `wormhole_token_bridge_custody_signer` | Wormhole Token Bridge | `["custody_signer"]` |
| `raydium_amm_authority` | Raydium AMM v4 | `["amm authority"]` |
| `orca_whirlpool` | Orca Whirlpool | `["whirlpool", config, mint_a, mint_b, tick_spacing u16 LE]` |
| `marinade_reserve` | Marinade | `[state, "reserve"]` |

## Adding a program

1. Add its ID to `KNOWN_PROGRAM_IDS`. This alone makes the program ID, and on-curve
   accounts reported as owned by it, non-user.
2. If its PDAs use fixed seeds, add the layout to `KNOWN_PDA_SEED_PATTERNS` and a fixture
   entry (address derived with `solders` or the program's SDK) to
   `tests/fixtures/solana_pda_addresses.json` under `program_vaults`.

## Limitations

- Any off-curve address is excluded even without a derivation hint (`pda_unverified`).
  That is safe because such an address cannot be a keypair wallet, but it cannot be
  attributed to a program without seeds or the account owner.
- On-curve program-owned accounts are only caught when the caller supplies the owner
  program, which requires an RPC `getAccountInfo` lookup.
