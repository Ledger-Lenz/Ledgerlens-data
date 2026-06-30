# MPC-based Collaborative Fraud Detection

## Overview

Multiple Stellar DEX ecosystem participants — exchanges, wallet providers,
custodians — each observe a partial view of wash-trade activity. Individually,
each participant's signal is incomplete: a wash-trading ring will spread its
activity across many venues to stay below any single platform's detection
threshold. Combining signals across participants would dramatically improve
detection accuracy, but participants cannot share raw trade data due to:

- **Competitive sensitivity**: trade flow data reveals customer bases, fee
  revenue, and liquidity strategies.
- **Regulatory constraints**: customer trading data is subject to data-residency
  and privacy regulations (GDPR, CCPA, FinCEN data minimisation rules).

**Secure Multi-Party Computation (MPC)** allows a group of parties to jointly
compute a function over their private inputs without any party revealing its
inputs to the others. `detection/mpc_aggregator.py` implements a 2- and 3-party
MPC protocol using the [mpyc](https://github.com/lschoe/mpyc) library to
compute the **mean and variance of wash-trade risk scores** across participants.

---

## Protocol description

### Inputs and outputs

| Entity | Input | Output |
|---|---|---|
| Each participant | A list of per-wallet risk scores (floats, 0–100) | Aggregate mean, variance, total count |
| No participant | Raw scores of any other participant | Nothing beyond the aggregate |

### Cryptographic primitive: Shamir secret sharing

The protocol uses **Shamir (t, n)-secret sharing** as implemented by mpyc:

1. Each party splits its input value `s` into `n` random shares such that
   any `t+1` shares reconstruct `s` but any `t` shares reveal nothing.
2. Each party sends one share to each other party.
3. Parties perform arithmetic (addition, multiplication) on the shares
   directly — the result shares decode to the correct plaintext result.
4. All parties reveal the output shares simultaneously to reconstruct the
   aggregate.

For `n=3` parties with threshold `t=1`, any single party may be compromised
without revealing any other party's input. For `n=2`, `t=0` (no tolerance for
a corrupt party — both must be semi-honest).

### Computation steps

```
Each party i holds:
  local_sum_i     = sum(scores_i)
  local_sum_sq_i  = sum(s^2 for s in scores_i)
  local_count_i   = len(scores_i)

Step 1: Secret-share these three values (one per party → n shares each)
Step 2: Sum the shares across all parties:
  total_sum     = Σ_i  [secret] local_sum_i
  total_sum_sq  = Σ_i  [secret] local_sum_sq_i
  total_count   = Σ_i  [secret] local_count_i
Step 3: Compute on shares:
  mean       = total_sum / total_count
  variance   = total_sum_sq / total_count - mean²
Step 4: All parties reconstruct mean, variance, total_count simultaneously
```

Crucially, **only the aggregate is revealed** — no intermediate per-party
sums are opened during the computation.

### Fixed-point arithmetic

Risk scores are real-valued. mpyc uses **fixed-point arithmetic** (32 integer
bits, 16 fractional bits by default) so all operations stay in a finite field.
The precision gives ~4 decimal places of accuracy; the MPC result is guaranteed
to match the plaintext aggregate within `1e-4` absolute tolerance.

---

## Trust model

### What each party learns

| Party | Learns | Does not learn |
|---|---|---|
| Party 0 | aggregate mean, variance, n_total | Party 1's scores, Party 2's scores, any individual party's sum |
| Party 1 | aggregate mean, variance, n_total | Party 0's scores, Party 2's scores |
| Party 2 | aggregate mean, variance, n_total | Party 0's scores, Party 1's scores |

**The aggregate itself** may carry some information. For example, if one party
has 1 score and the other has 1000, the aggregate mean is heavily influenced by
the majority party. This is unavoidable — it is a property of the aggregate
function, not a protocol leak.

### Semi-honest security model

The protocol is **information-theoretically secure in the semi-honest
(honest-but-curious) model**:

- A semi-honest party follows the protocol exactly but inspects all messages
  it receives, attempting to infer others' inputs.
- The Shamir shares a party receives are computationally and
  information-theoretically indistinguishable from random for any coalition of
  `t` or fewer parties.

**Limitations**:

| Threat | Protected? | Mitigation if needed |
|---|---|---|
| Semi-honest eavesdropper | ✅ Yes | Shamir IT-security |
| Malicious party (sends wrong shares) | ❌ No | Add SPDZ-style MACs (e.g. MP-SPDZ) |
| Collusion of ≥ t+1 parties | ❌ No | Reduce t; use MPC with stronger guarantees |
| Side-channel on the aggregator | ❌ No | Run in a TEE (Intel SGX, AWS Nitro) |

### Collusion tolerance

| n parties | threshold t | Colluders tolerated |
|---|---|---|
| 2 | 0 | 0 — both must be honest |
| 3 | 1 | 1 — any one party may be corrupt |
| 4+ | ⌊(n-1)/2⌋ | Up to half the parties |

For 4+ parties, `mpc_aggregate_scores_local` and `mpc_aggregate_scores` accept
`n_parties=4` and will run correctly, but the threshold and communication
overhead grow. For production at 4+ parties, consider MP-SPDZ or SCALE-MAMBA
for their actively-secure protocols.

---

## Party authentication

In distributed mode (`mpc_aggregate_scores`), each party connects to peers via
TLS. mpyc uses the same certificate infrastructure as
`detection/federated/coordinator.py`:

- Each participant generates an Ed25519 key pair and a self-signed certificate
  (or obtains one from a shared CA trusted by all participants).
- The certificate common name (`CN`) must match the participant's registered ID.
- mpyc's `Party` object accepts `ca_cert`, `cert`, and `key` parameters for
  mutual TLS authentication.

```python
from mpyc.runtime import Party

party = Party(
    pid=0,
    host="exchange-a.example.com",
    port=11000,
    # TLS configuration:
    ca_cert="certs/ca.crt",
    cert="certs/exchange-a.crt",
    key="certs/exchange-a.key",
)
```

All three certificates must be exchanged out-of-band before the first MPC run
(e.g. via a secure onboarding ceremony). This is the same PKI as the federated
gradient aggregation component.

---

## 4-party and larger configurations

`mpc_aggregate_scores` accepts `n_parties ≥ 4`. The Shamir threshold
automatically scales to `t = ⌊(n-1)/2⌋`. However:

1. **Communication overhead** grows as O(n²) shares per secret.
2. **Malicious security** is not provided — mpyc's default is semi-honest.
   With n ≥ 4 in a higher-risk environment, switch to an actively-secure
   framework (MP-SPDZ `bristol-fashion` or SCALE-MAMBA with honest-majority
   MAC-check).
3. **The test suite** in `tests/test_mpc_aggregator.py` covers n=2 and n=3.
   n=4 is not covered by the parametric regression tests; contributors adding
   4-party configurations should add corresponding fixtures.

---

## Running the protocol

### Local simulation (development / CI)

All parties run cooperatively in a single process — no network required:

```python
import asyncio
from detection.mpc_aggregator import mpc_aggregate_scores_local, plaintext_aggregate

# Each participant's private scores
party_scores = [
    [72.0, 45.0, 88.0, 31.0, 60.0],   # Exchange A
    [20.0, 40.0, 60.0, 80.0, 100.0],  # Wallet provider B
    [11.0, 22.0, 33.0, 44.0, 55.0],   # Custodian C
]

result = asyncio.run(mpc_aggregate_scores_local(party_scores))
print(result)
# {"mean": 50.8, "variance": 697.12, "n_total": 15, "n_parties": 3}

# Verify against plaintext:
plain = plaintext_aggregate(party_scores)
assert abs(result["mean"] - plain["mean"]) < 1e-4
```

### Distributed deployment

Each organisation runs independently on its own machine. Party 0 (the
coordinator) starts first; parties 1 and 2 connect to it.

```bash
# Party 0 (Exchange A):
python - <<'EOF'
import asyncio
from detection.mpc_aggregator import mpc_aggregate_scores

result = asyncio.run(mpc_aggregate_scores(
    local_scores=[72.0, 45.0, 88.0, 31.0, 60.0],
    n_parties=3,
    party_index=0,
    peers=[("exchange-a.example.com", 11000),
           ("wallet-b.example.com",   11001),
           ("custodian-c.example.com", 11002)],
))
print("Exchange A sees:", result)
EOF

# Party 1 (Wallet provider B) — same command with party_index=1
# Party 2 (Custodian C)       — same command with party_index=2
```

All three processes block until all parties have connected and the protocol
completes (~100–500 ms on a LAN, depending on score list sizes).

---

## Onboarding a new participant

1. **Agree on participation**: obtain signed agreement from all existing
   participants accepting the new party. Update the `n_parties` parameter
   in all deployments.
2. **Generate certificates**: the new party generates an Ed25519 key pair.
   The CA (or all existing parties) sign the certificate.
3. **Exchange public keys**: all parties exchange certificates out-of-band
   (e.g. via a secure governance channel).
4. **Configure network access**: the new party's host/port must be
   reachable from all other parties (firewall rules, VPN if needed).
5. **Test locally**: run `pytest tests/test_mpc_aggregator.py -v` to
   confirm the protocol produces correct results in local mode.
6. **Run a dry-run round** with synthetic scores before going live.
7. **Update the threshold**: with `n` parties the new threshold is
   `t = ⌊(n-1)/2⌋`. Verify this provides acceptable collusion tolerance
   for your threat model before proceeding.

---

## Handling party dropout

If a party drops out mid-protocol (network failure after sharing has started),
the current mpyc implementation will raise a `ConnectionError` and the round
fails. Recommended mitigations:

| Strategy | When to use | Notes |
|---|---|---|
| **Retry the round** | Transient network failures | mpyc supports reconnection; restart all parties from the beginning of the round |
| **Threshold increase** | Occasional unreliable parties | Increase `t` so the protocol tolerates more absentees |
| **Abort and alert** | Persistent failure | Log the dropout, alert the governance channel, exclude the party for the current cycle |
| **Asynchronous MPC** | High-latency environments | Replace mpyc with an async-MPC framework that supports mid-round party replacement (e.g. PRIO, Delphi) |

The current implementation does **not** automatically retry or exclude parties.
Adding retry logic around `mpc_aggregate_scores` is straightforward using the
existing `utils/retry.py` backoff decorator.

---

## References

- Shamir, A. (1979) "How to share a secret." *CACM*, 22(11), 612–613.
- Lschoe (2023) *mpyc — Multiparty Computation in Python*.
  https://github.com/lschoe/mpyc
- Bonawitz, K. et al. (2017) "Practical Secure Aggregation for Privacy-Preserving
  Machine Learning." *ACM CCS 2017*.
- Keller, M. (2020) "MP-SPDZ: A Versatile Framework for Multi-Party Computation."
  *ACM CCS 2020*.
