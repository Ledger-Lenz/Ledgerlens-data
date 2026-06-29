# LedgerLens Security

---

## Federated Learning Authentication

### Overview

The federated coordinator accepts gradient updates from registered participants.
All communication uses **mutual TLS (mTLS)**: both the server and each participant
present X.509 certificates, and both sides verify the other's certificate chain.

```
Participant                    Coordinator
    |                               |
    |  ClientHello                  |
    |------------------------------>|
    |  ServerHello + server cert    |
    |<------------------------------|
    |  client cert (signed by CA)   |
    |------------------------------>|
    |  TLS session established      |
    |  (OpenSSL rejects bad certs)  |
    |                               |
    |  POST /gradient_update        |
    |------------------------------>|
    |  App layer: CN revocation?    |
    |  App layer: model_id in scope?|
    |  202 Accepted / 401 / 403     |
    |<------------------------------|
```

### Certificate Issuance

1. **Bootstrap** — a new participant contacts the LedgerLens operator out-of-band
   (e.g. via a verified Signal message or physical key ceremony) and provides
   their public key fingerprint or CSR.
2. The operator runs `python -m scripts.manage_federated_certs issue --cn <CN> --models <models>`.
3. The signed certificate and private key are written to `certs/participants/`.
4. The operator transmits the private key to the participant over a secure channel
   (Signal, encrypted email, or direct physical handoff) and **deletes the local copy**.
5. The participant private key never appears in the coordinator at any point after step 4.

### Certificate Revocation

- Revocation is stored in the `federated_participant_certs` table (SQLite/PostgreSQL).
- The coordinator's `RevocationCache` reloads from the DB every 30 seconds,
  guaranteeing revocation takes effect **within 60 seconds**.
- No OCSP server is required; the revocation list is loaded in-process.
- To revoke: `python -m scripts.manage_federated_certs revoke --cn <CN>`

### Certificate Rotation

Participants should rotate certificates before expiry.  The `rotate` command
atomically revokes the old certificate and issues a replacement:

```bash
python -m scripts.manage_federated_certs rotate --cn participant-A --models benford,gnn
```

Expiry monitoring: `python -m scripts.manage_federated_certs check-expiry --days 30`
alerts on certificates expiring within 30 days.

### Authorisation Scopes

Each certificate encodes the set of model IDs the participant may submit
updates for in the X.509 `organizationalUnitName` (OU) field:

```
Subject: CN=participant-A, OU=benford,gnn, O=LedgerLens Participant
```

The coordinator's `require_participant` dependency rejects any gradient update
for a `model_id` not listed in the participant's OU.

### CA Private Key Storage Requirement

> **The CA private key MUST be stored in a hardware security module (HSM) or
> an encrypted secrets manager (e.g. HashiCorp Vault, AWS KMS, GCP Secret Manager).**

The `generate_ca_keypair()` function (and the `init-ca` CLI command) print the CA
private key to stdout exactly once.  The operator is responsible for loading it
into the secrets manager immediately.  It must never be written to plaintext files.

At runtime, inject the key via the `FEDERATED_CA_KEY_PEM` environment variable,
sourced from the secrets manager by the deployment pipeline (e.g. via
`vault kv get -field=key secret/ledgerlens/ca`).

---

## Webhook Security

- `ALERT_WEBHOOK_URL` must use `https://`; `http://` is rejected at startup.
- The URL is never written to log output.

## WebSocket Security

- The WebSocket server binds to `127.0.0.1` by default (loopback only).
- Setting `WS_BIND_HOST=0.0.0.0` requires `WS_ALLOW_EXTERNAL=1` to be explicitly
  set; the server raises `ValueError` otherwise.

## On-Chain Secret

- `LEDGERLENS_SUBMITTER_SECRET` (the service-account Stellar secret key) must be
  stored in a secrets manager and never committed to source control.
