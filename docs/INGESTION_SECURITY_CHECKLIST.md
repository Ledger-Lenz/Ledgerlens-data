# Security review checklist: new ingestion sources

Complete before merging any new external data source / connector (Issue #908).

- [ ] Raw responses go through `ingestion.untrusted_input.parse_untrusted_json` — never bare `json.loads` — so size (`MAX_PAYLOAD_BYTES`), depth (`MAX_JSON_DEPTH`) and container (`MAX_CONTAINER_ITEMS`) limits run **before** deserialization.
- [ ] Top-level schema (`expected_type`, `required_keys`) is declared for the source.
- [ ] Converted domain objects pass `validate_trade` / `validate_orderbook_event` / `validate_account_activity` before being yielded.
- [ ] `UntrustedInputError` is caught per record (skip + log), not allowed to crash the stream.
- [ ] Source added to the fuzz corpus in `scripts/fuzz_untrusted_input.py` if it has a novel payload shape; `python scripts/fuzz_untrusted_input.py` reports 0 crashes.
- [ ] Outbound calls wrapped with `call_with_retry(..., breaker=CircuitBreaker(...))` and paced by `AdaptiveRateLimiter` with a static `max_rps` ceiling.
- [ ] Trades carry a canonical identity (`ledger_sequence` + `operation_index`, or a Horizon TOID trade id) so `CrossSourceDeduplicator` can collapse redundant sources.
- [ ] No credentials, tokens, or raw payloads written to logs.
