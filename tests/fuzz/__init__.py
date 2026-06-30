"""Fuzz testing suite for LedgerLens data ingestion layer.

This package contains libFuzzer targets (via atheris) that systematically
explore the input space of critical parsing and deserialisation routines to
detect crashes, buffer overflows, and algorithmic complexity attacks.

Targets:
  - fuzz_avro_codec.py: Avro binary deserialization
  - fuzz_horizon_response.py: Horizon API response parsing (Pydantic models)

See README.md in this directory for setup and usage instructions.
"""
