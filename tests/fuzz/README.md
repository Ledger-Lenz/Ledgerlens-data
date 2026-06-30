# Fuzz Testing for LedgerLens Data Ingestion

This directory contains libFuzzer targets (via [atheris](https://github.com/google/atheris)) that systematically explore the input space of critical parsing and deserialization routines to detect crashes, buffer overflows, and algorithmic complexity attacks.

## Targets

### `fuzz_avro_codec.py`

Tests the Avro deserializer in `ingestion/avro_codec.py` which processes Kafka trade messages.

**What it tests:**
- Schemaless Avro binary deserialization
- Field validation and type conversion
- Robustness to truncated or corrupted binary data

**What it catches:**
- Unhandled exceptions in `deserialize()`
- Buffer overflows in fastavro's reader
- DoS via pathological encoding

### `fuzz_horizon_response.py`

Tests the Pydantic models in `ingestion/data_models.py` which parse Horizon API JSON responses.

**What it tests:**
- JSON parsing and validation against Pydantic models
- Type coercion and field conversion
- Edge cases in nested structures

**Models covered:**
- `Trade`: SDEX trade execution records
- `OrderBookEvent`: Order placement/cancellation events
- `AccountActivity`: Account lifecycle events
- `Asset`: Stellar asset identifiers
- `BotFingerprint`: Bot detection fingerprints

**What it catches:**
- Unhandled `ValidationError` exceptions
- Injection attacks via malformed JSON
- DoS via deeply nested or extremely large structures

## Setup

### Prerequisites

- Python 3.10+
- pip / virtualenv
- libFuzzer (optional, for maximum coverage with AddressSanitizer)

### Installation

1. Install atheris and dependencies:

```bash
pip install -r requirements.txt
python tests/fuzz/generate_corpus.py  # Generate seed corpus
```

2. (Optional) Compile with AddressSanitizer for memory-safety testing:

```bash
clang -fsanitize=fuzzer,address,undefined \
  -I/usr/include/python3.11 \
  -o /tmp/fuzz_avro_codec \
  tests/fuzz/fuzz_avro_codec.py
```

## Usage

### Interactive Mode (Python)

Run the fuzz target directly:

```bash
# Test Avro deserializer
python tests/fuzz/fuzz_avro_codec.py tests/fuzz/corpus/ -max_len=10000 -timeout=10

# Test Horizon API parsing
python tests/fuzz/fuzz_horizon_response.py tests/fuzz/corpus/ -max_len=50000 -timeout=10
```

Options:
- `-max_len=N`: Maximum input size in bytes
- `-timeout=N`: Timeout per input in seconds (libFuzzer will kill hanging tests)
- `-max_total_time=N`: Total time to run (seconds)
- `-artifact_prefix=DIR/`: Directory to save crash artifacts

### Makefile Target

Run both fuzz targets for 60 seconds each:

```bash
make fuzz
```

This is the recommended way to run fuzz testing in CI/CD.

### Continuous Fuzzing (OSS-Fuzz)

LedgerLens is designed to integrate with [OSS-Fuzz](https://github.com/google/oss-fuzz) for continuous fuzzing. See `.github/workflows/oss-fuzz.yml` for the integration.

## Corpus Management

The `corpus/` directory contains seed inputs that guide the fuzzer. Seeds should represent:
- Valid inputs for the positive case
- Edge cases and boundary conditions
- Previous crash-inducing inputs (for regression testing)

### Adding New Seeds

If a crash is found, add the reproducer to the corpus so it becomes part of the regression test:

```bash
cp /path/to/crash corpus/crash_<description>.bin
```

Commit the new seed so all developers have it.

### Corpus Size

Keep the corpus small (< 10 MB total) for fast iteration. libFuzzer will derive new inputs via mutations, so seed diversity matters more than quantity.

## Triage: What to Do When the Fuzzer Finds a Crash

1. **Reproduce locally:**
   ```bash
   python tests/fuzz/fuzz_avro_codec.py <crash_artifact>
   ```
   (libFuzzer writes crash artifacts to `crash-<hash>` in the current directory.)

2. **Classify the crash:**
   - **Acceptable:** `ValueError`, `ValidationError`, `JSONDecodeError` — these are caught gracefully by the caller. Add the input to the corpus as a regression test.
   - **Unacceptable:** Any other exception — this is a bug. Open an issue.

3. **Fix the root cause:**
   - Add input validation before the crash point
   - Handle the exception gracefully
   - Add a unit test in `tests/test_avro_codec.py` or `tests/test_data_models.py`

4. **Add regression test:**
   ```bash
   cp <crash_artifact> corpus/regression_<date>_<description>.bin
   ```

## Running in CI/CD

### GitHub Actions

Add to `.github/workflows/ci.yml`:

```yaml
- name: Run fuzz testing
  run: make fuzz
  timeout-minutes: 5
```

### Local Pre-Commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-push

make fuzz --always-make || {
  echo "Fuzz testing failed. Run 'make fuzz' locally to reproduce."
  exit 1
}
```

## Performance

### Typical Results

- **Avro codec:** 1000+ inputs/second on modern CPUs
- **Horizon parsing:** 500+ inputs/second (slower due to JSON parsing)
- **Total runtime (60s each):** ~60,000 – 120,000 mutations explored

### Optimization Tips

- Keep seed corpus small (< 10 items) to start fast
- Use `-max_len` to limit pathological cases (e.g., 10 KB max for binary, 50 KB for JSON)
- Run with `-timeout=5` to kill hanging inputs quickly
- Parallel fuzzing: `python -m pytest tests/fuzz/ -n auto` (requires pytest-xdist)

## References

- [atheris documentation](https://github.com/google/atheris)
- [libFuzzer documentation](https://llvm.org/docs/LibFuzzer/)
- [fastavro error handling](https://fastavro.readthedocs.io/)
- [Pydantic validation](https://docs.pydantic.dev/latest/usage/validators/)

## Troubleshooting

### ImportError: atheris

If you get `ImportError: No module named 'atheris'`:

```bash
pip install atheris
```

atheris requires Python 3.7+ and is available on Linux, macOS, and Windows.

### Slow fuzzing / Timeout

If inputs are timing out frequently:
1. Reduce `-max_len` to prevent pathological cases
2. Check for infinite loops in the code under test
3. Profile with `python -m cProfile` to find bottlenecks

### "No interesting coverage gained" (warning from libFuzzer)

This is normal after the fuzzer has explored the main code paths. It means:
- The corpus is well-exercised
- The fuzzer is mostly finding redundant inputs
- You can stop and move on to other testing

To improve coverage:
- Add more diverse seeds to the corpus
- Target new error paths (e.g., add a fuzz target for contract client code)

## Security Considerations

- **Differential privacy:** Fuzz testing does not leak information about labelled training data (fuzz targets only receive random bytes).
- **Artifact containment:** Crash artifacts are written locally; do not push them to the public repository unless they are sanitized.
- **Resource limits:** Run fuzz tests in sandboxed CI environments with CPU/memory limits to prevent DoS.

## Future Work

- Add fuzz targets for `streaming/kafka_worker.py` message handling
- Add fuzz targets for Soroban contract client (`integrations/contract_client.py`)
- Integrate with [OSS-Fuzz](https://github.com/google/oss-fuzz) for 24/7 fuzzing
- Implement differential fuzzing against multiple deserialization implementations
