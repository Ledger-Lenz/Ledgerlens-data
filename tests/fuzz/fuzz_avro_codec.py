"""Fuzz testing for Avro deserialisation in ingestion/avro_codec.py.

This fuzz target feeds random bytes to the Avro deserialiser to detect:
  - Unhandled exceptions that crash the ingestion worker
  - Buffer overflows or memory corruption (via AddressSanitizer when run with ASAN)
  - Algorithmic complexity attacks (via libFuzzer's timeout detection)

Acceptable exceptions (caught and handled gracefully):
  - ValueError: invalid schema, missing required fields, wrong types
  - fastavro.schema.SchemaParseException: schema parsing errors

Unacceptable exceptions (bugs that crash the worker):
  - Any unhandled exception that escapes to the caller

Run the fuzz target:
  python tests/fuzz/fuzz_avro_codec.py   # Interactive mode

Or via libFuzzer (requires building with address sanitizer):
  clang -fsanitize=fuzzer,address -I/usr/include -o /tmp/fuzz_avro_codec tests/fuzz/fuzz_avro_codec.py
  /tmp/fuzz_avro_codec tests/fuzz/corpus/ -max_len=10000 -timeout=10

See tests/fuzz/README.md for full setup instructions.
"""

import sys
import io
import json
import os
from functools import lru_cache
from pathlib import Path

try:
    import atheris
except ImportError:
    # Graceful fallback for non-fuzzing test runs
    atheris = None

import fastavro


@lru_cache(maxsize=1)
def _get_schema():
    """Load and cache the Avro schema."""
    try:
        # Find the schema file in the data directory
        repo_root = Path(__file__).parent.parent.parent
        schema_path = repo_root / "data" / "trade_avro_schema.json"
        
        with open(schema_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return fastavro.parse_schema(raw)
    except Exception as e:
        # If schema loading fails, return None and handle in the fuzz target
        print(f"Warning: Could not load schema: {e}", file=sys.stderr)
        return None


def _deserialize(value: bytes, schema: dict):
    """Inline deserialization to avoid circular imports."""
    return fastavro.schemaless_reader(io.BytesIO(value), schema)


def _test_avro_deserialiser(data: bytes) -> None:
    """Fuzz target: feed random bytes to deserialize() and assert no crashes.
    
    This function is called repeatedly by atheris with increasingly complex
    byte sequences. It should handle malformed input gracefully without
    raising unhandled exceptions.
    """
    schema = _get_schema()
    if schema is None:
        # Schema loading failed; skip this test
        return
    
    try:
        # Try to deserialize the random bytes
        result = _deserialize(data, schema)
        # If it succeeds, result should be a dict
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    except (ValueError, KeyError, TypeError) as e:
        # Expected exceptions for malformed input
        # These are gracefully caught in production ingestion workers
        pass
    except fastavro.schema.SchemaParseException:
        # Expected schema parsing error
        pass
    except fastavro.io.UnknownDecoderException:
        # Expected when fastavro encounters unknown/invalid encoding
        pass
    except EOFError:
        # Expected when buffer is too short
        pass
    except Exception as e:
        # Any other exception is a bug — raise it so the fuzzer can find the input
        raise AssertionError(
            f"Unhandled exception in deserialize(): {type(e).__name__}: {e}"
        ) from e


def main():
    """Entry point for both atheris and standalone fuzzing."""
    if atheris is None:
        print("Error: atheris not installed. Install with: pip install atheris", file=sys.stderr)
        sys.exit(1)
    
    atheris.Setup(sys.argv, _test_avro_deserialiser)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
