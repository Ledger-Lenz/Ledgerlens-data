"""Fuzz testing for Horizon API response parsing via Pydantic models.

This fuzz target feeds random JSON-like byte strings to the Pydantic models
in ingestion/data_models.py to detect:
  - Unhandled exceptions from validation errors
  - Injection attacks via malformed JSON
  - DoS via algorithmic complexity (e.g., deeply nested structures)

The models tested:
  - Trade: main trade execution record from Horizon SDEX
  - OrderBookEvent: order placement/cancellation events
  - AccountActivity: wallet lifecycle events
  - Asset: currency pair identifiers
  - BotFingerprint: bot detection fingerprints

Acceptable exceptions (caught and handled gracefully):
  - pydantic.ValidationError: type mismatch, missing required fields, etc.
  - json.JSONDecodeError: malformed JSON
  - TypeError: wrong input type

Unacceptable exceptions (bugs that crash the worker):
  - Any unhandled exception that escapes to the caller

Run the fuzz target:
  python tests/fuzz/fuzz_horizon_response.py   # Interactive mode

Or via libFuzzer:
  clang -fsanitize=fuzzer -o /tmp/fuzz_horizon tests/fuzz/fuzz_horizon_response.py
  /tmp/fuzz_horizon tests/fuzz/corpus/ -max_len=50000 -timeout=10

See tests/fuzz/README.md for full setup instructions.
"""

import sys
import json
from typing import Any

try:
    import atheris
except ImportError:
    atheris = None

from pydantic import ValidationError

from ingestion.data_models import Trade, OrderBookEvent, AccountActivity, Asset, BotFingerprint


def _fuzz_horizon_parsing(data: bytes) -> None:
    """Fuzz target: feed random bytes (parsed as JSON) to Pydantic models.
    
    The fuzzer explores the input space of malformed JSON and invalid objects
    to ensure the parser fails gracefully without raising unhandled exceptions.
    """
    # Try to decode as JSON first
    try:
        obj = json.loads(data.decode("utf-8", errors="ignore"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        # Malformed JSON — this is expected and ok
        return
    
    if not isinstance(obj, dict):
        # Not a dict, so it can't be a valid record
        return
    
    # Try to parse with each model
    models_to_test = [
        ("Trade", Trade),
        ("OrderBookEvent", OrderBookEvent),
        ("AccountActivity", AccountActivity),
        ("Asset", Asset),
        ("BotFingerprint", BotFingerprint),
    ]
    
    for model_name, model_class in models_to_test:
        try:
            # Try to instantiate the model with random data
            instance = model_class(**obj)
            # If it succeeds, validate the result is of expected type
            assert isinstance(instance, model_class), (
                f"{model_name} instantiation did not return correct type"
            )
        except ValidationError:
            # Expected — pydantic validation failed
            pass
        except TypeError:
            # Expected — wrong argument types
            pass
        except ValueError:
            # Expected — value conversion failed (e.g., invalid datetime)
            pass
        except Exception as e:
            # Any other exception is a bug — raise it so fuzzer can report
            raise AssertionError(
                f"Unhandled exception in {model_name} parsing: {type(e).__name__}: {e}"
            ) from e


def main():
    """Entry point for atheris fuzzing."""
    if atheris is None:
        print("Error: atheris not installed. Install with: pip install atheris", file=sys.stderr)
        sys.exit(1)
    
    atheris.Setup(sys.argv, _fuzz_horizon_parsing)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
