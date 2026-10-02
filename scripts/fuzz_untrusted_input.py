"""Fuzz harness for untrusted-input entry points (Issue #908).

Feeds a documented corpus of malformed payloads plus random mutations into
``parse_untrusted_json``. The only acceptable outcomes are a parsed value or
``UntrustedInputError``; any other exception is a crash and fails the run.

Usage: python scripts/fuzz_untrusted_input.py [iterations] [seed]
"""

from __future__ import annotations

import random
import sys

from ingestion.untrusted_input import UntrustedInputError, parse_untrusted_json

# Documented corpus of malformed / hostile inputs.
CORPUS: list[bytes | str] = [
    b"",
    b"{",
    b"}",
    b"null",
    b"NaN",
    b'{"a": Infinity}',
    b"\xff\xfe\x00",
    b'{"a": "\\ud800"}',
    "[" * 100_000 + "]" * 100_000,
    '{"a":' * 5_000 + "1" + "}" * 5_000,
    b"[" + b"1," * 20_000 + b"1]",
    b'"' + b"x" * (6 * 1024 * 1024) + b'"',
    b'{"trade_id": 1e999999}',
    b'{"id": "\\"]]]]]]"}',
    b"[1, 2, 3",
    b'{"a": 1,}',
]


def _mutate(rng: random.Random, data: bytes) -> bytes:
    buf = bytearray(data or b"{}")
    for _ in range(rng.randint(1, 8)):
        op = rng.random()
        pos = rng.randrange(len(buf) + 1)
        if op < 0.4:
            buf.insert(pos, rng.choice(b'{}[]",:\\0123456789aeflnrstu \x00\xff'))
        elif op < 0.7 and buf:
            del buf[min(pos, len(buf) - 1)]
        else:
            buf[pos:pos] = rng.choice([b"[" * 50, b'{"k":', b"]" * 10, b'"\\u'])
    return bytes(buf)


def run(iterations: int = 5_000, seed: int = 0) -> int:
    rng = random.Random(seed)
    seeds = [c if isinstance(c, bytes) else c.encode() for c in CORPUS] + [
        b'{"id": "1-0", "base_amount": "1.5", "nested": [1, {"x": null}]}'
    ]
    inputs: list[bytes | str] = list(CORPUS)
    inputs += [_mutate(rng, rng.choice(seeds[-1:] + seeds[:3])) for _ in range(iterations)]
    crashes = 0
    for payload in inputs:
        try:
            parse_untrusted_json(payload, source="fuzz", expected_type=object)
        except UntrustedInputError:
            pass
        except Exception as exc:  # noqa: BLE001 - any other exception is a finding
            crashes += 1
            print(f"CRASH {type(exc).__name__}: {exc!r} on {payload[:80]!r}")
    print(f"fuzzed {len(inputs)} inputs, {crashes} crashes")
    return crashes


if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(1 if run(int(args[0]) if args else 5_000, int(args[1]) if len(args) > 1 else 0) else 0)
