#!/usr/bin/env python
"""Scan configuration files and `.env.example` for accidentally committed
secret-shaped values.

Issue #956 — Env contract: add automated secret-detection scan integrated
with SECRETS_MANAGEMENT_PR.md guidance.

The scanner applies two complementary detection strategies:

1. **Regex patterns** — match well-known credential shapes:
   - Stellar secret keys (``S…`` Base32, 56 chars)
   - API keys (``sk_live_…``, ``ak_…``, ``Bearer …``, ``AKIA…``)
   - Generic high-entropy tokens (looks like a secret: 40+ hex chars, etc.)
   - Private key PEM headers
   - Database connection strings with embedded passwords
   - JWT tokens (``eyJ…``)

2. **Shannon entropy** — flag strings whose entropy exceeds a threshold
   (default 4.5 bits/char) inside an assignment context (``=``, ``:``) that
   are long enough to be a credential (≥ 20 chars).  This catches novel
   secret formats not covered by the regex list.

Target paths scanned by default:
- ``config/``
- ``config/environments/``
- ``.env.example``

The scanner intentionally **does not** scan the full repo (that is the job
of a dedicated git-history scanner like ``truffleHog`` or ``gitleaks``).  It
is scoped to config templates to catch secrets that would be visible to any
user who clones the repo.

Usage
-----
    python scripts/scan_secrets.py                       # scan default paths
    python scripts/scan_secrets.py --paths config/ .env.example
    python scripts/scan_secrets.py --entropy-threshold 4.2
    python scripts/scan_secrets.py --no-entropy          # regex only

Exit codes
----------
0  — no secrets found.
1  — at least one secret-shaped value found (CI blocks PR merge).
2  — unexpected error (file unreadable, etc.).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Default scan targets
# ---------------------------------------------------------------------------
DEFAULT_SCAN_PATHS: list[str] = [
    "config/",
    ".env.example",
]

# ---------------------------------------------------------------------------
# Secret patterns
# ---------------------------------------------------------------------------
# Each tuple: (pattern_name, compiled_regex)
# We look for these as raw values that appear in the scanned lines.
# Patterns are intentionally conservative to avoid false-positives on
# example/placeholder values.

_PLACEHOLDER_KEYWORDS = re.compile(
    r"placeholder|example|your[_\-]?|changeme|replace|todo|fixme|<.+>|xxx|yyy|"
    r"my[_\-]?secret|my[_\-]?key|fake|test[_\-]?key|sample|dummy|mock|ci[_\-]",
    re.IGNORECASE,
)

_SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Stellar secret key: starts with S, Base32, exactly 56 chars
    (
        "stellar_secret_key",
        re.compile(r"\bS[A-Z2-7]{55}\b"),
    ),
    # AWS access key
    (
        "aws_access_key",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ),
    # Stripe live secret key
    (
        "stripe_live_key",
        re.compile(r"\bsk_live_[0-9a-zA-Z]{24,}\b"),
    ),
    # Generic "Bearer <token>" with a long opaque value
    (
        "bearer_token",
        re.compile(r"\bBearer\s+[A-Za-z0-9+/=_\-]{30,}\b"),
    ),
    # PEM private key header
    (
        "pem_private_key",
        re.compile(r"-----BEGIN\s+(RSA |EC |OPENSSH |)PRIVATE KEY-----"),
    ),
    # Connection string with password embedded (postgres, mysql, redis, mongodb)
    (
        "connection_string_with_password",
        re.compile(
            r"(postgres|postgresql|mysql|mongodb|redis)://[^:@/\s]+:[^@/\s]{8,}@",
            re.IGNORECASE,
        ),
    ),
    # JWT token (three base64url segments)
    (
        "jwt_token",
        re.compile(r"\beyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\b"),
    ),
    # Generic long hex string in assignment context (SHA256/API keys)
    (
        "long_hex_value",
        re.compile(r"(?<=[=:\s])[0-9a-fA-F]{40,}(?=\s|$|[\"'])"),
    ),
    # Generic high-entropy base64 blob (longer than 40 chars, not just padding)
    (
        "long_base64_value",
        re.compile(r"(?<=[=:\s])[A-Za-z0-9+/]{40,}={0,2}(?=\s|$|[\"'])"),
    ),
]

# ---------------------------------------------------------------------------
# Entropy calculation
# ---------------------------------------------------------------------------
_ENTROPY_ASSIGNMENT_RE = re.compile(
    r"(?:^|[=:\s])([A-Za-z0-9+/=_\-]{20,})(?:\s|$|[\"'])"
)


def _shannon_entropy(s: str) -> float:
    """Return Shannon entropy in bits per character."""
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


# ---------------------------------------------------------------------------
# Finding type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecretFinding:
    file: Path
    lineno: int
    pattern: str
    matched_value: str
    line_preview: str  # sanitised — only first 80 chars, value partially masked

    def __str__(self) -> str:
        rel = self.file.relative_to(REPO_ROOT) if self.file.is_absolute() else self.file
        masked = self.matched_value[:6] + "***" if len(self.matched_value) > 6 else "***"
        return (
            f"{rel}:{self.lineno}: [{self.pattern}] matched value starting with "
            f"'{masked}' — {self.line_preview!r}"
        )


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


def _is_placeholder(value: str) -> bool:
    """Return True if the value looks like a template placeholder, not a real secret."""
    return bool(_PLACEHOLDER_KEYWORDS.search(value))


def _scan_file(
    path: Path,
    *,
    entropy_threshold: float = 4.5,
    use_entropy: bool = True,
) -> list[SecretFinding]:
    findings: list[SecretFinding] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"  [warning] Cannot read {path}: {exc}", file=sys.stderr)
        return findings

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        # Skip comment-only lines and blank lines
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        preview = raw_line[:80]

        # --- Regex patterns ---
        for pattern_name, regex in _SECRET_PATTERNS:
            for match in regex.finditer(raw_line):
                value = match.group(0).strip()
                if _is_placeholder(value) or _is_placeholder(raw_line):
                    continue
                findings.append(
                    SecretFinding(
                        file=path,
                        lineno=lineno,
                        pattern=pattern_name,
                        matched_value=value,
                        line_preview=preview,
                    )
                )

        # --- Entropy scan ---
        if use_entropy:
            for match in _ENTROPY_ASSIGNMENT_RE.finditer(raw_line):
                value = match.group(1)
                if _is_placeholder(value) or _is_placeholder(raw_line):
                    continue
                entropy = _shannon_entropy(value)
                if entropy >= entropy_threshold:
                    findings.append(
                        SecretFinding(
                            file=path,
                            lineno=lineno,
                            pattern=f"high_entropy({entropy:.2f}≥{entropy_threshold})",
                            matched_value=value,
                            line_preview=preview,
                        )
                    )

    return findings


def scan_paths(
    paths: list[Path],
    *,
    entropy_threshold: float = 4.5,
    use_entropy: bool = True,
    extensions: tuple[str, ...] = (".py", ".yaml", ".yml", ".env", ".cfg", ".ini", ".toml", ".json"),
) -> list[SecretFinding]:
    """Recursively scan the given paths for secret-shaped values.

    Parameters
    ----------
    paths:
        List of files or directories to scan.
    entropy_threshold:
        Minimum Shannon entropy (bits/char) to flag a token.
    use_entropy:
        If False, skip the entropy check and run regex patterns only.
    extensions:
        File extensions to include (ignored when a path is an explicit file).
    """
    all_findings: list[SecretFinding] = []

    for p in paths:
        if not p.exists():
            print(f"  [warning] Scan path does not exist: {p}", file=sys.stderr)
            continue
        if p.is_file():
            targets = [p]
        else:
            targets = sorted(
                f for ext in extensions for f in p.rglob(f"*{ext}")
                if "__pycache__" not in f.parts
            )
        for target in targets:
            all_findings.extend(
                _scan_file(target, entropy_threshold=entropy_threshold, use_entropy=use_entropy)
            )

    return all_findings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scan_secrets",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--paths",
        nargs="+",
        default=DEFAULT_SCAN_PATHS,
        metavar="PATH",
        help=(
            f"Files or directories to scan (default: {' '.join(DEFAULT_SCAN_PATHS)}). "
            "Directories are scanned recursively for .py, .yaml, .yml, .env, .cfg, "
            ".ini, .toml, .json files."
        ),
    )
    p.add_argument(
        "--entropy-threshold",
        type=float,
        default=4.5,
        metavar="BITS",
        help="Minimum Shannon entropy (bits/char) to flag a token (default: 4.5).",
    )
    p.add_argument(
        "--no-entropy",
        action="store_true",
        help="Disable entropy-based detection; run regex patterns only.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output; only print findings.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    scan_targets = [REPO_ROOT / p for p in args.paths]

    if not args.quiet:
        print(f"Secret-detection scan — target(s): {', '.join(str(t) for t in scan_targets)}")
        print(
            f"  Regex patterns: {len(_SECRET_PATTERNS)}  "
            f"Entropy: {'disabled' if args.no_entropy else f'≥{args.entropy_threshold} bits/char'}"
        )

    try:
        findings = scan_paths(
            scan_targets,
            entropy_threshold=args.entropy_threshold,
            use_entropy=not args.no_entropy,
        )
    except Exception as exc:
        print(f"ERROR: scanner crashed: {exc}", file=sys.stderr)
        return 2

    if findings:
        print(
            f"\nSecret-detection scan FAILED: {len(findings)} potential secret(s) found.\n",
            file=sys.stderr,
        )
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        print(
            "\nRemediation: see SECRETS_MANAGEMENT_PR.md#remediation-if-a-secret-is-detected",
            file=sys.stderr,
        )
        return 1

    if not args.quiet:
        n_files = sum(
            len(list(Path(t).rglob("*.py"))) if Path(t).is_dir() else 1
            for t in scan_targets
        )
        print(f"\nSecret-detection scan PASSED — no secret-shaped values found.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
