#!/usr/bin/env python3
"""Validate that dependency lockfiles are exact, complete, and current.

The check intentionally uses only the Python standard library so it can run
before a repository lockfile is trusted and installed.

Exit codes:
    0  Every configured lockfile is valid.
    1  At least one lockfile validation failed.
    2  Invalid command-line usage.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

LOCKFILE_VERSION = "1"
LOCKFILE_RESOLVER = "uv==0.8.3"
LOCKFILE_TARGET = "universal-python>=3.11,<3.13"
EXPECTED_PYTHON_POLICY = ">=3.11,<3.13"
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_TARGETS = {
    "requirements-linux-py311.txt": "linux-python==3.11",
    "requirements-linux-py312.txt": "linux-python==3.12",
    "requirements-macos-py311.txt": "macos-python==3.11",
    "requirements-macos-py312.txt": "macos-python==3.12",
}
DEFAULT_PAIRS = tuple(
    (REPO_ROOT / "requirements.in", REPO_ROOT / "requirements" / filename)
    for filename in LOCK_TARGETS
)

_NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_REQUIREMENT_RE = re.compile(
    rf"^(?P<name>{_NAME_PATTERN})" r"(?P<extras>\[[A-Za-z0-9._,\s-]+\])?" r"\s*(?P<specifier>.*)$"
)
_EXACT_PIN_RE = re.compile(r"^==\s*(?P<version>[A-Za-z0-9][A-Za-z0-9.!+_-]*)$")
_HASH_OPTION_RE = re.compile(r"\s+--hash=sha256:(?P<digest>[0-9a-fA-F]{64})(?=\s|$)")
_FINGERPRINT_RE = re.compile(r"^# source-sha256: (?P<digest>[0-9a-f]{64})$")
_VERSION_RE = re.compile(r"^# lockfile-version: (?P<version>\S+)$")
_SOURCE_RE = re.compile(r"^# source: (?P<source>\S+)$")
_RESOLVER_RE = re.compile(r"^# resolver: (?P<resolver>\S+)$")
_TARGET_RE = re.compile(r"^# target: (?P<target>\S+)$")
_PYTHON_POLICY_RE = re.compile(
    r'^\s*requires-python\s*=\s*"(?P<policy>[^"]+)"\s*(?:#.*)?$', re.MULTILINE
)
_SPECIFIER_RE = re.compile(r"^(?P<operator>~=|==|!=|<=|>=|<|>|===)\s*(?P<version>\S+)$")
_PEP440_VERSION_RE = re.compile(
    r"^\s*v?"
    r"(?:(?P<epoch>[0-9]+)!)?"
    r"(?P<release>[0-9]+(?:\.[0-9]+)*)"
    r"(?:(?:[-_.]?"
    r"(?P<pre_label>a|b|c|rc|alpha|beta|pre|preview)"
    r"[-_.]?(?P<pre_number>[0-9]+)?)"
    r")?"
    r"(?:(?:-(?P<post_number1>[0-9]+))|"
    r"(?:[-_.]?(?P<post_label>post|rev|r)[-_.]?(?P<post_number2>[0-9]+)?))?"
    r"(?:[-_.]?(?P<dev_label>dev)[-_.]?(?P<dev_number>[0-9]+)?)?"
    r"(?:\+(?P<local>[a-z0-9]+(?:[-_.][a-z0-9]+)*))?"
    r"\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Requirement:
    """A dependency declaration reduced to the fields needed by this check."""

    name: str
    extras: str
    marker: str
    specifier: str
    hashes: tuple[str, ...]
    line_number: int

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.name, self.extras, self.marker)


def canonicalize_name(name: str) -> str:
    """Apply the package-name normalization defined by PEP 503."""

    return re.sub(r"[-_.]+", "-", name).lower()


def source_digest(path: Path) -> str:
    """Return the SHA-256 fingerprint of a dependency source file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_target(path: Path) -> str:
    """Return the declared environment contract for a lockfile path."""

    return LOCK_TARGETS.get(path.name, LOCKFILE_TARGET)


def validate_python_policy(path: Path) -> list[str]:
    """Ensure project metadata and the supported lock targets cannot drift apart."""

    if not path.is_file():
        return [f"{path}: project metadata file does not exist"]
    policies = [match.group("policy") for match in _PYTHON_POLICY_RE.finditer(path.read_text())]
    if len(policies) != 1:
        return [f"{path}: expected exactly one requires-python declaration"]
    if policies[0] != EXPECTED_PYTHON_POLICY:
        return [
            f"{path}: requires-python is {policies[0]!r}; expected "
            f"{EXPECTED_PYTHON_POLICY!r} for the configured Python 3.11/3.12 locks"
        ]
    return []


def _logical_lines(text: str) -> Iterable[tuple[int, str]]:
    """Yield requirement lines with pip-style backslash continuations joined."""

    start = 0
    parts: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not parts and (not stripped or stripped.startswith("#")):
            continue

        if not parts:
            start = line_number

        continued = stripped.endswith("\\")
        if continued:
            stripped = stripped[:-1].rstrip()
        parts.append(stripped)

        if not continued:
            yield start, " ".join(parts)
            parts = []

    if parts:
        yield start, " ".join(parts)


def _parse_requirement(
    value: str, line_number: int, path: Path, errors: list[str]
) -> Requirement | None:
    if value.startswith(("-", ".")) or " @ " in value:
        errors.append(
            f"{path}:{line_number}: unsupported requirement form {value!r}; "
            "use a package name and version specifier"
        )
        return None

    hashes = tuple(match.group("digest").lower() for match in _HASH_OPTION_RE.finditer(value))
    without_hashes = _HASH_OPTION_RE.sub("", value).strip()
    if "--hash" in without_hashes:
        errors.append(
            f"{path}:{line_number}: malformed or unsupported hash option; "
            "expected --hash=sha256:<64 hexadecimal characters>"
        )
        return None
    requirement_text, separator, marker = without_hashes.partition(";")
    match = _REQUIREMENT_RE.fullmatch(requirement_text.strip())
    if not match:
        errors.append(f"{path}:{line_number}: could not parse requirement {value!r}")
        return None

    marker = " ".join(marker.split()) if separator else ""
    extras_value = match.group("extras") or ""
    extras = ""
    if extras_value:
        extras = ",".join(
            sorted(
                canonicalize_name(item.strip())
                for item in extras_value[1:-1].split(",")
                if item.strip()
            )
        )

    return Requirement(
        name=canonicalize_name(match.group("name")),
        extras=extras,
        marker=marker,
        specifier=match.group("specifier").strip(),
        hashes=hashes,
        line_number=line_number,
    )


def _version_key(value: str) -> tuple[object, ...]:
    """Return a comparison key for the PEP 440 version forms allowed in manifests."""

    match = _PEP440_VERSION_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"unsupported PEP 440 version {value!r}")

    release = tuple(int(part) for part in match.group("release").split("."))
    while len(release) > 1 and release[-1] == 0:
        release = release[:-1]

    pre_label = match.group("pre_label")
    dev_number = match.group("dev_number")
    has_dev = match.group("dev_label") is not None
    if pre_label is None:
        pre: tuple[object, ...] = (-1,) if has_dev else (1,)
    else:
        normalized_pre = {
            "alpha": "a",
            "beta": "b",
            "c": "rc",
            "pre": "rc",
            "preview": "rc",
        }.get(pre_label.lower(), pre_label.lower())
        pre = (
            0,
            {"a": 0, "b": 1, "rc": 2}[normalized_pre],
            int(match.group("pre_number") or 0),
        )

    post_number = match.group("post_number1") or match.group("post_number2")
    post: tuple[object, ...] = (
        (-1,)
        if match.group("post_label") is None and post_number is None
        else (0, int(post_number or 0))
    )
    dev: tuple[object, ...] = (1,) if not has_dev else (0, int(dev_number or 0))

    local_value = match.group("local")
    local: tuple[object, ...]
    if local_value is None:
        local = (-1,)
    else:
        local_parts: list[tuple[object, ...]] = []
        for part in re.split(r"[-_.]", local_value.lower()):
            local_parts.append((1, int(part)) if part.isdigit() else (0, part))
        local = (0, *local_parts)

    return (int(match.group("epoch") or 0), release, pre, post, dev, local)


def _release_prefix(value: str) -> tuple[int, ...]:
    match = _PEP440_VERSION_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"unsupported PEP 440 version {value!r}")
    return tuple(int(part) for part in match.group("release").split("."))


def _specifier_satisfied(version: str, specifier: str) -> bool:
    """Check an exact locked version against a comma-separated PEP 440 policy."""

    if not specifier:
        return True

    candidate_key = _version_key(version)
    for clause in specifier.split(","):
        match = _SPECIFIER_RE.fullmatch(clause.strip())
        if match is None:
            raise ValueError(f"unsupported version specifier {clause.strip()!r}")

        operator = match.group("operator")
        required = match.group("version")
        if operator == "===":
            matches = version.lower() == required.lower()
        elif required.endswith(".*"):
            if operator not in {"==", "!="}:
                raise ValueError(f"wildcard version {required!r} is only valid with == or !=")
            prefix = _release_prefix(required[:-2])
            matches = _release_prefix(version)[: len(prefix)] == prefix
            if operator == "!=":
                matches = not matches
        else:
            required_key = _version_key(required)
            if operator == "~=":
                release = _release_prefix(required)
                if len(release) < 2:
                    raise ValueError("compatible-release specifiers require at least two segments")
                upper_release = release[:-1]
                upper_release = (*upper_release[:-1], upper_release[-1] + 1)
                upper_key = _version_key(".".join(str(part) for part in upper_release))
                matches = candidate_key >= required_key and candidate_key < upper_key
            else:
                matches = {
                    "==": candidate_key == required_key,
                    "!=": candidate_key != required_key,
                    "<=": candidate_key <= required_key,
                    ">=": candidate_key >= required_key,
                    "<": candidate_key < required_key,
                    ">": candidate_key > required_key,
                }[operator]

        if not matches:
            return False
    return True


def _parse_requirements(path: Path, errors: list[str]) -> list[Requirement]:
    requirements: list[Requirement] = []
    seen: dict[tuple[str, str, str], Requirement] = {}

    for line_number, value in _logical_lines(path.read_text(encoding="utf-8")):
        requirement = _parse_requirement(value, line_number, path, errors)
        if requirement is None:
            continue

        previous = seen.get(requirement.key)
        if previous is not None:
            errors.append(
                f"{path}:{line_number}: duplicate requirement {requirement.name!r}; "
                f"first declared on line {previous.line_number}"
            )
            continue

        seen[requirement.key] = requirement
        requirements.append(requirement)

    return requirements


def _header_value(
    path: Path, lines: Sequence[str], pattern: re.Pattern[str], label: str, errors: list[str]
) -> str | None:
    matches = [match for line in lines if (match := pattern.fullmatch(line))]
    if len(matches) != 1:
        errors.append(f"{path}: expected exactly one valid {label} header")
        return None
    return next(iter(matches[0].groupdict().values()))


def validate_pair(manifest_path: Path, lockfile_path: Path) -> list[str]:
    """Return actionable validation errors for one manifest/lockfile pair."""

    errors: list[str] = []
    if not manifest_path.is_file():
        return [f"{manifest_path}: dependency source file does not exist"]
    if not lockfile_path.is_file():
        return [f"{lockfile_path}: lockfile does not exist; generate it from {manifest_path.name}"]

    lock_lines = lockfile_path.read_text(encoding="utf-8").splitlines()
    version = _header_value(lockfile_path, lock_lines, _VERSION_RE, "lockfile-version", errors)
    source = _header_value(lockfile_path, lock_lines, _SOURCE_RE, "source", errors)
    resolver = _header_value(lockfile_path, lock_lines, _RESOLVER_RE, "resolver", errors)
    target = _header_value(lockfile_path, lock_lines, _TARGET_RE, "target", errors)
    fingerprint = _header_value(lockfile_path, lock_lines, _FINGERPRINT_RE, "source-sha256", errors)

    if version is not None and version != LOCKFILE_VERSION:
        errors.append(
            f"{lockfile_path}: unsupported lockfile version {version!r}; "
            f"expected {LOCKFILE_VERSION!r}"
        )
    if source is not None and source != manifest_path.name:
        errors.append(
            f"{lockfile_path}: source header names {source!r}, expected {manifest_path.name!r}"
        )
    if resolver is not None and resolver != LOCKFILE_RESOLVER:
        errors.append(
            f"{lockfile_path}: resolver header names {resolver!r}, "
            f"expected {LOCKFILE_RESOLVER!r}"
        )
    expected_target = lock_target(lockfile_path)
    if target is not None and target != expected_target:
        errors.append(
            f"{lockfile_path}: target header names {target!r}, " f"expected {expected_target!r}"
        )

    expected_fingerprint = source_digest(manifest_path)
    if fingerprint is not None and fingerprint != expected_fingerprint:
        errors.append(
            f"{lockfile_path}: stale source fingerprint; regenerate the lockfile from "
            f"{manifest_path.name} and run `python scripts/validate_lockfiles.py --stamp`"
        )

    manifest_requirements = _parse_requirements(manifest_path, errors)
    lock_requirements = _parse_requirements(lockfile_path, errors)

    for requirement in lock_requirements:
        exact_pin = _EXACT_PIN_RE.fullmatch(requirement.specifier)
        if exact_pin is None:
            errors.append(
                f"{lockfile_path}:{requirement.line_number}: {requirement.name!r} is not "
                "locked to one exact version with =="
            )
        if not requirement.hashes:
            errors.append(
                f"{lockfile_path}:{requirement.line_number}: {requirement.name!r} has no "
                "sha256 artifact hash; regenerate with `make lock`"
            )
        elif len(set(requirement.hashes)) != len(requirement.hashes):
            errors.append(
                f"{lockfile_path}:{requirement.line_number}: {requirement.name!r} "
                "contains a duplicate sha256 artifact hash"
            )

    locked_by_dependency: dict[tuple[str, str], list[Requirement]] = {}
    for locked in lock_requirements:
        locked_by_dependency.setdefault((locked.name, locked.extras), []).append(locked)

    for requirement in manifest_requirements:
        candidates = locked_by_dependency.get((requirement.name, requirement.extras), [])
        if requirement.marker:
            candidates = [locked for locked in candidates if locked.marker == requirement.marker]
        if not candidates:
            errors.append(
                f"{lockfile_path}: missing direct dependency {requirement.name!r} "
                f"declared at {manifest_path}:{requirement.line_number}"
            )
            continue

        for locked in candidates:
            exact_pin = _EXACT_PIN_RE.fullmatch(locked.specifier)
            if exact_pin is None:
                continue
            locked_version = exact_pin.group("version")
            try:
                compatible = _specifier_satisfied(locked_version, requirement.specifier)
            except ValueError as exc:
                errors.append(
                    f"{manifest_path}:{requirement.line_number}: cannot validate "
                    f"{requirement.name!r}: {exc}"
                )
                break
            else:
                if not compatible:
                    errors.append(
                        f"{lockfile_path}:{locked.line_number}: locked version "
                        f"{locked_version!r} for {requirement.name!r} does not satisfy "
                        f"manifest specifier {requirement.specifier!r} declared at "
                        f"{manifest_path}:{requirement.line_number}"
                    )

    if not manifest_requirements:
        errors.append(f"{manifest_path}: dependency source contains no requirements")
    if not lock_requirements:
        errors.append(f"{lockfile_path}: lockfile contains no requirements")

    return errors


def stamp_pair(manifest_path: Path, lockfile_path: Path) -> str | None:
    """Add or refresh managed headers after dependency resolution."""

    if not manifest_path.is_file():
        return f"{manifest_path}: dependency source file does not exist"
    if not lockfile_path.is_file():
        return f"{lockfile_path}: lockfile does not exist"

    lines = lockfile_path.read_text(encoding="utf-8").splitlines()
    managed_prefixes = (
        "# lockfile-version:",
        "# source:",
        "# source-sha256:",
        "# resolver:",
        "# target:",
    )
    body = [line for line in lines if not line.startswith(managed_prefixes)]
    headers = [
        f"# lockfile-version: {LOCKFILE_VERSION}",
        f"# source: {manifest_path.name}",
        f"# source-sha256: {source_digest(manifest_path)}",
        f"# resolver: {LOCKFILE_RESOLVER}",
        f"# target: {lock_target(lockfile_path)}",
    ]
    lockfile_path.write_text("\n".join([*headers, *body]) + "\n", encoding="utf-8")
    return None


def _parse_pair(value: str) -> tuple[Path, Path]:
    manifest, separator, lockfile = value.partition(":")
    if not separator or not manifest or not lockfile:
        raise argparse.ArgumentTypeError("pairs must use MANIFEST:LOCKFILE syntax")
    return (Path(manifest).resolve(), Path(lockfile).resolve())


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate exact pins and source fingerprints in dependency lockfiles."
    )
    parser.add_argument(
        "--pair",
        action="append",
        type=_parse_pair,
        dest="pairs",
        metavar="MANIFEST:LOCKFILE",
        help="validate a custom pair (repeatable); defaults to all supported environment locks",
    )
    parser.add_argument(
        "--stamp",
        action="store_true",
        help="refresh source fingerprints before validating (use only after resolving locks)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    pairs = tuple(args.pairs) if args.pairs else DEFAULT_PAIRS

    if args.stamp:
        stamp_errors = [
            error
            for manifest_path, lockfile_path in pairs
            if (error := stamp_pair(manifest_path, lockfile_path)) is not None
        ]
        if stamp_errors:
            print("Lockfile stamping failed:", file=sys.stderr)
            for error in stamp_errors:
                print(f"  - {error}", file=sys.stderr)
            return 1

    errors = ([] if args.pairs else validate_python_policy(REPO_ROOT / "pyproject.toml")) + [
        error
        for manifest_path, lockfile_path in pairs
        for error in validate_pair(manifest_path, lockfile_path)
    ]
    if errors:
        print("Lockfile validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    for manifest_path, lockfile_path in pairs:
        print(f"Validated {lockfile_path.name} against {manifest_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
