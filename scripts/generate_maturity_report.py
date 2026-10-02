#!/usr/bin/env python
"""Generate a human-readable maturity report from ``config/repo_maturity.yaml``.

Issue #958 — Repo maturity: add automated badge/report generation from
``repo_maturity.yaml``.

This script performs three distinct jobs:

1. **Report generation** (``--output``): reads ``config/repo_maturity.yaml``,
   delegates scoring to :mod:`scripts.repo_maturity`, and writes a
   polished Markdown summary to a file (default:
   ``docs/maturity_report.md``).

2. **Badge generation** (``--badges``): for each module listed under a
   per-module maturity rating in ``repo_maturity.yaml``, writes a
   Shields.io badge snippet to the target directory so module READMEs
   can embed it.

3. **Stale-entry CI check** (``--check-stale``): scans every ``path:``
   entry in ``repo_maturity.yaml`` and fails (exit 1) if the referenced
   path no longer exists in the repository.  Designed to run as a
   required CI step so dangling references are caught automatically.

Usage
-----
    # Full report + badges
    python scripts/generate_maturity_report.py

    # Write report to a custom path
    python scripts/generate_maturity_report.py --output reports/maturity.md

    # CI stale-entry check only (fast, no I/O beyond the YAML)
    python scripts/generate_maturity_report.py --check-stale

    # Per-module badges written to docs/badges/
    python scripts/generate_maturity_report.py --badges docs/badges

    # All three in one shot
    python scripts/generate_maturity_report.py \\
        --output docs/maturity_report.md \\
        --badges docs/badges \\
        --check-stale

Exit codes
----------
0  — all checks passed, report/badges written (if requested).
1  — ``--check-stale`` found at least one dangling path reference.
2  — unexpected error (YAML parse failure, I/O error, etc.).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATURITY_YAML = REPO_ROOT / "config" / "repo_maturity.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "maturity_report.md"

# Colour palette for badge generation (Shields.io colour names)
_BADGE_COLOUR = {
    "production": "brightgreen",
    "stable": "green",
    "beta": "yellow",
    "experimental": "orange",
    "deprecated": "red",
    "unknown": "lightgrey",
}


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        print(f"ERROR: maturity config not found: {path}", file=sys.stderr)
        sys.exit(2)
    except yaml.YAMLError as exc:
        print(f"ERROR: failed to parse {path}: {exc}", file=sys.stderr)
        sys.exit(2)
    return data


# ---------------------------------------------------------------------------
# Stale-entry check
# ---------------------------------------------------------------------------


def _collect_path_entries(data: dict) -> list[tuple[str, Path]]:
    """Walk the YAML tree and collect every ``path:`` value → resolved Path."""
    entries: list[tuple[str, Path]] = []

    def _walk(node, breadcrumb: str):
        if isinstance(node, dict):
            if "path" in node and isinstance(node["path"], str):
                entries.append((breadcrumb, REPO_ROOT / node["path"]))
            for k, v in node.items():
                _walk(v, f"{breadcrumb}.{k}" if breadcrumb else k)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                _walk(item, f"{breadcrumb}[{i}]")

    _walk(data, "")
    return entries


def check_stale(data: dict) -> list[str]:
    """Return a list of human-readable error messages for dangling paths.

    Each message is in ``path:lineno``-style so CI annotations pick them up
    automatically::

        config/repo_maturity.yaml: stale entry 'docs.checks[0].path':
            'docs/drift_detection.md' does not exist
    """
    errors: list[str] = []
    for breadcrumb, resolved in _collect_path_entries(data):
        if not resolved.exists():
            rel = resolved.relative_to(REPO_ROOT) if resolved.is_absolute() else resolved
            errors.append(
                f"config/repo_maturity.yaml: stale entry '{breadcrumb}': "
                f"'{rel}' does not exist"
            )
    return errors


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _maturity_label(score: float) -> str:
    if score >= 90:
        return "production"
    if score >= 75:
        return "stable"
    if score >= 55:
        return "beta"
    if score >= 30:
        return "experimental"
    return "deprecated"


def _badge_url(label: str, status: str) -> str:
    """Return a Shields.io static-badge URL."""
    colour = _BADGE_COLOUR.get(status, "lightgrey")
    # URL-encode spaces as %20
    label_enc = label.replace(" ", "%20")
    status_enc = status.replace(" ", "%20")
    return f"https://img.shields.io/badge/{label_enc}-{status_enc}-{colour}"


def _render_report(data: dict, repo_maturity_module_report: dict | None = None) -> str:
    """Build the full Markdown maturity report."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    version = data.get("version", "—")
    description = data.get("description", "")
    threshold = data.get("default_threshold", 60)

    lines: list[str] = [
        "# LedgerLens Repository Maturity Report",
        "",
        "> Auto-generated by `scripts/generate_maturity_report.py` (Issue #958).  "
        "**Do not hand-edit.**  Regenerate with `make maturity-report`.",
        "",
        f"**Config version:** `{version}`  ",
        f"**Description:** {description}  ",
        f"**Default pass threshold:** {threshold}  ",
        f"**Generated:** {now}",
        "",
    ]

    # ------------------------------------------------------------------
    # Section: scoring model summary (from repo_maturity module if available)
    # ------------------------------------------------------------------
    if repo_maturity_module_report:
        composite = repo_maturity_module_report.get("composite_score", 0)
        passed = repo_maturity_module_report.get("passed", False)
        status_icon = "✅" if passed else "❌"
        lines += [
            "## Composite Score",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| **Composite score** | **{composite:.1f} / 100** |",
            f"| Pass threshold | {threshold} |",
            f"| Status | {status_icon} {'PASSED' if passed else 'FAILED'} |",
            "",
        ]
        dims = repo_maturity_module_report.get("dimensions", [])
        if dims:
            lines += [
                "### Dimension Breakdown",
                "",
                "| Dimension | Score | Weight | Status |",
                "|---|---|---|---|",
            ]
            for dim in dims:
                score = dim["score"]
                label = _maturity_label(score)
                badge = f"![{label}]({_badge_url(dim['name'], label)})"
                lines.append(
                    f"| {dim['name'].title()} | {score:.1f}/100 | {dim['weight']:.0%} | {badge} |"
                )
            lines.append("")

            # Dimension detail sections
            for dim in dims:
                lines += [
                    f"#### {dim['name'].title()} (score: {dim['score']:.1f}/100)",
                    "",
                ]
                if dim.get("deductions"):
                    lines.append("**Deductions:**")
                    for d in dim["deductions"]:
                        lines.append(f"- {d}")
                    lines.append("")
                if dim.get("details"):
                    lines.append("**Passing checks:**")
                    for d in dim["details"]:
                        lines.append(f"- {d.strip()}")
                    lines.append("")

    # ------------------------------------------------------------------
    # Section: dimensions from YAML (config reference)
    # ------------------------------------------------------------------
    dimensions = data.get("dimensions", {})
    if dimensions:
        lines += [
            "## Scoring Model Reference",
            "",
            "The following dimensions and checks are declared in "
            "`config/repo_maturity.yaml`.",
            "",
        ]
        for dim_name, dim_cfg in dimensions.items():
            weight_pct = int(dim_cfg.get("weight", 0) * 100)
            desc = dim_cfg.get("description", "").strip().replace("\n", " ")
            lines += [
                f"### {dim_name.title()} (weight: {weight_pct}%)",
                "",
                f"{desc}",
                "",
                "| Check ID | Description |",
                "|---|---|",
            ]
            for check in dim_cfg.get("checks", []):
                check_id = check.get("id", "—")
                check_desc = check.get("description", check.get("path", "—"))
                lines.append(f"| `{check_id}` | {check_desc} |")
            lines.append("")

    lines.append(f"---")
    lines.append(f"*Report generated at {now} by `scripts/generate_maturity_report.py`.*")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Badge generation
# ---------------------------------------------------------------------------


def generate_badges(data: dict, badges_dir: Path) -> list[str]:
    """Write per-module badge snippets.

    Returns a list of written file paths (as strings) for logging.
    """
    badges_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # Walk dimensions → checks to find module-level paths
    dimensions = data.get("dimensions", {})
    for dim_name, dim_cfg in dimensions.items():
        for check in dim_cfg.get("checks", []):
            path_val = check.get("path")
            if not path_val:
                continue
            module_name = Path(path_val).stem
            # Derive a maturity status from the dim weight as a rough proxy
            weight = dim_cfg.get("weight", 0.5)
            status = "stable" if weight >= 0.2 else "experimental"
            badge_md = (
                f"![{module_name}]({_badge_url(module_name, status)})\n"
                f"<!-- generated by scripts/generate_maturity_report.py -->\n"
            )
            out_path = badges_dir / f"{module_name}.md"
            out_path.write_text(badge_md, encoding="utf-8")
            written.append(str(out_path))

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_maturity_report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_MATURITY_YAML,
        metavar="YAML",
        help=f"Path to repo_maturity.yaml (default: {DEFAULT_MATURITY_YAML.relative_to(REPO_ROOT)})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help=f"Write Markdown report to this file (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})",
    )
    p.add_argument(
        "--badges",
        type=Path,
        default=None,
        metavar="DIR",
        help="Write per-module badge snippets to this directory (optional)",
    )
    p.add_argument(
        "--check-stale",
        action="store_true",
        help="Fail (exit 1) if repo_maturity.yaml references a path that does not exist",
    )
    p.add_argument(
        "--json-report",
        type=Path,
        default=None,
        metavar="PATH",
        help="Also write a JSON maturity report (via scripts.repo_maturity) to this path",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output",
    )
    return p


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    parser = _build_parser()
    args = parser.parse_args(argv)

    data = _load_yaml(args.config)

    # ------------------------------------------------------------------
    # 1. Stale-entry check
    # ------------------------------------------------------------------
    stale_errors: list[str] = []
    if args.check_stale:
        stale_errors = check_stale(data)
        if stale_errors:
            print(
                f"Stale-entry check FAILED: {len(stale_errors)} dangling path(s) in "
                f"{args.config.relative_to(REPO_ROOT)}:\n",
                file=sys.stderr,
            )
            for err in stale_errors:
                print(f"  {err}", file=sys.stderr)
            return 1
        if not args.quiet:
            print(
                "Stale-entry check PASSED: all path entries in "
                f"{args.config.relative_to(REPO_ROOT)} exist."
            )

    # ------------------------------------------------------------------
    # 2. Run the scoring engine (optional — graceful if not importable)
    # ------------------------------------------------------------------
    module_report: dict | None = None
    try:
        # Import the existing repo_maturity scorer
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from repo_maturity import compute_maturity  # type: ignore[import]

        threshold = float(data.get("default_threshold", 60))
        rpt = compute_maturity(REPO_ROOT, threshold=threshold)
        module_report = rpt.to_dict()

        if args.json_report:
            import json

            args.json_report.parent.mkdir(parents=True, exist_ok=True)
            args.json_report.write_text(
                json.dumps(module_report, indent=2, default=str), encoding="utf-8"
            )
            if not args.quiet:
                print(f"JSON report written to: {args.json_report}")

    except Exception as exc:
        if not args.quiet:
            print(
                f"  [warning] Could not run scoring engine: {exc}. "
                "Generating structural report only.",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # 3. Generate Markdown report
    # ------------------------------------------------------------------
    report_md = _render_report(data, module_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_md, encoding="utf-8")
    if not args.quiet:
        print(f"Maturity report written to: {args.output}")

    # ------------------------------------------------------------------
    # 4. Generate per-module badges (optional)
    # ------------------------------------------------------------------
    if args.badges:
        written = generate_badges(data, args.badges)
        if not args.quiet:
            print(f"Badge snippets written ({len(written)} files) to: {args.badges}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
