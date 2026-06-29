"""CLI to list archived model versions with training dates and metrics.

Usage:
    python -m scripts.list_model_versions
    python -m scripts.list_model_versions --max-rows 20
"""

import argparse
import json
import os

ARCHIVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "archive"
)


def load_version_metadata(version_dir: str) -> dict:
    """Load metrics.json and model_metadata.json from an archived version."""
    metrics_path = os.path.join(version_dir, "metrics.json")
    metadata_path = os.path.join(version_dir, "model_metadata.json")

    data = {"version": os.path.basename(version_dir)}

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data["metrics"] = json.load(f)

    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        data["trained_at"] = meta.get("trained_at", "unknown")
        data["n_training_rows"] = meta.get("n_training_rows", "?")
        data["n_test_rows"] = meta.get("n_test_rows", "?")

    return data


def list_versions(archive_dir: str = ARCHIVE_DIR) -> list[dict]:
    if not os.path.isdir(archive_dir):
        return []

    versions = []
    for entry in sorted(os.listdir(archive_dir), reverse=True):
        version_dir = os.path.join(archive_dir, entry)
        if os.path.isdir(version_dir):
            versions.append(load_version_metadata(version_dir))

    return versions


def print_table(versions: list[dict], max_rows: int | None = None) -> None:
    if not versions:
        print("No archived model versions found.")
        return

    if max_rows is not None:
        versions = versions[:max_rows]

    header = (
        f"{'Version':<20} {'Trained At':<28} {'Rows (train/test)':<20} {'AUC-ROC':<20} {'F1':<20}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for v in versions:
        version = v["version"]
        trained_at = v.get("trained_at", "unknown")
        n_train = v.get("n_training_rows", "?")
        n_test = v.get("n_test_rows", "?")
        rows_str = f"{n_train}/{n_test}"

        metrics = v.get("metrics", {})
        auc_strs = []
        f1_strs = []
        for model_name in sorted(metrics.keys()):
            m = metrics[model_name]
            auc_strs.append(f"{m.get('auc_roc', '?'):.4f}")
            f1_strs.append(f"{m.get('f1', '?'):.4f}")

        auc_display = ", ".join(auc_strs) if auc_strs else "?"
        f1_display = ", ".join(f1_strs) if f1_strs else "?"

        print(f"{version:<20} {trained_at:<28} {rows_str:<20} {auc_display:<20} {f1_display:<20}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List archived model versions")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of versions shown")
    parser.add_argument(
        "--archive-dir",
        default=ARCHIVE_DIR,
        help="Archive directory (default: models/archive)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    versions = list_versions(args.archive_dir)
    print_table(versions, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
