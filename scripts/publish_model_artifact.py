"""Sign a new model artifact and append its hash to the transparency log.

Usage:
    python -m scripts.publish_model_artifact \\
        --model-name rf \\
        --model-dir ./models \\
        --private-key-path /secrets/signing_key.pem \\
        --db-url sqlite:///ledgerlens.db

Security requirements:
    - The signing private key must be stored in an HSM or encrypted secrets
      manager (AWS Secrets Manager, HashiCorp Vault, etc.).  Never commit the
      key to source control or store it on disk unencrypted in production.
    - The transparency log DB must be backed up separately from the model
      artifact store so a coordinated attack cannot tamper with both.
"""

import argparse
import hashlib
import json
import os
import sys


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def publish(
    model_name: str,
    model_dir: str,
    private_key_path: str,
    db_url: str,
) -> str:
    """Sign *model_name* artifact, record in transparency log, return SHA-256."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from detection.persistence import (
        ModelIntegrityError,
        TransparencyLog,
        get_engine,
        get_session_factory,
        sign_metrics,
    )

    artifact_path = os.path.join(model_dir, f"{model_name}.joblib")
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    metrics_path = os.path.join(model_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.json not found in {model_dir}")

    # Load and validate the private key
    with open(private_key_path, "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ModelIntegrityError("Signing key is not an Ed25519 private key")

    # Compute artifact SHA-256
    artifact_sha = _sha256_file(artifact_path)

    # Update metrics.json with the new hash
    with open(metrics_path) as f:
        metrics = json.load(f)
    if model_name not in metrics:
        metrics[model_name] = {}
    metrics[model_name]["artifact_sha256"] = artifact_sha
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Sign metrics.json
    sign_metrics(metrics_path, private_key_path)

    # Append to transparency log
    engine = get_engine(db_url)
    session_factory = get_session_factory(engine)
    log = TransparencyLog(session_factory)
    log.append(model_name, artifact_sha)

    return artifact_sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="Model name (e.g. rf, xgb)")
    parser.add_argument("--model-dir", default="./models", help="Directory containing model artifacts")
    parser.add_argument("--private-key-path", required=True, help="Path to Ed25519 private key (PEM)")
    parser.add_argument("--db-url", default=None, help="SQLAlchemy DB URL (defaults to config)")
    args = parser.parse_args()

    if args.db_url is None:
        from config import config
        db_url = config.RISK_SCORE_DB_URL
    else:
        db_url = args.db_url

    try:
        sha = publish(
            model_name=args.model_name,
            model_dir=args.model_dir,
            private_key_path=args.private_key_path,
            db_url=db_url,
        )
        print(f"Published {args.model_name}: sha256={sha}")
        print(f"Transparency log updated in {db_url}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
