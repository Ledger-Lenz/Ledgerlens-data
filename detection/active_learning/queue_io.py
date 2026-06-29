import hashlib
import hmac
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_queue(path: Path, secret: str) -> list:
    """
    Loads and validates the annotation queue JSON file.

    Verifies the HMAC-SHA256 signature to prevent label poisoning.
    If the secret is empty, verification is skipped with a warning.
    """
    if not path.exists():
        return []

    raw = path.read_bytes()
    data = json.loads(raw)

    # Extract the signature block from the root object
    expected_mac = data.pop("_hmac", None)

    if not secret:
        logger.warning(
            f"ANNOTATION_HMAC_SECRET is empty. Skipping signature verification for {path}. "
            "This is insecure and should only be used in local development."
        )
        return data.get("annotations", [])

    # Serialize with sort_keys=True for deterministic signature matching
    serialized_data = json.dumps(data, sort_keys=True).encode("utf-8")
    computed_mac = hmac.new(secret.encode("utf-8"), serialized_data, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected_mac or "", computed_mac):
        raise ValueError(f"Annotation queue HMAC mismatch: {path}")

    return data.get("annotations", [])


def save_queue(path: Path, annotations: list, secret: str) -> None:
    """
    Saves the annotation queue to a JSON file and appends a valid HMAC-SHA256.
    """
    payload = {"annotations": annotations}

    if not secret:
        logger.warning(f"Saving queue without HMAC signature to {path} (Secret is empty).")
    else:
        # Generate signature on key-sorted data structure
        serialized_data = json.dumps(payload, sort_keys=True).encode("utf-8")
        mac = hmac.new(secret.encode("utf-8"), serialized_data, hashlib.sha256).hexdigest()
        payload["_hmac"] = mac

    # Write out the final payload file cleanly
    path.write_text(json.dumps(payload, indent=2))
