"""
Model provenance: metadata + integrity check for the deployed checkpoint.
Read-only with respect to the model itself -- never modifies best_lite_model.pth.
"""

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "artifacts" / "model" / "metadata.json"


def load_model_metadata():
    """Return the metadata dict, or None if it doesn't exist."""
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_integrity(model_path):
    """
    Compare the on-disk checkpoint's SHA-256 against metadata.json.
    Returns a dict: {"ok": bool, "reason": str}. Never raises -- integrity
    problems are surfaced to the caller as a warning, not a crash.
    """
    metadata = load_model_metadata()
    if metadata is None:
        return {"ok": False, "reason": "No metadata.json found -- provenance unverified."}

    expected = metadata.get("sha256")
    if not expected:
        return {"ok": False, "reason": "metadata.json has no sha256 field."}

    try:
        actual = _sha256(model_path)
    except OSError as e:
        return {"ok": False, "reason": f"Could not hash checkpoint: {e}"}

    if actual != expected:
        return {
            "ok": False,
            "reason": f"Checksum mismatch -- expected {expected[:12]}..., got {actual[:12]}...",
        }
    return {"ok": True, "reason": "Checksum matches metadata.json."}
