"""Data provenance and integrity helpers for reproducible research runs."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(data_path: str | Path) -> Path:
    """Return the canonical sidecar path for a research data product."""
    path = Path(data_path)
    return path.with_suffix(path.suffix + ".provenance.json")


def write_provenance(
    data_path: str | Path,
    *,
    source_url: str,
    source_product_id: str,
    source_sha256: str | None,
    processing: dict[str, Any],
) -> Path:
    """Write a canonical provenance sidecar for a derived data file."""
    path = Path(data_path)
    record = {
        "schema_version": 1,
        "product_path": path.name,
        "product_sha256": sha256_file(path),
        "source_url": source_url,
        "source_product_id": source_product_id,
        "source_sha256": source_sha256,
        "created_utc": datetime.now(UTC).isoformat(),
        "processing": processing,
    }
    sidecar = provenance_path(path)
    sidecar.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return sidecar


def verify_provenance(data_path: str | Path) -> dict[str, Any]:
    """Load a provenance sidecar and verify the derived product digest."""
    path = Path(data_path)
    sidecar = provenance_path(path)
    if not sidecar.exists():
        raise FileNotFoundError(
            f"Research data provenance sidecar not found: {sidecar}. "
            "Prepare the DEM with tools/prepare_lola_dem.py."
        )
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    expected = str(record.get("product_sha256", "")).lower()
    actual = sha256_file(path)
    if not expected or actual != expected:
        raise ValueError(
            f"Data integrity check failed for {path}: expected {expected or '<missing>'}, "
            f"computed {actual}."
        )
    if not record.get("source_url") or not record.get("source_product_id"):
        raise ValueError(f"Incomplete provenance record: {sidecar}")
    record["verified"] = True
    return record
