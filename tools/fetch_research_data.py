"""Fetch versioned research inputs from the official source manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_ingestion.provenance import sha256_file

ALLOWED_HOSTS = {"naif.jpl.nasa.gov", "planetarymaps.usgs.gov"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="config/data_sources.yaml")
    parser.add_argument("--group", choices=["ephemeris", "topography"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for name, source in manifest["sources"].items():
        if source["group"] != args.group:
            continue
        destination = Path(source["path"])
        source_url = str(source["url"])
        parsed = urlparse(source_url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
            raise ValueError(f"Source URL is not on the official allowlist: {source_url}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.force or not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".part")
            print(f"Downloading {name}: {source_url}")
            try:
                downloaded = temporary.stat().st_size if temporary.exists() else 0
                head = urllib.request.Request(source_url, method="HEAD")  # noqa: S310
                with urllib.request.urlopen(head, timeout=60) as response:  # noqa: S310
                    remote_size = int(response.headers.get("Content-Length", 0))
                if not downloaded or not remote_size or downloaded < remote_size:
                    request = urllib.request.Request(source_url)  # noqa: S310
                    if downloaded:
                        request.add_header("Range", f"bytes={downloaded}-")
                    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                        resume = downloaded > 0 and response.status == 206
                        mode = "ab" if resume else "wb"
                        with temporary.open(mode) as stream:
                            while chunk := response.read(1024 * 1024):
                                stream.write(chunk)
                for attempt in range(10):
                    try:
                        os.replace(temporary, destination)
                        break
                    except PermissionError:
                        if attempt == 9:
                            raise
                        time.sleep(0.2)
            finally:
                if temporary.exists():
                    try:
                        temporary.unlink()
                    except PermissionError:
                        pass
        records.append({
            "name": name,
            "product_id": source["product_id"],
            "url": source_url,
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        })

    lock_path = Path("data") / f"{args.group}.provenance.lock.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "records": records,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
