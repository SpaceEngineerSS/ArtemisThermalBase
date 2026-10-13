"""Download the pinned research inputs listed in config/data_sources.yaml.

    python tools/fetch_research_data.py --group ephemeris

Interrupted downloads resume from the ``.part`` file. Every file with a
``sha256`` in the manifest is verified; a mismatch stops the script and the
file is kept as ``.rejected`` for inspection. A lock file with sizes and
hashes is written to ``data/<group>.provenance.lock.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_ingestion.provenance import sha256_file  # noqa: E402

ALLOWED_HOSTS = {"naif.jpl.nasa.gov", "planetarymaps.usgs.gov"}
CHUNK = 1 << 20


def _download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    have = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "ArtemisThermalBase"})
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        resumed = have > 0 and response.status == 206
        total = int(response.headers.get("Content-Length", 0)) + (have if resumed else 0)
        done = have if resumed else 0
        with partial.open("ab" if resumed else "wb") as stream:
            while chunk := response.read(CHUNK):
                stream.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    os.replace(partial, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=str(ROOT / "config" / "data_sources.yaml"))
    parser.add_argument("--group", choices=["ephemeris", "topography"], required=True)
    parser.add_argument("--force", action="store_true", help="download even if present")
    args = parser.parse_args(argv)

    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for name, source in manifest["sources"].items():
        if source["group"] != args.group:
            continue
        url = str(source["url"])
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
            raise ValueError(f"{name}: {url} is not on the official host allowlist.")
        destination = ROOT / source["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.force or not destination.exists():
            print(f"{name}: {url}")
            _download(url, destination)
        digest = sha256_file(destination)
        expected = source.get("sha256")
        if expected and digest != expected:
            rejected = destination.with_suffix(destination.suffix + ".rejected")
            os.replace(destination, rejected)
            raise SystemExit(
                f"{name}: SHA-256 mismatch.\n  expected {expected}\n  got      {digest}\n"
                f"  kept as {rejected}"
            )
        print(f"{name}: ok ({'pinned' if expected else 'unpinned'} sha256 {digest[:12]}...)")
        records.append({
            "name": name,
            "product_id": source["product_id"],
            "url": url,
            "path": Path(source["path"]).as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": digest,
        })

    lock_path = ROOT / "data" / f"{args.group}.provenance.lock.json"
    lock_path.write_text(json.dumps({
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "records": records,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {lock_path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
