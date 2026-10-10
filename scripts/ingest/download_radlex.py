#!/usr/bin/env python3
"""Download RadLex OWL file from BioPortal.

Usage:
    python scripts/ingest/download_radlex.py [--output-dir data/raw]

Requires a BioPortal API key. Set via BIOPORTAL_API_KEY env var
or pass --api-key. Free registration at https://bioportal.bioontology.org/accounts/new
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path


BIOPORTAL_DOWNLOAD_URL = (
    "https://data.bioontology.org/ontologies/RADLEX/download"
    "?apikey={api_key}&download_format=rdf"
)

RADLEX_FILENAME = "radlex.owl"


def download_radlex(api_key: str, output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / RADLEX_FILENAME

    if dest.exists():
        print(f"[skip] {dest} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    url = BIOPORTAL_DOWNLOAD_URL.format(api_key=api_key)
    print(f"Downloading RadLex OWL to {dest} ...")

    req = urllib.request.Request(url)
    req.add_header("Accept", "application/rdf+xml")

    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            print(f"\r  {total / 1e6:.1f} MB downloaded", end="", flush=True)

    print(f"\n  Done. {dest.stat().st_size / 1e6:.1f} MB")
    return dest


def main():
    parser = argparse.ArgumentParser(description="Download RadLex OWL from BioPortal")
    parser.add_argument(
        "--output-dir", default="data/raw", help="Directory to save the OWL file"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BIOPORTAL_API_KEY", "dc114d57-336c-4c0c-9f61-eb534a7d0250"),
        help="BioPortal API key (or set BIOPORTAL_API_KEY env var)",
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "Error: BioPortal API key required.\n"
            "  Set BIOPORTAL_API_KEY env var or pass --api-key.\n"
            "  Register free at https://bioportal.bioontology.org/accounts/new",
            file=sys.stderr,
        )
        sys.exit(1)

    download_radlex(args.api_key, args.output_dir)


if __name__ == "__main__":
    main()