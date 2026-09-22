#!/usr/bin/env python3
"""Fetch the annual ATP/WTA result files used to construct the target list."""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive/"
    "resolve/main/{tour}/{tour}_matches_{year}.csv?download=true"
)


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "tennis-manifest/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                temporary.write_bytes(response.read())
            if temporary.stat().st_size:
                temporary.replace(destination)
                return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(".cache/match_records"))
    args = parser.parse_args()
    for tour in ("atp", "wta"):
        for year in range(2000, 2027):
            destination = args.output_dir / tour / f"{tour}_matches_{year}.csv"
            download(BASE_URL.format(tour=tour, year=year), destination)
    print(f"Fetched match records into {args.output_dir}")


if __name__ == "__main__":
    main()
