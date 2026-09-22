#!/usr/bin/env python3
"""Normalize a reviewed JSONL discovery catalog for inclusion in the repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = {
    "year",
    "draw",
    "round",
    "player1",
    "player2",
    "url",
    "platform",
    "video_id",
    "title",
    "confidence",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tournament")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for index, row in enumerate(rows, 1):
        missing = REQUIRED - row.keys()
        if missing:
            raise ValueError(f"Row {index} is missing: {sorted(missing)}")
        if args.tournament:
            row["tournament"] = args.tournament
    rows.sort(
        key=lambda row: (
            row["year"],
            row["draw"],
            row["round"],
            row["player1"],
            row["player2"],
            row["video_id"],
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Imported {len(rows)} records into {args.output}")


if __name__ == "__main__":
    main()
