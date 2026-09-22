#!/usr/bin/env python3
"""Search YouTube metadata for target matches not covered by official uploads."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROUND_NAMES = {"QF": "quarterfinal", "SF": "semifinal", "F": "final"}


def search(
    row: dict[str, str],
    command: list[str],
    output_dir: Path,
    result_count: int,
) -> tuple[str, int, str]:
    query = (
        f'"{row["player1"]}" "{row["player2"]}" {row["year"]} '
        f'{row["tournament"]} {ROUND_NAMES[row["round"]]} full match'
    )
    tournament_dir = "australian-open" if row["tournament"] == "Australian Open" else "us-open"
    destination = output_dir / tournament_dir / f'{row["match_id"]}.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        document = json.loads(destination.read_text(encoding="utf-8"))
        return row["match_id"], len(document.get("entries") or []), ""

    for attempt in range(3):
        try:
            result = subprocess.run(
                [
                    *command,
                    "--flat-playlist",
                    "--dump-single-json",
                    f"ytsearch{result_count}:{query}",
                ],
                capture_output=True,
                text=True,
                timeout=90,
            )
            if result.returncode == 0 and result.stdout.strip():
                document = json.loads(result.stdout)
                document["_target_match_id"] = row["match_id"]
                document["_target_tournament"] = row["tournament"]
                document["_search_query"] = query
                destination.write_text(
                    json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                return row["match_id"], len(document.get("entries") or []), ""
            error = (result.stderr or "empty output")[-500:]
        except Exception as exc:
            error = str(exc)
        time.sleep(2**attempt)
    return row["match_id"], 0, error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=Path, default=Path("data/matches.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(".cache/global_search"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--results", type=int, default=5)
    parser.add_argument(
        "--yt-dlp-command",
        default="uvx --from yt-dlp yt-dlp",
    )
    args = parser.parse_args()

    with args.matches.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["match_played"] == "True"
            and row["coverage_status"] != "verified_official_full_match"
        ]

    failures = []
    entries = 0
    command = shlex.split(args.yt_dlp_command)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(search, row, command, args.output_dir, args.results) for row in rows
        ]
        for index, future in enumerate(as_completed(futures), 1):
            match_id, count, error = future.result()
            entries += count
            if error:
                failures.append((match_id, error))
            if index % 20 == 0:
                print(
                    f"completed={index}/{len(rows)} entries={entries} "
                    f"failures={len(failures)}",
                    flush=True,
                )

    print(f"completed={len(rows)} entries={entries} failures={len(failures)}")
    for match_id, error in failures:
        print(f"FAILED {match_id}: {error}")


if __name__ == "__main__":
    main()
