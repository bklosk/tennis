#!/usr/bin/env python3
"""Refresh flat metadata from official Australian Open and US Open collections."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

COLLECTIONS = {
    "australian-open": {
        "channel-search": "https://www.youtube.com/@australianopen/search?query=full%20match",
        "classic-full-matches": "https://www.youtube.com/playlist?list=PL2RR--XMozwWcV6-VEEMbCNnJ-Xre1BU7",
        "full-matches-2022": "https://www.youtube.com/playlist?list=PL2RR--XMozwUsQ8ieTUc-0VkWdKpzV4mK",
        "full-matches-2023": "https://www.youtube.com/playlist?list=PL2RR--XMozwUKlrwbxNPUQAW-SU8AJvI_",
        "full-matches-2024": "https://www.youtube.com/playlist?list=PL2RR--XMozwUbAdvjCLuhe4Cuqgfv6X1U",
        "full-matches-2025": "https://www.youtube.com/playlist?list=PL2RR--XMozwVbfUzQxA_8mlkkqbR7c4Gs",
    },
    "us-open": {
        "channel-search": "https://www.youtube.com/@usopen/search?query=full%20match",
        "classic-full-matches": "https://www.youtube.com/playlist?list=PL_2A0MxHOgdZxZ3vK104p51lFOFVsEgrR",
        "iconic-full-matches": "https://www.youtube.com/playlist?list=PL_2A0MxHOgdaneHVlaxliKb0KqOJ5nJC4",
        "full-matches-2020": "https://www.youtube.com/playlist?list=PL_2A0MxHOgdZAYx1-kwgpR2ljiUElhk7y",
        "full-matches-2021": "https://www.youtube.com/playlist?list=PL_2A0MxHOgdY7HQMncvHtHLoE4PtMBnFy",
        "full-matches-2022": "https://www.youtube.com/playlist?list=PL_2A0MxHOgdYzDdqZaYu4nQ6jdl95rKvK",
        "full-matches-2023": "https://www.youtube.com/playlist?list=PL_2A0MxHOgdaYmi__7UiR8HQqkPaGdmqr",
        "full-matches-2024": "https://www.youtube.com/playlist?list=PL_2A0MxHOgda-1tlPgByVmCY2Ril6gMCQ",
        "full-matches-2025": "https://www.youtube.com/playlist?list=PL_2A0MxHOgdYNA0GLetzrLWZIp-8kWP_v",
        "full-matches-2026": "https://www.youtube.com/playlist?list=PLDq1x7qnJ698",
    },
}


def extract(command: list[str], url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        result = subprocess.run(
            [*command, "--flat-playlist", "--dump-single-json", url],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode == 0 and result.stdout.strip():
            document = json.loads(result.stdout)
            destination.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return
        if attempt == 3:
            raise RuntimeError(result.stderr.strip() or f"No metadata returned for {url}")
        time.sleep(2**attempt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(".cache/discovery"))
    parser.add_argument(
        "--yt-dlp-command",
        default="uvx --from yt-dlp yt-dlp",
        help="Command used to invoke a current yt-dlp release.",
    )
    args = parser.parse_args()
    command = shlex.split(args.yt_dlp_command)
    for tournament, collections in COLLECTIONS.items():
        for name, url in collections.items():
            destination = args.output_dir / tournament / f"{name}.json"
            print(f"{tournament}: {name}", flush=True)
            extract(command, url, destination)


if __name__ == "__main__":
    main()
