"""Download the dev/test match videos in data/eval_matches.csv (user-approved, 2026-10-08).

    uv run python scripts/fetch_videos.py [--role dev|test|all] [--parallel 4]

* Matches before 2011 are SD sources and are fetched at 720p; later matches at 1080p.
* Partial files are never resumed. A resumed download can splice two different encodes of the
  same format into one corrupt file (seen 2026-10-08), so every attempt starts from scratch.
* Each finished file is verified (duration within 1% of the listed one, no decode errors) and
  re-downloaded if it fails.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_videos import check  # noqa: E402

from uso.paths import DOWNLOADS, EVAL_MATCHES  # noqa: E402


def _clear_partials(vid: str) -> None:
    for p in DOWNLOADS.glob(f"{vid}.*"):
        if p.name != f"{vid}.mp4":
            p.unlink(missing_ok=True)


def fetch(vid: str, year: int, listed: float, attempts: int = 6) -> str:
    out = DOWNLOADS / f"{vid}.mp4"
    h = 720 if year <= 2010 else 1080
    for k in range(attempts):
        if out.exists():
            res = check(vid, listed)
            if res["status"] == "ok":
                return f"ok {vid} ({res['duration']:.0f}s)"
            print(f"bad {vid}: {res}; deleting", flush=True)
            out.unlink()
        _clear_partials(vid)
        cmd = ["uvx", "--from", "yt-dlp[default]", "yt-dlp", "--js-runtimes", "node", "--no-playlist", "--no-progress",
               "--no-continue", "-f", f"bv*[height<={h}][vcodec^=avc1]+ba[ext=m4a]/bv*[height<={h}]+ba/b",
               "--merge-output-format", "mp4", "-o", str(DOWNLOADS / "%(id)s.%(ext)s"), "--", vid]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"yt-dlp failed {vid} (attempt {k + 1}): {r.stderr.strip()[-300:]}", flush=True)
            time.sleep(min(300, 30 * 2 ** k))  # network outages: back off before a fresh attempt
    if out.exists() and check(vid, listed)["status"] == "ok":
        return f"ok {vid}"
    return f"FAILED {vid}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="all")
    ap.add_argument("--parallel", type=int, default=4)
    a = ap.parse_args()
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    em = pd.read_csv(EVAL_MATCHES)
    if a.role != "all":
        em = em[em.role == a.role]
    with ThreadPoolExecutor(a.parallel) as ex:
        futs = [ex.submit(fetch, r.video_id, int(r.year), float(r.duration_s)) for r in em.itertuples()]
        for f in futs:
            print(f.result(), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
