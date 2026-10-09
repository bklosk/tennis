"""Check downloaded videos: duration against the manifest and a full decode-error scan.

    uv run python scripts/verify_videos.py [VIDEO_ID ...]

Writes outputs/video_check.csv. A file is 'ok' when its duration is within 1% of the listed one and
a full software decode reports no errors.
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd

from uso.paths import DOWNLOADS, EVAL_MATCHES, OUTPUTS


def check(vid: str, listed: float) -> dict:
    path = DOWNLOADS / f"{vid}.mp4"
    if not path.exists():
        return dict(video_id=vid, status="missing")
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(path)], capture_output=True, text=True).stdout.strip() or 0)
    err = subprocess.run(["ffmpeg", "-v", "error", "-threads", "0", "-i", str(path), "-map", "0:v", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    n_err = len([line for line in err.splitlines() if line.strip()])
    ok = abs(dur - listed) / listed < 0.01 and n_err == 0
    return dict(video_id=vid, status="ok" if ok else "bad", duration=dur, listed=listed, decode_errors=n_err)


def main():
    em = pd.read_csv(EVAL_MATCHES)
    want = sys.argv[1:] or em.video_id.tolist()
    out_path = OUTPUTS / "video_check.csv"
    prev = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["video_id"])
    rows = []
    for r in em[em.video_id.isin(want)].itertuples():
        res = check(r.video_id, float(r.duration_s))
        print(res, flush=True)
        rows.append(res)
    new = pd.DataFrame(rows)
    merged = pd.concat([prev[~prev.video_id.isin(new.video_id)], new], ignore_index=True)
    merged.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
