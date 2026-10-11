"""Archive every US Open QF/SF/F full-match video (2001-2025) to the DigitalOcean Space.

    uv run python scripts/archive_videos.py [--parallel 3] [--limit N] [--dry-run]

For each match in data/archive_targets.csv (one full-match upload per match, from the manifest):
download it (yt-dlp, never resuming partial files; 720p for SD-era sources before 2011, 1080p
after), verify it (duration within 1% of the listing and a clean full decode), upload it as a
private object, check the stored size, and delete the local copy, keeping only the evaluation
videos that the pipeline reads from downloads/. Already-archived objects are skipped, so the run
can be stopped and restarted at any time.

Objects: s3://benklosky-data/tennis/usopen-video/YEAR/MATCH_ID__VIDEO_ID.mp4, plus manifest.csv
(match, video id, size, duration, sha256, status) and missing.csv (targets with no full-match
upload). Credentials are read from ~/.s3cfg and never printed.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_videos import check  # noqa: E402

from uso.paths import DOWNLOADS, EVAL_MATCHES, OUTPUTS, PROJECT  # noqa: E402

BUCKET = "benklosky-data"
ENDPOINT = "https://nyc3.digitaloceanspaces.com"
PREFIX = "tennis/usopen-video/"
TARGETS = PROJECT / "data" / "archive_targets.csv"
MANIFEST = OUTPUTS / "archive_manifest.csv"
FIELDS = ["match_id", "year", "draw", "round", "video_id", "key", "bytes", "duration_s", "sha256", "status", "note"]
_lock = threading.Lock()


HOSTS = ("www.youtube.com", "nyc3.digitaloceanspaces.com")
MIN_FREE_GB = 25


def wait_for_network(poll: float = 60.0) -> None:
    """Block until DNS works for YouTube and the Space. Outages (a closed lid, a dropped Wi-Fi)
    then cost waiting time instead of retries."""
    said = False
    while True:
        try:
            for h in HOSTS:
                socket.getaddrinfo(h, 443)
            if said:
                print(f"[archive] network back at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            return
        except OSError:
            if not said:
                print(f"[archive] network down at {time.strftime('%Y-%m-%d %H:%M:%S')}; waiting", flush=True)
                said = True
            time.sleep(poll)


def wait_for_disk(poll: float = 60.0) -> None:
    while shutil.disk_usage(DOWNLOADS).free / 1e9 < MIN_FREE_GB:
        time.sleep(poll)


def s3_client():
    import boto3
    from botocore.config import Config

    cfg = configparser.ConfigParser()
    cfg.read(Path.home() / ".s3cfg")
    d = cfg["default"]
    return boto3.client("s3", endpoint_url=ENDPOINT, region_name="nyc3", aws_access_key_id=d["access_key"],
                        aws_secret_access_key=d["secret_key"],
                        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))


def remote_size(s3, key: str) -> int | None:
    """Stored size, or None when the object does not exist. Other errors (network) propagate,
    so an outage is never mistaken for "not archived yet"."""
    from botocore.exceptions import ClientError

    wait_for_network()
    try:
        return int(s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"])
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def download(vid: str, year: int, listed: float, attempts: int = 5) -> tuple[Path | None, str]:
    out = DOWNLOADS / f"{vid}.mp4"
    h = 720 if year <= 2010 else 1080
    last = ""
    for k in range(attempts):
        if out.exists():
            res = check(vid, listed)
            if res["status"] == "ok":
                return out, ""
            last = f"verify failed: {res}"
            out.unlink()
        for p in DOWNLOADS.glob(f"{vid}.*"):
            if p.name != out.name:
                p.unlink(missing_ok=True)
        wait_for_network()
        wait_for_disk()
        r = subprocess.run([sys.executable, "-m", "yt_dlp", "--js-runtimes", "node", "--no-playlist",
                            "--no-progress", "--no-continue", "-f",
                            f"bv*[height<={h}][vcodec^=avc1]+ba[ext=m4a]/bv*[height<={h}]+ba/b",
                            "--merge-output-format", "mp4", "-o", str(DOWNLOADS / "%(id)s.%(ext)s"), "--", vid],
                           capture_output=True, text=True)
        if r.returncode != 0:
            last = r.stderr.strip()[-300:]
            if "Video unavailable" in last or "Private video" in last or "removed" in last:
                return None, last
            time.sleep(min(300, 30 * 2 ** k))
    if out.exists() and check(vid, listed)["status"] == "ok":
        return out, ""
    return None, last


def record(row: dict):
    with _lock:
        new = not MANIFEST.exists()
        with MANIFEST.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)


def archive_one(s3, r, keep: set[str], dry_run: bool) -> dict:
    """One match end to end; any failure is recorded and the run moves on."""
    try:
        return _archive_one(s3, r, keep, dry_run)
    except Exception as e:
        row = dict(match_id=r.match_id, year=int(r.year), draw=r.draw, round=r["round"], video_id=r.video_id,
                   key="", bytes="", duration_s="", sha256="", status="failed",
                   note=f"{type(e).__name__}: {str(e)[:200]}")
        record(row)
        print(f"[archive] FAILED {r.match_id}: {type(e).__name__}: {str(e)[:200]}\n{traceback.format_exc()[-800:]}",
              flush=True)
        return row


def _archive_one(s3, r, keep: set[str], dry_run: bool) -> dict:
    key = f"{PREFIX}{int(r.year)}/{r.match_id}__{r.video_id}.mp4"
    base = dict(match_id=r.match_id, year=int(r.year), draw=r.draw, round=r["round"], video_id=r.video_id, key=key,
                bytes="", duration_s="", sha256="", status="", note=r.match_status)
    size = remote_size(s3, key)
    if size:
        return {**base, "bytes": size, "status": "already archived"}
    if dry_run:
        return {**base, "status": "would archive"}
    t0 = time.time()
    path, err = download(r.video_id, int(r.year), float(r.duration_seconds))
    if path is None:
        row = {**base, "status": "download failed", "note": f"{r.match_status}; {err}"}
        record(row)
        return row
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(path)], capture_output=True, text=True).stdout.strip() or 0)
    digest = sha256(path)
    from boto3.s3.transfer import TransferConfig

    for k in range(8):
        wait_for_network()
        try:
            s3.upload_file(str(path), BUCKET, key,
                           ExtraArgs={"ACL": "private", "ContentType": "video/mp4",
                                      "Metadata": {"video-id": r.video_id, "match-id": r.match_id, "sha256": digest,
                                                   "source": f"https://www.youtube.com/watch?v={r.video_id}"}},
                           Config=TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20,
                                                 max_concurrency=8))
            break
        except Exception as e:  # dropped connections: wait for the network, then start the upload again
            print(f"[archive] upload retry {k + 1} for {r.match_id}: {type(e).__name__}", flush=True)
            time.sleep(min(300, 30 * 2 ** k))
    local = path.stat().st_size
    stored = remote_size(s3, key)
    if stored != local:
        row = {**base, "bytes": local, "status": f"upload size mismatch ({stored})"}
        record(row)
        return row
    if r.video_id not in keep:
        path.unlink()
    row = {**base, "bytes": local, "duration_s": round(dur, 1), "sha256": digest,
           "status": f"archived in {time.time() - t0:.0f}s"}
    record(row)
    print(f"[archive] {r.match_id}: {local / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)
    return row


def abort_stale_uploads(s3) -> int:
    """Multipart uploads left behind by a dropped connection still store their parts; drop them."""
    n = 0
    resp = s3.list_multipart_uploads(Bucket=BUCKET, Prefix=PREFIX)
    for u in resp.get("Uploads", []):
        s3.abort_multipart_upload(Bucket=BUCKET, Key=u["Key"], UploadId=u["UploadId"])
        n += 1
    return n


def latest_manifest() -> pd.DataFrame:
    """The manifest log keeps every attempt; the copy in the Space has the latest row per match."""
    m = pd.read_csv(MANIFEST)
    return m.drop_duplicates(["match_id", "video_id"], keep="last").sort_values(["year", "match_id"])


def write_missing(s3) -> int:
    """Every US Open QF/SF/F target 2001-2025 (data/matches.csv) without an archived full match,
    with the best video the manifest knows of (highlights only, nothing found, or a walkover)."""
    m = pd.read_csv(PROJECT.parent / "data" / "matches.csv")
    m = m[(m.tournament == "US Open") & m.year.between(2001, 2025)]
    done = set(latest_manifest().query("status.str.startswith('archived') or status == 'already archived'",
                                       engine="python").match_id) if MANIFEST.exists() else set()
    v = pd.read_csv(PROJECT / "data" / "video_candidates.csv")
    best = v.groupby("match_id").match_status.agg(lambda s: sorted(set(s))[-1] if len(s) else None)
    miss = m[~m.match_id.isin(done)].copy()
    miss["reason"] = [("walkover" if str(r.match_played) == "False" else
                       "no video found" if r.match_id not in best.index else f"best available: {best[r.match_id]}")
                      for r in miss.itertuples()]
    p = OUTPUTS / "archive_missing.csv"
    miss[["match_id", "year", "draw", "round", "player1", "player2", "reason"]].to_csv(p, index=False)
    s3.upload_file(str(p), BUCKET, PREFIX + "missing.csv", ExtraArgs={"ACL": "private", "ContentType": "text/csv"})
    return len(miss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--videos", nargs="*", help="only these video ids")
    a = ap.parse_args()
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(TARGETS).sort_values(["year", "draw", "round"])
    if a.videos:
        targets = targets[targets.video_id.isin(a.videos)]
    if a.limit:
        targets = targets.head(a.limit)
    keep = set(pd.read_csv(EVAL_MATCHES).video_id)
    wait_for_network()
    s3 = s3_client()
    if not a.dry_run:
        print(f"[archive] aborted {abort_stale_uploads(s3)} stale multipart uploads", flush=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        rows = list(ex.map(lambda r: archive_one(s3, r, keep, a.dry_run), [r for _, r in targets.iterrows()]))
    res = pd.DataFrame(rows)
    print(res.status.str.split(" in ").str[0].value_counts().to_string(), flush=True)
    if not a.dry_run and MANIFEST.exists():
        latest = OUTPUTS / "archive_manifest_latest.csv"
        latest_manifest().to_csv(latest, index=False)
        s3.upload_file(str(latest), BUCKET, PREFIX + "manifest.csv", ExtraArgs={"ACL": "private", "ContentType": "text/csv"})
        print("missing targets:", write_missing(s3), flush=True)


if __name__ == "__main__":
    main()
