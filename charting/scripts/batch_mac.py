"""Process archived US Open videos on this Mac, a few at a time, for a fixed time budget.

    uv run --project charting python charting/scripts/batch_mac.py [--hours 16] [--slots 2] [--port 8765]

Videos are prefetched from the Space (sha256-checked against the archive manifest) and run through
the free stages (scene..candidates; never the paid onsets stage). Each finished match's artifacts are
uploaded to s3://benklosky-data/tennis/charting-outputs/YEAR/MATCH_ID__VIDEO_ID/, the small ones are
committed to charting/results/ and pushed to the charting-results branch, and the local video and
audio.wav are deleted. No new match starts after the deadline (or once a STOP file appears next to
the state file); the Mac is kept awake with caffeinate until the last match is stored. Progress is
served on http://127.0.0.1:PORT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from archive_videos import BUCKET, s3_client  # noqa: E402

from uso.paths import DOWNLOADS, OUTPUTS, PROJECT, REPO  # noqa: E402

STAGES = ["scene", "audio", "people", "tracks", "pose", "candidates"]
# seconds per second of video at two concurrent matches (Oct 9 logs); only used for progress bars
EXPECTED = {"scene": 0.122, "audio": 0.004, "people": 0.176, "tracks": 0.001, "pose": 0.062, "candidates": 0.003}
SPACE_PREFIX = "tennis/charting-outputs"
GIT_FILES = ["run_log.jsonl", "scene.parquet", "tracks.parquet"]  # each well under 1 MB
SKIP_UPLOAD = {"audio.wav"}  # 300+ MB and regenerated from the video in seconds
GIT_ROOT = HERE.parent.parent  # the checkout this script lives in
RESULTS = HERE.parent / "results"
STATE_DIR = OUTPUTS / "batch"


class Batch:
    def __init__(self, hours: float, slots: int):
        self.lock = threading.Lock()
        self.s3 = s3_client()
        self.slots = slots
        self.t0 = time.time()
        self.deadline = self.t0 + hours * 3600
        self.events: list[str] = []
        self.matches: dict[str, dict] = {}
        self.order: list[str] = []
        self.ready: list[str] = []  # downloaded, waiting for a slot
        self.running: dict[str, subprocess.Popen] = {}
        self.git = dict(commits=0, pushed=0, last_error="")
        self.git_lock = threading.Lock()
        self.system: dict = {}
        self.caffeinate: subprocess.Popen | None = None
        self.finished = False
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self._load_queue()

    # ---- bookkeeping ------------------------------------------------------------------------
    def log(self, msg: str):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with self.lock:
            self.events.append(line)
            self.events = self.events[-200:]
        with (STATE_DIR / "events.log").open("a") as f:
            f.write(line + "\n")

    def _load_queue(self):
        rows = [r for r in csv.DictReader(open(OUTPUTS / "archive_manifest_latest.csv")) if r["duration_s"] and r["key"]]
        rows.sort(key=lambda r: (-int(r["year"]), r["key"]))
        for r in rows:
            vid = r["video_id"]
            d = OUTPUTS / vid
            done_before = (d / "pose.parquet").exists()
            self.matches[vid] = dict(
                vid=vid, match_id=r["match_id"], year=int(r["year"]), draw=r["draw"], round=r["round"],
                key=r["key"], bytes=int(float(r["bytes"])), sha256=r["sha256"], duration=float(r["duration_s"]),
                status="prior" if done_before else "queued", stages={}, attempts=0, got=0,
                started=None, finished=None, uploaded=False, committed=False, error="")
            self.order.append(vid)
            if done_before:
                self.matches[vid]["stages"] = self._stage_seconds(vid)

    def _stage_seconds(self, vid: str) -> dict:
        p = OUTPUTS / vid / "run_log.jsonl"
        out = {}
        if p.exists():
            for line in p.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("stage") in STAGES:
                    out[r["stage"]] = r["seconds"]
        return out

    def stop_requested(self) -> bool:
        return time.time() >= self.deadline or (STATE_DIR / "STOP").exists()

    # ---- downloads --------------------------------------------------------------------------
    def next_queued(self) -> str | None:
        with self.lock:
            for vid in self.order:
                if self.matches[vid]["status"] == "queued":
                    self.matches[vid]["status"] = "downloading"
                    return vid
        return None

    def download(self, vid: str):
        m = self.matches[vid]
        dst = DOWNLOADS / f"{vid}.mp4"
        DOWNLOADS.mkdir(parents=True, exist_ok=True)
        m["dl_started"] = time.time()
        m["got"] = 0

        def cb(n):
            m["got"] += n

        if not (dst.exists() and dst.stat().st_size == m["bytes"]):
            self.s3.download_file(BUCKET, m["key"], str(dst), Callback=cb)
        m["got"] = m["bytes"]
        h = hashlib.sha256()
        with dst.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != m["sha256"]:
            dst.unlink(missing_ok=True)
            raise RuntimeError("sha256 mismatch after download")
        m["dl_seconds"] = round(time.time() - m["dl_started"], 1)

    def downloader(self):
        while not self.stop_requested():
            with self.lock:
                waiting = len(self.ready) + sum(1 for m in self.matches.values() if m["status"] == "downloading")
            if waiting >= self.slots:
                time.sleep(2)
                continue
            vid = self.next_queued()
            if vid is None:
                if not self.running and not self.ready:
                    return
                time.sleep(5)  # a failed match may still be re-queued for its retry
                continue
            try:
                self.download(vid)
                with self.lock:
                    self.matches[vid]["status"] = "ready"
                    self.ready.append(vid)
                self.log(f"downloaded {vid} ({self.matches[vid]['bytes'] / 1e9:.1f} GB in {self.matches[vid]['dl_seconds']:.0f}s)")
            except Exception as e:
                self.fail(vid, f"download: {type(e).__name__}: {e}")

    # ---- processing -------------------------------------------------------------------------
    def start(self, vid: str):
        m = self.matches[vid]
        m.update(status="processing", started=time.time(), attempts=m["attempts"] + 1)
        log = (OUTPUTS / vid).resolve()
        log.mkdir(parents=True, exist_ok=True)
        f = open(log / "batch.log", "a")
        self.running[vid] = subprocess.Popen(
            [sys.executable, "-m", "uso.run", vid, "--stages", ",".join(STAGES)],
            cwd=PROJECT, stdout=f, stderr=subprocess.STDOUT)
        self.log(f"started {vid} {m['match_id']} ({m['duration'] / 3600:.2f} h)")

    def fail(self, vid: str, err: str):
        m = self.matches[vid]
        retry = m["attempts"] < 2 and not err.startswith("download: RuntimeError")
        with self.lock:
            m["error"] = err[-800:]
            m["status"] = "queued" if retry else "failed"
            if vid in self.ready:
                self.ready.remove(vid)
        self.log(f"{'retrying' if retry else 'FAILED'} {vid}: {err.splitlines()[-1] if err else ''}")
        if not retry:
            (DOWNLOADS / f"{vid}.mp4").unlink(missing_ok=True)

    def store(self, vid: str):
        """Upload artifacts to the Space, commit the small ones, then free the disk."""
        m = self.matches[vid]
        m["status"] = "storing"
        d = OUTPUTS / vid
        base = f"{SPACE_PREFIX}/{m['year']}/{m['match_id']}__{vid}"
        for f in sorted(d.iterdir()):
            if f.is_file() and f.name not in SKIP_UPLOAD:
                self.s3.upload_file(str(f), BUCKET, f"{base}/{f.name}")
        m["uploaded"] = True
        self.commit(vid)
        if m["status"] == "storing":
            (DOWNLOADS / f"{vid}.mp4").unlink(missing_ok=True)
            (d / "audio.wav").unlink(missing_ok=True)

    def commit(self, vid: str):
        m = self.matches[vid]
        dst = RESULTS / f"{m['match_id']}__{vid}"
        dst.mkdir(parents=True, exist_ok=True)
        for name in GIT_FILES:
            if (OUTPUTS / vid / name).exists():
                shutil.copy2(OUTPUTS / vid / name, dst / name)
        git = ["git", "-C", str(GIT_ROOT)]
        msg = f"Chart {m['match_id']} ({vid}) on the Mac batch run.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
        try:
            with self.git_lock:
                self.write_index()
                subprocess.run(git + ["add", str(RESULTS)], check=True, capture_output=True)
                r = subprocess.run(git + ["commit", "-q", "-m", msg], capture_output=True, text=True)
                if r.returncode == 0:
                    self.git["commits"] += 1
                m["committed"] = True
                self.push()
        except subprocess.CalledProcessError as e:
            self.git["last_error"] = (e.stderr or b"").decode()[-300:]

    def push(self):
        r = subprocess.run(["git", "-C", str(GIT_ROOT), "push", "-q", "origin", "HEAD:charting-results"],
                           capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            self.git["pushed"] = self.git["commits"]
            self.git["last_error"] = ""
        else:
            self.git["last_error"] = r.stderr[-300:]

    def write_index(self):
        RESULTS.mkdir(parents=True, exist_ok=True)
        with (RESULTS / "index.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["match_id", "video_id", "year", "draw", "round", "duration_s", *[f"{s}_s" for s in STAGES],
                        "space_prefix"])
            for vid in self.order:
                m = self.matches[vid]
                if m["status"] in ("done", "storing", "prior") and m["stages"]:
                    w.writerow([m["match_id"], vid, m["year"], m["draw"], m["round"], m["duration"],
                                *[m["stages"].get(s, "") for s in STAGES],
                                f"s3://{BUCKET}/{SPACE_PREFIX}/{m['year']}/{m['match_id']}__{vid}/"])

    def poll(self):
        for vid, p in list(self.running.items()):
            self.matches[vid]["stages"] = self._stage_seconds(vid)
            if p.poll() is None:
                continue
            del self.running[vid]
            m = self.matches[vid]
            if p.returncode != 0:
                tail = (OUTPUTS / vid / "batch.log").read_text()[-1500:] if (OUTPUTS / vid / "batch.log").exists() else ""
                tb = tail[tail.rfind("Traceback"):] if "Traceback" in tail else tail[-400:]
                self.fail(vid, f"exit {p.returncode}\n{tb}")
                continue
            m["finished"] = time.time()
            try:
                self.store(vid)
                m["status"] = "done"
                self.log(f"done {vid} in {(m['finished'] - m['started']) / 60:.1f} min, stored")
            except Exception as e:
                m["status"] = "done"
                m["error"] = f"store: {type(e).__name__}: {e}"
                self.log(f"done {vid} but storing failed: {e}")

    # ---- system ---------------------------------------------------------------------------------
    def sample_system(self):
        du = shutil.disk_usage(REPO)
        power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        pct = re.search(r"(\d+)%", power)
        self.system = dict(load=round(os.getloadavg()[0], 1), cpus=os.cpu_count(), free_gb=round(du.free / 1e9, 1),
                           ac="AC Power" in power, battery=int(pct.group(1)) if pct else None,
                           caffeinate=self.caffeinate is not None and self.caffeinate.poll() is None)

    def snapshot(self) -> dict:
        now = time.time()
        ms = []
        for vid in self.order:
            m = dict(self.matches[vid])
            if m["status"] == "processing":
                done_s = sum(m["stages"].values())
                cur = next((s for s in STAGES if s not in m["stages"]), None)
                m["current"] = cur
                if cur:
                    el = now - m["started"] - done_s
                    m["current_elapsed"] = round(el)
                    m["current_expected"] = round(EXPECTED[cur] * m["duration"])
            ms.append(m)
        done = [m for m in ms if m["status"] == "done"]
        vh = sum(m["duration"] for m in done) / 3600
        el = (now - self.t0) / 3600
        rate = vh / el if el > 0.25 else None
        left = sum(m["duration"] for m in ms if m["status"] in ("queued", "downloading", "ready", "processing")) / 3600
        return dict(now=now, t0=self.t0, deadline=self.deadline, slots=self.slots, finished=self.finished,
                    stopping=self.stop_requested(), matches=ms, events=self.events[-40:], git=self.git,
                    system=self.system, stages=STAGES, expected=EXPECTED,
                    totals=dict(done=len(done), video_h=round(vh, 2), rate=rate, left_h=round(left, 1),
                                failed=sum(m["status"] == "failed" for m in ms),
                                prior=sum(m["status"] == "prior" for m in ms)))

    # ---- main loop --------------------------------------------------------------------------
    def store_prior(self):
        """Matches processed before this run: upload and commit them once."""
        marker = STATE_DIR / "prior_stored.json"
        stored = set(json.loads(marker.read_text())) if marker.exists() else set()
        for vid in self.order:
            m = self.matches[vid]
            if m["status"] == "prior" and vid not in stored:
                try:
                    base = f"{SPACE_PREFIX}/{m['year']}/{m['match_id']}__{vid}"
                    for f in sorted((OUTPUTS / vid).iterdir()):
                        if f.is_file() and f.name not in SKIP_UPLOAD:
                            self.s3.upload_file(str(f), BUCKET, f"{base}/{f.name}")
                    stored.add(vid)
                    marker.write_text(json.dumps(sorted(stored)))
                except Exception as e:
                    self.log(f"prior upload failed {vid}: {e}")
        for vid in self.order:
            if self.matches[vid]["status"] == "prior":
                self.matches[vid]["uploaded"] = vid in stored
                dst = RESULTS / f"{self.matches[vid]['match_id']}__{vid}"
                dst.mkdir(parents=True, exist_ok=True)
                for name in GIT_FILES:
                    if (OUTPUTS / vid / name).exists():
                        shutil.copy2(OUTPUTS / vid / name, dst / name)
        with self.git_lock:
            self.write_index()
            subprocess.run(["git", "-C", str(GIT_ROOT), "add", str(RESULTS)], capture_output=True)
            r = subprocess.run(["git", "-C", str(GIT_ROOT), "commit", "-q", "-m",
                                "Add charting results for matches processed before the Mac batch run.\n\n"
                                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"], capture_output=True)
            if r.returncode == 0:
                self.git["commits"] += 1
                self.push()
        self.log(f"stored {len(stored)} previously processed matches")

    def run(self):
        self.caffeinate = subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])
        self.log(f"batch start: {self.slots} slots, deadline {time.strftime('%a %H:%M', time.localtime(self.deadline))}")
        self.sample_system()
        threading.Thread(target=self.store_prior, daemon=True).start()
        threading.Thread(target=self.downloader, daemon=True).start()
        last_sys = 0
        while True:
            try:
                self.poll()
                while len(self.running) < self.slots and self.ready and not self.stop_requested():
                    with self.lock:
                        vid = self.ready.pop(0)
                    self.start(vid)
                if time.time() - last_sys > 15:
                    self.sample_system()
                    last_sys = time.time()
                    (STATE_DIR / "state.json").write_text(json.dumps(self.snapshot()))
                busy = self.running or any(m["status"] in ("downloading", "storing") for m in self.matches.values())
                if self.stop_requested() and not busy:
                    break
                if not busy and not self.ready and not any(m["status"] == "queued" for m in self.matches.values()):
                    break
            except Exception:
                self.log("loop error: " + traceback.format_exc().splitlines()[-1])
            time.sleep(3)
        for vid in list(self.ready):  # prefetched but never started
            (DOWNLOADS / f"{vid}.mp4").unlink(missing_ok=True)
            self.matches[vid]["status"] = "queued"
        self.ready.clear()
        self.finished = True
        self.caffeinate.terminate()
        self.sample_system()
        (STATE_DIR / "state.json").write_text(json.dumps(self.snapshot()))
        t = self.snapshot()["totals"]
        self.log(f"batch finished: {t['done']} matches, {t['video_h']} video-h, {t['failed']} failed; caffeinate released")


def serve(batch: Batch, port: int):
    page = (HERE / "batch_dashboard.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/state.json"):
                body, ctype = json.dumps(batch.snapshot()).encode(), "application/json"
            elif self.path in ("/", "/index.html"):
                body, ctype = page, "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=16)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    b = Batch(a.hours, a.slots)
    threading.Thread(target=serve, args=(b, a.port), daemon=True).start()
    b.run()
    # keep the final page up (the Mac may sleep again now); Ctrl-C or kill to exit
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
