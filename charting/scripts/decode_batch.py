"""Decode archive matches that have been through the vision stages: Decisions API onset labels,
rally decoding, stroke features and stroke predictions.

    uv run python scripts/decode_batch.py --write-table          # (re)build data/batch_matches.csv
    uv run python scripts/decode_batch.py --dry-run              # what would run, and the estimated spend
    uv run python scripts/decode_batch.py [--slots 2] [--only VIDEO_ID ...]

Matches with an MCP record ('charted') are decoded and aligned like the evaluation set and get
outputs/VIDEO_ID/shots.csv and points.csv; the rest ('uncharted') go through `uso.uncharted` and
get uncharted_shots.csv / uncharted_points.csv (hitters by court end only). Each video is fetched
from the Space and sha256-checked against the archive manifest, decoded in its own process, its new
outputs uploaded next to the vision outputs in s3://benklosky-data/tennis/charting-outputs/, and the
local video deleted. Paid requests go through `uso.decisions` and stop at its cap; a match that hits
the cap stops the batch. Resumable (finished matches are skipped, API answers are cached); a STOP file
in outputs/decode/ ends it after the running matches. The Mac is kept awake with caffeinate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import unicodedata
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from archive_videos import BUCKET, s3_client  # noqa: E402

from uso.paths import BATCH_MATCHES, DOWNLOADS, MCP_DIR, OUTPUTS, PROJECT, match_dir  # noqa: E402

STATE = OUTPUTS / "decode"
SPACE_PREFIX = "tennis/charting-outputs"
SKIP_UPLOAD = {"audio.wav", "rally_p.parquet"}
BUDGET_EXIT = 3
# Measured on the 14 evaluation matches: onset requests per candidate with z >= 8, dollars per onset
# request, rally contacts per candidate, and dollars per rally contact for the two stroke questions.
ONSETS_PER_Z8, USD_PER_ONSET, RALLY_PER_Z8, USD_PER_RALLY = 1.505, 0.000067, 0.099, 0.000177


# ---------------------------------------------------------------- match table

def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower())


def write_table() -> pd.DataFrame:
    """Every archive match with vision outputs and no decode yet, with its MCP record if there is one.
    Columns follow data/eval_matches.csv; role is 'charted' or 'uncharted'."""
    at = pd.read_csv(PROJECT / "data" / "archive_targets.csv")
    em = pd.read_csv(PROJECT / "data" / "eval_matches.csv")
    mm = pd.concat([pd.read_csv(MCP_DIR / f"charting-{g}-matches.csv", encoding_errors="replace").assign(g=g)
                    for g in ("m", "w")])
    hands: dict[str, str] = {}
    for _, r in mm.iterrows():
        for p, h in ((r["Player 1"], r["Pl 1 hand"]), (r["Player 2"], r["Pl 2 hand"])):
            if isinstance(p, str) and h in ("R", "L"):
                hands.setdefault(_norm(p), h)
    uso_mm = mm[mm.match_id.str.contains("-US_Open-")]
    rows = []
    for _, r in at.iterrows():
        d = OUTPUTS / r.video_id
        if r.video_id in set(em.video_id) or not (d / "candidates.parquet").exists():
            continue
        g = "m" if r.draw == "men" else "w"
        l1, l2 = _norm(r.player1.split()[-1]), _norm(r.player2.split()[-1])
        hit = uso_mm[uso_mm.match_id.str.startswith(str(r.year)) & uso_mm.match_id.str.contains(f"-US_Open-{r['round']}-")
                     & (uso_mm.g == g)]
        hit = hit[[l1 in _norm(m) and l2 in _norm(m) for m in hit.match_id]]
        row = dict(video_id=r.video_id, year=int(r.year), gender=g, round=r["round"], best_of=5 if g == "m" else 3,
                   duration_s=int(r.duration_seconds), match_id=r.match_id)
        if len(hit) == 1:
            m = hit.iloc[0]
            row.update(role="charted", mcp_match_id=m.match_id, player1=m["Player 1"], player2=m["Player 2"],
                       hand1=m["Pl 1 hand"], hand2=m["Pl 2 hand"])  # MCP's "Best of" has slips; slams are 5/3
        else:
            row.update(role="uncharted", mcp_match_id=None, player1=r.player1, player2=r.player2,
                       hand1=hands.get(_norm(r.player1), "R"), hand2=hands.get(_norm(r.player2), "R"))
        rows.append(row)
    cols = ["role", "video_id", "mcp_match_id", "year", "gender", "round", "player1", "player2", "hand1", "hand2",
            "best_of", "duration_s", "match_id"]
    t = pd.DataFrame(rows)[cols].sort_values(["role", "year", "gender", "round", "video_id"])
    t.to_csv(BATCH_MATCHES, index=False)
    return t


def done(r) -> bool:
    d = match_dir(r.video_id)
    return (d / ("shots.csv" if r.role == "charted" else "uncharted_shots.csv")).exists()


def estimate(vid: str) -> float:
    c = pd.read_parquet(match_dir(vid) / "candidates.parquet", columns=["z"])
    z8 = int((c.z >= 8).sum())
    return z8 * (ONSETS_PER_Z8 * USD_PER_ONSET + RALLY_PER_Z8 * USD_PER_RALLY)


# ---------------------------------------------------------------- one match (child process)

def decode_one(vid: str) -> dict:
    from uso import report
    from uso import stroke_model as sm
    from uso.decisions import BudgetExceeded, DecisionClient

    r = pd.read_csv(BATCH_MATCHES).set_index("video_id").loc[vid]
    t0, spent0 = time.time(), DecisionClient.spent()
    try:
        if r.role == "charted":
            import post

            post.run(vid, ["onsets", "shots", "strokefeats", "strokeapi"])
            tab = sm.match_table(vid)
            pred = sm.load().predict(tab)
            keep = ["hit_id", "p_forehand", "side", "family"] + [c for c in pred.columns
                                                                 if c.startswith("p_") and c != "p_forehand"]
            report.export(vid, pred[keep])
        else:
            from uso import uncharted

            # one hand per court end: right/right is exact unless a left-hander plays (flagged in the table)
            uncharted.chart(vid, {"near": "R", "far": "R"})
    except BudgetExceeded as e:
        print(f"[decode] {vid}: budget cap reached: {e}", flush=True)
        sys.exit(BUDGET_EXIT)
    info = dict(video_id=vid, role=r.role, seconds=round(time.time() - t0, 1),
                usd=round(DecisionClient.spent() - spent0, 3), spent_total=round(DecisionClient.spent(), 3))
    print("[decode] " + json.dumps(info), flush=True)
    return info


# ---------------------------------------------------------------- batch (parent process)

class Batch:
    def __init__(self, todo: pd.DataFrame, slots: int, keep_video: bool):
        self.s3 = s3_client()
        self.todo = todo
        self.slots = slots
        self.keep_video = keep_video
        man = pd.read_csv(OUTPUTS / "archive_manifest_latest.csv")
        self.man = man[man.status.str.startswith("archived")].set_index("video_id")
        self.ready: list[str] = []
        self.lock = threading.Lock()
        self.stop = False
        self.state = {v: dict(status="queued") for v in todo.video_id}
        STATE.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with (STATE / "events.log").open("a") as f:
            f.write(line + "\n")
        (STATE / "state.json").write_text(json.dumps(self.state, indent=1))

    def stopping(self) -> bool:
        return self.stop or (STATE / "STOP").exists()

    def fetch(self, vid: str):
        m = self.man.loc[vid]
        dst = DOWNLOADS / f"{vid}.mp4"
        DOWNLOADS.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if not (dst.exists() and dst.stat().st_size == int(m.bytes)):
            self.s3.download_file(BUCKET, m.key, str(dst))
        h = hashlib.sha256()
        with dst.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != m.sha256:
            dst.unlink(missing_ok=True)
            raise RuntimeError("sha256 mismatch")
        self.log(f"fetched {vid} ({int(m.bytes) / 1e9:.1f} GB in {time.time() - t0:.0f}s)")

    def prefetcher(self):
        for vid in self.todo.video_id:
            while not self.stopping():
                with self.lock:
                    if len(self.ready) < self.slots:
                        break
                time.sleep(5)
            if self.stopping():
                return
            for attempt in range(3):
                try:
                    self.fetch(vid)
                    break
                except Exception as e:  # network or checksum: retry, then skip the match
                    self.log(f"fetch {vid} failed ({type(e).__name__}: {e}); attempt {attempt + 1}")
                    time.sleep(30)
            else:
                self.state[vid]["status"] = "fetch failed"
                continue
            with self.lock:
                self.ready.append(vid)

    def upload(self, vid: str) -> int:
        r = self.todo.set_index("video_id").loc[vid]
        prefix = f"{SPACE_PREFIX}/{r.year}/{r.match_id}__{vid}/"
        have = {o["Key"][len(prefix):]: o["Size"] for page in self.s3.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET, Prefix=prefix) for o in page.get("Contents", [])}
        n = 0
        for p in sorted(match_dir(vid).iterdir()):
            if p.is_file() and p.name not in SKIP_UPLOAD and have.get(p.name) != p.stat().st_size:
                self.s3.upload_file(str(p), BUCKET, prefix + p.name, ExtraArgs={"ACL": "private"})
                n += 1
        return n

    def finish(self, vid: str, proc: subprocess.Popen):
        code = proc.wait()
        st = self.state[vid]
        if code == 0:
            st["status"] = "done"
            st["uploaded"] = self.upload(vid)
            msg = f"done {vid} ({st['uploaded']} files to the Space)"
        else:
            st["status"] = "budget" if code == BUDGET_EXIT else f"failed (exit {code})"
            msg = f"{st['status']}: {vid}; see {STATE / (vid + '.log')}"
            if code == BUDGET_EXIT:
                self.stop = True
        if not self.keep_video:
            (DOWNLOADS / f"{vid}.mp4").unlink(missing_ok=True)
        self.log(msg)

    def run(self):
        caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        threading.Thread(target=self.prefetcher, daemon=True).start()
        running: dict[str, tuple[subprocess.Popen, threading.Thread]] = {}
        left = list(self.todo.video_id)
        try:
            while left or running:
                for vid, (p, _) in list(running.items()):
                    if p.poll() is not None:
                        self.finish(vid, p)
                        running.pop(vid)
                with self.lock:
                    start = [v for v in self.ready if v in left][: max(0, self.slots - len(running))]
                    for v in start:
                        self.ready.remove(v)
                for vid in start:
                    left.remove(vid)
                    if self.stopping():
                        continue
                    log = (STATE / f"{vid}.log").open("a")
                    p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--one", vid],
                                         stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT)
                    running[vid] = (p, None)
                    self.state[vid].update(status="running", started=time.strftime("%H:%M:%S"))
                    self.log(f"started {vid} ({self.todo.set_index('video_id').at[vid, 'role']})")
                if self.stopping() and not running:
                    break
                if not start and not running and left and all(self.state[v]["status"] == "fetch failed" for v in left):
                    break
                time.sleep(5)
        finally:
            caff.terminate()
        from uso.decisions import DecisionClient

        n = sum(s["status"] == "done" for s in self.state.values())
        self.log(f"batch finished: {n} of {len(self.todo)} decoded; total API spend ${DecisionClient.spent():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-table", action="store_true", help="rebuild data/batch_matches.csv and exit")
    ap.add_argument("--dry-run", action="store_true", help="list what would run and the estimated spend")
    ap.add_argument("--only", nargs="*", help="video ids to decode (default: every undecoded batch match)")
    ap.add_argument("--slots", type=int, default=2, help="matches decoded at once")
    ap.add_argument("--keep-video", action="store_true", help="keep downloaded videos after decoding")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        decode_one(a.one)
        return
    if a.write_table or not BATCH_MATCHES.exists():
        t = write_table()
        print(f"[decode] {BATCH_MATCHES.name}: {len(t)} matches ({(t.role == 'charted').sum()} with an MCP record)")
        if a.write_table:
            return
    from uso.decisions import HARD_CAP_USD, DecisionClient

    t = pd.read_csv(BATCH_MATCHES)
    t = t[[not done(r) for r in t.itertuples()]]
    if a.only:
        t = t[t.video_id.isin(a.only)]
    est = sum(estimate(v) for v in t.video_id)
    spent = DecisionClient.spent()
    print(f"[decode] {len(t)} matches to decode ({(t.role == 'charted').sum()} charted); estimated ${est:.2f}; "
          f"spent ${spent:.2f}; cap ${HARD_CAP_USD:.2f}")
    if a.dry_run:
        for r in t.itertuples():
            print(f"  {r.video_id}  {r.role:9s}  {r.year} {r.gender} {r.round:2s}  {r.player1} – {r.player2}  ~${estimate(r.video_id):.2f}")
        return
    if spent + est > HARD_CAP_USD:
        print(f"[decode] the estimate would cross the cap (${spent + est:.2f} > ${HARD_CAP_USD:.2f}); "
              "matches run in order until the cap stops them")
    Batch(t, a.slots, a.keep_video).run()


if __name__ == "__main__":
    main()
