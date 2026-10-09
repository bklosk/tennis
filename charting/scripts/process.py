"""Process every evaluation match whose video has been verified, stage by stage, resumably.

    uv run python scripts/process.py [--role dev|test|all] [--stages scene,audio,people,tracks,pose] [--wait]

With --wait it keeps polling for newly verified downloads until every listed match is processed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_videos import check  # noqa: E402

from uso.paths import DOWNLOADS, EVAL_MATCHES, OUTPUTS, match_dir
from uso.run import STAGES, run_stage

ARTIFACT = {"scene": "scene.parquet", "audio": "audio_onsets.parquet", "people": "people_rfdetr.parquet",
            "tracks": "tracks.parquet", "pose": "pose.parquet", "candidates": "candidates.parquet",
            "onsets": "onset_api.parquet"}


def verified(em: pd.DataFrame) -> set[str]:
    """Verified-ok videos. Newly finished downloads (merged file present, no partial siblings,
    size stable) are checked here and recorded in outputs/video_check.csv."""
    p = OUTPUTS / "video_check.csv"
    v = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["video_id", "status"])
    known = set(v[v.status == "ok"].video_id)  # bad files are re-checked once the downloader replaces them
    new = []
    for r in em.itertuples():
        f = DOWNLOADS / f"{r.video_id}.mp4"
        if r.video_id in known or not f.exists() or any(DOWNLOADS.glob(f"{r.video_id}.f*")):
            continue
        size = f.stat().st_size
        time.sleep(5)
        if f.stat().st_size != size:
            continue
        res = check(r.video_id, float(r.duration_s))
        print(json.dumps(dict(verify=res)), flush=True)
        new.append(res)
    if new:
        nd = pd.DataFrame(new)
        v = pd.concat([v[~v.video_id.isin(nd.video_id)], nd], ignore_index=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        v.to_csv(p, index=False)
    return set(v[v.status == "ok"].video_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="all")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="walk the match list backwards (for a second driver)")
    ap.add_argument("--videos", nargs="*", help="only these video ids, in this order")
    a = ap.parse_args()
    em = pd.read_csv(EVAL_MATCHES)
    if a.role != "all":
        em = em[em.role == a.role]
    if a.videos:
        em = em.set_index("video_id").loc[a.videos].reset_index()
    if a.reverse:
        em = em.iloc[::-1]
    stages = a.stages.split(",")
    while True:
        ok = verified(em)
        pending = []
        for vid in em.video_id:
            todo = [s for s in stages if not (match_dir(vid) / ARTIFACT[s]).exists()]
            if not todo:
                continue
            pending.append(vid)
            if vid not in ok:
                continue
            lock = match_dir(vid) / ".lock"  # one driver per match
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
            except FileExistsError:
                continue
            try:
                run_match(vid, todo)
            finally:
                lock.unlink(missing_ok=True)
        if not pending or not a.wait:
            break
        time.sleep(30)
    print("all done", flush=True)


def run_match(vid: str, todo: list[str]):
    for st in todo:
        t0 = time.time()
        try:
            info = run_stage(st, vid, False)
        except Exception as e:  # keep going with other matches; the log says what broke
            info = dict(error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
            print(json.dumps(dict(stage=st, video=vid, **info)), flush=True)
            return
        info.update(stage=st, video=vid, seconds=round(time.time() - t0, 1))
        print(json.dumps(info), flush=True)
        with (match_dir(vid) / "run_log.jsonl").open("a") as f:
            f.write(json.dumps(info) + "\n")


if __name__ == "__main__":
    main()
