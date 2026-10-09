"""Run pipeline stages for one or more matches. Each stage caches its artifact under
outputs/VIDEO_ID/ and is skipped when the artifact exists (use --force STAGE to redo).

    uv run python -m uso.run VIDEO_ID [VIDEO_ID ...] [--stages scene,audio,people,tracks,pose]
"""

from __future__ import annotations

import argparse
import json
import time

from uso.paths import match_dir

STAGES = ["scene", "audio", "people", "tracks", "pose", "candidates", "onsets"]


def run_stage(stage: str, vid: str, force: bool):
    if stage == "scene":
        from uso import scene
        df = scene.run(vid, force=force)
        segs = scene.segments(df)
        return dict(samples=len(df), main=int(df["main"].sum()), segments=len(segs), main_s=float(segs.dur.sum()))
    if stage == "audio":
        from uso import audio
        df = audio.run(vid, force=force)
        return dict(onsets=len(df))
    if stage == "people":
        from uso import people
        df = people.detect(vid, force=force)
        return dict(detections=len(df))
    if stage == "tracks":
        from uso import players
        df = players.run(vid, force=force)
        return dict(rows=len(df))
    if stage == "pose":
        from uso import pose
        df = pose.run(vid, force=force)
        return dict(rows=len(df))
    if stage == "candidates":
        from uso import rally
        df = rally.features(vid, force=force)
        return dict(rows=len(df))
    if stage == "onsets":  # paid: Decisions API labels for strong onsets (cached, budget-capped)
        from uso import onset_api
        from uso.decisions import DecisionClient
        cl = DecisionClient()
        df = onset_api.run(vid, z_min=8.0, client=cl)
        return dict(rows=len(df), spent_total_usd=round(cl.spent(), 3))
    raise ValueError(stage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--force", default="")
    a = ap.parse_args()
    force = set(a.force.split(",")) if a.force else set()
    for vid in a.videos:
        log_path = match_dir(vid) / "run_log.jsonl"
        for st in a.stages.split(","):
            t0 = time.time()
            info = run_stage(st, vid, st in force)
            info.update(stage=st, video=vid, seconds=round(time.time() - t0, 1), at=time.strftime("%Y-%m-%d %H:%M:%S"))
            print(json.dumps(info), flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(info) + "\n")


if __name__ == "__main__":
    main()
