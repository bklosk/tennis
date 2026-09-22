"""Summarise pilot quality and throughput metrics across processed matches."""
import json
import sys

import numpy as np
import pandas as pd

from tennis_pipeline.paths import OUTPUTS
from tennis_pipeline.video import probe
from tennis_pipeline.cli import video_path


def match_metrics(vid: str) -> dict:
    d = OUTPUTS / vid
    segs = pd.read_csv(d / "segments.csv")
    align = json.loads((d / "align_summary.json").read_text())
    track = json.loads((d / "track_stats.json").read_text())
    points = pd.read_csv(d / "points.csv")
    shots = pd.read_csv(d / "shots.csv")
    ball = pd.read_parquet(d / "ball.parquet")
    players = pd.read_parquet(d / "players.parquet")
    aligned = points[points.point_number.notna()]
    in_play = np.zeros(len(ball), bool)
    for r in aligned.itertuples():
        in_play |= ((ball.t >= r.t_start) & (ball.t <= r.t_end)).to_numpy()
    sampled = ball[ball.frame % 4 == 0]
    sampled_in_play = sampled[in_play[sampled.index]]
    cover = {}
    for side in ("near", "far"):
        p = players[players.side == side][["chunk_id", "frame"]]
        key = set(zip(p.chunk_id, p.frame))
        cover[side] = float(np.mean([(c, f) in key for c, f in zip(sampled_in_play.chunk_id, sampled_in_play.frame)]))
    rally = aligned.dropna(subset=["official_rally_count"])
    diff = rally.n_shots - rally.official_rally_count
    return {
        "video_min": round(probe(video_path(vid))["duration"] / 60, 1),
        "segments_processed": int(len({c.split("_")[0] for c in ball.chunk_id})),
        "main_camera_min_processed": round(len(ball) / 30 / 60, 1),
        "main_camera_min_total": round(float(segs.duration.sum()) / 60, 1),
        "tracking_fps": track["fps"],
        "aligned_points": align["aligned_points"],
        "official_points": align["official_points"],
        "serve_detected_rate": round(align["serve_detected_rate"], 3),
        "server_end_agreement_when_detected": round(align["server_side_agreement"], 3),
        "rally_exact": round(float((diff == 0).mean()), 3),
        "rally_within_1": round(float((diff.abs() <= 1).mean()), 3),
        "ball_detected_in_play": round(float(ball.raw_detected[in_play].mean()), 3),
        "near_player_coverage": round(cover["near"], 3),
        "far_player_coverage": round(cover["far"], 3),
        "shots_in_aligned_points": int(len(shots)),
        "shots_with_bounce_xy": round(float(shots.bounce_x_m.notna().mean()), 3),
        "shots_audio_confirmed": round(float(shots.audio_confirmed.mean()), 3),
        "points_with_official_serve_speed": int((aligned.serve_speed_kmh > 0).sum()),
    }


if __name__ == "__main__":
    out = {vid: match_metrics(vid) for vid in sys.argv[1:]}
    print(json.dumps(out, indent=2))
