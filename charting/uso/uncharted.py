"""Chart a match that has no MCP record.

Runs the same stages and decoder as the evaluation (no MCP input anywhere) and writes
outputs/VIDEO_ID/uncharted_shots.csv and uncharted_points.csv. Hitters are reported by court end
(near = camera end); naming them needs who served first and the score, which a video alone does
not give reliably, so it is left to the caller. Handedness is per end: pass `hands` when a
left-hander is involved (players swap ends at changeovers, so a single left-hander needs the
per-point end mapping; otherwise all-right-handed is assumed).

    uv run python -m uso.uncharted VIDEO_ID [--hand-near R --hand-far R]
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from uso import points as points_mod
from uso import rally, strokes
from uso.paths import OUTPUTS, match_dir
from uso.run import run_stage
from uso.shots import blend_learned
from uso.stroke_model import load


def chart(video_id: str, hands: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    hands = hands or {"near": "R", "far": "R"}
    for st in ("scene", "audio", "people", "tracks", "pose", "candidates", "onsets"):
        run_stage(st, video_id, False)
    from uso import onset_api
    onset_api.run_soft(video_id)
    d = match_dir(video_id)
    p = {**rally.DECODER, **json.loads((OUTPUTS / "decoder_params.json").read_text())}
    c = rally.with_api(rally.features(video_id), pd.read_parquet(d / "onset_api.parquet"), p)
    c = blend_learned("__test__", c, p)
    sv = rally.pick_serves_api(c, p_min=p["serve_p"], quiet=p["quiet"])
    vp, contacts = points_mod.assemble(c, sv, w_prior=p["w_prior"], phantom=p.get("phantom"))
    contacts = points_mod.enrich_contacts(contacts, pd.read_parquet(d / "tracks.parquet"))
    contacts["hitter_hand"] = contacts.hitter_end.map(hands)
    rc = contacts[contacts.kind == "rally"].copy()
    rc["hand"] = rc.hitter_hand
    feats = strokes.local_features(video_id, rc, pd.read_parquet(d / "tracks.parquet"))
    tab = rc.merge(feats.drop(columns=["dist_net"], errors="ignore"), on="hit_id", how="left")
    # the stroke model was trained with the API's stroke answers as features, so ask them here too
    a = strokes.run_api(video_id, shots=contacts, out_name="uncharted_api_strokes.parquet")
    b = strokes.run_api_imgside(video_id, shots=contacts, out_name="uncharted_api_imgside.parquet")
    tab = tab.merge(a, on="hit_id", how="left").merge(b, on="hit_id", how="left")
    tab = tab.sort_values(["pid", "t"])
    tab["dt_prev"] = tab.groupby("pid").t.diff()
    tab["opp_dnet"] = tab.oy.abs()
    tab["hit_speed"] = (tab.hvx ** 2 + tab.hvy ** 2) ** 0.5
    pred = load().predict(tab)
    out = contacts.merge(pred, on="hit_id", how="left")
    out["source"] = "model:uso-0.1"
    out.to_csv(d / "uncharted_shots.csv", index=False)
    vp["source"] = "model:uso-0.1"
    vp.to_csv(d / "uncharted_points.csv", index=False)
    return out, vp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("--hand-near", default="R")
    ap.add_argument("--hand-far", default="R")
    a = ap.parse_args()
    shots, pts = chart(a.video_id, {"near": a.hand_near, "far": a.hand_far})
    print(f"{len(pts)} points, {len(shots)} contacts -> {match_dir(a.video_id)}")


if __name__ == "__main__":
    main()
