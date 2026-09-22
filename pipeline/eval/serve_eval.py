"""Serve detection against the official record, with and without the body-based serve detector.

    uv run python -m eval.serve_eval KCcKkUnjbzA Fl33UXv6jKI SINNER_FRITZ_ID --sweep 0.35,0.5,0.65

Re-runs events + align from cached tracks (CPU only), so it overwrites those outputs; the
last configuration run is the default detector so outputs end in the production state.
For each aligned point without a detected serve it reports why the best nearby candidate
was rejected, which is the list to work from when tuning.
"""
import argparse
import json

import pandas as pd

from tennis_pipeline import align, serve
from tennis_pipeline.cli import video_path
from tennis_pipeline.paths import match_dir
from tennis_pipeline.process import events_match


def run_config(vid: str, detector: bool, threshold: float) -> dict:
    path = video_path(vid)
    events_match(vid, path if path.exists() else None, serve_detector=detector,
                 serve_params=serve.ServeParams(threshold=threshold))
    summary = align.run(vid)
    hits = pd.read_parquet(match_dir(vid) / "hits_raw.parquet")
    src = hits.serve_source.value_counts().to_dict() if "serve_source" in hits else {}
    return {k: summary.get(k) for k in ("aligned_points", "official_points", "serve_detected_rate",
                                         "server_side_agreement", "rally_count_within_1")} | {"serve_sources": src}


def miss_reasons(vid: str) -> dict:
    out_dir = match_dir(vid)
    points = pd.read_csv(out_dir / "points.csv")
    cands_path = out_dir / "serve_candidates.parquet"
    cands = pd.read_parquet(cands_path) if cands_path.exists() else pd.DataFrame()
    missed = points[points.point_number.notna() & points.server_side.isna()]
    reasons = []
    for p in missed.itertuples():
        near = cands[(cands.t >= p.t_start - 4.0) & (cands.t <= p.t_start + 0.5)
                     & (cands.side == p.official_server_side)] if len(cands) else cands
        if near.empty:
            reasons.append("no_extension_candidate")
        else:
            best = near.sort_values("score", ascending=False).iloc[0]
            reasons.append(best.reject or "accepted_but_not_first_hit")
    return pd.Series(reasons, dtype=object).value_counts().to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_ids", nargs="+")
    ap.add_argument("--sweep", default="", help="comma-separated extra thresholds to try")
    args = ap.parse_args()
    default = serve.ServeParams().threshold
    thresholds = [float(x) for x in args.sweep.split(",") if x] + [default]
    report = {}
    for vid in args.video_ids:
        rows = {"baseline_rule_only": run_config(vid, False, default)}
        for th in thresholds:
            rows[f"detector@{th:g}"] = run_config(vid, True, th)
        rows["misses_at_default"] = miss_reasons(vid)
        report[vid] = rows
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
