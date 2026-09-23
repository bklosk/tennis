"""Fit the serve detector's candidate weights against the official point-by-point record.

    uv run python -m eval.serve_fit KCcKkUnjbzA Fl33UXv6jKI Ce3dRYHWIBI

Each match is first aligned with rule-only serve detection (fewer serves, but precise, so the
alignment is trustworthy). Serve candidates are then labeled against it: a candidate on the
official server's side within 1.5 s before to 0.5 s after an aligned point's first hit is a
serve; one inside an aligned point's rally is not; anything else (faults, between points) is
left unlabeled. Candidates rejected by the hard gates (stance inside the court, opponent hit
just before) are not scored and are left out. Prints leave-one-match-out quality and the
weights for `serve.DEFAULT_WEIGHTS`. Overwrites events/align outputs, finishing with the
production configuration.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from tennis_pipeline import align, serve
from tennis_pipeline.cli import video_path
from tennis_pipeline.paths import match_dir
from tennis_pipeline.process import events_match

GATES = {"not_behind_baseline", "opponent_hit_recently", "no_ball_or_audio_evidence"}


def labeled_candidates(vid: str) -> pd.DataFrame:
    path = video_path(vid)
    path = path if path.exists() else None
    events_match(vid, path, serve_detector=False)
    align.run(vid)
    ref = pd.read_csv(match_dir(vid) / "points.csv")
    ref = ref[ref.point_number.notna()]
    events_match(vid, path, serve_detector=True)
    cands = pd.read_parquet(match_dir(vid) / "serve_candidates.parquet")
    labels = []
    for c in cands.itertuples():
        start = ref[(c.t >= ref.t_start - 1.5) & (c.t <= ref.t_start + 0.5)]
        if len(start):
            labels.append(float((start.official_server_side == c.side).any()))
        elif ((c.t > ref.t_start + 1.0) & (c.t < ref.t_end + 0.5)).any():
            labels.append(0.0)
        else:
            labels.append(np.nan)
    cands["label"] = labels
    cands["video_id"] = vid
    return cands


def fit(df: pd.DataFrame, C: float = 0.5) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=5000).fit(df[list(serve.FEATURES)].to_numpy(float), df.label.astype(int))


def weights(clf: LogisticRegression) -> dict:
    return {"bias": round(float(clf.intercept_[0]), 2), **{k: round(float(v), 2) for k, v in zip(serve.FEATURES, clf.coef_[0])}}


def quality(y: np.ndarray, p: np.ndarray) -> dict:
    out = {"n": int(len(y)), "serves": int(y.sum()), "auc": round(float(roc_auc_score(y, p)), 3) if 0 < y.sum() < len(y) else None}
    for th in (0.5, 0.7):
        acc = p >= th
        out[f"precision@{th}"] = round(float(y[acc].mean()), 3) if acc.any() else None
        out[f"recall@{th}"] = round(float(acc[y == 1].mean()), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_ids", nargs="+")
    ap.add_argument("--C", type=float, default=0.5, help="inverse L2 strength")
    args = ap.parse_args()
    cands = pd.concat([labeled_candidates(v) for v in args.video_ids], ignore_index=True)
    df = cands[cands.label.notna() & ~cands.reject.isin(GATES)]
    X = lambda d: d[list(serve.FEATURES)].to_numpy(float)  # noqa: E731
    w = serve.DEFAULT_WEIGHTS
    current = lambda d: 1 / (1 + np.exp(-(w["bias"] + X(d) @ np.array([w[k] for k in serve.FEATURES]))))  # noqa: E731
    report = {"current_weights": quality(df.label.to_numpy(int), current(df))}
    if len(args.video_ids) > 1:
        lomo = {}
        for vid in args.video_ids:
            train, test = df[df.video_id != vid], df[df.video_id == vid]
            clf = fit(train, args.C)
            lomo[vid] = {"fitted_on_others": quality(test.label.to_numpy(int), clf.predict_proba(X(test))[:, 1]),
                         "current_weights": quality(test.label.to_numpy(int), current(test)),
                         "weights": weights(clf)}
        report["leave_one_match_out"] = lomo
    report["weights"] = weights(fit(df, args.C))
    print(json.dumps(report, indent=2))
    for vid in args.video_ids:  # leave outputs in the production configuration
        path = video_path(vid)
        events_match(vid, path if path.exists() else None)
        align.run(vid)


if __name__ == "__main__":
    main()
