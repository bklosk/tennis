"""Main-camera detection and per-sample court registration.

Frames are sampled at a low rate and registered with the court network. A sample is "main view"
when it registers well AND its court layout matches the match's dominant camera layout (the
high behind-the-baseline camera). That rejects other angles that also show court lines (low
baseline cameras, side cameras) and all close-ups, crowd shots and graphics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uso.court import KPS_M, CourtDetector, register
from uso.paths import match_dir
from uso.video import Meta, ff_frames, video_path

SAMPLE_W, SAMPLE_H = 640, 360
CORNERS_M = np.array([(-4.115, 11.885), (4.115, 11.885), (-4.115, -11.885), (4.115, -11.885)], np.float32)


def run(video_id: str, every: float = 1.0, det: CourtDetector | None = None, force: bool = False) -> pd.DataFrame:
    out = match_dir(video_id) / "scene.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    path = video_path(video_id)
    meta = Meta.probe(path)
    det = det or CourtDetector()
    sx, sy = meta.width / SAMPLE_W, meta.height / SAMPLE_H
    S = np.diag([sx, sy, 1.0])
    rows, buf = [], []

    def flush():
        res = det.keypoints([f for _, f in buf])
        for (t, f), (kps, sc) in zip(buf, res):
            r = register(f, kps)
            row = dict(t=t, n_kps=r.n_kps, n_inliers=r.n_inliers, reproj=r.reproj_px, line_score=r.line_score, reg_ok=r.ok)
            if r.H is not None:
                Hf = S @ r.H
                row.update({f"h{i}": v for i, v in enumerate((Hf / Hf[2, 2]).ravel())})
                c = r.to_image(CORNERS_M) / [SAMPLE_W, SAMPLE_H]
                row.update({f"c{i}": v for i, v in enumerate(c.ravel())})
            rows.append(row)
        buf.clear()

    for t, f in ff_frames(path, fps=1.0 / every, size=(SAMPLE_W, SAMPLE_H)):
        buf.append((t, f))
        if len(buf) == 16:
            flush()
    if buf:
        flush()
    df = pd.DataFrame(rows)
    df = classify(df)
    df.to_parquet(out)
    return df


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Mark live-view samples: registered, and looking down the court from behind the near
    baseline (near baseline low in the frame, far baseline high, both baselines' midpoints inside
    the frame). Broadcasts switch between a high centred camera, a low centred one and low
    cameras offset to one side for live points; all of these pass. Close-ups, crowd shots,
    graphics and true side views do not.
    Corner columns: c0,c1 far-left; c2,c3 far-right; c4,c5 near-left; c6,c7 near-right (x, y as
    fractions of the frame)."""
    df = df.copy()
    if "c0" not in df:
        df["main"] = False
        return df
    ok = df["reg_ok"].fillna(False).astype(bool) & df["c0"].notna()
    far_y = (df.c1 + df.c3) / 2
    near_y = (df.c5 + df.c7) / 2
    cx_far = (df.c0 + df.c2) / 2
    cx_near = (df.c4 + df.c6) / 2
    near_w = df.c6 - df.c4
    df["main"] = (ok & far_y.between(0.04, 0.55) & near_y.between(0.5, 1.05) & (near_y - far_y).ge(0.25)
                  & cx_far.between(0.1, 0.9) & cx_near.between(0.05, 0.95) & near_w.ge(0.3))
    return df


def segments(df: pd.DataFrame, max_gap: float = 2.01, min_len: float = 2.0) -> pd.DataFrame:
    """Contiguous main-view runs. Single-sample dropouts are bridged."""
    t = df["t"].to_numpy()
    m = df["main"].to_numpy()
    segs, start, last = [], None, None
    for ti, mi in zip(t, m):
        if mi:
            if start is None:
                start = ti
            elif ti - last > max_gap:
                segs.append((start, last))
                start = ti
            last = ti
    if start is not None:
        segs.append((start, last))
    step = np.median(np.diff(t)) if len(t) > 1 else 0.5
    out = pd.DataFrame(
        [(a - step / 2, b + step / 2) for a, b in segs if b - a >= min_len], columns=["start", "end"]
    )
    out["dur"] = out["end"] - out["start"]
    return out


def homography_at(df: pd.DataFrame, t: float) -> np.ndarray | None:
    """Homography (court m -> full-res px) from the nearest main-view samples, linearly blended."""
    m = df[df["main"]]
    if m.empty:
        return None
    ts = m["t"].to_numpy()
    i = np.searchsorted(ts, t)
    hcols = [f"h{k}" for k in range(9)]
    if i <= 0:
        return m.iloc[0][hcols].to_numpy(float).reshape(3, 3)
    if i >= len(ts):
        return m.iloc[-1][hcols].to_numpy(float).reshape(3, 3)
    a, b = m.iloc[i - 1], m.iloc[i]
    if ts[i] - ts[i - 1] > 2.0:  # not the same shot; take the nearer one
        r = a if t - ts[i - 1] < ts[i] - t else b
        return r[hcols].to_numpy(float).reshape(3, 3)
    w = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    Ha, Hb = a[hcols].to_numpy(float).reshape(3, 3), b[hcols].to_numpy(float).reshape(3, 3)
    return (1 - w) * Ha + w * Hb


__all__ = ["run", "classify", "segments", "homography_at", "KPS_M"]
