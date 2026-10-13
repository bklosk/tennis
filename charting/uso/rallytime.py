"""Rally-time classifier: is a point being played at this track sample?

Player tracks cover every live-view second, but more than half of that is between points
(walking, towelling, bouncing the ball). Decoded matches know their points; matches that have
only been through the vision stages (no Decisions API labels yet) do not. This model scores each
4 fps track sample from what every processed match has: both players' speeds and positions, and
the density of audio onsets around it. Labels come from decoded points: from the in-play serve
(minus 0.3 s for the swing) to the last contact (plus 0.8 s for the ball in flight).

Leave-one-match-out on the 14 decoded matches: 89% of samples right, precision 0.84, recall 0.82,
predicted rally share 0.33 vs 0.33 true.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from uso.paths import match_dir

SPEED_WINDOWS = (3, 7, 13, 25)  # samples at 4 fps
ONSET_Z = (6, 12, 30)
ONSET_WINDOWS = (1.0, 2.5, 5.0)  # seconds either side
KEEP = ("seg", "t", "y", "p", "n_cx", "n_cy", "f_cx", "f_cy")  # not features


def features(video_id: str) -> pd.DataFrame:
    """One row per track sample: positions (n_/f_ cx, cy in court meters) plus model features."""
    tr = pd.read_parquet(match_dir(video_id) / "tracks.parquet", columns=["seg", "t", "half", "cx_s", "cy_s"])
    w = tr.pivot_table(index=["seg", "t"], columns="half", values=["cx_s", "cy_s"])
    w.columns = [f"{h[0]}_{c[:2]}" for c, h in w.columns]
    w = w.reset_index().sort_values("t")
    out = []
    for _, g in w.groupby("seg"):
        g = g.copy()
        for p in "nf":
            g[f"{p}_v"] = np.hypot(g[f"{p}_cx"].diff(), g[f"{p}_cy"].diff()) / g.t.diff()
            for k in SPEED_WINDOWS:
                g[f"{p}_v{k}"] = g[f"{p}_v"].rolling(k, center=True, min_periods=1).mean()
            g[f"{p}_ay"] = g[f"{p}_cy"].abs()
            g[f"{p}_ax"] = g[f"{p}_cx"].abs()
            g[f"{p}_ay_sd13"] = g[f"{p}_cy"].rolling(13, center=True, min_periods=2).std()
        g["seg_age"] = g.t - g.t.min()
        g["seg_left"] = g.t.max() - g.t
        out.append(g)
    f = pd.concat(out).reset_index(drop=True)
    c = pd.read_parquet(match_dir(video_id) / "candidates.parquet", columns=["t", "z"])
    for zmin in ONSET_Z:
        tt = np.sort(c.t[c.z >= zmin].to_numpy())
        for win in ONSET_WINDOWS:
            f[f"on{zmin}_{win}"] = np.searchsorted(tt, f.t + win) - np.searchsorted(tt, f.t - win)
        i = np.searchsorted(tt, f.t)
        prev = np.where(i > 0, f.t - tt[np.maximum(i - 1, 0)], 99) if len(tt) else 99
        nxt = np.where(i < len(tt), tt[np.minimum(i, len(tt) - 1)] - f.t, 99) if len(tt) else 99
        f[f"since{zmin}"] = np.minimum(prev, 30)
        f[f"until{zmin}"] = np.minimum(nxt, 30)
    return f


def feature_names(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if c not in KEEP]


def labels(video_id: str, t: np.ndarray) -> np.ndarray:
    """True where a decoded point is in play."""
    s = pd.read_parquet(match_dir(video_id) / "shots.parquet", columns=["pid", "t", "kind", "in_play"])
    y = np.zeros(len(t), bool)
    for _, g in s.groupby("pid"):
        sv = g[(g.kind == "serve") & g.in_play]
        if len(sv):
            y |= (t >= sv.t.min() - 0.3) & (t <= g.t.max() + 0.8)
    return y


def fit(frames: list[pd.DataFrame]) -> HistGradientBoostingClassifier:
    tr = pd.concat(frames)
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          random_state=0).fit(tr[feature_names(tr)], tr.y)


def predict(model: HistGradientBoostingClassifier, f: pd.DataFrame) -> np.ndarray:
    """p(rally) per sample, smoothed over ±0.5 s."""
    p = model.predict_proba(f[feature_names(f)])[:, 1]
    return pd.Series(p).rolling(5, center=True, min_periods=1).mean().to_numpy()


def score(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    yh = p > thr
    tp = int((yh & y).sum())
    return dict(acc=float((yh == y).mean()), precision=tp / max(int(yh.sum()), 1), recall=tp / max(int(y.sum()), 1),
                share_true=float(y.mean()), share_pred=float(yh.mean()))
