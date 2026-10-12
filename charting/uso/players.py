"""Near/far player tracks in court coordinates.

Per main-view segment and court half, a Viterbi pass picks one detection per sampled frame,
trading a position/confidence prior against implausible jumps. That keeps ball kids, the chair
umpire and line judges out of the player tracks. Positions are then smoothed and can be sampled at
any time (e.g. at a racket contact).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uso import people as people_mod
from uso.paths import match_dir


def _cand_score(cx: np.ndarray, cy: np.ndarray, conf: np.ndarray, hgt: np.ndarray) -> np.ndarray:
    x, y = np.abs(cx), np.abs(cy)
    s = np.log(np.clip(conf, 1e-3, 1))
    bad = (x > 8.5) | (y > 21.0)
    s = s - np.where((x > 5.6) & (y < 3.5), 6.0, 0.0)  # umpire chair / net-post ball kids
    s = s - np.where(x > 5.2, 1.5 * (x - 5.2), 0.0)  # outside the doubles alley
    s = s - np.where(y > 16.0, 0.6 * (y - 16.0), 0.0)  # far behind the baseline
    s = s + 0.5 * np.log(np.clip(hgt / np.median(hgt), 0.2, 5))  # crouching kids are small
    return np.where(bad, -np.inf, s)


def tracks(people: pd.DataFrame, max_speed: float = 9.0) -> pd.DataFrame:
    """One row per (seg, t, half) with the chosen detection."""
    if people.empty:
        return pd.DataFrame()
    df = people.copy()
    df["half"] = np.where(df.cy < 0, "near", "far")
    df["h"] = df.y2 - df.y1
    out = []
    for (seg, half), g in df.groupby(["seg", "half"]):
        g = g.assign(score=_cand_score(g.cx.to_numpy(), g.cy.to_numpy(), g.conf.to_numpy(), g.h.to_numpy()))
        g = g[np.isfinite(g.score)]
        if g.empty:
            continue
        times = np.sort(g.t.unique())
        groups = [gg for _, gg in g.groupby("t", sort=True)]
        costs, back, prev = [], [], None
        for k, cand in enumerate(groups):
            em = -cand.score.to_numpy()
            pos = cand[["cx", "cy"]].to_numpy()
            if prev is None:
                c, b = em.copy(), -np.ones(len(cand), int)
            else:
                ppos, pc, pt = prev
                dt = max(1e-3, times[k] - pt)
                d = np.linalg.norm(pos[:, None, :] - ppos[None, :, :], axis=2)
                trans = np.where(d / dt > max_speed, 20.0 + 2 * d, 1.5 * d)
                tot = pc[None, :] + trans
                b = tot.argmin(1)
                c = em + tot.min(1)
            costs.append(c)
            back.append(b)
            prev = (pos, c, times[k])
        j = int(np.argmin(costs[-1]))
        path = [j]
        for k in range(len(groups) - 1, 0, -1):
            j = int(back[k][j])
            path.append(j)
        for k, j in enumerate(path[::-1]):
            r = groups[k].iloc[j]
            out.append(dict(seg=seg, t=times[k], half=half, cx=r.cx, cy=r.cy, conf=r.conf, score=r.score,
                            x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2))
    trk = pd.DataFrame(out).sort_values(["seg", "half", "t"]).reset_index(drop=True)
    # light smoothing of court positions (centred 3-sample median, then mean)
    for c in ("cx", "cy"):
        trk[c + "_s"] = trk.groupby(["seg", "half"])[c].transform(
            lambda s: s.rolling(3, center=True, min_periods=1).median().rolling(3, center=True, min_periods=1).mean())
    return trk


def state_at(trk: pd.DataFrame, seg: int, half: str, t: float) -> dict | None:
    """Interpolated court position, velocity and box of one player at time t."""
    g = trk[(trk.seg == seg) & (trk.half == half)]
    if g.empty:
        return None
    ts = g.t.to_numpy()
    if t < ts[0] - 0.5 or t > ts[-1] + 0.5:
        return None
    x = np.interp(t, ts, g.cx_s.to_numpy())
    y = np.interp(t, ts, g.cy_s.to_numpy())
    i = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
    dt = max(1e-3, ts[i] - ts[i - 1])
    vx = (g.cx_s.iloc[i] - g.cx_s.iloc[i - 1]) / dt
    vy = (g.cy_s.iloc[i] - g.cy_s.iloc[i - 1]) / dt
    box = [float(np.interp(t, ts, g[c].to_numpy())) for c in ("x1", "y1", "x2", "y2")]
    return dict(x=float(x), y=float(y), vx=float(vx), vy=float(vy), box=box)


def run(video_id: str, backend: str = "rfdetr", force: bool = False) -> pd.DataFrame:
    out = match_dir(video_id) / "tracks.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    trk = tracks(people_mod.detect(video_id, backend=backend, force=force))
    trk.to_parquet(out)
    return trk
