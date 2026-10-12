"""Serves, rally contacts and points from audio onsets plus player tracks and pose.

Every audio onset inside a main-view segment becomes a candidate. Candidates get features from
the audio (transient strength and spectrum), both players' positions and speeds at that moment,
and pose cues (a wrist above the head for serves and overheads, wrist extension for groundstrokes).
Serves are found first; each serve opens a point, and the rally after it is decoded as the best
alternating sequence of hits (server's opponent first) under an inter-contact interval prior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uso import scene as scene_mod
from uso.paths import match_dir

NOSE, L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP, L_AN, R_AN = 0, 5, 6, 9, 10, 11, 12, 15, 16


def _kps(row) -> np.ndarray | None:
    k = row
    if k is None or (isinstance(k, float) and np.isnan(k)):
        return None
    k = np.asarray(k, np.float32).reshape(17, 3)
    if np.isnan(k[:, 0]).all():
        return None
    return k


def pose_cues(k: np.ndarray) -> dict:
    """Scale-free arm cues from one pose (image y grows downward)."""
    c = k[:, 2]
    sh = k[[L_SH, R_SH]]
    hip = k[[L_HIP, R_HIP]]
    if min(c[L_SH], c[R_SH], c[L_HIP], c[R_HIP]) < 0.25:
        return {}
    torso = float(np.linalg.norm(sh[:, :2].mean(0) - hip[:, :2].mean(0))) + 1e-3
    head_y = k[NOSE, 1] if c[NOSE] > 0.3 else sh[:, 1].mean() - 0.5 * torso
    wr = k[[L_WR, R_WR]]
    ok = wr[:, 2] > 0.25
    if not ok.any():
        return {}
    wy = wr[ok, 1]
    wx = wr[ok, 0]
    mid_x = hip[:, 0].mean()
    return dict(
        arm_up=float((head_y - wy.min()) / torso),  # >0: a wrist above the head
        both_up=float((head_y - wy.max()) / torso) if ok.all() else np.nan,
        wrist_ext=float(np.abs(wx - mid_x).max() / torso),
        wrists_apart=float(np.linalg.norm(wr[0, :2] - wr[1, :2]) / torso) if ok.all() else np.nan,
        torso_px=torso,
    )


def _player_cues(pose_g: pd.DataFrame, t: float, lo: float, hi: float) -> dict:
    """Max of each pose cue over pose samples in [t+lo, t+hi]."""
    w = pose_g[(pose_g.t >= t + lo) & (pose_g.t <= t + hi)]
    out: dict[str, float] = {}
    for kp in w.kps:
        k = _kps(kp)
        if k is None:
            continue
        for key, v in pose_cues(k).items():
            if not np.isnan(v):
                out[key] = max(out.get(key, -np.inf), v)
    return out


def features(video_id: str, force: bool = False) -> pd.DataFrame:
    out_path = match_dir(video_id) / "candidates.parquet"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)
    d = match_dir(video_id)
    on = pd.read_parquet(d / "audio_onsets.parquet")
    sc = pd.read_parquet(d / "scene.parquet")
    segs = scene_mod.segments(sc)
    pose = pd.read_parquet(d / "pose.parquet")
    starts, ends = segs.start.to_numpy(), segs.end.to_numpy()
    k = np.searchsorted(starts, on.t.to_numpy(), side="right") - 1
    inside = (k >= 0) & (on.t.to_numpy() <= ends[np.clip(k, 0, None)])
    on = on[inside].copy()
    on["seg"] = k[inside]
    rows = []
    groups = {key: g.sort_values("t") for key, g in pose.groupby(["seg", "half"])}
    for r in on.itertuples():
        row = r._asdict()
        row.pop("Index")
        for half in ("near", "far"):
            g = groups.get((r.seg, half))
            if g is None or len(g) < 2:
                continue
            ts = g.t.to_numpy()
            if r.t < ts[0] - 0.5 or r.t > ts[-1] + 0.5:
                continue
            x = np.interp(r.t, ts, g.cx_s.to_numpy())
            y = np.interp(r.t, ts, g.cy_s.to_numpy())
            x1 = np.interp(r.t - 1.0, ts, g.cx_s.to_numpy())
            y1 = np.interp(r.t - 1.0, ts, g.cy_s.to_numpy())
            x2 = np.interp(r.t + 0.6, ts, g.cx_s.to_numpy())
            y2 = np.interp(r.t + 0.6, ts, g.cy_s.to_numpy())
            p = half[0]
            row.update({f"{p}_x": x, f"{p}_y": y, f"{p}_spd_before": np.hypot(x - x1, y - y1),
                        f"{p}_spd_after": np.hypot(x2 - x, y2 - y) / 0.6})
            for key, v in _player_cues(g, r.t, -0.55, 0.1).items():
                row[f"{p}_{key}_pre"] = v
            for key, v in _player_cues(g, r.t, -0.15, 0.15).items():
                row[f"{p}_{key}_at"] = v
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    df["dt_prev"] = df.groupby("seg").t.diff()
    df["dt_next"] = -df.groupby("seg").t.diff(-1)
    df.to_parquet(out_path)
    return df


# ---------------------------------------------------------------------------------------------
# Serves

def serve_scores(c: pd.DataFrame) -> pd.DataFrame:
    """Log-odds-style score that each onset is a serve by the near ('n') or far ('f') player."""
    out = {}
    for p, o in (("n", "f"), ("f", "n")):
        y, x = c.get(f"{p}_y"), c.get(f"{p}_x")
        oy, ox = c.get(f"{o}_y"), c.get(f"{o}_x")
        if y is None or oy is None:
            out[p] = pd.Series(-np.inf, index=c.index)
            continue
        ay, ax, aoy = y.abs(), x.abs(), oy.abs()
        s = pd.Series(0.0, index=c.index)
        s += np.where(ay >= 10.3, 1.0, -4.0)  # at/behind the baseline
        s += np.where(ax <= 4.8, 0.5, -3.0)  # between centre mark and sideline
        s += np.where(aoy >= 7.0, 0.5, -3.0)  # returner back
        s += np.where(x * ox <= 0.6, 0.5, -1.5)  # diagonal
        s += np.where(c[f"{p}_spd_before"].fillna(9) <= 1.6, 0.5, -1.0)  # set before serving
        arm = c.get(f"{p}_arm_up_pre", pd.Series(np.nan, index=c.index)).fillna(0.0)  # no pose: neutral
        s += np.clip(2.5 * arm, -2.5, 2.5)
        s += np.clip(0.25 * (c.z - 8.0), -1.5, 1.5)
        out[p] = s
    return pd.DataFrame({"serve_n": out["n"], "serve_f": out["f"]}, index=c.index)


def pick_serves(c: pd.DataFrame, thr: float = 2.0, refractory: float = 3.5) -> pd.DataFrame:
    """Greedy non-maximum suppression over serve scores."""
    sc = serve_scores(c)
    best = sc.max(axis=1)
    who = np.where(sc.serve_n >= sc.serve_f, "near", "far")
    cand = c.assign(serve_score=best, server=who)[best >= thr].sort_values("serve_score", ascending=False)
    taken: list[float] = []
    rows = []
    for r in cand.itertuples():
        if any(abs(r.t - t) < refractory for t in taken):
            continue
        taken.append(r.t)
        rows.append(dict(t=r.t, seg=r.seg, server=r.server, serve_score=r.serve_score, z=r.z,
                         sx=getattr(r, f"{r.server[0]}_x"), sy=getattr(r, f"{r.server[0]}_y")))
    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True) if rows else pd.DataFrame()


def serve_side(server: str, sx: float) -> str:
    """Deuce court: the near server stands right of centre (+x); the far server, facing the
    camera, stands at -x."""
    return "deuce" if (sx > 0) == (server == "near") else "ad"


# ---------------------------------------------------------------------------------------------
# Rally hits

def hit_scores(c: pd.DataFrame, w: dict | None = None) -> pd.DataFrame:
    """Heuristic log-odds that each onset is a rally contact by near / far."""
    w = w or {}
    out = {}
    for p in ("n", "f"):
        s = w.get("bias", -1.0) + w.get("z", 0.25) * (c.z - w.get("z0", 6.0))
        ext = c.get(f"{p}_wrist_ext_at", pd.Series(np.nan, index=c.index)).fillna(0.6)
        s = s + w.get("ext", 1.0) * (ext - 0.8)
        out[p] = s.clip(-6, 6)
    return pd.DataFrame({"hit_n": out["n"], "hit_f": out["f"]}, index=c.index)


def interval_logprior(dt: np.ndarray) -> np.ndarray:
    """Log density (up to a constant) of the time between consecutive contacts."""
    dt = np.asarray(dt, float)
    lp = -0.5 * ((np.log(np.clip(dt, 1e-3, None)) - np.log(1.15)) / 0.40) ** 2
    return np.where((dt >= 0.3) & (dt <= 2.8), lp, -np.inf)


def decode_rally(t: np.ndarray, s_near: np.ndarray, s_far: np.ndarray, t_serve: float, first: str,
                 max_gap: float = 2.8, w_prior: float = 0.5, phantom: float | None = None,
                 return_phantoms: bool = False):
    """Onsets chosen as rally contacts after a serve at t_serve. Contacts alternate, starting with
    `first` ('near'/'far'); the score is sum(hit log-odds) + w_prior * interval prior.

    With `phantom` (a log-odds penalty), the decoder may bridge one missed contact between two
    chosen onsets (the other player's contact went undetected): the parity then advances by two and
    the rally length by one extra. Returns indices (and, if return_phantoms, a list of phantom
    times inserted midway)."""
    n = len(t)
    if n == 0:
        return ([], []) if return_phantoms else []
    best = np.full(n, -np.inf)
    back = -np.ones(n, int)
    parity = np.zeros(n, int)
    ph_before = np.zeros(n, bool)  # a phantom sits between back[i] and i

    def score_i(i, par):
        hitter_near = (first == "near") == (par % 2 == 0)
        return s_near[i] if hitter_near else s_far[i]

    for i in range(n):
        cands = []
        dt0 = t[i] - t_serve
        if 0.3 <= dt0 <= max_gap:  # i is the return (parity 0)
            cands.append((score_i(i, 0) + w_prior * interval_logprior(np.array([dt0]))[0], -1, 0, False))
        if phantom is not None and 2 * 0.45 <= dt0 <= 2 * max_gap:  # return missed, i is shot 3
            cands.append((phantom + score_i(i, 1) + 2 * w_prior * interval_logprior(np.array([dt0 / 2]))[0], -1, 1, True))
        for j in range(i):
            if best[j] == -np.inf:
                continue
            dt = t[i] - t[j]
            if 0.3 <= dt <= max_gap:
                par = parity[j] + 1
                cands.append((best[j] + score_i(i, par) + w_prior * interval_logprior(np.array([dt]))[0], j, par, False))
            if phantom is not None and 2 * 0.45 <= dt <= 2 * max_gap:
                par = parity[j] + 2
                cands.append((best[j] + phantom + score_i(i, par) + 2 * w_prior * interval_logprior(np.array([dt / 2]))[0],
                              j, par, True))
        if cands:
            sc, j, par, ph = max(cands, key=lambda x: x[0])
            best[i], back[i], parity[i], ph_before[i] = sc, j, par, ph
    i = int(np.argmax(best))
    if best[i] <= 0:
        return ([], []) if return_phantoms else []
    path, phantoms = [], []
    while i >= 0:
        path.append(i)
        if ph_before[i]:
            prev_t = t[back[i]] if back[i] >= 0 else t_serve
            phantoms.append((prev_t + t[i]) / 2)
        i = back[i]
    path = path[::-1]
    return (path, sorted(phantoms)) if return_phantoms else path


# ---------------------------------------------------------------------------------------------
# Decisions-API onset labels as evidence

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


DECODER = dict(audio_w=0.0, z0=20.0, bias=0.0, unlabeled=-4.0, w_prior=0.5, serve_p=0.5, quiet=2.5, phantom=None)


def with_api(c: pd.DataFrame, lab: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Join API action probabilities onto candidates and derive hit scores. Overheads mid-rally
    are often labelled 'serve', so serve + hit counts as a contact for rally decoding. A real
    contact is also loud, so audio strength adds audio_w * log(z / z0)."""
    p = {**DECODER, **(params or {})}
    c = c.drop(columns=[x for x in c.columns if x.startswith(("near_", "far_")) or x in ("hit_n", "hit_f")], errors="ignore")
    c = c.join(lab.set_index("oid").drop(columns=["tokens"], errors="ignore"))
    loud = p["audio_w"] * np.log(np.clip(c.z.to_numpy(), 1.0, None) / p["z0"])
    for who, k in (("near", "n"), ("far", "f")):
        contact = c[f"{who}_hit"].fillna(0) + c[f"{who}_serve"].fillna(0)
        s = _logit(contact.to_numpy()) + loud + p["bias"]
        c[f"hit_{k}"] = np.where(c[f"{who}_hit"].isna(), p["unlabeled"], s)
    return c


def pick_serves_api(c: pd.DataFrame, p_min: float = 0.5, quiet: float = 2.5, nms: float = 2.0) -> pd.DataFrame:
    rows = []
    for who, p, o in (("near", "n", "f"), ("far", "f", "n")):
        ps = c[f"{who}_serve"].fillna(0)
        y, x = c[f"{p}_y"].abs(), c[f"{p}_x"].abs()
        oy = c[f"{o}_y"].abs()
        ok = (ps >= p_min) & (y >= 10.0) & (x <= 5.0) & (oy.fillna(12) >= 6.0)
        for i in c.index[ok]:
            rows.append(dict(i=i, t=c.at[i, "t"], seg=c.at[i, "seg"], server=who, p=ps[i],
                             sx=c.at[i, f"{p}_x"], sy=c.at[i, f"{p}_y"], z=c.at[i, "z"]))
    if not rows:
        return pd.DataFrame()
    cand = pd.DataFrame(rows).sort_values("p", ascending=False)
    # quiet before: no likely contact by either player in the previous `quiet` seconds
    hit_p = np.maximum(c.near_hit.fillna(0), c.far_hit.fillna(0)).to_numpy()
    ts, segs = c.t.to_numpy(), c.seg.to_numpy()
    keep, taken = [], []
    for r in cand.itertuples():
        m = (segs == r.seg) & (ts >= r.t - quiet) & (ts <= r.t - 0.3)
        if (hit_p[m] >= 0.5).any():
            continue
        if any(abs(r.t - t) < nms for t in taken):
            continue
        taken.append(r.t)
        keep.append(dict(t=r.t, seg=r.seg, server=r.server, serve_score=r.p, z=r.z, sx=r.sx, sy=r.sy))
    return pd.DataFrame(keep).sort_values("t").reset_index(drop=True)


def apply_v2(c: pd.DataFrame, lab2: pd.DataFrame, params: dict | None = None, w_contact: float = 0.5) -> pd.DataFrame:
    """Overwrite hit scores with v2 single-player labels where they exist. Contact evidence is
    the 'stroke/serve contact' choice probability blended with the contact predicate."""
    p = {**DECODER, **(params or {})}
    c = c.copy()
    loud = p["audio_w"] * np.log(np.clip(c.z.to_numpy(), 1.0, None) / p["z0"])
    loud = pd.Series(loud, index=c.index)
    for who, k in (("near", "n"), ("far", "f")):
        L = lab2[lab2.half == who].set_index("oid")
        if L.empty:
            continue
        pc = (L.p_stroke_contact.fillna(0) + L.p_serve_contact.fillna(0)).clip(0, 1)
        pc = (1 - w_contact) * pc + w_contact * L.p_contact.fillna(pc)
        idx = c.index.intersection(L.index)
        c.loc[idx, f"hit_{k}"] = _logit(pc.loc[idx].to_numpy()) + loud.loc[idx].to_numpy() + p["bias"]
    return c
