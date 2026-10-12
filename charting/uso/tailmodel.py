"""Rally-end model: is a decoded contact still part of the rally?

After a point ends players often hit the dead ball (to a ball kid, back to the server, a
knock-away after a fault, a lunge at a winner). The onset labeller rightly calls these hits, but
MCP does not count them. For aligned dev points, decoded contact k (k >= 2) is labelled real when
k <= MCP's contact count. A classifier on per-contact evidence learns p(real), and at inference the
rally is cut at the first contact below a threshold chosen by leave-one-match-out validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATS = ["k", "n_serves", "dt_prev", "dt_next", "z", "sharp", "p_hit", "p_serve", "p_other", "h_y", "h_x",
         "h_spd_before", "h_spd_after", "o_y", "o_spd_before", "o_spd_after", "dt_serve", "prev_p", "prev_z"]


def contact_features(contacts: pd.DataFrame, vp: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """One row per decoded rally contact with evidence joined from the candidates table by time."""
    rc = contacts[(contacts.kind == "rally") & contacts.in_play].sort_values(["pid", "t"]).copy()
    if rc.empty:
        return rc
    ct = c.t.to_numpy()
    idx = np.clip(np.searchsorted(ct, rc.t.to_numpy()), 0, len(ct) - 1)
    near = np.abs(ct[idx] - rc.t.to_numpy()) < 1e-6
    cc = c.iloc[idx].reset_index(drop=True)
    rc = rc.reset_index(drop=True)
    is_near = rc.hitter_end == "near"
    pick = lambda a, b: np.where(is_near, cc[a].to_numpy(dtype=float), cc[b].to_numpy(dtype=float))  # noqa: E731
    out = pd.DataFrame({
        "pid": rc.pid, "t": rc.t, "k": rc.shot_no.astype(float),
        "z": np.where(near, cc.z, np.nan), "sharp": np.where(near, cc.sharp, np.nan),
        "p_hit": pick("near_hit", "far_hit"), "p_serve": pick("near_serve", "far_serve"),
        "p_other": np.where(is_near, cc.far_hit.to_numpy(dtype=float), cc.near_hit.to_numpy(dtype=float)),
        "h_y": np.abs(pick("n_y", "f_y")), "h_x": np.abs(pick("n_x", "f_x")),
        "h_spd_before": pick("n_spd_before", "f_spd_before"), "h_spd_after": pick("n_spd_after", "f_spd_after"),
        "o_y": np.abs(np.where(is_near, cc.f_y, cc.n_y)),
        "o_spd_before": np.where(is_near, cc.f_spd_before, cc.n_spd_before),
        "o_spd_after": np.where(is_near, cc.f_spd_after, cc.n_spd_after),
        "phantom": rc.get("phantom", pd.Series(False, index=rc.index)).fillna(False).astype(bool),
    })
    out["dt_prev"] = out.groupby("pid").t.diff()
    serve_t = contacts[(contacts.kind == "serve") & contacts.in_play].set_index("pid").t
    out["dt_serve"] = out.t - out.pid.map(serve_t)
    out["dt_prev"] = out.dt_prev.fillna(out.dt_serve)
    out["dt_next"] = -out.groupby("pid").t.diff(-1)
    out["prev_p"] = out.groupby("pid").p_hit.shift(1)
    out["prev_z"] = out.groupby("pid").z.shift(1)
    out["n_serves"] = out.pid.map(vp.set_index("pid").n_serves).astype(float)
    return out


def labels(feats: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """y = 1 if the contact's shot number is within MCP's count for its aligned point."""
    ok = pairs[pairs.server_ok & pairs.m_rally.notna()]
    m = dict(zip(ok.v_pid, ok.m_rally.astype(float)))
    f = feats[feats.pid.isin(m)].copy()
    f["y"] = (f.k <= f.pid.map(m)).astype(int)
    return f


class TailModel:
    def __init__(self, thr: float = 0.5):
        self.clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                  l2_regularization=1.0, random_state=0)
        self.thr = thr

    def fit(self, df: pd.DataFrame):
        self.clf.fit(df[FEATS], df.y)
        return self

    def p_real(self, feats: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(feats[FEATS])[:, 1]


def truncate(vp: pd.DataFrame, contacts: pd.DataFrame, feats: pd.DataFrame, p: np.ndarray, thr: float):
    """Cut each rally at its first contact with p_real < thr; returns updated (vp, contacts)."""
    f = feats.assign(p_real=p)
    cut_t = {}
    for pid, g in f.groupby("pid"):
        bad = g[g.p_real < thr]
        if len(bad):
            cut_t[pid] = bad.t.min()
    keep = ~contacts.apply(lambda r: r.kind == "rally" and r.pid in cut_t and r.t >= cut_t[r.pid] - 1e-9, axis=1)
    contacts = contacts[keep].copy()
    vp = vp.copy()
    n_rally = contacts[(contacts.kind == "rally") & contacts.in_play].groupby("pid").size()
    vp["rally"] = np.where(vp.get("double_fault", False), 0, 1 + vp.pid.map(n_rally).fillna(0).astype(int))
    return vp, contacts
