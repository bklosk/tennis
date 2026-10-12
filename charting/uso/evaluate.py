"""Metrics against the Match Charting Project for the evaluation matches.

MCP has no coordinates, so positions are checked through charted facts that depend on them:
the serve side (deuce/ad) against the server's stance, the serve direction (wide/body/T)
against where the returner makes contact, the shot direction (1/2/3) against where the
receiver makes contact, and net play (volleys, '-' modifiers) against distance from the net.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from uso.paths import match_dir


def paired_shots(video_id: str, pairs: pd.DataFrame, contacts: pd.DataFrame, mcp_shots: pd.DataFrame,
                 exact_only: bool = False) -> pd.DataFrame:
    """Pair video contacts with MCP shots inside aligned points by shot number (serve = 1).
    Contacts carry hitter positions; MCP shots carry letters, direction, depth, modifiers."""
    rows = []
    vc = contacts[contacts.in_play].groupby("pid")
    ms = mcp_shots[~mcp_shots.is_fault_serve.astype(bool)].groupby("Pt")
    for r in pairs.itertuples():
        if not r.server_ok:
            continue
        if exact_only and r.v_rally != r.m_rally:
            continue
        try:
            cv = vc.get_group(r.v_pid).sort_values("shot_no")
            cm = ms.get_group(r.m_Pt).sort_values("shot_no")
        except KeyError:
            continue
        n = min(len(cv), len(cm))
        for k in range(n):
            a, b = cv.iloc[k], cm.iloc[k]
            row = {**{f"v_{c}": a[c] for c in cv.columns}, **{f"m_{c}": b[c] for c in cm.columns}}
            row.update(video_id=video_id, exact=bool(r.v_rally == r.m_rally), m_rally=r.m_rally, v_rally=r.v_rally)
            rows.append(row)
    return pd.DataFrame(rows)


def rally_metrics(pairs: pd.DataFrame) -> dict:
    ok = pairs[pairs.server_ok & pairs.m_rally.notna()]
    d = ok.v_rally - ok.m_rally.astype(float)
    return dict(n=len(ok), exact=float((d == 0).mean()), within1=float((d.abs() <= 1).mean()),
                bias=float(d.mean()), mae=float(d.abs().mean()))


def side_metrics(y_true: pd.Series, p_fh: pd.Series) -> dict:
    m = y_true.isin(["F", "B"]) & p_fh.notna()
    if m.sum() == 0:
        return dict(n=0)
    yt = (y_true[m] == "F").astype(int)
    pred = (p_fh[m] >= 0.5).astype(int)
    out = dict(n=int(m.sum()), acc=float((yt == pred).mean()))
    if yt.nunique() == 2:
        out["auc"] = float(roc_auc_score(yt, p_fh[m]))
    out["ece"] = ece(yt.to_numpy(), p_fh[m].to_numpy())
    return out


def family_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    m = y_true.notna() & y_pred.notna()
    yt, yp = y_true[m], y_pred[m]
    labels = sorted(set(yt))
    rec = {lab: float((yp[yt == lab] == lab).mean()) for lab in labels}
    return dict(n=int(m.sum()), acc=float((yt == yp).mean()),
                macro_f1=float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
                recall=rec, support={lab: int((yt == lab).sum()) for lab in labels})


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of the predicted class probability."""
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = (p >= 0.5).astype(int) == y
    edges = np.linspace(0.5, 1.0, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def position_checks(shots: pd.DataFrame) -> dict:
    """shots: paired shots with v_hx, v_hy (hitter court position at contact, meters), v_hitter_end,
    and MCP fields. Returns separability of charted categories by position."""
    out = {}
    s = shots.copy()
    # distance from the net and lateral position in the hitter's own frame (x positive to the hitter's right)
    s["dnet"] = s.v_hy.abs()
    s["xr"] = np.where(s.v_hitter_end == "near", s.v_hx, -s.v_hx)
    serve = s[s.m_is_serve.astype(bool)]
    if len(serve):
        out["server_behind_baseline"] = float((serve.dnet >= 11.0).mean())
    rally = s[~s.m_is_serve.astype(bool)]
    net = rally.m_family.isin(["volley", "half_volley", "swinging_volley", "overhead"]) | rally.m_modifiers.fillna("").str.contains("-")
    if net.nunique() == 2:
        out["net_vs_dnet_auc"] = float(roc_auc_score(net.astype(int), -rally.dnet))
        out["median_dnet_net_shots"] = float(rally.dnet[net].median())
        out["median_dnet_other"] = float(rally.dnet[~net].median())
    # return position vs serve direction: previous shot is the serve
    ret = rally[rally.m_shot_no == 2]
    if len(ret):
        prev_dir = ret.m_prev_direction if "m_prev_direction" in ret else None
        if prev_dir is not None:
            out["return_x_by_serve_dir"] = ret.groupby(prev_dir).xr.median().round(2).to_dict()
    return out
