"""Assemble video points from serves and decoded rallies, align them to MCP, and score them.

A point is one or two serve attempts by the same server from the same side, then the rally after
the last attempt. A serve that draws no rally (or only a knock-away) and is followed within
`fault_gap` seconds by another serve from the same server and side is taken as a fault.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uso import align as align_mod
from uso import rally
from uso.paths import match_dir


def assemble(c: pd.DataFrame, serves: pd.DataFrame, hit_w: dict | None = None, fault_gap: float = 40.0,
             rally_end_gap: float = 2.8, w_prior: float = 0.5, phantom: float | None = None
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (points, contacts). contacts has one row per serve attempt and rally contact."""
    if serves.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "hit_n" not in c or "hit_f" not in c:  # no learned scores supplied: heuristic ones
        c = c.join(rally.hit_scores(c, hit_w))
    serves = serves.sort_values("t").reset_index(drop=True)
    serves["side"] = [rally.serve_side(s, x) for s, x in zip(serves.server, serves.sx)]
    attempts = []
    for i, s in serves.iterrows():
        nxt = serves.t.iloc[i + 1] if i + 1 < len(serves) else np.inf
        w = c[(c.seg == s.seg) & (c.t > s.t + 0.05) & (c.t < nxt - 0.5)]
        first = "far" if s.server == "near" else "near"
        idx, ph = rally.decode_rally(w.t.to_numpy(), w.hit_n.to_numpy(), w.hit_f.to_numpy(), s.t, first,
                                     max_gap=rally_end_gap, w_prior=w_prior, phantom=phantom, return_phantoms=True)
        hits = w.iloc[idx]
        times = sorted([(x, False) for x in hits.t.tolist()] + [(x, True) for x in ph])
        attempts.append(dict(t=s.t, seg=s.seg, server=s.server, side=s.side, serve_score=s.serve_score,
                             hits=[x for x, _ in times], phantom=[p_ for _, p_ in times], n_hits=len(times)))
    # group attempts into points
    pts, contacts = [], []
    i = 0
    while i < len(attempts):
        a = attempts[i]
        group = [a]
        if i + 1 < len(attempts):
            b = attempts[i + 1]
            same = b["server"] == a["server"] and b["side"] == a["side"] and b["t"] - a["t"] < fault_gap
            if same and a["n_hits"] <= 1:
                group.append(b)
        last = group[-1]
        pid = len(pts)
        # an unreturned second serve is ten times likelier a double fault (MCP: 0 contacts) than an
        # ace (10.2% vs 0.9% of second-serve points in the charted targets)
        double_fault = len(group) == 2 and last["n_hits"] == 0
        pts.append(dict(pid=pid, t_start=group[0]["t"], t_end=(last["hits"][-1] if last["hits"] else last["t"]),
                        seg=last["seg"], server_end=a["server"], side=a["side"], n_serves=len(group),
                        rally=0 if double_fault else 1 + last["n_hits"], double_fault=double_fault))
        for k, g in enumerate(group):
            in_play = g is last and not double_fault
            contacts.append(dict(pid=pid, t=g["t"], seg=g["seg"], kind="serve", serve_no=k + 1,
                                 hitter_end=g["server"], in_play=in_play, shot_no=1 if in_play else 0))
        hitter = "far" if a["server"] == "near" else "near"
        for k, (t, phm) in enumerate(zip(last["hits"], last["phantom"])):
            contacts.append(dict(pid=pid, t=t, seg=last["seg"], kind="rally", serve_no=None, hitter_end=hitter,
                                 in_play=True, shot_no=k + 2, phantom=phm))
            hitter = "far" if hitter == "near" else "near"
        i += len(group)
    return pd.DataFrame(pts), pd.DataFrame(contacts)


def evaluate(video_pts: pd.DataFrame, mcp_pts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    al, info = align_mod.align(video_pts[["server_end", "side", "n_serves", "rally"]], mcp_pts)
    m = al.dropna().astype(int)
    v = video_pts.iloc[m.vi.to_numpy()].reset_index(drop=True)
    t = mcp_pts.iloc[m.mj.to_numpy()].reset_index(drop=True)
    pairs = pd.concat([v.add_prefix("v_"), t.add_prefix("m_")], axis=1)
    layout = info["layout"]
    pairs["server_ok"] = pairs.v_server_end == pairs[f"m_server_end_{layout}"]
    ok = pairs.m_rally.notna() & pairs.server_ok
    d = (pairs.v_rally - pairs.m_rally.astype(float))[ok]
    info.update(
        rally_exact=float((d == 0).mean()) if len(d) else np.nan,
        rally_within1=float((d.abs() <= 1).mean()) if len(d) else np.nan,
        rally_bias=float(d.mean()) if len(d) else np.nan,
        server_agree=float(pairs.server_ok.mean()) if len(pairs) else np.nan,
        side_agree=float((pairs.v_side == pairs.m_side).mean()) if len(pairs) else np.nan,
        n_serves_agree=float((pairs.v_n_serves == pairs.m_n_serves).mean()) if len(pairs) else np.nan,
        mcp_coverage=len(pairs) / max(1, len(mcp_pts)),
    )
    return pairs, info


def run(video_id: str, mcp_pts: pd.DataFrame, hit_w: dict | None = None, serve_thr: float = 2.0):
    c = rally.features(video_id)
    serves = rally.pick_serves(c, thr=serve_thr)
    vp, contacts = assemble(c, serves, hit_w)
    pairs, info = evaluate(vp, mcp_pts)
    d = match_dir(video_id)
    vp.to_parquet(d / "video_points.parquet")
    contacts.to_parquet(d / "contacts.parquet")
    pairs.to_parquet(d / "aligned_points.parquet")
    return vp, contacts, pairs, info


def enrich_contacts(contacts: pd.DataFrame, trk: pd.DataFrame) -> pd.DataFrame:
    """Add hitter and opponent court positions (meters) and velocities at each contact."""
    from uso.players import state_at

    rows = []
    for r in contacts.itertuples():
        opp = "far" if r.hitter_end == "near" else "near"
        h = state_at(trk, r.seg, r.hitter_end, r.t)
        o = state_at(trk, r.seg, opp, r.t)
        rows.append(dict(
            hx=h["x"] if h else np.nan, hy=h["y"] if h else np.nan,
            hvx=h["vx"] if h else np.nan, hvy=h["vy"] if h else np.nan,
            ox=o["x"] if o else np.nan, oy=o["y"] if o else np.nan,
        ))
    out = pd.concat([contacts.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["dist_net"] = out.hy.abs()
    # the time is part of the id so answers cached for an earlier decoding never attach to a different contact
    out["hit_id"] = [f"{p}_{k}_{s}_{t:.3f}" for p, k, s, t in
                     zip(out.pid, out.shot_no, out.serve_no.fillna(0).astype(int), out.t)]
    out["half"] = out.hitter_end
    return out
