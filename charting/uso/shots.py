"""The deliverable table: one row per detected contact.

Columns: video time, point (MCP number when aligned), shot number (serve = 1), hitter end and
named hitter with handedness, hitter and opponent court positions in meters at contact, and the
paired MCP shot (letter, side, family) for evaluation. Stroke predictions are merged on by
`uso.strokes`. Everything here is machine-generated; MCP fields are kept separately as truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uso import points as points_mod
from uso import rally, truth
from uso.paths import match_dir


def blend_learned(video_id: str, c: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Mix the learned contact model into the API-based hit scores (weight learned_w). Dev matches
    use the model trained without them, so their metrics stay out-of-sample."""
    import pickle

    from uso.paths import OUTPUTS

    w = float(p.get("learned_w") or 0.0)
    if w <= 0:
        return c
    loo = OUTPUTS / f"hitmodel_loo_{video_id}.pkl"
    path = loo if loo.exists() else OUTPUTS / "hitmodel.pkl"
    if not path.exists():
        return c
    with open(path, "rb") as f:
        m = pickle.load(f)
    sc = m.scores(c, bias=float(p.get("learned_bias", 0.0)))
    c = c.copy()
    for k in ("hit_n", "hit_f"):
        c[k] = (1 - w) * c[k] + w * sc[k]
    return c


def build(video_id: str, params: dict | None = None) -> dict:
    d = match_dir(video_id)
    p = {**rally.DECODER, **(params or {})}
    c = rally.with_api(rally.features(video_id), pd.read_parquet(d / "onset_api.parquet"), p)
    c = blend_learned(video_id, c, p)
    sv = rally.pick_serves_api(c, p_min=p["serve_p"], quiet=p["quiet"])
    vp, contacts = points_mod.assemble(c, sv, w_prior=p["w_prior"], phantom=p.get("phantom"))
    m = truth.mcp_points(video_id)
    pairs, info = points_mod.evaluate(vp, m)
    trk = pd.read_parquet(d / "tracks.parquet")
    contacts = points_mod.enrich_contacts(contacts, trk)
    # attach MCP point and player identity through the alignment
    mrow = truth.match_row(video_id)
    names = {1: mrow.player1, 2: mrow.player2}
    hands = {1: mrow.hand1, 2: mrow.hand2}
    layout = info["layout"]
    pt_of = dict(zip(pairs.v_pid, pairs.m_Pt))
    contacts["mcp_pt"] = contacts.pid.map(pt_of)
    mi = m.set_index("Pt")
    ids, nm, hd = [], [], []
    for r in contacts.itertuples():
        pt = r.mcp_pt
        if pd.isna(pt):
            ids.append(np.nan); nm.append(None); hd.append(None)
            continue
        srv = int(mi.at[pt, "server"])
        srv_end = mi.at[pt, f"server_end_{layout}"]
        hitter = srv if r.hitter_end == srv_end else 3 - srv
        ids.append(hitter); nm.append(names[hitter]); hd.append(hands[hitter])
    contacts["hitter"] = ids
    contacts["hitter_name"] = nm
    contacts["hitter_hand"] = hd
    vp["mcp_pt"] = vp.pid.map(pt_of)
    # pair with MCP shots by shot number inside aligned points
    ms = truth.mcp_shots(video_id)
    ms = ms[~ms.is_fault_serve.astype(bool)]
    key = ms.set_index(["Pt", "shot_no"])
    rows = []
    for r in contacts.itertuples():
        if pd.isna(r.mcp_pt) or not r.in_play:
            rows.append((None, None, None, None))
            continue
        k = (int(r.mcp_pt), int(r.shot_no))
        if k in key.index:
            s = key.loc[k]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            rows.append((s.letter, s.side, s.family, s.hitter))
        else:
            rows.append((None, None, None, None))
    contacts[["mcp_letter", "mcp_side", "mcp_family", "mcp_hitter"]] = pd.DataFrame(rows, index=contacts.index)
    exact = pairs.set_index("v_pid").apply(lambda r: r.v_rally == r.m_rally, axis=1).to_dict()
    contacts["point_count_exact"] = contacts.pid.map(exact)
    contacts.to_parquet(d / "shots.parquet")
    vp.to_parquet(d / "video_points.parquet")
    pairs.to_parquet(d / "aligned_points.parquet")
    return dict(info=info, contacts=contacts, vp=vp, pairs=pairs)
