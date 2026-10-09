"""Evaluation against MCP for the processed matches, plus deliverable exports.

Rally length, point alignment and serve side come from the aligned points. Positions are checked
through charted facts that depend on them (MCP has no coordinates):
  * serve side: the server's stance (deuce/ad) against MCP's score-implied side;
  * serve direction (4 wide / 5 body / 6 T) against how far outward the returner makes contact;
  * shot direction (1/2/3, relative to a right-hander) against the receiver's lateral position at
    their next contact;
  * net play (volleys, overheads, '-' modifiers) against the hitter's distance from the net.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from uso import truth
from uso.paths import match_dir

ERAS = [(2001, 2005, "2001-05"), (2006, 2010, "2006-10"), (2011, 2015, "2011-15"), (2016, 2020, "2016-20"),
        (2021, 2025, "2021-25")]


def era(year: int) -> str:
    return next(lab for a, b, lab in ERAS if a <= year <= b)


def rally_metrics(pairs: pd.DataFrame) -> dict:
    ok = pairs[pairs.server_ok & pairs.m_rally.notna()]
    d = ok.v_rally - ok.m_rally.astype(float)
    return dict(points=len(ok), rally_exact=float((d == 0).mean()), rally_within1=float((d.abs() <= 1).mean()),
                rally_bias=float(d.mean()), rally_mae=float(d.abs().mean()))


def _auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else np.nan


def position_checks(video_id: str, shots: pd.DataFrame) -> dict:
    """Uses shots.parquet joined with MCP shot fields (direction, modifiers) by (Pt, shot_no)."""
    ms = truth.mcp_shots(video_id)
    ms = ms[~ms.is_fault_serve.astype(bool)][["Pt", "shot_no", "direction", "modifiers", "family", "is_serve"]]
    s = shots[shots.in_play & shots.mcp_pt.notna()].copy()
    s["Pt"] = s.mcp_pt.astype(int)
    s = s.merge(ms, on=["Pt", "shot_no"], how="left")
    out = {}
    serve = s[s.kind == "serve"]
    out["server_behind_baseline"] = float((serve.hy.abs() >= 10.8).mean()) if len(serve) else np.nan
    # serve direction vs returner's outward contact position
    mp = truth.mcp_points(video_id).set_index("Pt")
    ret = s[s.shot_no == 2].copy()
    if len(ret):
        ret["side"] = ret.Pt.map(mp.side)
        sdir = s[s.shot_no == 1].set_index("Pt").direction
        ret["serve_dir"] = ret.Pt.map(sdir)
        sign = np.where((ret.hitter_end == "near") == (ret.side == "deuce"), 1.0, -1.0)
        ret["outward"] = ret.hx * sign
        g = ret[ret.serve_dir.isin(["4", "5", "6"])]
        out["return_outward_median_m"] = g.groupby("serve_dir").outward.median().round(2).to_dict()
        wt = g[g.serve_dir.isin(["4", "6"])]
        out["serve_wide_vs_T_auc"] = _auc((wt.serve_dir == "4").astype(int), wt.outward)
    # rally shot direction vs receiver's lateral position at the next contact
    s = s.sort_values(["pid", "shot_no"])
    s["next_hx"] = s.groupby("pid").hx.shift(-1)
    s["next_end"] = s.groupby("pid").hitter_end.shift(-1)
    r = s[(s.kind == "rally") & s.direction.isin(["1", "2", "3"]) & s.next_hx.notna()].copy()
    if len(r):
        r["recv_right_x"] = np.where(r.next_end == "near", r.next_hx, -r.next_hx)
        out["recv_right_x_median_by_dir"] = r.groupby("direction").recv_right_x.median().round(2).to_dict()
        r13 = r[r.direction.isin(["1", "3"])]
        out["shot_dir_1_vs_3_auc"] = _auc((r13.direction == "1").astype(int), r13.recv_right_x)
    # net play vs distance from the net
    rr = s[s.kind == "rally"].copy()
    if len(rr):
        net = rr.family.isin(["volley", "half_volley", "swinging_volley", "overhead"]) | \
            rr.modifiers.fillna("").str.contains("-", regex=False)
        rr["dnet"] = rr.hy.abs()
        out["net_shot_auc"] = _auc(net.astype(int), -rr.dnet)
        out["dnet_median_net_shots"] = float(rr.dnet[net].median()) if net.any() else np.nan
        out["dnet_median_other"] = float(rr.dnet[~net].median())
    return out


def match_summary(video_id: str) -> dict:
    d = match_dir(video_id)
    m = truth.match_row(video_id)
    pairs = pd.read_parquet(d / "aligned_points.parquet")
    shots = pd.read_parquet(d / "shots.parquet")
    vp = pd.read_parquet(d / "video_points.parquet")
    mcp = truth.mcp_points(video_id)
    row = dict(video_id=video_id, role=m.role, year=int(m.year), era=era(int(m.year)), gender=m.gender,
               round=m["round"], match=f"{m.player1} v {m.player2}", mcp_points=len(mcp), video_points=len(vp),
               matched=len(pairs), coverage=len(pairs) / len(mcp),
               side_agree=float((pairs.v_side == pairs.m_side).mean()),
               serves_agree=float((pairs.v_n_serves == pairs.m_n_serves).mean()))
    row.update(rally_metrics(pairs))
    row.update(position_checks(video_id, shots))
    return row


SOURCE = "model:uso-0.1"


def export(video_id: str, stroke_pred: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deliverable tables for one match. Every value is machine-generated (source column), match ids
    carry a '-machine' suffix, and MCP fields are kept only as clearly prefixed truth columns."""
    d = match_dir(video_id)
    m = truth.match_row(video_id)
    mid = f"{m.mcp_match_id}-machine"
    sh = pd.read_parquet(d / "shots.parquet")
    if stroke_pred is not None:
        sh = sh.merge(stroke_pred, on="hit_id", how="left")
    cols = dict(pid="video_point", mcp_pt="mcp_point", shot_no="shot_no", t="t_video_s", kind="kind",
                serve_no="serve_attempt", phantom="interpolated", hitter_end="hitter_end", hitter="hitter_player_no",
                hitter_name="hitter", hitter_hand="hitter_hand", hx="hitter_x_m", hy="hitter_y_m",
                ox="opponent_x_m", oy="opponent_y_m", dist_net="hitter_dist_net_m")
    sh = sh.copy()
    if "phantom" not in sh:
        sh["phantom"] = False
    sh["phantom"] = sh.phantom.fillna(False).astype(bool)
    for c in ("mcp_pt", "hitter", "serve_no"):
        sh[c] = pd.array(sh[c], dtype="Int64")
    out = sh.rename(columns=cols)
    keep = list(cols.values())
    for c in ("side", "p_forehand", "family", "p_groundstroke", "p_slice", "p_volley", "p_overhead", "p_lob",
              "p_drop_shot"):
        if c in out:
            out = out.rename(columns={c: f"stroke_{c}" if not c.startswith("p_") else f"stroke_{c}"})
            keep.append(f"stroke_{c}")
    out.insert(0, "match_id", mid)
    out.insert(1, "video_id", video_id)
    out["source"] = SOURCE
    truth_cols = [c for c in ("mcp_letter", "mcp_side", "mcp_family") if c in out]
    out = out[["match_id", "video_id"] + keep + ["source"] + truth_cols]
    vp = pd.read_parquet(d / "video_points.parquet")
    mp = truth.mcp_points(video_id).set_index("Pt")
    pts = vp.rename(columns={"pid": "video_point", "mcp_pt": "mcp_point", "rally": "rally_length_pred",
                             "n_serves": "serve_attempts_pred", "server_end": "server_end", "side": "serve_side_pred"})
    pts["mcp_point"] = pd.array(pts.mcp_point, dtype="Int64")
    pts["mcp_rally_length"] = pd.array(pts.mcp_point.map(mp.rally), dtype="Int64")
    pts.insert(0, "match_id", mid)
    pts.insert(1, "video_id", video_id)
    pts["source"] = SOURCE
    out.to_csv(d / "shots.csv", index=False)
    pts.to_csv(d / "points.csv", index=False)
    return out, pts
