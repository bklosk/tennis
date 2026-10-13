"""Analytics over every processed match: court heatmaps and charts as PNG images, plus CSV tables.

    uv run python scripts/analytics.py [--sync] [--refit]

Three tiers of data:
  * charted (decoded, aligned to an MCP record): every contact with both players' positions,
    stroke side and family, named hitters, and the point winner from MCP;
  * uncharted (decoded, no MCP record): contacts, positions and strokes by court end only; used
    where identity and point winners are not needed (all-right-handed matches only for
    forehand/backhand charts);
  * tracked (vision stages only): both players' positions at 4 fps. Rally time comes from
    `uso.rallytime`, trained on the charted matches; every match uses the classifier so the tiers
    are comparable, and charted matches are scored out-of-fold.

Frames are court meters, x across (0 = centre line), y along (0 = net). The hitter frame puts the
player at the near end (y < 0) facing the net with x > 0 on their right; the forehand frame also
mirrors left-handers so x > 0 is the forehand side. Depth is meters behind the baseline.

--sync downloads the small per-match tables from the Space for match folders not on disk;
--refit retrains the rally-time model even when the cached predictions match the charted set.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analytics_plots as P  # noqa: E402

from uso import rallytime  # noqa: E402
from uso.paths import BATCH_MATCHES, OUTPUTS, PROJECT, match_dir  # noqa: E402

REPORT = PROJECT / "reports" / "analytics"
SPACE_PREFIX = "tennis/charting-outputs/"
SYNC_FILES = ("tracks.parquet", "candidates.parquet", "shots.parquet", "shots.csv", "points.csv",
              "aligned_points.parquet", "uncharted_shots.csv", "uncharted_points.csv")

BASELINE = 11.885
SERVICE_LINE = 6.40
HALF = dict(x=(-8.0, 8.0), y=(-20.0, 0.0))  # hitter frame, net at the top edge
FULL = dict(x=(-8.0, 8.0), y=(-20.0, 20.0))
CV_FOLDS = 5  # rally-time model: grouped folds once there are more charted matches than this x3


# ---------------------------------------------------------------- data

def sync_from_space() -> int:
    """Fetch the analysis tables for Space match folders that are missing on disk."""
    from archive_videos import BUCKET, s3_client

    s3 = s3_client()
    got = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=SPACE_PREFIX):
        for o in page.get("Contents", []):
            parts = o["Key"][len(SPACE_PREFIX):].split("/")  # YEAR/MATCH__VID/FILE
            if len(parts) != 3 or parts[2] not in SYNC_FILES:
                continue
            dest = match_dir(parts[1].rsplit("__", 1)[-1]) / parts[2]
            if not dest.exists():
                s3.download_file(BUCKET, o["Key"], str(dest))
                got += 1
    return got


def matches() -> pd.DataFrame:
    """Every match folder with tracks, its metadata and tier (charted / uncharted / tracked)."""
    at = pd.read_csv(PROJECT / "data" / "archive_targets.csv")
    em = pd.read_csv(PROJECT / "data" / "eval_matches.csv")
    bm = pd.read_csv(BATCH_MATCHES) if BATCH_MATCHES.exists() else em.iloc[0:0].assign(match_id=None)
    meta = {r.video_id: dict(year=int(r.year), gender=r.draw, round=r["round"], p1=r.player1, p2=r.player2)
            for _, r in at.iterrows()}
    for _, r in em.iterrows():
        meta.setdefault(r.video_id, dict(year=int(r.year), gender="men" if r.gender == "m" else "women",
                                         round=r["round"], p1=r.player1, p2=r.player2))
    mcp = set(em.video_id) | set(bm.video_id[bm.mcp_match_id.notna()])
    hands = {r.video_id: (r.hand1, r.hand2) for t in (em, bm) for _, r in t.iterrows()}
    rows = []
    for d in sorted(OUTPUTS.iterdir()):
        if not (d / "tracks.parquet").exists() or not (d / "candidates.parquet").exists() or d.name not in meta:
            continue
        v, m = d.name, meta[d.name]
        tier = ("charted" if (d / "shots.csv").exists() and v in mcp
                else "uncharted" if (d / "uncharted_shots.csv").exists() else "tracked")
        last = lambda n: n.split()[-1]  # noqa: E731
        rows.append(dict(video_id=v, **m, tier=tier, decoded=tier != "tracked",
                         all_right=hands.get(v, ("?", "?")) == ("R", "R"),
                         label=f"{m['year']} {m['round']} {last(m['p1'])}–{last(m['p2'])}"))
    return pd.DataFrame(rows).sort_values(["year", "gender", "round", "label"]).reset_index(drop=True)


def rally_samples(ms: pd.DataFrame, refit: bool) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Per match: track samples with p(rally). Charted matches are predicted out-of-fold (leave one
    match out, or grouped folds when there are many); the others by a model trained on all of them."""
    cached = {v: match_dir(v) / "rally_p.parquet" for v in ms.video_id}
    evals = OUTPUTS / "rallytime_eval.csv"
    dec = list(ms.video_id[ms.tier == "charted"])
    if not refit and all(p.exists() for p in cached.values()) and evals.exists():
        ev = pd.read_csv(evals)
        if set(ev.video_id) == set(dec):
            return {v: pd.read_parquet(p) for v, p in cached.items()}, ev
    frames = {v: rallytime.features(v) for v in ms.video_id}
    for v in dec:
        frames[v]["y"] = rallytime.labels(v, frames[v].t.to_numpy())
    folds = [[v] for v in dec] if len(dec) <= 3 * CV_FOLDS else [dec[k::CV_FOLDS] for k in range(CV_FOLDS)]
    res = []
    for fold in folds:
        m = rallytime.fit([frames[u] for u in dec if u not in fold])
        for v in fold:
            frames[v]["p"] = rallytime.predict(m, frames[v])
            res.append(dict(video_id=v, **rallytime.score(frames[v].y.to_numpy(), frames[v].p.to_numpy())))
    full = rallytime.fit([frames[u] for u in dec])
    for v in ms.video_id[ms.tier != "charted"]:
        frames[v]["p"] = rallytime.predict(full, frames[v])
    keep = ["seg", "t", "n_cx", "n_cy", "f_cx", "f_cy", "n_v", "f_v", "p"]
    out = {}
    for v, f in frames.items():
        out[v] = f[keep + (["y"] if "y" in f else [])].reset_index(drop=True)
        out[v].to_parquet(cached[v])
    ev = pd.DataFrame(res)
    ev.to_csv(evals, index=False)
    return out, ev


def occupancy(f: pd.DataFrame, thr: float = 0.5) -> pd.DataFrame:
    """Both players during rally time, in their own hitter frame."""
    r = f[f.p > thr]
    near = pd.DataFrame(dict(x=r.n_cx, y=r.n_cy, end="near")).dropna()
    far = pd.DataFrame(dict(x=-r.f_cx, y=-r.f_cy, end="far")).dropna()
    o = pd.concat([near, far], ignore_index=True)
    return o[o.y < 0]


UNCHARTED_COLS = {"pid": "video_point", "t": "t_video_s", "serve_no": "serve_attempt", "hx": "hitter_x_m",
                  "hy": "hitter_y_m", "ox": "opponent_x_m", "oy": "opponent_y_m", "side": "stroke_side",
                  "family": "stroke_family", "dist_net": "hitter_dist_net_m"}


def shots_table(ms: pd.DataFrame) -> pd.DataFrame:
    """Decoded contacts with hitter-frame and forehand-frame positions. Charted matches carry the
    named hitter and the MCP point winner; uncharted ones only the court end (and a hand only when
    both players are right-handed)."""
    out = []
    for _, m in ms[ms.decoded].iterrows():
        v, d = m.video_id, match_dir(m.video_id)
        if m.tier == "charted":
            s = pd.read_csv(d / "shots.csv", low_memory=False)
            sp = pd.read_parquet(d / "shots.parquet", columns=["pid", "t", "in_play"])
            s = s.merge(sp.rename(columns={"pid": "video_point", "t": "t_video_s"}),
                        on=["video_point", "t_video_s"], how="left")
            pts = pd.read_csv(d / "points.csv")
            s = s.merge(pts[["video_point", "serve_side_pred", "rally_length_pred", "double_fault"]], on="video_point")
            ap = pd.read_parquet(d / "aligned_points.parquet")
            ap = ap[ap.server_ok][["m_Pt", "m_winner", "m_server", "m_rally"]].rename(columns={"m_Pt": "mcp_point"})
            s = s.merge(ap, on="mcp_point", how="left")
        else:
            s = pd.read_csv(d / "uncharted_shots.csv").rename(columns=UNCHARTED_COLS)
            up = pd.read_csv(d / "uncharted_points.csv")
            s = s.merge(up[["pid", "rally", "side", "double_fault"]].rename(
                columns={"pid": "video_point", "rally": "rally_length_pred", "side": "serve_side_pred"}), on="video_point")
            s["hitter_hand"] = "R" if m.all_right else np.nan
            for c in ("hitter", "hitter_player_no", "m_winner", "m_server", "m_rally", "mcp_point"):
                s[c] = np.nan
        for k, val in m.items():
            s[f"m_{k}"] = val
        out.append(s)
    s = pd.concat(out, ignore_index=True)
    far = s.hitter_end == "far"
    s["hx"] = np.where(far, -s.hitter_x_m, s.hitter_x_m)
    s["hy"] = np.where(far, -s.hitter_y_m, s.hitter_y_m)
    s["ox"] = np.where(far, -s.opponent_x_m, s.opponent_x_m)  # opponent in the hitter's frame (y > 0)
    s["oy"] = np.where(far, -s.opponent_y_m, s.opponent_y_m)
    s["fx"] = np.where(s.hitter_hand == "L", -s.hx, s.hx)
    s["depth"] = -s.hy - BASELINE
    s["won"] = np.where(s.m_winner.notna() & s.hitter_player_no.notna(),
                        (s.m_winner == s.hitter_player_no).astype(float), np.nan)
    return s


# ---------------------------------------------------------------- grids

def grid(x, y, cell: float, extent=HALF, sigma: float | None = None) -> np.ndarray:
    """Counts on a cell grid, row 0 at the top (largest y), optionally Gaussian-smoothed."""
    (x0, x1), (y0, y1) = extent["x"], extent["y"]
    nx, ny = round((x1 - x0) / cell), round((y1 - y0) / cell)
    h, _, _ = np.histogram2d(np.asarray(y, float), np.asarray(x, float), bins=[ny, nx], range=[[y0, y1], [x0, x1]])
    h = h[::-1]
    return gaussian_filter(h, sigma / cell, mode="constant") if sigma else h


def heat_panels(groups: dict[str, pd.DataFrame], cell=0.25, sigma=0.45, extent=HALF) -> dict:
    """Density panels on one shared scale (each normalized to its own sample count); dens in [0, 1]."""
    dens = {k: grid(g.x, g.y, cell, extent, sigma) / max(len(g), 1) for k, g in groups.items()}
    vmax = max(float(d.max()) for d in dens.values())
    return {k: dict(n=len(g), dens=dens[k] / vmax, x0=extent["x"][0], x1=extent["x"][1], y0=extent["y"][0],
                    y1=extent["y"][1], peak=100 * vmax / cell**2) for k, g in groups.items()}


def mini_panel(x, y, cell=0.25, sigma=0.6) -> dict:
    d = grid(x, y, cell, HALF, sigma)
    return dict(dens=d / max(float(d.max()), 1e-12), x0=HALF["x"][0], x1=HALF["x"][1], y0=HALF["y"][0], y1=HALF["y"][1])


# ---------------------------------------------------------------- analyses

def zone_table(g: pd.DataFrame) -> pd.DataFrame:
    """Share of samples by depth band (rows) and third of the court width (columns)."""
    depth = -g.y - BASELINE
    bands = [("Inside the service line", depth < SERVICE_LINE - BASELINE),
             ("Inside the baseline", (depth >= SERVICE_LINE - BASELINE) & (depth < 0)),
             ("0–1 m behind", (depth >= 0) & (depth < 1)), ("1–2 m behind", (depth >= 1) & (depth < 2)),
             ("2–3 m behind", (depth >= 2) & (depth < 3)), ("3+ m behind", depth >= 3)]
    cols = [("left_third", g.x < -1.37), ("middle_third", g.x.abs() <= 1.37), ("right_third", g.x > 1.37)]
    n = max(len(g), 1)
    return pd.DataFrame([dict(zone=name, **{c: round(100 * float((b & cm).sum()) / n, 1) for c, cm in cols},
                              total=round(100 * float(b.sum()) / n, 1)) for name, b in bands])


def win_map(r: pd.DataFrame, cell=1.5, min_n=40) -> dict:
    """Share of rally shots whose hitter went on to win the point, by contact cell. Cells are
    centred on the centre line and start at the baseline."""
    r = r[r.won.notna() & (r.hy < 0)]
    xe = np.arange(-6.75, 6.76, cell)
    de = np.arange(-4.5, 7.6, cell)  # depth edges, meters behind the baseline
    cells = []
    for (i, j), g in r.groupby([np.digitize(r.depth, de), np.digitize(r.hx, xe)]):
        if 0 < i < len(de) and 0 < j < len(xe) and len(g) >= min_n:
            cells.append(dict(x0=float(xe[j - 1]), x1=float(xe[j]), d0=float(de[i - 1]), d1=float(de[i]), n=len(g),
                              win=round(100 * g.won.mean(), 1)))
    bands = []
    for name, lo, hi in [("Inside the baseline", -99, 0), ("0–1 m behind", 0, 1), ("1–2 m behind", 1, 2),
                         ("2–3 m behind", 2, 3), ("3–4 m behind", 3, 4), ("4+ m behind", 4, 99)]:
        g = r[(r.depth >= lo) & (r.depth < hi)]
        bands.append(dict(zone=name, shots=len(g), win=round(100 * g.won.mean(), 1) if len(g) else None))
    return dict(cells=cells, bands=bands, n=len(r), overall=round(100 * r.won.mean(), 1), min_n=min_n)


def serve_setup(s: pd.DataFrame) -> dict:
    """Server and returner at the serve contact, server frame (server y < 0, returner y > 0)."""
    sv = s[(s.kind == "serve") & s.serve_attempt.isin([1, 2]) & s.ox.notna()]
    out, tab = {}, []
    for k in (1, 2):
        g = sv[sv.serve_attempt == k]
        out[f"serve{k}"] = pd.DataFrame(dict(x=np.r_[g.hx, g.ox], y=np.r_[g.hy, g.oy]))
        tab.append(dict(serve=f"{'1st' if k == 1 else '2nd'} serve", n=len(g),
                        server_depth=round(float((-g.hy - BASELINE).median()), 2),
                        returner_depth=round(float((g.oy - BASELINE).median()), 2),
                        returner_wide=round(float(g.ox.abs().median()), 2)))
    return dict(panels=heat_panels(out, cell=0.25, sigma=0.35, extent=FULL), table=pd.DataFrame(tab))


def return_dumbbell(s: pd.DataFrame, min_first=30, min_second=12) -> pd.DataFrame:
    """Median returner depth at serve contact, 1st vs 2nd serve, per returner over all their matches."""
    sv = s[(s.kind == "serve") & s.serve_attempt.isin([1, 2]) & s.oy.notna() & s.hitter.notna()].copy()
    players = s.dropna(subset=["hitter"]).groupby("m_video_id").hitter.unique()
    sv["returner"] = [next((p for p in players[v] if p != h), None) for v, h in zip(sv.m_video_id, sv.hitter)]
    sv["rdepth"] = sv.oy - BASELINE
    rows = []
    for ret, g in sv.groupby("returner"):
        a, b = g[g.serve_attempt == 1].rdepth, g[g.serve_attempt == 2].rdepth
        if len(a) >= min_first and len(b) >= min_second:
            rows.append(dict(player=ret, gender=g.m_gender.iloc[0], matches=g.m_video_id.nunique(),
                             first=round(float(a.median()), 2), second=round(float(b.median()), 2),
                             n_first=len(a), n_second=len(b)))
    d = pd.DataFrame(rows)
    return d.assign(step=d["first"] - d["second"]).sort_values("step", ascending=False).reset_index(drop=True)


def rally_lengths(ms: pd.DataFrame) -> dict:
    pts = []
    for _, m in ms[ms.tier == "charted"].iterrows():
        ap = pd.read_parquet(match_dir(m.video_id) / "aligned_points.parquet")
        ap = ap[ap.server_ok & ap.m_rally.notna()]
        pts.append(pd.DataFrame(dict(machine=ap.v_rally, mcp=ap.m_rally.astype(int), winner=ap.m_winner,
                                     server=ap.m_server, gender=m.gender)))
    p = pd.concat(pts, ignore_index=True)
    cap = 15
    dist = pd.DataFrame([dict(shots=f"{cap}+" if k == cap else str(k),
                              machine=round(100 * float((p.machine.clip(upper=cap) == k).mean()), 1),
                              mcp=round(100 * float((p.mcp.clip(upper=cap) == k).mean()), 1)) for k in range(cap + 1)])
    srv = []
    for g_name in ("men", "women"):
        g = p[p.gender == g_name]
        for name, lo, hi in [("1–4 shots", 1, 4), ("5–8 shots", 5, 8), ("9+ shots", 9, 999)]:
            b = g[(g.machine >= lo) & (g.machine <= hi)]
            srv.append(dict(gender=g_name, bucket=name, points=len(b),
                            server_win=round(100 * float((b.winner == b.server).mean()), 1)))
    means = {g: dict(machine=round(float(p[p.gender == g].machine.mean()), 2),
                     mcp=round(float(p[p.gender == g].mcp.mean()), 2)) for g in ("men", "women")}
    return dict(dist=dist, server=pd.DataFrame(srv), means=means, n=len(p),
                exact=round(100 * float((p.machine == p.mcp).mean()), 1),
                within1=round(100 * float(((p.machine - p.mcp).abs() <= 1).mean()), 1))


def forehand_by_x(r: pd.DataFrame) -> pd.DataFrame:
    """Forehand share of groundstrokes by lateral contact position (forehand frame). Returns are
    split off: a returner starts wide in the serve box and moves toward the ball."""
    g = r[r.stroke_side.isin(["F", "B"]) & r.stroke_family.isin(["groundstroke", "slice"]) & r.hitter_hand.notna()]
    kinds = {"rally": g[g.shot_no >= 4], "returns": g[g.shot_no == 2]}
    out = []
    for lo in np.arange(-5.0, 5.0, 0.5):
        row = dict(x=round(lo + 0.25, 2))
        for k, gk in kinds.items():
            b = gk[(gk.fx >= lo) & (gk.fx < lo + 0.5)]
            row[k] = round(100 * float((b.stroke_side == "F").mean()), 1) if len(b) >= 25 else None
            row[f"n_{k}"] = len(b)
        out.append(row)
    return pd.DataFrame(out)


def distance_run(ms: pd.DataFrame, samples: dict[str, pd.DataFrame], s: pd.DataFrame) -> dict:
    """Meters each player covers during a charted point (in-play serve to last contact), split by
    who won it. Steps are from the smoothed tracks at 4 fps."""
    rows = []
    for _, m in ms[ms.tier == "charted"].iterrows():
        f = samples[m.video_id].sort_values("t")
        sh = s[s.m_video_id == m.video_id]
        for _, g in sh.groupby("video_point"):
            sv = g[(g.kind == "serve") & g.in_play.eq(True)]
            if not len(sv) or g.won.isna().all() or g.hitter.isna().all():
                continue
            a, b = sv.t_video_s.min(), g.t_video_s.max()
            w = f[(f.t >= a) & (f.t <= b)]
            if len(w) < 3 or w.seg.nunique() > 1:
                continue
            d = {}
            for p in "nf":
                st = np.hypot(w[f"{p}_cx"].diff(), w[f"{p}_cy"].diff())
                d[p] = float(st[st < 2.5].sum())  # a >10 m/s step is a track jump, not running
            first = g.iloc[0]
            near_player = first.hitter_player_no if first.hitter_end == "near" else 3 - first.hitter_player_no
            near_won = near_player == first.m_winner
            rows.append(dict(rally=int(first.rally_length_pred), seconds=b - a,
                             winner_m=d["n"] if near_won else d["f"], loser_m=d["f"] if near_won else d["n"]))
    d = pd.DataFrame(rows)
    d = d[d.rally >= 3]
    by = []
    for lo, hi, name in [(3, 4, "3–4"), (5, 6, "5–6"), (7, 8, "7–8"), (9, 11, "9–11"), (12, 999, "12+")]:
        b = d[(d.rally >= lo) & (d.rally <= hi)]
        by.append(dict(rally=name, points=len(b), winner=round(float(b.winner_m.median()), 1),
                       loser=round(float(b.loser_m.median()), 1)))
    return dict(by_rally=pd.DataFrame(by), n=len(d), loser_more=round(100 * float((d.loser_m > d.winner_m).mean()), 1),
                per_sec=round(float((d.winner_m + d.loser_m).sum() / d.seconds.sum() / 2), 2))


def per_match(ms: pd.DataFrame, samples: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    rows, minis = [], {}
    for _, m in ms.iterrows():
        f = samples[m.video_id]
        o = occupancy(f)
        depth = -o.y - BASELINE
        rows.append(dict(video_id=m.video_id, label=m.label, year=m.year, gender=m.gender, round=m["round"],
                         tier=m.tier, rally_hours=round(float((f.p > 0.5).sum()) * 0.25 / 3600, 2),
                         depth_median=round(float(depth.median()), 2),
                         net_share=round(100 * float((-o.y < SERVICE_LINE).mean()), 2),
                         lateral_sd=round(float(o.x.std()), 2)))
        minis[m.video_id] = mini_panel(o.x, o.y)
    return pd.DataFrame(rows), minis


def player_prints(r: pd.DataFrame, min_shots=150) -> list[dict]:
    """Rally-shot positions per player over all their charted matches (hitter frame, not mirrored)."""
    out = []
    for who, g in r[r.hitter.notna()].groupby("hitter"):
        if len(g) < min_shots:
            continue
        gs = g[g.stroke_side.isin(["F", "B"])]
        out.append(dict(player=who, hand=g.hitter_hand.mode().iloc[0], gender=g.m_gender.iloc[0],
                        matches=g.m_video_id.nunique(), shots=len(g), depth=round(float(g.depth.median()), 2),
                        inside=round(100 * float((g.depth < 0).mean()), 1),
                        fh=round(100 * float((gs.stroke_side == "F").mean()), 1), panel=mini_panel(g.hx, g.hy, sigma=0.7)))
    return sorted(out, key=lambda d: d["depth"])


def build(refit: bool) -> dict:
    ms = matches()
    samples, rt_eval = rally_samples(ms, refit)
    s = shots_table(ms)
    r = s[(s.kind == "rally") & (s.hy < 0)]
    pm, minis = per_match(ms, samples)
    occ = {g: pd.concat([occupancy(samples[v]) for v in ms.video_id[ms.gender == g]])[["x", "y"]] for g in ("men", "women")}
    fhb = r[r.stroke_side.isin(["F", "B"]) & r.hitter_hand.notna()]
    contacts = {"forehand": fhb[fhb.stroke_side == "F"].rename(columns={"fx": "x", "hy": "y"}),
                "backhand": fhb[fhb.stroke_side == "B"].rename(columns={"fx": "x", "hy": "y"})}
    sv = s[(s.kind == "serve") & s.m_tier.eq("charted")]
    man = pd.read_csv(OUTPUTS / "archive_manifest_latest.csv")
    archived = set(man.video_id[man.status.str.startswith("archived")])
    return dict(
        generated=time.strftime("%Y-%m-%d"), ms=ms, pm=pm, minis=minis, rt_eval=rt_eval,
        kpi=dict(matches=len(ms), charted=int((ms.tier == "charted").sum()), uncharted=int((ms.tier == "uncharted").sum()),
                 tracked=int((ms.tier == "tracked").sum()), rally_shots=len(r), serves=int((s.kind == "serve").sum()),
                 rally_hours=round(float(pm.rally_hours.sum()), 1), archived=len(archived),
                 unprocessed=len(archived - set(ms.video_id)), years=f"{ms.year.min()}–{ms.year.max()}"),
        contacts=dict(panels=heat_panels(contacts), table={k: zone_table(g) for k, g in contacts.items()},
                      depth={k: round(float(g.depth.median()), 2) for k, g in contacts.items()},
                      matches=int(fhb.m_video_id.nunique())),
        occupancy=dict(panels=heat_panels(occ), table={k: zone_table(g) for k, g in occ.items()},
                       depth={k: round(float((-g.y - BASELINE).median()), 2) for k, g in occ.items()},
                       matches={g: int((ms.gender == g).sum()) for g in ("men", "women")}),
        winmap=win_map(r), serve=serve_setup(s), dumbbell=return_dumbbell(s), rally=rally_lengths(ms),
        forehand=forehand_by_x(r), distance=distance_run(ms, samples, s), players=player_prints(r),
        calib=dict(server={e: round(float(g.depth.median()), 2) for e, g in sv.groupby("hitter_end")},
                   returner={("far" if e == "near" else "near"): round(float((g.oy - BASELINE).median()), 2)
                             for e, g in sv[sv.serve_attempt == 1].groupby("hitter_end")}),
    )


# ---------------------------------------------------------------- images

def pct(v) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:.1f}%"


def fig_contacts(D, path):
    C = D["contacts"]
    fh, bh = C["table"]["forehand"], C["table"]["backhand"]
    sub = (f"Backhands come from the backhand third of the court {bh.left_third.sum():.0f}% of the time. Forehands spread "
           f"across the width: {fh.left_third.sum():.0f}% are struck from the backhand third (run-around forehands), "
           f"{fh.right_third.sum():.0f}% from the forehand third. Median contact depth: {C['depth']['forehand']:.2f} m "
           f"(forehands) and {C['depth']['backhand']:.2f} m (backhands) behind the baseline.")
    note = (f"Rally shots from {C['matches']} decoded matches, folded into the hitter's own half (net at the top) and "
            "mirrored for left-handers, so the forehand side is always on the right. Density smoothed over ~45 cm.")
    fig, top = P.figure(7.8, 7.2, "Where forehands and backhands are struck", sub, note)
    p = C["panels"]
    _, pw = P.court_row(fig, top, 0.13, [(p["forehand"], "Forehands", f"{p['forehand']['n']:,} shots"),
                                         (p["backhand"], "Backhands", f"{p['backhand']['n']:,} shots")])
    P.heat_scale(fig, 0.3 / 7.8 + 0.02, 0.105, 0.22, p["forehand"]["peak"], "shots")
    P.save(fig, path)


def fig_occupancy(D, path):
    O, RT = D["occupancy"], D["rt_eval"]
    deep = {g: O["table"][g].set_index("zone").at["3+ m behind", "total"] for g in ("men", "women")}
    sub = (f"Men play a little deeper than women: a median {O['depth']['men']:.2f} m behind the baseline against "
           f"{O['depth']['women']:.2f} m, and they spend {deep['men']:.1f}% of rally time 3 m or more back "
           f"(women {deep['women']:.1f}%). The hot spots either side of the centre mark are serve and return stances.")
    note = (f"All {D['kpi']['matches']} processed matches, both players folded into one half. Rally time (serve to last "
            "contact) comes from a classifier on player speed, position and audio onsets, trained on the charted "
            f"matches; held out it labels {100 * RT.acc.mean():.1f}% of samples right (precision "
            f"{100 * RT.precision.mean():.1f}%, recall {100 * RT.recall.mean():.1f}%).")
    fig, top = P.figure(7.8, 7.3, "Where players stand while the ball is in play", sub, note)
    p = O["panels"]
    P.court_row(fig, top, 0.14, [(p["men"], "Men", f"{O['matches']['men']} matches · {p['men']['n']:,} samples"),
                                 (p["women"], "Women", f"{O['matches']['women']} matches · {p['women']['n']:,} samples")])
    P.heat_scale(fig, 0.3 / 7.8 + 0.02, 0.115, 0.22, p["men"]["peak"], "samples")
    P.save(fig, path)


def fig_winmap(D, path):
    W = D["winmap"]
    b = W["bands"]
    sub = (f"Hitters who strike from inside the baseline go on to win {b[0]['win']:.1f}% of those points; from 4 m or "
           f"more behind, {b[-1]['win']:.1f}%. Across all {W['n']:,} rally shots the hitter wins {W['overall']:.1f}%.")
    note = (f"Charted matches only (point winner from the Match Charting Project). Hitter frame, 1.5 m cells with at "
            f"least {W['min_n']} shots. Descriptive: being pushed deep is both a cause and a symptom of losing the rally.")
    fig, top = P.figure(9.2, 6.9, "Pressure map: hit from here, win the point?", sub, note)
    H = 6.9
    ph = (top - 0.17) * H - 0.2
    pw = ph * 15 / 20.1
    ax = fig.add_axes([0.3 / 9.2, 0.17, pw / 9.2, ph / H])
    P.court(ax)
    for c in W["cells"]:
        y_top, y_bot = -(P.BL + c["d0"]), -(P.BL + c["d1"])
        g = 0.06
        ax.add_patch(P.Rectangle((c["x0"] + g, y_bot + g), c["x1"] - c["x0"] - 2 * g, y_top - y_bot - 2 * g,
                                 color=P.div_color((c["win"] - 50) / 15), lw=0, zorder=2))
    cax = fig.add_axes([0.3 / 9.2 + 0.035, 0.135, 0.2, 0.014])
    t = np.linspace(-1, 1, 256)
    cax.imshow(np.array([[P.div_color(u) for u in t]]), aspect="auto", extent=(0, 1, 0, 1))
    cax.axis("off")
    fig.text(0.3 / 9.2 + 0.027, 0.142, "35%", ha="right", va="center", fontsize=7.5, color=P.MUTED)
    fig.text(0.3 / 9.2 + 0.243, 0.142, "65% of points won by the hitter (50% = even)", va="center", fontsize=7.5,
             color=P.MUTED)
    bx = (0.3 + pw + 0.9) / 9.2
    fig.text(bx - 0.04, top, "Hitter's win rate by contact depth", fontsize=10, fontweight="bold", va="top")
    ax2 = fig.add_axes([bx, 0.25, 0.93 - bx, top - 0.33])
    ax2.set_ylim(0, 70)
    cats = [z["zone"].replace(" behind", "").replace("Inside the baseline", "inside") for z in b]
    P.value_axis(ax2, [0, 10, 20, 30, 40, 50, 60, 70], lambda v: f"{v}%", "contact depth, meters behind the baseline")
    P.columns(ax2, cats, [("win", P.BLUE)], {"win": [z["win"] for z in b]}, labels=lambda v: f"{v:.0f}", ref=50)
    P.save(fig, path)


def fig_serve(D, path):
    S = D["serve"]
    a, b = S["table"].iloc[0], S["table"].iloc[1]
    sub = (f"Both players at the moment of serve contact, server at the bottom. On first serves the returner waits a "
           f"median {a.returner_depth:.2f} m behind the baseline; on second serves {b.returner_depth:.2f} m, a step of "
           f"{a.returner_depth - b.returner_depth:.1f} m in. The two returner plumes are the deuce and ad courts.")
    fig, top = P.figure(6.6, 8.6, "The serve standoff", sub, "Decoded matches; serves by attempt number, faults included.")
    p = S["panels"]
    view = dict(x=(-7, 7), y=(-13.5, 18.5))
    axes, _ = P.court_row(fig, top, 0.11, [(p["serve1"], "1st serve", f"{int(a.n):,} serves"),
                                           (p["serve2"], "2nd serve", f"{int(b.n):,} serves")], view=view, full=True)
    for ax in axes:
        ax.text(-6.8, 18.2, "Returner", fontsize=7.5, color=P.MUTED, va="top")
        ax.text(-6.8, -13.2, "Server", fontsize=7.5, color=P.MUTED, va="bottom")
    P.heat_scale(fig, 0.3 / 6.6 + 0.02, 0.085, 0.25, p["serve1"]["peak"], "positions")
    P.save(fig, path)


def fig_dumbbell(D, path):
    d = D["dumbbell"]
    closer = int((d["second"] < d["first"]).sum())
    top_ = d.iloc[0]
    title = ("Every returner steps in on the second serve" if closer == len(d)
             else f"{closer} of {len(d)} returners step in on the second serve")
    sub = (f"{closer} of {len(d)} returners stand closer on second serves. {top_.player} moves the most: from "
           f"{top_['first']:.2f} m behind the baseline to {top_['second']:.2f} m.")
    rh = 0.22
    H = 1.6 + rh * len(d) + 1.1
    fig, top = P.figure(7.4, H, title, sub, "Median returner position at serve contact over all of each player's "
                        "charted matches; players with at least 30 first-serve and 12 second-serve returns.")
    ax = fig.add_axes([2.3 / 7.4, 1.0 / H, 4.8 / 7.4, top - 0.3 / H - 1.0 / H])
    lo = min(-1.0, np.floor(d["second"].min())); hi = max(4.0, np.ceil(d["first"].max()))
    ax.set_xlim(lo - 0.2, hi + 0.2)
    ax.set_ylim(len(d) - 0.5, -0.5)
    for t in np.arange(lo, hi + 1):
        ax.axvline(t, color=P.AXIS if t == 0 else P.GRID, lw=0.75, zorder=0)
    ax.text(0.05, -0.55, "baseline", fontsize=7, color=P.MUTED, va="bottom")
    for i, row in d.iterrows():
        ax.plot([row["second"], row["first"]], [i, i], color=P.AXIS, lw=1.5, zorder=1, solid_capstyle="round")
        ax.plot([row["first"]], [i], "o", ms=6, color=P.ORD1, mec=P.SURFACE, mew=1.5, zorder=2)
        ax.plot([row["second"]], [i], "o", ms=6, color=P.ORD2, mec=P.SURFACE, mew=1.5, zorder=2)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{p}  ({m})" if m > 1 else p for p, m in zip(d.player, d.matches)], fontsize=8, color=P.INK)
    ax.set_xticks(np.arange(lo, hi + 1))
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("meters behind the baseline at serve contact (median); (n) = matches", fontsize=8, color=P.MUTED)
    P.legend(fig, 2.3 / 7.4, top + 0.12 / H, [("dot", P.ORD1, "1st serve"), ("dot", P.ORD2, "2nd serve")])
    P.save(fig, path)


def fig_rally(D, path):
    R = D["rally"]
    sv = R["server"].set_index(["gender", "bucket"]).server_win
    sub = (f"Mean rally length (serve = 1) is {R['means']['men']['machine']:.2f} shots for men and "
           f"{R['means']['women']['machine']:.2f} for women by machine, against {R['means']['men']['mcp']:.2f} and "
           f"{R['means']['women']['mcp']:.2f} from human charters on the same {R['n']:,} points; the machine count is "
           f"exact on {R['exact']:.1f}% of points and within one shot on {R['within1']:.1f}%. Servers win "
           f"{sv['men', '1–4 shots']:.0f}% (men) and {sv['women', '1–4 shots']:.0f}% (women) of points over within four "
           "shots; beyond that it is close to a coin flip.")
    fig, top = P.figure(10, 5.4, "Rally length", sub, "Charted matches; points aligned to MCP with the right server.")
    fig.text(0.3 / 10, top, "Share of points by rally length", fontsize=10, fontweight="bold", va="top")
    P.legend(fig, 0.3 / 10, top - 0.28 / 5.4, [("box", P.BLUE, "Machine"), ("box", P.DEEMPH, "Human (MCP)")])
    ax = fig.add_axes([0.75 / 10, 0.2, 5.6 / 10, top - 0.2 - 0.62 / 5.4])
    ymax = 10 * np.ceil(max(R["dist"].machine.max(), R["dist"].mcp.max()) / 10)
    P.value_axis(ax, list(np.arange(0, ymax + 1, 10)), lambda v: f"{v:.0f}%", "shots in the rally (0 = double fault)")
    P.columns(ax, list(R["dist"].shots), [("machine", P.BLUE), ("mcp", P.DEEMPH)],
              {"machine": list(R["dist"].machine), "mcp": list(R["dist"].mcp)})
    x2 = 7.0 / 10
    fig.text(x2 - 0.04, top, "Server's win rate by rally length", fontsize=10, fontweight="bold", va="top")
    P.legend(fig, x2 - 0.04, top - 0.28 / 5.4, [("box", P.BLUE, "Men"), ("box", P.ORANGE, "Women")])
    ax2 = fig.add_axes([x2, 0.2, 0.27, top - 0.2 - 0.62 / 5.4])
    cats = ["1–4 shots", "5–8 shots", "9+ shots"]
    P.value_axis(ax2, [0, 20, 40, 60, 80], lambda v: f"{v:.0f}%")
    P.columns(ax2, cats, [("men", P.BLUE), ("women", P.ORANGE)],
              {g: [sv[g, c] for c in cats] for g in ("men", "women")}, labels=lambda v: f"{v:.0f}", ref=50)
    P.save(fig, path)


def fig_forehand(D, path):
    F = D["forehand"]
    cross = None
    for a, b in zip(F.itertuples(), list(F.itertuples())[1:]):
        if a.rally is not None and b.rally is not None and np.isfinite(a.rally) and np.isfinite(b.rally) \
                and a.rally < 50 <= b.rally:
            cross = a.x + (50 - a.rally) / (b.rally - a.rally) * (b.x - a.x)
            break

    def avg(k, lo, hi):
        g = F[(F.x > lo) & (F.x < hi) & F[k].notna()]
        return float((g[k] * g[f"n_{k}"]).sum() / g[f"n_{k}"].sum())

    where = "near the centre line" if cross is None else \
        f"{abs(cross):.1f} m into the {'backhand' if cross < 0 else 'forehand'} side"
    sub = (f"From the fourth shot on, groundstrokes turn mostly into forehands {where}: players run around the "
           "backhand. Returns go the other way. A deuce-court returner waits about 3.5 m out, so a serve down the T "
           f"reaches the backhand while they are still on the forehand side: 1.5–3 m to that side only "
           f"{avg('returns', 1.5, 3):.0f}% of returns are forehands, against {avg('rally', 1.5, 3):.0f}% of rally shots. "
           "The ad court mirrors it.")
    fig, top = P.figure(8.6, 5.3, "How far players run around the backhand", sub,
                        "Decoded matches, groundstrokes and slices, forehand frame (left-handers mirrored); 0.5 m bins "
                        "with at least 25 shots.")
    P.legend(fig, 0.3 / 8.6, top, [("line", P.BLUE, "Rally shots (4th shot on)"), ("line", P.DEEMPH, "Returns")])
    ax = fig.add_axes([0.75 / 8.6, 0.22, 0.88, top - 0.22 - 0.3 / 5.3])
    ax.set_xlim(-5.1, 5.1)
    P.value_axis(ax, [0, 25, 50, 75, 100], lambda v: f"{v}%", "meters from the centre line at contact")
    ax.axvline(0, color=P.AXIS, lw=0.75)
    for k, color in (("rally", P.BLUE), ("returns", P.DEEMPH)):
        g = F[F[k].notna()]
        ax.plot(g.x, g[k], color=color, lw=1.5, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.text(-5.05, -16, "← backhand side", fontsize=8, color=P.MUTED, ha="left", clip_on=False)
    ax.text(5.05, -16, "forehand side →", fontsize=8, color=P.MUTED, ha="right", clip_on=False)
    ax.set_xticks(range(-5, 6))
    P.save(fig, path)


def fig_era(D, path):
    M = D["pm"]
    groups = [(2001, 2010), (2011, 2020), (2021, 2025)]
    med = [(lo, hi, len(g), float(g.depth_median.median())) for lo, hi in groups
           for g in [M[(M.year >= lo) & (M.year <= hi)]]]
    sub = ("Median depth during rallies, one dot per match: " + "; ".join(
        f"{n} matches from {lo}–{str(hi)[2:]} at {m:.2f} m" for lo, hi, n, m in med) +
        ". Net play is rare throughout. The early years rest on a handful of finals, so read this as a hint, not a trend.")
    fig, top = P.figure(10, 4.9, "Twenty-five years of court position", sub,
                        "All processed matches, rally time from the classifier. Filled dots: decoded matches; rings: positions only.")
    P.legend(fig, 0.3 / 10, top, [("dot", P.BLUE, "Men"), ("dot", P.ORANGE, "Women")])
    for k, (col, ymax, fmt, title) in enumerate([
            ("depth_median", 2.5, lambda v: f"{v:.1f} m", "Median depth behind the baseline"),
            ("net_share", 6, lambda v: f"{v:.0f}%", "Share of rally time inside the service line")]):
        x0 = 0.75 / 10 + k * 0.49
        fig.text(x0 - 0.045, top - 0.33 / 4.9, title, fontsize=10, fontweight="bold", va="top")
        ax = fig.add_axes([x0, 0.2, 0.41, top - 0.2 - 0.62 / 4.9])
        ax.set_xlim(2000, 2026)
        ticks = list(np.linspace(0, ymax, 6 if ymax == 2.5 else 4))
        P.value_axis(ax, ticks, fmt)
        ax.set_xticks([2001, 2005, 2010, 2015, 2020, 2025])
        ax.spines["bottom"].set_color(P.AXIS)
        for _, g in M.groupby(["year", "gender"]):
            for i, (_, m) in enumerate(g.sort_values(col).iterrows()):
                x = m.year + (-0.22 if m.gender == "men" else 0.22) + (i - (len(g) - 1) / 2) * min(0.06, 0.3 / len(g))
                c = P.BLUE if m.gender == "men" else P.ORANGE
                ax.plot([x], [min(m[col], ymax)], "o", ms=5.5, color=c if m.tier != "tracked" else P.SURFACE,
                        mec=c if m.tier == "tracked" else P.SURFACE, mew=1.3, zorder=3, clip_on=False)
    P.save(fig, path)


def fig_distance(D, path):
    Ds = D["distance"]
    by = Ds["by_rally"]
    last = by.iloc[-1]
    sub = (f"About the same. The player who lost the point covered more ground in {Ds['loser_more']:.1f}% of "
           f"{Ds['n']:,} rallies of three shots or more. Distance grows with rally length to about "
           f"{(last.winner + last.loser) / 2:.0f} m each in rallies of {last.rally} shots; players average "
           f"{Ds['per_sec']:.2f} m/s while the ball is in play.")
    fig, top = P.figure(7.4, 4.6, "Who runs more, the winner or the loser of the point?", sub,
                        "Charted matches; median meters covered from the in-play serve to the last contact, smoothed tracks at 4 fps.")
    P.legend(fig, 0.3 / 7.4, top, [("box", P.BLUE, "Point winner"), ("box", P.DEEMPH, "Point loser")])
    ax = fig.add_axes([0.8 / 7.4, 0.2, 0.85, top - 0.2 - 0.3 / 4.6])
    ymax = 10 * np.ceil(by[["winner", "loser"]].max().max() / 10)
    P.value_axis(ax, list(np.arange(0, ymax + 1, 10)), lambda v: f"{v:.0f} m", "shots in the rally")
    P.columns(ax, list(by.rally), [("winner", P.BLUE), ("loser", P.DEEMPH)],
              {"winner": list(by.winner), "loser": list(by.loser)})
    P.save(fig, path)


def fig_players(D, path):
    pl = D["players"]
    groups = [(("Men" if g == "men" else "Women") + f" · {sum(p['gender'] == g for p in pl)} players",
               [p for p in pl if p["gender"] == g]) for g in ("men", "women")]
    P.mini_grid(path, pl, "Player fingerprints",
                "Where each player struck their rally shots, over all their charted matches, ordered from closest to the "
                "baseline to deepest. Each court is scaled to its own peak, so compare shapes, not darkness.",
                "Hitter frame (net at the top, the player's right on the right; left-handers not mirrored, marked L). "
                "Players with at least 150 rally shots.",
                label=lambda p: p["player"] + (" (L)" if p["hand"] == "L" else ""),
                sub=lambda p: f"{p['depth']:.1f} m deep · {p['matches']} match{'es' if p['matches'] > 1 else ''}",
                groups=groups)


def fig_matches(D, path):
    pm = D["pm"]
    items = [dict(**r, panel=D["minis"][r["video_id"]]) for r in pm.to_dict("records")]
    groups = [(("Men" if g == "men" else "Women") + f" · {sum(i['gender'] == g for i in items)} matches",
               [i for i in items if i["gender"] == g]) for g in ("men", "women")]
    P.mini_grid(path, items, "Match fingerprints",
                "Both players' positions during rally time in every processed match, folded into one half, oldest first. "
                "Each court is scaled to its own peak.",
                "Rally time from the classifier for every match. Labels: year, round, players; depth is the median "
                "meters behind the baseline; pos. = positions only (not decoded yet).",
                label=lambda m: " ".join(m["label"].split(" ")[2:]),
                sub=lambda m: f"{m['year']} {m['round']} · {m['depth_median']:.1f} m" + ("" if m["tier"] != "tracked" else " · pos."),
                groups=groups)


FIGURES = [("01_stroke_heatmaps.png", fig_contacts), ("02_rally_positions.png", fig_occupancy),
           ("03_pressure_map.png", fig_winmap), ("04_serve_standoff.png", fig_serve),
           ("05_return_depth.png", fig_dumbbell), ("06_rally_length.png", fig_rally),
           ("07_forehand_share.png", fig_forehand), ("08_era.png", fig_era), ("09_distance_run.png", fig_distance),
           ("10_player_fingerprints.png", fig_players), ("11_match_fingerprints.png", fig_matches)]


def write_tables(D):
    t = dict(per_match=D["pm"], rallytime_eval=D["rt_eval"], rally_length_distribution=D["rally"]["dist"],
             server_win_by_rally=D["rally"]["server"], return_depth_by_returner=D["dumbbell"],
             forehand_share_by_x=D["forehand"], win_by_depth=pd.DataFrame(D["winmap"]["bands"]),
             distance_by_rally=D["distance"]["by_rally"], serve_setup=D["serve"]["table"],
             players=pd.DataFrame([{k: v for k, v in p.items() if k != "panel"} for p in D["players"]]))
    for k in ("forehand", "backhand"):
        t[f"zones_{k}"] = D["contacts"]["table"][k]
    for k in ("men", "women"):
        t[f"zones_rally_time_{k}"] = D["occupancy"]["table"][k]
    for name, df in t.items():
        df.to_csv(REPORT / f"{name}.csv", index=False)
    return list(t)


def write_readme(D, tables):
    K, c = D["kpi"], D["calib"]
    lines = [
        "# Court analytics", "",
        f"{K['matches']} US Open quarterfinals, semifinals and finals from broadcast video ({K['years']}): "
        f"{K['charted']} decoded and aligned to the Match Charting Project, {K['uncharted']} decoded without an MCP record, "
        f"{K['tracked']} with player positions only. {K['rally_shots']:,} rally shots and {K['serves']:,} serves located; "
        f"{K['rally_hours']} h of rally time tracked. {K['unprocessed']} of the {K['archived']} archived videos have not "
        f"been processed yet. Generated {D['generated']} by `scripts/analytics.py`.", "",
    ]
    for name, _ in FIGURES:
        lines += [f"![{name}]({name})", ""]
    lines += [
        "## Caveats", "",
        "* Rally length from the machine is exact on about 63% of points and within one shot on 87% (held-out "
        "evaluation), and runs low on long rallies. Forehand/backhand is about 91% accurate; slices, volleys and touch "
        "shots are weak, so they are not broken out.",
        "* Positions are each player's ground point from the tracker, in court meters: where the player stood, not where "
        "the ball was.",
        f"* Depth is calibrated at the baseline (servers read {c['server'].get('near', float('nan')):.2f} m near, "
        f"{c['server'].get('far', float('nan')):.2f} m far) but the ends disagree behind it: first-serve returners read "
        f"{c['returner'].get('near', float('nan')):.2f} m deep at the near end and {c['returner'].get('far', float('nan')):.2f} m "
        "at the far end, and the near player leaves the frame beyond about 4.5 m. Players change ends every other game, "
        "so comparisons between groups are like for like; treat absolute depths as roughly ±0.4 m.",
        "* Point winners come from MCP through the point alignment; everything else is machine-made.", "",
        "Tables: " + ", ".join(f"`{t}.csv`" for t in tables) + ".", "",
        "Regenerate: `cd charting && uv run python scripts/analytics.py [--sync] [--refit]`.",
    ]
    (REPORT / "README.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true", help="download missing match tables from the Space first")
    ap.add_argument("--refit", action="store_true", help="retrain the rally-time model")
    a = ap.parse_args()
    if a.sync:
        print(f"[analytics] {sync_from_space()} files fetched from the Space", flush=True)
    D = build(a.refit)
    REPORT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES:
        fn(D, REPORT / name)
    tables = write_tables(D)
    write_readme(D, tables)
    print(f"[analytics] {len(FIGURES)} images and {len(tables)} tables in {REPORT}")


if __name__ == "__main__":
    main()
