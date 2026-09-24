"""Stage 5: group hits into points and align them to official point-by-point data.

Official full-match uploads are edited (changeovers and dead time removed), so a single time
offset does not work. Points are aligned with dynamic programming using rally length,
which end the server is on (from the change-of-ends rule), and elapsed-time consistency.
"""
import csv
import json
import re
import urllib.error
import urllib.request
from functools import lru_cache

import numpy as np
import pandas as pd

from .court import HALF_LENGTH, SINGLES_HALF_WIDTH
from .paths import DATA, SACKMANN, SACKMANN_BASE, match_dir


def _fetch(rel: str):
    path = SACKMANN / rel.split("/")[-1]
    if not path.exists():
        SACKMANN.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{SACKMANN_BASE}/{rel}", path)
    return path


def _norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def video_match(video_id: str) -> dict:
    with open(DATA / "videos.csv") as fh:
        for row in csv.DictReader(fh):
            if row["video_id"] == video_id:
                return row
    raise KeyError(video_id)


SLAM_SLUG = {"US Open": "usopen", "Australian Open": "ausopen"}


def _ends_parity(pts: pd.DataFrame) -> np.ndarray:
    """Players change ends after odd games; parity of end changes before each point."""
    games_per_set = pts.groupby("SetNo").GameNo.max().to_dict()
    changes = []
    for r in pts.itertuples():
        before = sum(int(np.ceil(games_per_set[s] / 2)) for s in games_per_set if s < r.SetNo)
        changes.append(before + r.GameNo // 2)
    return np.array(changes, int) % 2


def sackmann_points(video_id: str) -> tuple[dict, pd.DataFrame]:
    meta = video_match(video_id)
    year, slug = meta["year"], SLAM_SLUG[meta["tournament"]]
    try:
        matches = pd.read_csv(_fetch(f"slam_pointbypoint/{year}-{slug}-matches.csv"))
    except urllib.error.HTTPError as e:
        raise KeyError(f"no official point-by-point file for {year} {slug}") from e
    names = {_norm(meta["player1"]), _norm(meta["player2"])}
    hit = matches[matches.apply(lambda r: {_norm(str(r.player1)), _norm(str(r.player2))} == names, axis=1)]
    if hit.empty:
        raise KeyError(f"no point-by-point match for {video_id}")
    m = hit.iloc[0]
    pts = pd.read_csv(_fetch(f"slam_pointbypoint/{year}-{slug}-points.csv"), low_memory=False)
    pts = pts[(pts.match_id == m.match_id) & ~pts.PointNumber.astype(str).isin(["0", "0X", "0Y"])
              & (pts.PointServer.astype(int) > 0)].copy()
    pts["elapsed_s"] = pts.ElapsedTime.map(lambda s: sum(int(x) * 60 ** i for i, x in enumerate(reversed(s.split(":")))))
    # Older files (2011-2017) have no RallyCount/ServeNumber/ServeWidth; Rally held the count then.
    if "RallyCount" not in pts or pts.RallyCount.isna().all():
        pts["RallyCount"] = pd.to_numeric(pts.get("Rally"), errors="coerce")
    for col in ("ServeNumber", "ServeWidth", "ServeDepth", "ReturnDepth", "Speed_KMH"):
        if col not in pts:
            pts[col] = np.nan
    pts = pts.reset_index(drop=True)
    pts["ends_parity"] = _ends_parity(pts)
    return {"match_id": m.match_id, "player1": m.player1, "player2": m.player2, "source": "official"}, pts


def ocr_points(video_id: str) -> tuple[dict, pd.DataFrame]:
    """Point table read from the broadcast's score graphics (see `ocr.py`)."""
    meta = video_match(video_id)
    path = match_dir(video_id) / "ocr_points.csv"
    if not path.exists():
        raise KeyError(f"no official data for {video_id}; run the ocr stage first")
    pts = pd.read_csv(path)
    if pts.empty:
        raise KeyError(f"ocr found no points for {video_id}")
    pts["PointServer"] = pts.server_p1_first
    for col in ("RallyCount", "ServeWidth", "ServeDepth", "ReturnDepth"):
        pts[col] = np.nan
    pts["ends_parity"] = _ends_parity(pts)
    return {"match_id": pts.match_id.iloc[0], "player1": meta["player1"], "player2": meta["player2"],
            "source": "ocr"}, pts


def official_points(video_id: str) -> tuple[dict, pd.DataFrame]:
    try:
        return sackmann_points(video_id)
    except KeyError:
        return ocr_points(video_id)


@lru_cache
def handedness() -> dict:
    hands = {}
    for rel in ("atp/atp_players.csv", "wta/wta_players.csv"):
        df = pd.read_csv(_fetch(rel), encoding="latin-1", low_memory=False)
        for r in df.itertuples():
            hands[_norm(f"{r.name_first}{r.name_last}")] = r.hand
    return hands


def video_points(hits: pd.DataFrame, impacts=None) -> list[dict]:
    """Group hits into serve attempts, then merge faults with the following second serve."""
    hits = hits.sort_values("t").reset_index(drop=True)
    attempts = []
    for h in hits.itertuples():
        segment = h.chunk_id.split("_")[0]  # 30 s chunks of one camera segment are continuous
        new = h.is_serve or not attempts or h.t - attempts[-1]["hits"][-1].t > 3.0 or segment != attempts[-1]["segment"]
        if new:
            attempts.append({"segment": segment, "hits": [h], "server_side": h.side if h.is_serve else None,
                             "serve_detected": bool(h.is_serve), "closed": False})
        elif not attempts[-1]["closed"]:
            attempts[-1]["hits"].append(h)
        # A rally shot that bounces outside the singles court ends the point; later hits are
        # balls knocked back to the ball kids.
        if not h.is_serve and _bounced_out(h):
            attempts[-1]["closed"] = True
    points = []
    for a in attempts:
        first = a["hits"][0]
        fault = a["serve_detected"] and len(a["hits"]) == 1 and first.serve_in is False
        prev = points[-1] if points else None
        if (prev and prev["pending_fault"] and a["server_side"] == prev["server_side"]
                and first.t - prev["t_last"] < 30):
            prev["attempts"].append(a)
            prev["pending_fault"] = fault
            prev["t_last"] = a["hits"][-1].t
            continue
        points.append({"attempts": [a], "server_side": a["server_side"], "t_start": first.t,
                       "t_last": a["hits"][-1].t, "pending_fault": fault})
    out = []
    for i, p in enumerate(points):
        last = p["attempts"][-1]
        n = len(last["hits"])
        if p["pending_fault"]:
            n = 0
        elif not last["serve_detected"]:
            n += 1  # serve itself was missed
        out.append({"vp": i, "t_start": p["t_start"], "t_end": p["t_last"], "server_side": p["server_side"],
                    "n_serves": len(p["attempts"]), "n_shots": n, "n_shots_audio": None,
                    "hit_ids": [h.hit_id for a in p["attempts"] for h in a["hits"]]})
    if impacts is not None:
        from . import audio

        for i, v in enumerate(out):
            nxt = out[i + 1]["t_start"] if i + 1 < len(out) else None
            # A short gap is usually a false serve splitting this rally, so keep listening.
            # A long gap is the next point; stop before its serve.
            if nxt is not None and nxt - v["t_end"] > audio.GAP_STOP:
                limit = nxt - 0.3
            else:
                limit = v["t_start"] + 12
            v["n_shots_audio"] = audio.shot_count(impacts, v["t_start"], limit)
    return out


def _bounced_out(h, tol: float = 0.4) -> bool:
    bx, by = getattr(h, "bounce_x_m", np.nan), getattr(h, "bounce_y_m", np.nan)
    if pd.isna(bx) or pd.isna(by):
        return False
    return abs(bx) > SINGLES_HALF_WIDTH + tol or abs(by) > HALF_LENGTH + tol


def _align(vps: list[dict], pts: pd.DataFrame, p1_start: str):
    n, m = len(vps), len(pts)
    tv = np.array([v["t_start"] for v in vps])
    tp = pts.elapsed_s.to_numpy(float)
    rally = pts.RallyCount.fillna(-1).to_numpy(int)
    server = pts.PointServer.to_numpy(int)
    parity = pts.ends_parity.to_numpy(int)
    other = {"near": "far", "far": "near"}
    p1_side = np.where(parity == 0, p1_start, other[p1_start])
    exp_side = np.where(server == 1, p1_side, [other[s] for s in p1_side])
    # OCR-derived points carry the video time at which the new score first appeared.
    t_hi = pts.video_t_hi.to_numpy(float) if "video_t_hi" in pts else np.full(m, np.nan)

    def s(i, j):
        v = vps[i]
        sc = 0.0
        if v["server_side"] is not None:
            sc += 1.5 if v["server_side"] == exp_side[j] else -2.5
        if rally[j] >= 0:
            sc += 1.5 - 0.6 * min(abs(v["n_shots"] - rally[j]), 5)
            # Court-mic impacts after the last tracked hit. Weaker than the visual
            # count (it also hears bounces) and only a tie-break beside it.
            na = v.get("n_shots_audio")
            if na is not None:
                sc += 0.35 * (1 - min(abs(na - rally[j]), 4) / 4)
        if not np.isnan(t_hi[j]):
            sc += 1.5 if t_hi[j] - 40 <= v["t_end"] <= t_hi[j] + 2 else -2.0
        return sc

    def trans(i0, j0, i1, j1):
        dv, dp = tv[i1] - tv[i0], tp[j1] - tp[j0]
        if dv <= 0:
            return -10.0
        if dv > dp + 8:
            return -3.0 - (dv - dp) / 20
        if dv >= dp - 8:
            return 1.0
        return -0.4  # the broadcast edit removed time here

    NEG = -1e9
    best = np.full((n, m), NEG)
    back = {}
    for i in range(n):
        for j in range(m):
            base = s(i, j)
            # Full-match uploads start at the first point, so skipping points at the start costs
            # the same as skipping them mid-match.
            cand = base - 0.6 * i - 1.0 * j
            arg = None
            for i0 in range(max(0, i - 5), i):
                for j0 in range(max(0, j - 8), j):
                    if best[i0, j0] == NEG:
                        continue
                    val = best[i0, j0] + base + trans(i0, j0, i, j) - 0.6 * (i - i0 - 1) - 1.0 * (j - j0 - 1)
                    if val > cand:
                        cand, arg = val, (i0, j0)
            best[i, j] = cand
            back[(i, j)] = arg
    i, j = np.unravel_index(np.argmax(best), best.shape)
    score = best[i, j]
    pairs = []
    cur = (int(i), int(j))
    while cur is not None:
        pairs.append(cur)
        cur = back[cur]
    return score, pairs[::-1], exp_side


def best_alignment(vps: list[dict], pts: pd.DataFrame, source: str):
    """Align under each unknown: which end player 1 starts at, and (OCR only) who served first.

    For OCR points the two unknowns are coupled: "player 1 starts far, player 2 serves first"
    puts the server at the same end on every point as "player 1 starts near and serves first",
    so video alone cannot tell them apart. A server named on the speed graphic settles it;
    otherwise the result is flagged ambiguous (near/far player names may be swapped).
    """
    ambiguous = False
    if source == "official":
        firsts = [None]
    else:
        hints = pts.get("server_hint", pd.Series(dtype=float)).dropna()
        if len(hints) >= 3:
            agree = float((hints == pts.loc[hints.index, "server_p1_first"]).mean())
            firsts = [1 if agree >= 0.5 else 2]
        else:
            firsts = [1, 2]
            ambiguous = True
    results = []
    for first in firsts:
        t = pts if first is None else pts.assign(
            PointServer=pts.server_p1_first if first == 1 else 3 - pts.server_p1_first)
        results += [(_align(vps, t, start), start, t) for start in ("near", "far")]
    (score, pairs, exp_side), p1_start, table = max(results, key=lambda r: r[0][0])
    return score, pairs, exp_side, p1_start, table, ambiguous


def _torso_hist(path: str) -> np.ndarray | None:
    import cv2

    strip = cv2.imread(path)
    if strip is None:
        return None
    w = strip.shape[1] // 3
    crop = strip[:, w:2 * w]
    h, cw = crop.shape[:2]
    torso = cv2.cvtColor(crop[int(0.33 * h):int(0.55 * h), int(0.42 * cw):int(0.58 * cw)], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([torso], [0, 1], None, [18, 4], [0, 180, 0, 256]).ravel()
    return hist / max(hist.sum(), 1)


def fix_identity(shots: pd.DataFrame, points: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Correct near/far -> player mapping per point using the hitters' clothing colour.

    Two colour clusters are named by majority vote of the official-data mapping; a point whose
    shots mostly contradict its mapping has its near/far players swapped.
    """
    from sklearn.cluster import KMeans

    s = shots[shots.point_number.notna() & shots.crop_path.notna()].copy()
    feats = [(_torso_hist(p) if isinstance(p, str) else None) for p in s.crop_path]
    keep = [f is not None for f in feats]
    s, feats = s[keep], [f for f in feats if f is not None]
    s["feat_idx"] = range(len(s))
    X = np.sqrt(np.stack(feats)) if feats else np.zeros((0, 72))
    # Cluster within each end separately: one end is often in shadow, which would otherwise
    # dominate the colour clusters.
    s["appearance_player"] = None
    purity = {}
    for side, g in s.groupby("side"):
        if len(g) < 20:
            continue
        labels = KMeans(2, n_init=10, random_state=0).fit_predict(X[g.feat_idx.to_numpy()])
        votes = pd.crosstab(labels, g.player)
        names = votes.idxmax(axis=1).to_dict()
        agree = float(votes.max(axis=1).sum() / votes.to_numpy().sum())
        purity[side] = agree
        if len(set(names.values())) == 2 and agree >= 0.7:
            s.loc[g.index, "appearance_player"] = [names[k] for k in labels]
    s = s[s.appearance_player.notna()]
    if s.empty:
        return points, {"identity_fixed_points": 0, "appearance_purity": purity}
    swapped = 0
    points = points.copy()
    for vp, g in s.groupby("vp"):
        agree = (g.appearance_player == g.player).sum()
        if len(g) >= 2 and agree < len(g) / 2 - 0.5:
            i = points.index[points.vp == vp][0]
            points.loc[i, ["near_player", "far_player"]] = points.loc[i, ["far_player", "near_player"]].to_numpy()
            swapped += 1
    return points, {"identity_fixed_points": swapped, "appearance_purity": purity}


def run(video_id: str) -> dict:
    out_dir = match_dir(video_id)
    hits = pd.read_parquet(out_dir / "hits_raw.parquet")
    if hits.empty:
        raise ValueError(f"{video_id}: no hits were detected, so there is nothing to align")
    feats_path = out_dir / "hit_features.parquet"
    meta, pts = official_points(video_id)
    from . import audio
    from .cli import video_path

    impacts = audio.onsets(video_id, video_path(video_id))
    vps = video_points(hits, impacts)
    _, pairs, exp_side, p1_start, pts, ambiguous = best_alignment(vps, pts, meta["source"])

    matched = {i: j for i, j in pairs}
    names = {1: meta["player1"], 2: meta["player2"]}
    hands = handedness()
    rows = []
    for v in vps:
        j = matched.get(v["vp"])
        rec = {k: v[k] for k in ("vp", "t_start", "t_end", "server_side", "n_serves", "n_shots", "n_shots_audio")}
        if j is not None:
            p = pts.iloc[j]
            server = int(p.PointServer)
            server_side = exp_side[j]
            rec.update({
                "point_number": int(p.PointNumber), "set": int(p.SetNo), "game": int(p.GameNo),
                "server": names[server], "returner": names[3 - server],
                "official_server_side": server_side,
                "near_player": names[server] if server_side == "near" else names[3 - server],
                "far_player": names[3 - server] if server_side == "near" else names[server],
                "official_rally_count": int(p.RallyCount) if not pd.isna(p.RallyCount) else None,
                "serve_number": int(p.ServeNumber) if not pd.isna(p.ServeNumber) else None,
                "serve_speed_kmh": float(p.Speed_KMH) if not pd.isna(p.Speed_KMH) and p.Speed_KMH else None,
                "points_source": meta["source"],
                "serve_width": p.ServeWidth, "serve_depth": p.ServeDepth, "return_depth": p.ReturnDepth,
                "point_winner": names.get(int(p.PointWinner)),
                "score_after": f"{p.P1Score}-{p.P2Score}", "elapsed_s": int(p.elapsed_s),
            })
        rows.append(rec)
    points = pd.DataFrame(rows)
    points.to_csv(out_dir / "points.csv", index=False)

    hit_point = {hid: v["vp"] for v in vps for hid in v["hit_ids"]}
    feats = pd.read_parquet(feats_path) if feats_path.exists() else None

    def attach(points_df):
        sh = hits.copy()
        sh["vp"] = sh.hit_id.map(hit_point)
        sh = sh.merge(points_df[["vp", "point_number", "near_player", "far_player", "server"]], on="vp", how="left")
        sh["player"] = np.where(sh.side == "near", sh.near_player, sh.far_player)
        sh["hand"] = sh.player.map(lambda n: hands.get(_norm(n)) if isinstance(n, str) else None)
        if feats is not None:
            sh = sh.merge(feats, on="hit_id", how="left")
        return sh

    shots = attach(points)
    identity = {}
    if feats is not None and "crop_path" in shots:
        points, identity = fix_identity(shots, points)
        points.to_csv(out_dir / "points.csv", index=False)
        shots = attach(points)
    shots.to_parquet(out_dir / "shots_aligned.parquet", index=False)

    aligned = points.point_number.notna()
    both = points[aligned & points.official_rally_count.notna()]
    served = points[aligned & points.server_side.notna()]
    summary = {
        "video_id": video_id, "official_match_id": meta["match_id"], "points_source": meta["source"],
        "server_identity_ambiguous": ambiguous,
        "official_points": int(len(pts)), "video_points": int(len(points)),
        "aligned_points": int(aligned.sum()), "p1_starts": p1_start,
        "serve_detected_rate": float(len(served) / max(aligned.sum(), 1)),
        "server_side_agreement": float((served.server_side == served.official_server_side).mean()) if len(served) else None,
        "rally_count_exact": float((both.n_shots == both.official_rally_count).mean()) if len(both) else None,
        "rally_count_within_1": float(((both.n_shots - both.official_rally_count).abs() <= 1).mean()) if len(both) else None,
        "rally_audio_exact": float((both.n_shots_audio == both.official_rally_count).mean()) if len(both) and both.n_shots_audio.notna().any() else None,
        "rally_audio_within_1": float(((both.n_shots_audio - both.official_rally_count).abs() <= 1).mean()) if len(both) and both.n_shots_audio.notna().any() else None,
        **identity,
    }
    (out_dir / "align_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
