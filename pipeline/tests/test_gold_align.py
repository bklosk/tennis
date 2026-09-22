import numpy as np
import pandas as pd

from tennis_pipeline import align, gold, score
from tennis_pipeline.score import Rules, State


def test_image_side_to_stroke_truth_table():
    f = gold.image_side_to_stroke
    assert f("right", "near", "R") == "forehand"  # back to camera: image right is the player's right
    assert f("left", "near", "R") == "backhand"
    assert f("right", "far", "R") == "backhand"  # facing the camera: image right is the player's left
    assert f("left", "far", "R") == "forehand"
    assert f("left", "near", "L") == "forehand"
    assert f("right", "far", "L") == "forehand"


def votes(side, side_m, pt=None, pt_m=None, two="no"):
    return {"side": side, "side_mirror": side_m, "point": pt, "point_mirror": pt_m, "two_hands": two}


def test_decide_requires_mirror_consistency():
    # A model that always says "left" never flips under mirroring: abstain.
    assert gold.decide(votes("left", "left", "left", "left"), "near", "R")[0] is None
    assert gold.decide(votes("right", "left"), "near", "R") == ("forehand", "medium", "ok")
    assert gold.decide(votes("right", "left", "right", "left"), "near", "R") == ("forehand", "high", "ok")
    assert gold.decide(votes("right", "left", "left", "right"), "near", "R")[2] == "votes_disagree"
    assert gold.decide(votes("right", "left", two="yes"), "near", "R")[2] == "two_hands_on_forehand"
    assert gold.decide(votes("left", "right", two="yes"), "near", "R")[0] == "backhand"


def test_point_side_parsing():
    assert gold._point_side('{"racket": [812, 300], "chest": [500, 420]}') == "right"
    assert gold._point_side('```json\n{"racket":[120,90],"torso":[256,260]}\n```') == "left"
    assert gold._point_side('{"racket": [500, 300], "chest": [505, 420]}') is None
    assert gold._point_side("no idea") is None


def test_plan_quotas_spreads_over_eras_sides_and_hands():
    rows = []
    for era, n in (("2000-06", 300), ("2013-26", 300), ("2007-12", 30)):
        for k in range(n):
            rows.append({"era": era, "side": "near" if k % 2 else "far", "hand": "L" if k % 5 == 0 else "R"})
    pool = pd.DataFrame(rows)
    q = gold.plan_quotas(pool, 149)
    assert sum(q.values()) == 149
    by_era = {e: sum(v for k, v in q.items() if k[0] == e) for e in ("2000-06", "2007-12", "2013-26")}
    assert by_era["2007-12"] <= 30 and by_era["2000-06"] >= 49
    near = sum(v for k, v in q.items() if k[1] == "near")
    lefty = sum(v for k, v in q.items() if k[2] == "L")
    assert near / 149 > 0.5 and lefty / 149 >= 0.25
    avail = pool.groupby(["era", "side", "hand"]).size().to_dict()
    assert all(v <= avail.get(k, 0) for k, v in q.items())


def ocr_table(first_server=2, n=40, seed=0):
    rules = Rules(3, "tb7")
    rng = np.random.default_rng(seed)
    s, t, snaps = State(), 0.0, []
    for _ in range(n):
        s = score.win_point(s, int(rng.integers(1, 3)), rules)
        t += 30
        snaps.append({"state": s, "first_seen": t, "last_seen": t + 5, "n_reads": 5})
    pts = score.points_from_states([{"state": State(), "first_seen": 0.0, "last_seen": 5.0, "n_reads": 5}] + snaps,
                                   rules, "ocr-x")
    pts["ends_parity"] = align._ends_parity(pts)
    for c in ("RallyCount", "ServeWidth", "ServeDepth", "ReturnDepth"):
        pts[c] = np.nan
    true_server = pts.server_p1_first if first_server == 1 else 3 - pts.server_p1_first
    return pts, true_server.to_numpy()


def video_points_for(pts, true_server, p1_start="far"):
    other = {"near": "far", "far": "near"}
    vps, sides = [], []
    for j, p in pts.iterrows():
        p1_side = p1_start if p.ends_parity == 0 else other[p1_start]
        side = p1_side if true_server[j] == 1 else other[p1_side]
        sides.append(side)
        # The rally ends a few seconds before the new score appears; a few serves are missed.
        vps.append({"vp": j, "t_start": p.video_t_hi - 14, "t_end": p.video_t_hi - 6,
                    "server_side": None if j % 5 == 0 else side, "n_serves": 1, "n_shots": 3, "hit_ids": []})
    return vps, np.array(sides)


def test_ocr_alignment_without_hints_gets_ends_right_and_flags_names():
    pts, true_server = ocr_table(first_server=2)
    vps, sides = video_points_for(pts, true_server)
    _, pairs, exp_side, _, _, ambiguous = align.best_alignment(vps, pts, "ocr")
    assert ambiguous
    assert (exp_side == sides).all()
    assert sum(i == j for i, j in pairs) >= 0.95 * len(pts)


def test_ocr_alignment_with_speed_graphic_server_names():
    pts, true_server = ocr_table(first_server=2)
    vps, _ = video_points_for(pts, true_server)
    pts["server_hint"] = np.nan
    pts.loc[pts.index[[3, 11, 20, 27]], "server_hint"] = true_server[[3, 11, 20, 27]]
    _, _, _, start, table, ambiguous = align.best_alignment(vps, pts, "ocr")
    assert not ambiguous and start == "far"
    assert (table.PointServer.to_numpy() == true_server).all()


def test_speed_graphic_server_name():
    from tennis_pipeline import ocr
    from tennis_pipeline.ocr import Word
    assert ocr.speed_server([Word("SAMPRAS 128 MPH", 0, 0, 10, 10)], "Andre Agassi", "Pete Sampras") == 2
    assert ocr.speed_server([Word("128 MPH", 0, 0, 10, 10)], "Andre Agassi", "Pete Sampras") is None


def test_sackmann_old_schema(tmp_path, monkeypatch):
    matches = pd.DataFrame({"match_id": ["2011-usopen-1701"], "player1": ["Novak Djokovic"],
                            "player2": ["Rafael Nadal"]})
    points = pd.DataFrame({
        "match_id": ["2011-usopen-1701"] * 3, "ElapsedTime": ["00:00:00", "0:00:00", "0:00:31"],
        "SetNo": [1, 1, 1], "GameNo": [1, 1, 1], "PointNumber": [0, 1, 2], "PointWinner": [0, 1, 2],
        "PointServer": [0, 1, 1], "Speed_KMH": [0, 190, 185], "Rally": [0, 7, 3], "P1Score": [0, 15, 15],
        "P2Score": [0, 0, 15]})
    files = {"matches": tmp_path / "m.csv", "points": tmp_path / "p.csv"}
    matches.to_csv(files["matches"], index=False)
    points.to_csv(files["points"], index=False)
    monkeypatch.setattr(align, "_fetch", lambda rel: files["matches" if rel.endswith("matches.csv") else "points"])
    monkeypatch.setattr(align, "video_match", lambda vid: {"year": "2011", "tournament": "US Open",
                                                           "player1": "Novak Djokovic", "player2": "Rafael Nadal"})
    meta, pts = align.sackmann_points("x")
    assert meta["source"] == "official" and len(pts) == 2
    assert pts.RallyCount.tolist() == [7, 3] and pts.ServeNumber.isna().all()
