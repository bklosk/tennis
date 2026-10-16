"""Score the Jev benchmark against gpt-6-luna, RF-DETR and MCP on one match.

Reads outputs/VIDEO_ID/jev/*.parquet (from scripts/jev_bench.py) next to the luna outputs, and
writes reports/jev/metrics.json, plus spot-check images in outputs/VIDEO_ID/jev/. The luna outputs are not changed.

    uv run python scripts/jev_score.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from uso import hitmodel, rally, report, scene, shots as shots_mod, strokes, truth
from uso import labeler as labeler_mod
from uso import points as points_mod
from uso.jev import LEDGER as JEV_LEDGER
from uso.paths import OUTPUTS, PROJECT, match_dir
from uso.people import _to_court

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_bench import BOX_H, BOX_W, VIDEO  # noqa: E402

REPORT = PROJECT / "reports" / "jev"
FULL_W, FULL_H = 1920, 1080
LUNA_USD_PER_TOKEN = 0.10 / 1_000_000


def _auc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s, float)
    ok = ~np.isnan(s)
    return float(roc_auc_score(y[ok], s[ok])) if len(set(y[ok])) == 2 else float("nan")


def params() -> dict:
    return json.loads((OUTPUTS / "decoder_params.json").read_text())


# ------------------------------------------------------------------ rally length


def decode(vid: str, lab: pd.DataFrame, p: dict, learned: bool, with_pairs: bool = False, labeler: str = "luna"):
    """Production decoding (shots.build) without writing anything. `labeler` picks the contact model."""
    p = {**rally.DECODER, **p}
    if not learned:
        p["learned_w"] = 0.0
    c = rally.with_api(rally.features(vid), lab, p)
    c = shots_mod.blend_learned(vid, c, p, labeler)
    sv = rally.serves(c, p)
    vp, _ = points_mod.assemble(c, sv, w_prior=p["w_prior"], phantom=p.get("phantom"))
    pairs, info = points_mod.evaluate(vp, truth.mcp_points(vid))
    out = report.rally_metrics(pairs)
    out.update(video_points=len(vp), mcp_points=len(truth.mcp_points(vid)), aligned=len(pairs),
               coverage=info["mcp_coverage"], server_agree=info["server_agree"], serves_agree=info["n_serves_agree"])
    out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
    return (out, pairs) if with_pairs else out


def paired(vid: str, luna: pd.DataFrame, jev: pd.DataFrame, setups: dict | None = None) -> dict:
    """Rally length on the MCP points both decodes aligned with the right server. setups: name ->
    (params, learned, labeler); default both API-only with luna's decoder parameters."""
    setups = setups or {"luna": (params(), False, "luna"), "jev": (params(), False, "luna")}
    res = {}
    for name, lab in (("luna", luna), ("jev", jev)):
        p, learned, lb = setups[name]
        _, pr = decode(vid, lab, p, learned=learned, with_pairs=True, labeler=lb)
        ok = pr[pr.server_ok & pr.m_rally.notna()]
        res[name] = (ok.v_rally - ok.m_rally.astype(float)).set_axis(ok.m_Pt.to_numpy())
    common = res["luna"].index.intersection(res["jev"].index)
    lu, je = res["luna"].loc[common], res["jev"].loc[common]
    return dict(common_points=len(common), luna_exact=round(float((lu == 0).mean()), 4),
                jev_exact=round(float((je == 0).mean()), 4), only_jev_exact=int(((je == 0) & (lu != 0)).sum()),
                only_luna_exact=int(((lu == 0) & (je != 0)).sum()), luna_within1=round(float((lu.abs() <= 1).mean()), 4),
                jev_within1=round(float((je.abs() <= 1).mean()), 4), luna_off_by_2plus=int((lu.abs() >= 2).sum()),
                jev_off_by_2plus=int((je.abs() >= 2).sum()))


def rally_section(vid: str, luna: pd.DataFrame, jev: pd.DataFrame) -> dict:
    p = params()
    out = {"luna_production (API + learned contact model)": decode(vid, luna, p, learned=True),
           "luna_api_only": decode(vid, luna, p, learned=False),
           "jev_api_only": decode(vid, jev, p, learned=False),
           "jev + learned contact model (trained on luna answers)": decode(vid, jev, p, learned=True)}
    # Decoder thresholds were tuned on dev matches with luna's answers. Retune bias and the serve
    # threshold on this match for each model: an optimistic ceiling, but the same one for both.
    sweep = {}
    for name, lab in (("luna", luna), ("jev", jev)):
        best = None
        for bias in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0):
            for sp in (0.2, 0.3, 0.4, 0.5, 0.6):
                r = decode(vid, lab, {**p, "bias": bias, "serve_p": sp}, learned=False)
                key = (r["rally_exact"] * r["coverage"], r["rally_within1"])
                if best is None or key > best[0]:
                    best = (key, dict(bias=bias, serve_p=sp, **r))
        sweep[name] = best[1]
    out["retuned_on_this_match_api_only"] = sweep
    out["paired_api_only"] = paired(vid, luna, jev)
    if labeler_mod.params_path("jev").exists():  # Jev's own decoder parameters and contact model, tuned on dev
        pj = labeler_mod.params("jev")
        out["jev_tuned_api_only"] = decode(vid, jev, pj, learned=False)
        out["jev_tuned + jev contact model (production jev)"] = decode(vid, jev, pj, learned=True, labeler="jev")
        out["paired_production"] = paired(vid, luna, jev, {"luna": (params(), True, "luna"), "jev": (pj, True, "jev")})
        out["jev_params"] = pj
    return out


def onset_section(vid: str, luna: pd.DataFrame, jev: pd.DataFrame) -> dict:
    """Per-onset contact labels from MCP-constrained decoding inside aligned points (as the learned
    contact model is trained). The decoder picks which onsets carry MCP's contact count using the
    mean of both models' scores, so neither model's own answers define its truth."""
    d = match_dir(vid)
    p = {**rally.DECODER, **params()}
    feats = rally.features(vid)
    cl = rally.with_api(feats, luna, p)
    cj = rally.with_api(feats, jev, p)
    both = luna.oid[luna.oid.isin(jev.oid)]
    scores = pd.DataFrame({k: (cl[k] + cj[k]) / 2 for k in ("hit_n", "hit_f")})
    labels = hitmodel.make_labels(cl, pd.read_parquet(d / "aligned_points.parquet"),
                                  pd.read_parquet(d / "video_points.parquet"), pd.read_parquet(d / "shots.parquet"),
                                  scores)
    labels = labels[labels.onset.isin(both)]
    out = {"labelled_pairs": len(labels), "positives": int(labels.y.sum())}
    for name, lab in (("luna", luna), ("jev", jev)):
        L = lab.set_index("oid")
        res = {}
        for who in ("near", "far", "all"):
            g = labels if who == "all" else labels[labels.hitter == who]
            pc = [min(1.0, L.at[o, f"{h}_hit"] + L.at[o, f"{h}_serve"]) for o, h in zip(g.onset, g.hitter)]
            res[f"contact_auc_{who}"] = round(_auc(g.y, pc), 4)
        out[name] = res
    # how often the two agree on each player's top action
    agree = {}
    L, J = luna.set_index("oid").loc[both], jev.set_index("oid").loc[both]
    for who in ("near", "far"):
        cols = [f"{who}_{a}" for a in ("serve", "hit", "bounce", "none")]
        agree[who] = round(float((L[cols].to_numpy().argmax(1) == J[cols].to_numpy().argmax(1)).mean()), 4)
    out["top_action_agreement"] = agree
    out["jev_answered"] = int(jev.oid.isin(luna.oid).sum())
    out["luna_answered"] = len(luna)
    return out


# ------------------------------------------------------------------ strokes


def stroke_section(vid: str, luna_s, jev_s, luna_i, jev_i) -> dict:
    d = match_dir(vid)
    sh = pd.read_parquet(d / "shots.parquet")
    rc = sh[(sh.kind == "rally") & sh.mcp_side.isin(["F", "B"])][["hit_id", "half", "mcp_side", "mcp_family"]]
    prod = pd.read_csv(d / "shots.csv")  # same rows, same order as shots.parquet (report.export)
    assert len(prod) == len(sh) and np.allclose(prod.t_video_s, sh.t)
    prod["hit_id"] = sh.hit_id.to_numpy()
    out = {"n_contacts_with_mcp_side": len(rc)}
    g = rc.merge(prod[["hit_id", "stroke_side"]], on="hit_id", how="left")
    out["pose_model_production_side_acc"] = round(float((g.stroke_side == g.mcp_side).mean()), 4)
    for name, s, i in (("luna", luna_s, luna_i), ("jev", jev_s, jev_i)):
        g = rc.merge(s, on="hit_id", how="inner").merge(i[["hit_id", "p_fh_img"]], on="hit_id", how="left")
        y = (g.mcp_side == "F").astype(int)
        pf = g.api_fh / (g.api_fh + g.api_bh)
        res = dict(n=len(g), side_acc=round(float(((pf > 0.5).astype(int) == y).mean()), 4), side_auc=round(_auc(y, pf), 4),
                   unclear_mean=round(float(g.api_unclear.mean()), 4),
                   imgside_acc=round(float(((g.p_fh_img > 0.5).astype(int) == y)[g.p_fh_img.notna()].mean()), 4),
                   imgside_auc=round(_auc(y, g.p_fh_img), 4))
        for half in ("near", "far"):
            h = g.half == half
            res[f"side_acc_{half}"] = round(float(((pf[h] > 0.5).astype(int) == y[h]).mean()), 4)
        # stroke family: the classes MCP has on this match
        fam = g.mcp_family.map(strokes.mcp_family)
        fcols = [f"api_fam_{f}" for f in strokes.FAMILIES]
        pred = g[fcols].to_numpy().argmax(1)
        pred = pd.Series([strokes.FAMILIES[k] for k in pred], index=g.index).replace({"half_volley": "volley"})
        m = fam.notna()
        res["family_acc"] = round(float((pred[m] == fam[m]).mean()), 4)
        res["family_confusion_mcp_to_pred"] = pd.crosstab(fam[m], pred[m]).to_dict(orient="index")
        res["slice_auc"] = round(_auc((fam[m] == "slice").astype(int), g.loc[m, "api_fam_slice"]), 4)
        # two hands vs MCP backhand (all players here hit two-handed backhands)
        res["two_hands_auc_vs_backhand"] = round(_auc((g.mcp_side == "B").astype(int), g.api_two_hands), 4)
        out[name] = res
    return out


# ------------------------------------------------------------------ Box


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def box_table(vid: str, bx: pd.DataFrame) -> pd.DataFrame:
    """One row per (frame, player): Jev box and RF-DETR track box in full-res px, IoU, court positions."""
    d = match_dir(vid)
    trk = pd.read_parquet(d / "tracks.parquet").set_index(["seg", "t", "half"])
    sc = pd.read_parquet(d / "scene.parquet")
    rows = []
    for r in bx.itertuples():
        H = scene.homography_at(sc, r.t)
        for half, name in (("near", "near_player"), ("far", "far_player")):
            ref = trk.loc[(r.seg, r.t, half)]
            jb = np.array([getattr(r, f"{name}_left") * FULL_W, getattr(r, f"{name}_top") * FULL_H,
                           getattr(r, f"{name}_right") * FULL_W, getattr(r, f"{name}_bottom") * FULL_H])
            rb = ref[["x1", "y1", "x2", "y2"]].to_numpy(float)
            jg = _to_court(H, np.array([[(jb[0] + jb[2]) / 2, jb[3]]]))[0]
            rows.append(dict(seg=r.seg, t=r.t, hit_id=r.hit_id, kind=r.kind, hitter_end=r.hitter_end, half=half,
                             iou=_iou(jb, rb), conf=getattr(r, f"{name}_confidence"),
                             jx=jg[0], jy=jg[1], rx=ref.cx, ry=ref.cy,
                             jh=jb[3] - jb[1], rh=rb[3] - rb[1],
                             foot_px=float(np.hypot((jb[0] + jb[2]) / 2 - (rb[0] + rb[2]) / 2, jb[3] - rb[3])),
                             **{f"j{k}": v for k, v in zip(("x1", "y1", "x2", "y2"), jb)},
                             **{f"r{k}": v for k, v in zip(("x1", "y1", "x2", "y2"), rb)}))
    t = pd.DataFrame(rows)
    t["court_err_m"] = np.hypot(t.jx - t.rx, t.jy - t.ry)
    return t


def position_checks_with(vid: str, bt: pd.DataFrame, source: str) -> dict:
    """report.position_checks with hitter/opponent positions taken from one source at the snapped frame."""
    sh = pd.read_parquet(match_dir(vid) / "shots.parquet")
    xcol, ycol = ("jx", "jy") if source == "jev" else ("rx", "ry")
    pos = bt.set_index(["hit_id", "half"])[[xcol, ycol]]
    s = sh[sh.hit_id.isin(bt.hit_id)].copy()
    opp = np.where(s.hitter_end == "near", "far", "near")
    s["hx"] = [pos.at[(h, e), xcol] for h, e in zip(s.hit_id, s.hitter_end)]
    s["hy"] = [pos.at[(h, e), ycol] for h, e in zip(s.hit_id, s.hitter_end)]
    s["ox"] = [pos.at[(h, e), xcol] for h, e in zip(s.hit_id, opp)]
    s["oy"] = [pos.at[(h, e), ycol] for h, e in zip(s.hit_id, opp)]
    r = report.position_checks(vid, s)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}


def box_section(vid: str, bx: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    bt = box_table(vid, bx)
    out = {"frames": len(bx)}
    for half in ("near", "far"):
        g = bt[bt.half == half]
        out[half] = dict(iou_median=round(float(g.iou.median()), 3), iou_ge_0_5=round(float((g.iou >= 0.5).mean()), 3),
                         iou_ge_0_7=round(float((g.iou >= 0.7).mean()), 3), wrong_person=round(float((g.iou == 0).mean()), 3),
                         foot_px_median=round(float(g.foot_px.median()), 1),
                         court_err_m_median=round(float(g.court_err_m.median()), 3),
                         court_err_m_p90=round(float(g.court_err_m.quantile(0.9)), 3),
                         height_ratio_median=round(float((g.jh / g.rh).median()), 3),
                         conf_median=round(float(g.conf.median()), 3),
                         conf_vs_iou_spearman=round(float(g.conf.corr(g.iou, method="spearman")), 3))
    prod = report.match_summary(vid)
    out["position_checks"] = {
        "production (RF-DETR, interpolated at contact)": {
            k: v for k, v in prod.items()
            if k in ("server_behind_baseline", "serve_wide_vs_T_auc", "shot_dir_1_vs_3_auc", "net_shot_auc")},
        "rfdetr_at_snapped_frame": position_checks_with(vid, bt, "rfdetr"),
        "jev_box_at_snapped_frame": position_checks_with(vid, bt, "jev")}
    return out, bt


def ball_section(vid: str, bx: pd.DataFrame, bt: pd.DataFrame) -> dict:
    """No ball ground truth. Weak automatic check on frames within 1.5 video frames of a detected
    contact: the ball should be at the hitter, i.e. inside the hitter's RF-DETR box grown by half."""
    b = bx.copy()
    b["bw"] = (b.ball_right - b.ball_left) * FULL_W
    b["bh"] = (b.ball_bottom - b.ball_top) * FULL_H
    b["bx"] = (b.ball_left + b.ball_right) / 2 * FULL_W
    b["by"] = (b.ball_top + b.ball_bottom) / 2 * FULL_H
    hit = bt[bt.half == bt.hitter_end].set_index("hit_id")
    shots = pd.read_parquet(match_dir(vid) / "shots.parquet").set_index("hit_id")
    b["dt"] = (b.t - b.hit_id.map(shots.t)).abs()
    near_contact = b[b.dt <= 0.05]
    inside = []
    for r in near_contact.itertuples():
        h = hit.loc[r.hit_id]
        w, hh = h.rx2 - h.rx1, h.ry2 - h.ry1
        inside.append(h.rx1 - 0.5 * w <= r.bx <= h.rx2 + 0.5 * w and h.ry1 - 0.5 * hh <= r.by <= h.ry2 + 0.25 * hh)
    return dict(frames=len(b), ball_visible_median=round(float(b.ball_visible.median()), 3),
                ball_box_px_median=[round(float(b.bw.median()), 1), round(float(b.bh.median()), 1)],
                ball_conf_median=round(float(b.ball_confidence.median()), 3),
                frames_within_0_05s_of_contact=len(near_contact),
                ball_at_hitter_rate=round(float(np.mean(inside)), 3) if inside else float("nan"))


# ------------------------------------------------------------------ images


def spot_sheets(vid: str, bx: pd.DataFrame, bt: pd.DataFrame, n: int = 24):
    """Box overlays (green Jev, magenta RF-DETR) and zoomed ball crops for eyeballing. They are
    broadcast frames, so they stay in outputs/ (gitignored) and are not redistributed."""
    fdir = match_dir(vid) / "jev" / "box_frames"
    dst = match_dir(vid) / "jev"
    rng = np.random.default_rng(0)
    pick = bx.iloc[np.sort(rng.choice(len(bx), size=min(n, len(bx)), replace=False))]
    tiles, balls = [], []
    for r in pick.itertuples():
        im = cv2.imread(str(fdir / f"{r.seg}_{r.t:.2f}.jpg"))
        s = BOX_W / FULL_W
        for row in bt[(bt.seg == r.seg) & (bt.t == r.t)].itertuples():
            cv2.rectangle(im, (int(row.rx1 * s), int(row.ry1 * s)), (int(row.rx2 * s), int(row.ry2 * s)), (255, 0, 255), 2)
            cv2.rectangle(im, (int(row.jx1 * s), int(row.jy1 * s)), (int(row.jx2 * s), int(row.jy2 * s)), (0, 220, 0), 2)
        x1, y1 = int(r.ball_left * BOX_W), int(r.ball_top * BOX_H)
        x2, y2 = int(r.ball_right * BOX_W), int(r.ball_bottom * BOX_H)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(im, f"t={r.t:.2f} {r.kind} ball_visible={r.ball_visible:.2f}", (12, 34), 0, 1.0, (0, 255, 255), 2,
                    cv2.LINE_AA)
        tiles.append(cv2.resize(im, (784, 441), interpolation=cv2.INTER_AREA))
        cx, cy, half = (x1 + x2) // 2, (y1 + y2) // 2, 80
        pad = cv2.copyMakeBorder(im, half, half, half, half, cv2.BORDER_CONSTANT)
        crop = pad[cy:cy + 2 * half, cx:cx + 2 * half]
        balls.append(cv2.resize(crop, (240, 240), interpolation=cv2.INTER_NEAREST))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles) - len(tiles) % 3, 3)]
    cv2.imwrite(str(dst / "box_overlays.jpg"), np.vstack(rows[:6]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    brow = [np.hstack(balls[i:i + 6]) for i in range(0, len(balls) - len(balls) % 6, 6)]
    cv2.imwrite(str(dst / "ball_crops.jpg"), np.vstack(brow), [cv2.IMWRITE_JPEG_QUALITY, 90])


# ------------------------------------------------------------------ cost


def cost_section(vid: str) -> dict:
    out = {}
    if JEV_LEDGER.exists():
        led = pd.read_json(JEV_LEDGER, lines=True)
        led["kind"] = led.label.str.split(":").str[0]
        g = led.groupby("kind").agg(requests=("usd", "size"), tokens_per_request=("input_tokens", "mean"), usd=("usd", "sum"))
        out["jev"] = {k: {kk: round(float(vv), 5) for kk, vv in v.items()} for k, v in g.to_dict("index").items()}
        out["jev_total_usd"] = round(float(led.usd.sum()), 4)
    luna = {}
    d = match_dir(vid)
    for kind, f, col in (("onsets", "onset_api.parquet", "tokens"), ("strokes", "api_strokes.parquet", "api_tokens")):
        x = pd.read_parquet(d / f)
        luna[kind] = dict(requests=len(x), tokens_per_request=round(float(x[col].mean()), 1),
                          usd=round(float(x[col].sum() * LUNA_USD_PER_TOKEN), 5))
    n = len(pd.read_parquet(d / "api_imgside.parquet"))
    luna["imgside"] = dict(requests=n, tokens_per_request=639, usd=round(n * 639 * LUNA_USD_PER_TOKEN, 5))
    out["luna_same_requests"] = luna
    return out


def main():
    vid = VIDEO
    d = match_dir(vid)
    j = d / "jev"
    luna_on, luna_s, luna_i = (pd.read_parquet(d / f) for f in ("onset_api.parquet", "api_strokes.parquet", "api_imgside.parquet"))
    have = lambda f: (j / f).exists()  # noqa: E731
    out = {"video_id": vid, "match": "2023 US Open women's QF, Ostapenko v Gauff (held-out test match)"}
    jev_path = labeler_mod.labels_path(vid, "jev")  # production route (Jev's own soft pass), else the benchmark set
    if not jev_path.exists():
        jev_path = j / "onset_api.parquet"
    if jev_path.exists():
        jev_on = pd.read_parquet(jev_path)
        out["jev_onset_labels"] = str(jev_path.relative_to(OUTPUTS))
        out["onsets"] = onset_section(vid, luna_on, jev_on)
        out["rally_length"] = rally_section(vid, luna_on, jev_on)
    else:  # luna only: checks that decoding reproduces the production numbers
        out["rally_length"] = {"luna_production": decode(vid, luna_on, params(), learned=True),
                               "luna_api_only": decode(vid, luna_on, params(), learned=False)}
    if have("api_strokes.parquet") and have("api_imgside.parquet"):
        out["strokes"] = stroke_section(vid, luna_s, pd.read_parquet(j / "api_strokes.parquet"), luna_i,
                                        pd.read_parquet(j / "api_imgside.parquet"))
    if have("box.parquet"):
        bx = pd.read_parquet(j / "box.parquet")
        out["box"], bt = box_section(vid, bx)
        out["ball"] = ball_section(vid, bx, bt)
        bt.to_parquet(j / "box_table.parquet")
        spot_sheets(vid, bx, bt)
    if have("box_person.parquet"):
        bxp = pd.read_parquet(j / "box_person.parquet")
        out["box_person_only"], btp = box_section(vid, bxp)
        btp.to_parquet(j / "box_table_person.parquet")
    out["cost"] = cost_section(vid)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "metrics.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
