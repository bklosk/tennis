"""Benchmark TypeSafe Jev (images + Box) against the Decisions API stages and RF-DETR on one charted match.

Jev gets exactly what gpt-6-luna got: the same onset contact sheets and questions (`uso.onset_api`),
the same stroke contact sheets and questions (`uso.strokes`). Box questions find both players (and,
as an experiment, the ball) on full frames at each detected contact, scored against the RF-DETR
track boxes. Results go to outputs/VIDEO_ID/jev/ and never overwrite the luna outputs.

    uv run python scripts/jev_bench.py probe            # a few requests: measured tokens, projected cost
    uv run python scripts/jev_bench.py onsets strokes imgside box
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np
import pandas as pd

from uso import crops as crops_mod
from uso import onset_api, strokes
from uso.jev import USD_PER_INPUT_TOKEN, BudgetExceeded, JevClient, Request, box, choice_probs, image_ref, noul, \
    questions_from_decisions
from uso.paths import match_dir
from uso.video import frames_at, video_path

VIDEO = "8gP591_wx_k"
BOX_W, BOX_H = 1568, 882  # 1920x1080 scaled to Jev's 1568 px maximum side
CTX_MIN = 280  # Jev's minimum image side; the 384x216 stroke context frame is upscaled to 498x280
PROBE_ABORT_USD = 1.00


def out_dir(vid: str):
    d = match_dir(vid) / "jev"
    d.mkdir(exist_ok=True)
    return d


# ------------------------------------------------------------------ onset labels


def onset_rows(vid: str) -> pd.DataFrame:
    """The onsets luna labelled (strong pass plus the soft pass), so both models see the same set."""
    d = match_dir(vid)
    c = pd.read_parquet(d / "candidates.parquet")
    oids = pd.read_parquet(d / "onset_api.parquet").oid
    on = c.loc[oids, ["seg", "t"]].copy()
    on["oid"] = on.index
    return on


def onset_requests(vid: str, on: pd.DataFrame, trk: pd.DataFrame) -> list[Request]:
    imgs = onset_api.build_images(vid, on, trk)
    qs = questions_from_decisions(onset_api.QUESTIONS)
    reqs = []
    for o in on.oid:
        ref = image_ref(imgs[o])
        reqs.append(Request(state={"description": onset_api.TEXT, "image": ref}, questions=qs, images={ref: imgs[o]}))
    return reqs


def onset_parse(oid, raw) -> dict:
    row = {"oid": oid, "tokens": (raw.get("usage") or {}).get("input_tokens")}
    for who in ("near", "far"):
        pr = choice_probs(raw, f"{who}_action")
        for act in ("serve", "hit", "bounce", "none"):
            row[f"{who}_{act}"] = pr.get(act, np.nan)
    return row


def run_onsets(vid: str, cl: JevClient, limit: int | None = None) -> pd.DataFrame:
    d = match_dir(vid)
    trk = pd.read_parquet(d / "tracks.parquet")
    on = onset_rows(vid)
    if limit:
        on = on.iloc[:limit]
    rows = []
    for i in range(0, len(on), 200):
        part = on.iloc[i:i + 200]
        ans = cl.ask_many(onset_requests(vid, part, trk), label=f"onsets:{vid}")
        rows += [onset_parse(o, a) for o, a in zip(part.oid, ans) if a is not None]
        print(f"[onsets] {min(i + 200, len(on))}/{len(on)} spent ${cl.spent():.4f}", flush=True)
    res = pd.DataFrame(rows)
    if not limit:
        res.to_parquet(out_dir(vid) / "onset_api.parquet")
    return res


# ------------------------------------------------------------------ strokes


def rally_contacts(vid: str) -> pd.DataFrame:
    shots = pd.read_parquet(match_dir(vid) / "shots.parquet")
    rc = shots[shots.kind == "rally"].copy()
    rc["hand"] = rc.hitter_hand.fillna("U")
    return rc


def _context_jpeg(ctx: np.ndarray) -> bytes:
    h, w = ctx.shape[:2]
    s = max(1.0, CTX_MIN / min(h, w))
    if s > 1:
        ctx = cv2.resize(ctx, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_CUBIC)
    return crops_mod.save_jpeg(ctx, quality=80)


def stroke_requests(vid: str, rc: pd.DataFrame, trk: pd.DataFrame, kind: str) -> tuple[list, list[Request]]:
    ent = crops_mod.hit_crops(vid, rc[["hit_id", "seg", "t", "half"]], trk)
    ids, reqs = [], []
    for h in rc.itertuples():
        e = ent.get(h.hit_id)
        if e is None:
            continue
        if kind == "strokes":
            luna = strokes.request_for(e, h.half, h.hand, getattr(h, "dist_net", None))
            if luna is None:
                continue
            sheet = luna.images[0]
            ims = {image_ref(sheet): sheet}
            state = {"description": luna.text, "image_1": image_ref(sheet)}
            if len(luna.images) > 1:
                ctx = _context_jpeg(e["context"])
                ims[image_ref(ctx)] = ctx
                state["image_2"] = image_ref(ctx)
            qs = questions_from_decisions(luna.questions)
        else:
            luna = strokes.request_image_side(e, h.half)
            if luna is None:
                continue
            sheet = luna.images[0]
            ims = {image_ref(sheet): sheet}
            state = {"description": luna.text, "image": image_ref(sheet)}
            qs = questions_from_decisions(luna.questions)
        ids.append((h.hit_id, h.half, h.hitter_hand if isinstance(h.hitter_hand, str) else "R"))
        reqs.append(Request(state=state, questions=qs, images=ims))
    return ids, reqs


def stroke_parse(kind: str, ident, raw) -> dict:
    hid, half, hand = ident
    if kind == "strokes":
        side = choice_probs(raw, "stroke_side")
        fam = choice_probs(raw, "stroke_family")
        row = dict(hit_id=hid, api_fh=side.get("forehand", np.nan), api_bh=side.get("backhand", np.nan),
                   api_unclear=side.get("unclear", np.nan), api_two_hands=noul(raw, "two_hands"),
                   api_tokens=(raw.get("usage") or {}).get("input_tokens"))
        for f in strokes.FAMILIES:
            row[f"api_fam_{f}"] = fam.get(f, np.nan)
        return row
    pr = choice_probs(raw, "racket_image_side")
    return dict(hit_id=hid, p_fh_img=strokes.side_from_image(pr.get("image_left", np.nan), pr.get("image_right", np.nan),
                                                             half, hand),
                p_overhead_img=pr.get("overhead_or_unclear", np.nan), two_hands=noul(raw, "two_hands"),
                tokens=(raw.get("usage") or {}).get("input_tokens"))


def run_strokes(vid: str, cl: JevClient, kind: str, limit: int | None = None) -> pd.DataFrame:
    trk = pd.read_parquet(match_dir(vid) / "tracks.parquet")
    rc = rally_contacts(vid)
    if limit:
        rc = rc.iloc[:limit]
    ids, reqs = stroke_requests(vid, rc, trk, kind)
    ans = cl.ask_many(reqs, label=f"{kind}:{vid}")
    res = pd.DataFrame([stroke_parse(kind, i, a) for i, a in zip(ids, ans) if a is not None])
    if not limit:
        res.to_parquet(out_dir(vid) / ("api_strokes.parquet" if kind == "strokes" else "api_imgside.parquet"))
    print(f"[{kind}] {len(res)} answered, spent ${cl.spent():.4f}", flush=True)
    return res


# ------------------------------------------------------------------ Box


BOX_TEXT = "Tennis broadcast frame from the main camera, which looks down the court from behind the near baseline."
BOX_QS = {
    "near_player": {"type": "box", "instructions": "the tennis player at the near end of the court in `frame` (the "
                    "end closest to the camera; they have their back to the camera). Include their whole body and racket."},
    "far_player": {"type": "box", "instructions": "the tennis player at the far end of the court in `frame` (beyond "
                   "the net, facing the camera). Include their whole body and racket."},
    "ball": {"type": "box", "instructions": "the tennis ball in `frame`"},
    "ball_visible": {"type": "noul", "instructions": "Is the tennis ball itself visible in `frame`?"},
}
# Second pass: the person only, matching what RF-DETR's person boxes cover (the first pass asked for
# body and racket, which widens the box and shifts its centre toward the racket side).
BOX_QS_PERSON = {
    "near_player": {"type": "box", "instructions": "the tennis player at the near end of the court in `frame` (the "
                    "end closest to the camera; they have their back to the camera). Box the person only, head to "
                    "feet, not the racket."},
    "far_player": {"type": "box", "instructions": "the tennis player at the far end of the court in `frame` (beyond "
                   "the net, facing the camera). Box the person only, head to feet, not the racket."},
}
BOX_VARIANTS = {"box": (BOX_QS, "box.parquet"), "box_person": (BOX_QS_PERSON, "box_person.parquet")}


def box_frames(vid: str) -> pd.DataFrame:
    """One frame per detected contact (serves and rally shots), snapped to the nearest RF-DETR sample
    time where both players are tracked, so the reference boxes are detections, not interpolations."""
    d = match_dir(vid)
    shots = pd.read_parquet(d / "shots.parquet")
    trk = pd.read_parquet(d / "tracks.parquet")
    both = trk.groupby(["seg", "t"]).half.nunique()
    both = both[both == 2].reset_index()[["seg", "t"]]
    rows = []
    for s in shots.itertuples():
        g = both[both.seg == s.seg]
        if g.empty:
            continue
        j = int(np.abs(g.t.to_numpy() - s.t).argmin())
        ts = float(g.t.iloc[j])
        if abs(ts - s.t) > 0.13:
            continue
        rows.append(dict(hit_id=s.hit_id, seg=s.seg, t_contact=s.t, t=ts, kind=s.kind, hitter_end=s.hitter_end))
    return pd.DataFrame(rows).drop_duplicates(["seg", "t"]).reset_index(drop=True)


def box_requests(vid: str, fr: pd.DataFrame, qs: dict = BOX_QS) -> list[Request]:
    frames = frames_at(video_path(vid), list(fr.t))
    fdir = out_dir(vid) / "box_frames"
    fdir.mkdir(exist_ok=True)
    reqs = []
    for r in fr.itertuples():
        _, f = frames[r.t]
        im = cv2.resize(f, (BOX_W, BOX_H), interpolation=cv2.INTER_AREA)
        jpg = crops_mod.save_jpeg(im, quality=90)
        (fdir / f"{r.seg}_{r.t:.2f}.jpg").write_bytes(jpg)
        ref = image_ref(jpg)
        reqs.append(Request(state={"description": BOX_TEXT, "frame": ref}, questions=qs, images={ref: jpg}))
    return reqs


def box_parse(r, raw) -> dict:
    row = dict(seg=r.seg, t=r.t, hit_id=r.hit_id, kind=r.kind, hitter_end=r.hitter_end,
               tokens=(raw.get("usage") or {}).get("input_tokens"), ball_visible=noul(raw, "ball_visible"))
    for name in ("near_player", "far_player", "ball"):
        b = box(raw, name) or {}
        for k in ("left", "top", "right", "bottom", "confidence"):
            row[f"{name}_{k}"] = b.get(k, np.nan)
    return row


def run_box(vid: str, cl: JevClient, limit: int | None = None, variant: str = "box") -> pd.DataFrame:
    qs, out_name = BOX_VARIANTS[variant]
    fr = box_frames(vid)
    if limit:
        fr = fr.iloc[:limit]
    rows = []
    for i in range(0, len(fr), 100):
        part = fr.iloc[i:i + 100]
        ans = cl.ask_many(box_requests(vid, part, qs), label=f"{variant}:{vid}")
        rows += [box_parse(r, a) for r, a in zip(part.itertuples(), ans) if a is not None]
        print(f"[{variant}] {min(i + 100, len(fr))}/{len(fr)} spent ${cl.spent():.4f}", flush=True)
    res = pd.DataFrame(rows)
    if not limit:
        res.to_parquet(out_dir(vid) / out_name)
    return res


# ------------------------------------------------------------------ probe


def probe(vid: str, cl: JevClient, n: int = 3) -> dict:
    """A few requests of each kind: measured tokens per request and the projected cost of the full run."""
    counts = dict(onsets=len(onset_rows(vid)), strokes=len(rally_contacts(vid)), imgside=len(rally_contacts(vid)),
                  box=len(box_frames(vid)))
    toks = {}
    toks["onsets"] = run_onsets(vid, cl, limit=n).tokens.mean()
    toks["strokes"] = run_strokes(vid, cl, "strokes", limit=n).api_tokens.mean()
    toks["imgside"] = run_strokes(vid, cl, "imgside", limit=n).tokens.mean()
    toks["box"] = run_box(vid, cl, limit=n).tokens.mean()
    proj = {k: counts[k] * toks[k] * USD_PER_INPUT_TOKEN for k in counts}
    out = dict(counts=counts, tokens_per_request={k: round(float(v)) for k, v in toks.items()},
               projected_usd={k: round(v, 4) for k, v in proj.items()}, projected_total=round(sum(proj.values()), 4),
               spent=round(cl.spent(), 5))
    (out_dir(vid) / "probe.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["probe", "onsets", "strokes", "imgside", "box", "box_person"])
    ap.add_argument("--video", default=VIDEO)
    a = ap.parse_args()
    cl = JevClient()
    try:
        for st in a.steps:
            if st == "probe":
                if probe(a.video, cl)["projected_total"] > PROBE_ABORT_USD:
                    print(f"projected cost is over ${PROBE_ABORT_USD:.2f}; stopping", file=sys.stderr)
                    sys.exit(3)
            elif st == "onsets":
                run_onsets(a.video, cl)
            elif st in ("strokes", "imgside"):
                run_strokes(a.video, cl, st)
            elif st in BOX_VARIANTS:
                run_box(a.video, cl, variant=st)
    except BudgetExceeded as e:
        print(f"budget: {e}", file=sys.stderr)
        sys.exit(3)
    print(json.dumps(dict(spent_usd=round(cl.spent(), 4), **{k: v for k, v in cl.tally().items() if k != "offset"})))


if __name__ == "__main__":
    main()
