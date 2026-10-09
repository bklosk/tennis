"""Ask the Decisions API what each player is doing at an audio onset.

Audio alone cannot count contacts: pre-serve ball bounces are as loud as hits, and a single hit
often brings several onsets (grunt, squeak, bounce). For each candidate onset the API sees both
players' crops at -0.1 s, 0 and +0.1 s and answers, per player, whether they are serving,
hitting, bouncing the ball, or doing none of these at time 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uso.crops import _box_at, cut, save_jpeg, square
from uso.decisions import DecisionClient, Request, choice_probs
from uso.paths import match_dir
from uso.video import frames_at, video_path

OFFS = (-0.1, 0.0, 0.1)
SIZE = 160

ACTIONS = [
    {"value": "serve", "description": "Serving: overhead service swing from behind the baseline after a ball toss, striking the ball around the middle frame."},
    {"value": "hit", "description": "Swinging at and striking the ball around the middle frame: groundstroke, return, volley, overhead or lob."},
    {"value": "bounce", "description": "Bouncing the ball on the ground (pre-serve routine), catching it, or tapping it to a ball kid."},
    {"value": "none", "description": "Not striking the ball at the middle frame: waiting, moving, split step, recovering, or follow-through of an earlier stroke."},
]


def _question(name: str, who: str) -> dict:
    return {"type": "choice", "name": name,
            "instructions": f"What is the {who} player (row {'1, top' if who == 'near' else '2, bottom'}) doing at the middle frame?",
            "choices": ACTIONS}


QUESTIONS = [_question("near_action", "near"), _question("far_action", "far")]
TEXT = ("Tennis broadcast. A sharp sound was heard at time 0. Row 1 (top) shows the NEAR player (back to the "
        "camera) at -0.1 s, 0 s and +0.1 s; row 2 (bottom) shows the FAR player (facing the camera) at the same "
        "three moments. Each crop is centred on that player.")


def build_images(video_id: str, onsets: pd.DataFrame, trk: pd.DataFrame) -> dict:
    """onsets: columns oid, seg, t. Returns {oid: jpeg bytes}."""
    want = [t + o for t in onsets.t for o in OFFS]
    frames = frames_at(video_path(video_id), want)
    out = {}
    for r in onsets.itertuples():
        rows = []
        for half in ("near", "far"):
            tiles = []
            for o in OFFS:
                ft, f = frames[r.t + o]
                box = _box_at(trk, r.seg, half, ft)
                if box is None:
                    tiles.append(np.zeros((SIZE, SIZE, 3), np.uint8))
                    continue
                x0, y0, side = square(box, f.shape[1], f.shape[0], scale=1.6)
                tiles.append(cut(f, x0, y0, side, SIZE))
            rows.append(np.hstack(tiles))
        out[r.oid] = save_jpeg(np.vstack(rows), quality=85)
    return out


def label(video_id: str, onsets: pd.DataFrame, trk: pd.DataFrame, client: DecisionClient | None = None,
          chunk: int = 200) -> pd.DataFrame:
    """API action probabilities per onset. Cached per request, so re-runs are free."""
    client = client or DecisionClient()
    rows = []
    for i in range(0, len(onsets), chunk):
        part = onsets.iloc[i:i + chunk]
        imgs = build_images(video_id, part, trk)
        reqs = [Request(text=TEXT, images=[imgs[o]], questions=QUESTIONS) for o in part.oid]
        ans = client.ask_many(reqs, label=f"onsets:{video_id}", workers=16)
        for oid, a in zip(part.oid, ans):
            if a is None:
                continue
            row = {"oid": oid, "tokens": (a.get("_usage") or {}).get("input_tokens")}
            for who in ("near", "far"):
                pr = choice_probs(a.get(f"{who}_action"))
                for act in ("serve", "hit", "bounce", "none"):
                    row[f"{who}_{act}"] = pr.get(act, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def run(video_id: str, z_min: float = 10.0, client: DecisionClient | None = None, max_onsets: int | None = None,
        dry_run: bool = False) -> pd.DataFrame:
    """Label every candidate onset (inside live segments, z >= z_min). Cached as onset_api.parquet."""
    d = match_dir(video_id)
    out = d / "onset_api.parquet"
    c = pd.read_parquet(d / "candidates.parquet")
    trk = pd.read_parquet(d / "tracks.parquet")
    on = c[c.z >= z_min][["seg", "t"]].copy()
    on["oid"] = on.index
    if max_onsets:
        on = on.iloc[:max_onsets]
    if out.exists():
        prev = pd.read_parquet(out)
        on = on[~on.oid.isin(prev.oid)]
    else:
        prev = pd.DataFrame()
    client = client or DecisionClient()
    client.check_estimate(len(on), 700, label=f"onset labels {video_id}")
    if dry_run or on.empty:
        return prev
    lab = label(video_id, on, trk, client)
    res = pd.concat([prev, lab], ignore_index=True)
    res.to_parquet(out)
    return res


def run_soft(video_id: str, z_lo: float = 5.0, z_hi: float = 8.0, after: float = 3.0, p_ctx: float = 0.4,
             client: DecisionClient | None = None) -> pd.DataFrame:
    """Second pass: soft onsets (volleys, touch shots) inside rally context, i.e. within `after`
    seconds after an already-labelled likely contact. Appends to onset_api.parquet."""
    d = match_dir(video_id)
    out = d / "onset_api.parquet"
    c = pd.read_parquet(d / "candidates.parquet")
    trk = pd.read_parquet(d / "tracks.parquet")
    lab = pd.read_parquet(out)
    L = lab.set_index("oid")
    pc = (L[["near_hit", "near_serve"]].sum(axis=1).clip(upper=1)).combine(L[["far_hit", "far_serve"]].sum(axis=1).clip(upper=1), max)
    ctx_oid = pc[pc >= p_ctx].index
    ctx_t = np.sort(c.loc[ctx_oid, "t"].to_numpy())
    soft = c[(c.z >= z_lo) & (c.z < z_hi) & ~c.index.isin(L.index)]
    i = np.searchsorted(ctx_t, soft.t.to_numpy()) - 1
    near_ctx = (i >= 0) & (soft.t.to_numpy() - ctx_t[np.clip(i, 0, None)] <= after)
    on = soft[near_ctx][["seg", "t"]].copy()
    on["oid"] = on.index
    client = client or DecisionClient()
    client.check_estimate(len(on), 700, label=f"soft onsets {video_id}")
    if on.empty:
        return lab
    new = label(video_id, on, trk, client)
    res = pd.concat([lab, new], ignore_index=True)
    res.to_parquet(out)
    return res


# ---------------------------------------------------------------------------------------------
# v2: one player per request, five frames, explicit "missed swing" and "ball handling" classes

OFFS2 = (-0.15, -0.075, 0.0, 0.075, 0.15)
SIZE2 = 176
ACTIONS2 = [
    {"value": "serve_contact", "description": "Serving: the racket strikes the tossed ball overhead from behind the baseline."},
    {"value": "stroke_contact", "description": "A rally stroke or return in which the racket strikes the ball around frame 3 (groundstroke, volley, overhead, lob, drop shot)."},
    {"value": "swing_miss", "description": "Swings or lunges at a ball but does not touch it (e.g. reaching for an ace or a passing shot)."},
    {"value": "ball_handling", "description": "Casual ball handling: bouncing it before serving, catching it, or tapping/pushing it away after the point is over."},
    {"value": "no_stroke", "description": "No stroke near frame 3: waiting, moving, split step, recovering, or the follow-through of an earlier stroke."},
]


def _q2(end: str) -> list[dict]:
    return [{"type": "choice", "name": "action", "instructions": "What is this player doing around frame 3?",
             "choices": ACTIONS2},
            {"type": "predicate", "name": "racket_ball_contact",
             "instructions": "The player's racket strikes the ball between frames 2 and 4."}]


def text2(end: str) -> str:
    who = "the near player, back to the camera" if end == "near" else "the far player, facing the camera"
    return (f"Tennis broadcast. Five frames (numbered 1-5, left to right) of one player ({who}), 0.075 s apart; "
            f"a sharp sound was heard at frame 3. During rallies players hit about once a second; after a point "
            f"ends they may bounce, catch or casually tap the ball.")


def build_images2(video_id: str, onsets: pd.DataFrame, trk: pd.DataFrame) -> dict:
    """onsets: oid, seg, t, half. Returns {(oid, half): jpeg}."""
    want = sorted({t + o for t in onsets.t for o in OFFS2})
    frames = frames_at(video_path(video_id), want)
    out = {}
    for r in onsets.itertuples():
        tiles = []
        for o in OFFS2:
            ft, f = frames[r.t + o]
            box = _box_at(trk, r.seg, r.half, ft)
            if box is None:
                tiles.append(np.zeros((SIZE2, SIZE2, 3), np.uint8))
                continue
            x0, y0, side = square(box, f.shape[1], f.shape[0], scale=1.6)
            tiles.append(cut(f, x0, y0, side, SIZE2))
        out[(r.oid, r.half)] = save_jpeg(np.hstack(tiles), quality=85)
    return out


def label2(video_id: str, onsets: pd.DataFrame, trk: pd.DataFrame, client: DecisionClient | None = None,
           chunk: int = 200) -> pd.DataFrame:
    client = client or DecisionClient()
    rows = []
    for i in range(0, len(onsets), chunk):
        part = onsets.iloc[i:i + chunk]
        imgs = build_images2(video_id, part, trk)
        reqs = [Request(text=text2(h), images=[imgs[(o, h)]], questions=_q2(h)) for o, h in zip(part.oid, part.half)]
        ans = client.ask_many(reqs, label=f"onsets2:{video_id}")
        for (o, h), a in zip(zip(part.oid, part.half), ans):
            if a is None:
                continue
            pr = choice_probs(a.get("action"))
            row = {"oid": o, "half": h, "tokens2": (a.get("_usage") or {}).get("input_tokens"),
                   "p_contact": (a.get("racket_ball_contact") or {}).get("probability", np.nan)}
            for act in ACTIONS2:
                row[f"p_{act['value']}"] = pr.get(act["value"], np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def run2(video_id: str, p_min: float = 0.15, client: DecisionClient | None = None,
         oids: list | None = None) -> pd.DataFrame:
    """v2 labels for (onset, player) pairs the first pass found at all plausible (hit+serve >= p_min)."""
    d = match_dir(video_id)
    out = d / "onset_api2.parquet"
    c = pd.read_parquet(d / "candidates.parquet")
    trk = pd.read_parquet(d / "tracks.parquet")
    lab = pd.read_parquet(d / "onset_api.parquet").set_index("oid")
    rows = []
    for who in ("near", "far"):
        pl = lab[f"{who}_hit"].fillna(0) + lab[f"{who}_serve"].fillna(0)
        for o in pl.index[pl >= p_min]:
            rows.append((o, who))
    pairs = pd.DataFrame(rows, columns=["oid", "half"])
    if oids is not None:
        pairs = pairs[pairs.oid.isin(oids)]
    pairs = pairs.join(c[["seg", "t"]], on="oid")
    prev = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["oid", "half"])
    if len(prev):
        done = set(zip(prev.oid, prev.half))
        pairs = pairs[[(o, h) not in done for o, h in zip(pairs.oid, pairs.half)]]
    client = client or DecisionClient()
    client.check_estimate(len(pairs), 600, label=f"onsets v2 {video_id}")
    if pairs.empty:
        return prev
    new = label2(video_id, pairs, trk, client)
    res = pd.concat([prev, new], ignore_index=True)
    res.to_parquet(out)
    return res
