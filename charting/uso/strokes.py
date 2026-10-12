"""Stroke side (forehand/backhand) and stroke family for rally contacts.

Two predictors, fused later on validation matches:
  * the Decisions API on a contact sheet (six hitter crops around contact) plus a context frame;
  * a local classifier on pose and court-position features.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from uso import crops as crops_mod
from uso.decisions import DecisionClient, Request, choice_probs
from uso.paths import match_dir

FAMILIES = ["groundstroke", "slice", "volley", "half_volley", "overhead", "lob", "drop_shot", "other"]

SIDE_Q = {
    "type": "choice", "name": "stroke_side",
    "instructions": ("Which side of the highlighted player's body is the racket on when it meets the ball "
                     "(frame 4)? Use the player's own left and right, and their stated playing hand."),
    "choices": [
        {"value": "forehand", "description": "Racket-hand side of the body (a right-hander's right side)."},
        {"value": "backhand", "description": "Non-racket-hand side, hit with one or two hands."},
        {"value": "unclear", "description": "Cannot be determined from these frames."},
    ],
}

FAMILY_Q = {
    "type": "choice", "name": "stroke_family",
    "instructions": "What kind of shot is the highlighted player hitting?",
    "choices": [
        {"value": "groundstroke", "description": "Full swing after the bounce, low-to-high topspin or flat drive."},
        {"value": "slice", "description": "Underspin groundstroke or chip: high-to-low swing, open racket face."},
        {"value": "volley", "description": "Ball taken out of the air before it bounces, short punching motion, usually near the net."},
        {"value": "half_volley", "description": "Ball taken immediately after the bounce, very low, near the feet."},
        {"value": "overhead", "description": "Smash: racket swung from above the head at a high ball."},
        {"value": "lob", "description": "Ball lifted high and deep over the opponent."},
        {"value": "drop_shot", "description": "Soft, short touch shot meant to land just over the net."},
        {"value": "other", "description": "Trick shot, mishit, or cannot tell."},
    ],
}

TWO_HANDS_Q = {
    "type": "predicate", "name": "two_hands",
    "instructions": "At contact (frame 4) the highlighted player holds the racket with both hands.",
}


def request_for(entry: dict, end: str, hand: str, dist_net: float | None, with_context: bool = True) -> Request | None:
    sheet = crops_mod.contact_sheet(entry)
    if sheet is None:
        return None
    hand_word = {"R": "right-handed", "L": "left-handed"}.get(hand, "of unknown handedness")
    facing = ("at the near end, back to the camera, so their right is image right" if end == "near" else
              "at the far end, facing the camera, so their right is image left")
    text = (f"Tennis broadcast. Image 1: six frames (numbered 1-6, left to right, top to bottom) cropped around one "
            f"player, from 0.30 s before to 0.20 s after racket contact; frame 4 is the contact. The player is "
            f"{hand_word} and is {facing}.")
    if dist_net is not None:
        text += f" At contact they are about {dist_net:.1f} m from the net."
    images = [crops_mod.save_jpeg(sheet)]
    if with_context and entry.get("context") is not None:
        text += " Image 2: the full court at contact, with this player boxed in yellow."
        images.append(crops_mod.save_jpeg(entry["context"], quality=80))
    return Request(text=text, images=images, questions=[SIDE_Q, FAMILY_Q, TWO_HANDS_Q])


def api_predict(video_id: str, contacts: pd.DataFrame, trk: pd.DataFrame, hands: dict[str, str],
                client: DecisionClient | None = None, label: str = "", dry_run: bool = False,
                save_sheets: bool = True) -> pd.DataFrame:
    """contacts: hit_id, seg, t, half ('near'/'far'), dist_net. hands: {'near': 'R', 'far': 'L'} per row via
    column 'hand' if present. Returns per-hit probabilities."""
    client = client or DecisionClient()
    ent = crops_mod.hit_crops(video_id, contacts, trk)
    reqs, ids = [], []
    sheet_dir = crops_mod.crops_dir(video_id)
    for h in contacts.itertuples():
        e = ent.get(h.hit_id)
        if e is None:
            continue
        hand = getattr(h, "hand", None) or hands.get(h.half, "U")
        r = request_for(e, h.half, hand, getattr(h, "dist_net", None))
        if r is None:
            continue
        if save_sheets:
            (sheet_dir / f"{h.hit_id}.jpg").write_bytes(r.images[0])
        reqs.append(r)
        ids.append(h.hit_id)
    if dry_run:
        return pd.DataFrame({"hit_id": ids, "n_images": [len(r.images) for r in reqs]})
    answers = client.ask_many(reqs, label=label or f"strokes:{video_id}")
    rows = []
    for hid, a in zip(ids, answers):
        if a is None:
            continue
        side = choice_probs(a.get("stroke_side"))
        fam = choice_probs(a.get("stroke_family"))
        th = a.get("two_hands") or {}
        row = dict(hit_id=hid, api_fh=side.get("forehand", np.nan), api_bh=side.get("backhand", np.nan),
                   api_unclear=side.get("unclear", np.nan), api_two_hands=th.get("probability", np.nan),
                   api_tokens=(a.get("_usage") or {}).get("input_tokens"))
        for f in FAMILIES:
            row[f"api_fam_{f}"] = fam.get(f, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# Model classes. Half-volleys and swinging volleys are scored as volleys (too rare to learn apart);
# MCP's unknown ('q') and trick ('t') shots carry no family and are left out of training and scoring.
MCP_FAMILY = {"groundstroke": "groundstroke", "slice": "slice", "volley": "volley", "half_volley": "volley",
              "swinging_volley": "volley", "overhead": "overhead", "drop": "drop_shot", "lob": "lob"}


def mcp_family(fam: str):
    return MCP_FAMILY.get(fam)


def dump(obj) -> str:
    return json.dumps(obj, default=str)


# ---------------------------------------------------------------------------------------------
# Local model: pose features from the contact-sheet crops

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12


def pose_features(kps_list: list[np.ndarray | None], end: str, hand: str) -> dict:
    """Body-frame features from up to six crops' keypoints. x is mirrored so the hitter's
    forehand side is always +x: s = +1 for a right-hander at the near end (back to the camera)
    or a left-hander at the far end."""
    s = 1.0 if (end == "near") == (hand != "L") else -1.0
    racket, other = (R_WR, L_WR) if hand != "L" else (L_WR, R_WR)
    feats: dict[str, float] = {}
    for k, kp in enumerate(kps_list):
        if kp is None or np.isnan(kp[:, 0]).all():
            continue
        c = kp[:, 2]
        if min(c[L_SH], c[R_SH], c[L_HIP], c[R_HIP]) < 0.2:
            continue
        hip = kp[[L_HIP, R_HIP], :2].mean(0)
        sh = kp[[L_SH, R_SH], :2].mean(0)
        torso = np.linalg.norm(sh - hip) + 1e-3
        def rel(i):
            return (s * (kp[i, 0] - hip[0]) / torso, (hip[1] - kp[i, 1]) / torso)
        for name, i in (("rw", racket), ("ow", other), ("re", R_EL if hand != "L" else L_EL)):
            x, y = rel(i)
            feats[f"{name}_x_{k}"], feats[f"{name}_y_{k}"] = x, y
            feats[f"{name}_c_{k}"] = float(c[i])
        feats[f"wdist_{k}"] = float(np.linalg.norm(kp[racket, :2] - kp[other, :2]) / torso)
        # shoulder line: signed image width of shoulders relative to torso (turn proxy)
        feats[f"shw_{k}"] = float(s * (kp[R_SH, 0] - kp[L_SH, 0]) / torso)
    return feats


def run_api(video_id: str, client: DecisionClient | None = None, dry_run: bool = False,
            shots: pd.DataFrame | None = None, out_name: str = "api_strokes.parquet") -> pd.DataFrame:
    """Decisions API stroke predictions for every detected rally contact of a match (cached)."""
    d = match_dir(video_id)
    out = d / out_name
    shots = shots if shots is not None else pd.read_parquet(d / "shots.parquet")
    trk = pd.read_parquet(d / "tracks.parquet")
    rally_c = shots[(shots.kind == "rally")].copy()
    rally_c["hand"] = rally_c.hitter_hand.fillna("U")
    prev = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["hit_id"])
    todo = rally_c[~rally_c.hit_id.isin(prev.hit_id)]
    client = client or DecisionClient()
    client.check_estimate(len(todo), 1100, label=f"strokes {video_id}")
    if dry_run or todo.empty:
        return prev
    res = []
    for i in range(0, len(todo), 150):
        part = todo.iloc[i:i + 150]
        res.append(api_predict(video_id, part[["hit_id", "seg", "t", "half", "dist_net", "hand"]], trk, {}, client=client,
                               label="part"))
    new = pd.concat(res, ignore_index=True) if res else pd.DataFrame()
    allp = pd.concat([prev, new], ignore_index=True)
    allp.to_parquet(out)
    return allp


IMG_SIDE_Q = {
    "type": "choice", "name": "racket_image_side",
    "instructions": ("Look at frame 4 (the moment of contact) and frames 3 and 5 around it. On which side of the "
                     "IMAGE is the racket relative to the highlighted player's body when it strikes the ball? "
                     "Answer in image coordinates (viewer's left/right), not the player's."),
    "choices": [
        {"value": "image_left", "description": "Racket and ball are to the viewer's left of the player's torso."},
        {"value": "image_right", "description": "Racket and ball are to the viewer's right of the player's torso."},
        {"value": "overhead_or_unclear", "description": "Racket above the head, or cannot tell."},
    ],
}


def side_from_image(p_left: float, p_right: float, end: str, hand: str) -> float:
    """P(forehand) from the racket's image side. A right-hander's forehand is image-right at the
    near end (back to the camera) and image-left at the far end (facing it); left-handers mirror."""
    tot = p_left + p_right
    if not tot or np.isnan(tot):
        return np.nan
    p_right_n = p_right / tot
    fh_is_right = (end == "near") == (hand != "L")
    return p_right_n if fh_is_right else 1 - p_right_n


def request_image_side(entry: dict, end: str) -> Request | None:
    sheet = crops_mod.contact_sheet(entry)
    if sheet is None:
        return None
    facing = "with their back to the camera" if end == "near" else "facing the camera"
    text = (f"Tennis broadcast. Six frames (numbered 1-6, left to right, top to bottom) cropped around one player "
            f"{facing}, from 0.30 s before to 0.20 s after racket contact; frame 4 is the contact.")
    return Request(text=text, images=[crops_mod.save_jpeg(sheet)], questions=[IMG_SIDE_Q, TWO_HANDS_Q])


def axis_features(kp: np.ndarray, hand: str) -> dict:
    """Orientation-free cues from one pose: the racket wrist projected on the person's own
    left-to-right shoulder axis (anatomical keypoints, so valid facing toward or away from the
    camera). Positive = racket on the racket-hand side (forehand side)."""
    c = kp[:, 2]
    if min(c[L_SH], c[R_SH]) < 0.25:
        return {}
    lsh, rsh = kp[L_SH, :2], kp[R_SH, :2]
    u = rsh - lsh
    n = np.linalg.norm(u)
    if n < 1e-3:
        return {}
    u = u / n
    ctr = (lsh + rsh) / 2
    hips = kp[[L_HIP, R_HIP], :2].mean(0) if min(c[L_HIP], c[R_HIP]) >= 0.25 else ctr + np.array([0, n])
    torso = np.linalg.norm(ctr - hips) + 1e-3
    rw, ow = (R_WR, L_WR) if hand != "L" else (L_WR, R_WR)
    sgn = 1.0 if hand != "L" else -1.0  # project onto the racket-hand direction
    out = {"shw": n / torso}
    for name, i in (("rw", rw), ("ow", ow)):
        if c[i] >= 0.2:
            v = kp[i, :2] - ctr
            out[f"{name}_ax"] = sgn * float(v @ u) / torso
            out[f"{name}_up"] = float(-(v[1])) / torso
    if "rw_ax" in out and "ow_ax" in out:
        out["wdist"] = float(np.linalg.norm(kp[rw, :2] - kp[ow, :2]) / torso)
    return out


def racket_features(boxes: list, kps_list: list, end: str, hand: str) -> dict:
    """Racket-head position per crop in the hitter's body frame (mirrored so the forehand side is
    +x; y up), normalised by torso length, plus its height change into contact (swing path:
    high-to-low for slices, low-to-high for topspin)."""
    s = 1.0 if (end == "near") == (hand != "L") else -1.0
    out = {}
    ys = {}
    for k, (b, kp) in enumerate(zip(boxes, kps_list)):
        if b is None or kp is None or np.isnan(kp[:, 0]).all():
            continue
        c = kp[:, 2]
        if min(c[L_SH], c[R_SH], c[L_HIP], c[R_HIP]) < 0.2:
            continue
        hip = kp[[L_HIP, R_HIP], :2].mean(0)
        sh = kp[[L_SH, R_SH], :2].mean(0)
        torso = np.linalg.norm(sh - hip) + 1e-3
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        out[f"rk_x_{k}"] = s * (cx - hip[0]) / torso
        out[f"rk_y_{k}"] = (sh[1] - cy) / torso  # >0: racket above the shoulders
        out[f"rk_ar_{k}"] = (b[2] - b[0]) / max(1.0, b[3] - b[1])
        ys[k] = out[f"rk_y_{k}"]
    for a, b in ((0, 3), (1, 3), (2, 3), (3, 5)):
        if a in ys and b in ys:
            out[f"rk_dy_{a}{b}"] = ys[b] - ys[a]
    return out


class RacketDetector:
    """Tennis-racket boxes (COCO category 43) from RF-DETR on player crops."""

    def __init__(self):
        from uso.people import BoxDetector

        self.det = BoxDetector("rfdetr", conf=0.15)
        self.det.person_ids = {43}

    def __call__(self, crops: list[np.ndarray]) -> list:
        res = self.det(crops)
        out = []
        for boxes, conf in res:
            out.append(boxes[int(np.argmax(conf))] if len(conf) else None)
        return out


def local_features(video_id: str, contacts: pd.DataFrame, trk: pd.DataFrame, model=None, size: int = 256,
                   rackets: "RacketDetector | None" = None) -> pd.DataFrame:
    """Pose (and racket) features from the six contact-sheet times for each contact
    (hit_id, seg, t, half, hand)."""
    from uso.pose import PoseModel

    model = model or PoseModel()
    ent = crops_mod.hit_crops(video_id, contacts[["hit_id", "seg", "t", "half"]], trk, out_size=size)
    rows = []
    batch, keys = [], []
    for h in contacts.itertuples():
        e = ent.get(h.hit_id)
        if e is None:
            continue
        for k, cimg in enumerate(e["crops"]):
            if cimg is not None:
                batch.append(cimg)
                keys.append((h.hit_id, k))
    kps = {}
    for i in range(0, len(batch), 64):
        for key, kp in zip(keys[i:i + 64], model(batch[i:i + 64])):
            kps[key] = kp
    rk = {}
    if rackets is not None:
        for i in range(0, len(batch), 64):
            for key, b in zip(keys[i:i + 64], rackets(batch[i:i + 64])):
                rk[key] = b
    for h in contacts.itertuples():
        hand = getattr(h, "hand", "R") or "R"
        f = {"hit_id": h.hit_id}
        klist = [kps.get((h.hit_id, k)) for k in range(len(crops_mod.OFFSETS))]
        f.update(pose_features(klist, h.half, hand))
        for k, kp in enumerate(klist):
            if kp is None or np.isnan(kp[:, 0]).all():
                continue
            for key, v in axis_features(kp, hand).items():
                f[f"{key}_{k}"] = v
        if rackets is not None:
            f.update(racket_features([rk.get((h.hit_id, k)) for k in range(len(crops_mod.OFFSETS))], klist, h.half, hand))
        f["dist_net"] = getattr(h, "dist_net", np.nan)
        f["is_far"] = 1.0 if h.half == "far" else 0.0
        rows.append(f)
    return pd.DataFrame(rows)


def run_local_features(video_id: str, force: bool = False, rackets: bool = False) -> pd.DataFrame:
    d = match_dir(video_id)
    out = d / "stroke_feats.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    shots = pd.read_parquet(d / "shots.parquet")
    trk = pd.read_parquet(d / "tracks.parquet")
    rc = shots[shots.kind == "rally"].copy()
    rc["hand"] = rc.hitter_hand.fillna("R")
    feats = local_features(video_id, rc, trk, rackets=RacketDetector() if rackets else None)
    feats.to_parquet(out)
    return feats


def run_api_imgside(video_id: str, client: DecisionClient | None = None, shots: pd.DataFrame | None = None,
                    out_name: str = "api_imgside.parquet") -> pd.DataFrame:
    """Image-side racket question for every rally contact (cached); converted to P(forehand)."""
    d = match_dir(video_id)
    out = d / out_name
    shots = shots if shots is not None else pd.read_parquet(d / "shots.parquet")
    trk = pd.read_parquet(d / "tracks.parquet")
    rc = shots[shots.kind == "rally"].copy()
    prev = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["hit_id"])
    todo = rc[~rc.hit_id.isin(prev.hit_id)]
    client = client or DecisionClient()
    client.check_estimate(len(todo), 700, label=f"image side {video_id}")
    if todo.empty:
        return prev
    ent = crops_mod.hit_crops(video_id, todo[["hit_id", "seg", "t", "half"]], trk)
    reqs, ids = [], []
    for h in todo.itertuples():
        r = request_image_side(ent[h.hit_id], h.half) if h.hit_id in ent else None
        if r is not None:
            reqs.append(r)
            ids.append((h.hit_id, h.half, h.hitter_hand if isinstance(h.hitter_hand, str) else "R"))
    ans = client.ask_many(reqs, label=f"imgside:{video_id}")
    rows = []
    for (hid, half, hand), a in zip(ids, ans):
        if a is None:
            continue
        pr = choice_probs(a.get("racket_image_side"))
        rows.append(dict(hit_id=hid, p_fh_img=side_from_image(pr.get("image_left", np.nan), pr.get("image_right", np.nan), half, hand),
                         p_overhead_img=pr.get("overhead_or_unclear", np.nan),
                         two_hands=(a.get("two_hands") or {}).get("probability", np.nan)))
    res = pd.concat([prev, pd.DataFrame(rows)], ignore_index=True)
    res.to_parquet(out)
    return res
