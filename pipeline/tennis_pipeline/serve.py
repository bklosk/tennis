"""Serve detection from the server's body, for serves the ball tracker loses.

The rule in `events.annotate_hits` needs the toss and the outgoing flight tracked around
contact. At 200+ km/h the ball is a faint streak right after contact, so big servers lose
most serves (54% detected in the 2024 men's final). This detector keys on the server
instead:

- stance: settled behind the baseline near the centre mark for about a second;
- extension: the person box grows upward as the tossing and hitting arms go above the
  head (contact is at the box's tallest point);
- quiet: the opponent has not hit the ball in the previous 2 s (not a rally shot).

At least one piece of ball or sound evidence is also required: a tracked toss, a racket
onset in the audio, the ball leaving toward the opponent, a return hit, or a raw hit
candidate at contact. Candidates are scored with a small logistic model whose default
weights are hand-set; `eval/serve_eval.py` sweeps the threshold against official data.
"""
from dataclasses import dataclass

import numpy as np

from .court import HALF_LENGTH, SINGLES_HALF_WIDTH
from .track import FPS, SegmentTracks

FEATURES = ("ext", "still", "toss", "audio", "response", "returner_back", "hit_near")
DEFAULT_WEIGHTS = {"bias": -3.0, "ext": 1.5, "still": 1.0, "toss": 1.5, "audio": 1.0,
                   "response": 1.2, "returner_back": 0.8, "hit_near": 1.0}


@dataclass
class ServeParams:
    min_extension: float = 1.15  # box height at contact / settled stance height
    behind_baseline_m: float = 0.8  # stance feet at least HALF_LENGTH - this from the net
    max_abs_x_m: float = SINGLES_HALF_WIDTH + 1.0
    quiet_s: float = 2.0
    threshold: float = 0.5
    merge_frames: int = 12


def _feet_court(tr: SegmentTracks, side: str) -> tuple[np.ndarray, np.ndarray]:
    box = tr.players.get(side)
    n = tr.n
    feet = np.full((n, 2), np.nan)
    if box is None:
        return np.full((n, 4), np.nan), feet
    for f in range(n):
        c = tr.calibs[f]
        if c is None or np.isnan(box[f]).any():
            continue
        feet[f] = c.to_court_m(np.array([[(box[f, 0] + box[f, 2]) / 2, box[f, 3]]]))[0]
    return box, feet


def _nanmedian(a: np.ndarray, axis=0):
    if a.size == 0 or np.isnan(a).all():
        return np.full(a.shape[1:] if a.ndim > 1 else (), np.nan)
    return np.nanmedian(a, axis=axis)


def _runs(mask: np.ndarray, max_gap: int = 4) -> list[tuple[int, int]]:
    idx = np.where(mask)[0]
    runs = []
    for i in idx:
        if runs and i - runs[-1][1] <= max_gap:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [(a, b) for a, b in runs]


def _ball_leaves(b: np.ndarray, f: int, side: str) -> float:
    """1 if the ball is tracked moving from the server toward the opponent just after contact."""
    seg = b[f + 2:f + 21]
    ok = seg[~np.isnan(seg[:, 1])]
    if len(ok) < 3:
        return 0.0
    dy = ok[-1, 1] - ok[0, 1]
    return float(dy < -20 if side == "near" else dy > 15)


def detect(tr: SegmentTracks, b: np.ndarray, raw_hits: list[dict], onsets: tuple | None = None,
           params: ServeParams | None = None, weights: dict | None = None) -> list[dict]:
    """Scored serve candidates for one chunk (accepted and rejected, for diagnostics)."""
    p = params or ServeParams()
    w = weights or DEFAULT_WEIGHTS
    n = tr.n
    if n < 60 or not tr.players:
        return []
    on_t, on_s = onsets if onsets is not None else (np.zeros(0), np.zeros(0))
    other = {"near": "far", "far": "near"}
    feet_by_side = {s: _feet_court(tr, s) for s in ("near", "far")}
    cands = []
    for side in ("near", "far"):
        box, feet = feet_by_side[side]
        if np.isnan(box).all():
            continue
        h = box[:, 3] - box[:, 1]
        sign = 1 if side == "near" else -1
        h_ref = np.full(n, np.nan)
        for f in range(n):
            h_ref[f] = _nanmedian(h[max(0, f - 60):max(0, f - 20)])
        ext = h / h_ref
        for a, z in _runs(np.nan_to_num(ext) >= p.min_extension):
            f = a + int(np.nanargmax(h[a:z + 1]))
            if f < 30 or f > n - 3:
                continue
            stance = _nanmedian(feet[max(0, f - 45):f - 15])
            if np.isnan(stance).any():
                continue
            behind = stance[1] * sign >= HALF_LENGTH - p.behind_baseline_m and abs(stance[0]) <= p.max_abs_x_m
            opp_recent = [hh for hh in raw_hits if hh["side"] == other[side]
                          and f - p.quiet_s * FPS <= hh["frame"] <= f - p.merge_frames]
            pre = feet[max(0, f - 60):f - 15]
            spread = float(np.nanstd(pre, axis=0).max()) if (~np.isnan(pre[:, 0])).sum() >= 5 else np.inf
            ref_box = box[max(0, f - 30)] if not np.isnan(box[max(0, f - 30)]).any() else box[f]
            toss = 0.0
            for g in range(max(0, f - 40), min(n, f + 3)):
                bg = b[g]
                if not np.isnan(bg[0]) and not np.isnan(box[g]).any():
                    cx = (box[g, 0] + box[g, 2]) / 2
                    if bg[1] < ref_box[1] and abs(bg[0] - cx) < 0.8 * h_ref[f]:
                        toss = 1.0
                        break
            t_c = tr.t0 + f / FPS
            lo, hi = np.searchsorted(on_t, [t_c - 0.3, t_c + 0.3])
            onset_t = float(on_t[lo + int(np.argmax(on_s[lo:hi]))]) if hi > lo else None
            ret_hits = [hh for hh in raw_hits if hh["side"] == other[side] and f + 12 <= hh["frame"] <= f + 60]
            near_hits = [hh for hh in raw_hits if hh["side"] == side and abs(hh["frame"] - f) <= p.merge_frames]
            opp_feet = _nanmedian(feet_by_side[other[side]][1][max(0, f - 30):f + 1])
            feats = {
                "ext": float(np.clip((ext[f] - 1.05) / 0.3, 0, 1)),
                "still": float(np.clip(1 - spread / 0.8, 0, 1)) if np.isfinite(spread) else 0.0,
                "toss": toss or float(any(hh.get("toss") for hh in near_hits)),
                "audio": float(onset_t is not None),
                "response": max(_ball_leaves(b, f, side), float(bool(ret_hits))),
                "returner_back": float(not np.isnan(opp_feet).any() and abs(opp_feet[1]) >= HALF_LENGTH - 3.5),
                "hit_near": float(bool(near_hits)),
            }
            logit = w["bias"] + sum(w[k] * feats[k] for k in FEATURES)
            score = float(1 / (1 + np.exp(-logit)))
            evidence = feats["toss"] + feats["audio"] + feats["response"] + feats["hit_near"]
            reject = None
            if not behind:
                reject = "not_behind_baseline"
            elif opp_recent:
                reject = "opponent_hit_recently"
            elif evidence < 1:
                reject = "no_ball_or_audio_evidence"
            elif score < p.threshold:
                reject = "low_score"
            if near_hits:
                contact = min(near_hits, key=lambda hh: abs(hh["frame"] - f))["frame"]
            elif onset_t is not None:
                contact = int(np.clip(round((onset_t - tr.t0) * FPS), 0, n - 1))
            else:
                contact = f
            cands.append({"frame": int(contact), "peak_frame": int(f), "side": side, "t": tr.t0 + contact / FPS,
                          "score": score, "extension": float(ext[f]), "stance_x_m": float(stance[0]),
                          "stance_y_m": float(stance[1]), "accepted": reject is None, "reject": reject, **feats})
    return _dedupe(cands)


def _dedupe(cands: list[dict], window: int = 45) -> list[dict]:
    """Keep the best accepted candidate within `window` frames (nobody serves twice in 1.5 s)."""
    cands = sorted(cands, key=lambda c: (-c["accepted"], -c["score"]))
    kept = []
    for c in cands:
        if c["accepted"] and any(k["accepted"] and abs(k["frame"] - c["frame"]) < window for k in kept):
            c = {**c, "accepted": False, "reject": "duplicate"}
        kept.append(c)
    return sorted(kept, key=lambda c: c["frame"])


def merge(raw_hits: list[dict], serves: list[dict], b: np.ndarray, tr: SegmentTracks,
          params: ServeParams | None = None) -> list[dict]:
    """Flag or insert accepted serves among the raw hits; drop the server's pre-serve "hits".

    A player cannot hit a shot and then serve within 2.5 s, so same-side hit candidates just
    before a serve are ball bounces from the pre-serve routine.
    """
    p = params or ServeParams()
    out = [dict(h) for h in raw_hits]
    for s in (s for s in serves if s["accepted"]):
        f, side = s["frame"], s["side"]
        out = [h for h in out if not (h["side"] == side and f - 75 <= h["frame"] < f - p.merge_frames)]
        match = [h for h in out if h["side"] == side and abs(h["frame"] - f) <= p.merge_frames]
        if match:
            m = min(match, key=lambda h: abs(h["frame"] - f))
            m["serve_score"] = s["score"]
            continue
        box = tr.players.get(side)
        bx = None if box is None else box[f]
        pos = b[f] if not np.isnan(b[f, 0]) else (
            np.array([(bx[0] + bx[2]) / 2, bx[1]]) if bx is not None and not np.isnan(bx).any() else np.array([np.nan, np.nan]))
        out.append({"frame": int(f), "side": side, "dv": 0.0, "pos": pos, "gap_frames": 0,
                    "player_seen": bx is not None and not np.isnan(bx).any(), "toss": False,
                    "inferred": True, "serve_score": s["score"]})
    out.sort(key=lambda h: h["frame"])
    deduped = []
    for h in out:
        if deduped and h["frame"] - deduped[-1]["frame"] < p.merge_frames:
            keep_new = h.get("serve_score") is not None and deduped[-1].get("serve_score") is None
            if keep_new:
                deduped[-1] = h
            continue
        deduped.append(h)
    return deduped
