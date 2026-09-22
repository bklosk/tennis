"""Stage 3: bounces, hits, serves, and per-hit stroke features from segment tracks."""
import catboost as ctb
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

from .court import HALF_LENGTH, SERVICE_LINE, SINGLES_HALF_WIDTH
from .paths import WEIGHTS
from .track import FPS, SegmentTracks, interp_nan

COCO = {"nose": 0, "l_sh": 5, "r_sh": 6, "l_el": 7, "r_el": 8, "l_wr": 9, "r_wr": 10, "l_hip": 11, "r_hip": 12}


def clean_ball(ball: np.ndarray, max_jump: float = 90.0) -> np.ndarray:
    b = ball.copy()
    valid = ~np.isnan(b[:, 0])
    for i in np.where(valid)[0]:
        if valid[max(0, i - 2):i + 3].sum() <= 1:
            b[i] = np.nan
    idx = np.where(~np.isnan(b[:, 0]))[0]
    for k in range(1, len(idx) - 1):
        i0, i, i1 = idx[k - 1], idx[k], idx[k + 1]
        if i1 - i0 <= 6:
            pred = b[i0] + (b[i1] - b[i0]) * (i - i0) / (i1 - i0)
            if np.linalg.norm(b[i] - pred) > max_jump and np.linalg.norm(b[i] - b[i0]) > max_jump:
                b[i] = np.nan
    b = interp_nan(b, max_gap=6)
    out = b.copy()
    good = ~np.isnan(b[:, 0])
    # Light smoothing inside each contiguous run.
    start = None
    for i, g in enumerate(np.append(good, False)):
        if g and start is None:
            start = i
        elif not g and start is not None:
            if i - start >= 7:
                out[start:i] = savgol_filter(b[start:i], 5, 2, axis=0)
            start = None
    return out


class BounceModel:
    """Pretrained CatBoost bounce regressor from yastrebksv/TennisProject (30 fps, 1280x720)."""

    def __init__(self):
        self.model = ctb.CatBoostRegressor()
        self.model.load_model(str(WEIGHTS / "bounce.cbm"))

    def predict(self, ball: np.ndarray, threshold: float = 0.45) -> list[int]:
        x = [None if np.isnan(v) else float(v) for v in ball[:, 0]]
        y = [None if np.isnan(v) else float(v) for v in ball[:, 1]]
        x, y = _extrapolate_gaps(x, y)
        df = pd.DataFrame({"x": x, "y": y}, dtype=float)
        feats = {}
        for i in (1, 2):
            for c in ("x", "y"):
                lag, inv = df[c].shift(i), df[c].shift(-i)
                d, dinv = lag - df[c], inv - df[c]
                if c == "x":
                    d, dinv = d.abs(), dinv.abs()
                feats[f"{c}_diff_{i}"] = d
                feats[f"{c}_diff_inv_{i}"] = dinv
                ratio = d / (dinv + 1e-15)
                feats[f"{c}_div_{i}"] = ratio.abs() if c == "x" else ratio
        feats = pd.DataFrame(feats)
        cols = ([f"x_diff_{i}" for i in (1, 2)] + [f"x_diff_inv_{i}" for i in (1, 2)] + [f"x_div_{i}" for i in (1, 2)]
                + [f"y_diff_{i}" for i in (1, 2)] + [f"y_diff_inv_{i}" for i in (1, 2)] + [f"y_div_{i}" for i in (1, 2)])
        feats = feats[cols].dropna()
        if feats.empty:
            return []
        preds = self.model.predict(feats)
        frames = feats.index.to_numpy()
        hits = np.where(preds > threshold)[0]
        keep = []
        for h in hits:
            if keep and frames[h] - frames[keep[-1]] <= 1:
                if preds[h] > preds[keep[-1]]:
                    keep[-1] = h
            else:
                keep.append(h)
        return [int(frames[k]) for k in keep]


def _extrapolate_gaps(x, y, interp: int = 5):
    missing = [v is None for v in x]
    counter = 0
    for i in range(interp, len(x) - 1):
        if x[i] is None and not any(missing[i - interp:i]) and counter < 3:
            fx = CubicSpline(range(interp), x[i - interp:i], bc_type="natural")
            fy = CubicSpline(range(interp), y[i - interp:i], bc_type="natural")
            x[i], y[i] = float(fx(interp)), float(fy(interp))
            missing[i] = False
            counter += 1
        else:
            counter = 0
    return x, y


def _runs(sig: np.ndarray):
    runs = []
    cur, start, last = 0, None, None
    for i, v in enumerate(sig):
        if np.isnan(v):
            if cur:
                runs.append((cur, start, last))
            cur, start = 0, None
            continue
        if v == 0:
            continue
        if v != cur:
            if cur:
                runs.append((cur, start, last))
            cur, start = v, i
        last = i
    if cur:
        runs.append((cur, start, last))
    return runs


def _reach(ball_xy, box, margin_x: float = 0.9, above: float = 0.8, below: float = 0.1) -> bool:
    if box is None or np.isnan(box).any() or np.isnan(ball_xy).any():
        return False
    x1, y1, x2, y2 = box
    h = y2 - y1
    return (x1 - margin_x * h <= ball_xy[0] <= x2 + margin_x * h) and (y1 - above * h <= ball_xy[1] <= y2 + below * h)


def _flights(b: np.ndarray, lag: int = 3, dead: float = 4.0, min_len: int = 4, max_gap: int = 30):
    """Split the ball track into flights with a consistent image-y direction.

    Returns (direction, first_frame, last_frame); +1 = toward the near baseline.
    Same-direction pieces separated by short gaps (missed detections, bounces) are merged.
    """
    n = len(b)
    by = b[:, 1]
    sig = np.full(n, np.nan)
    for t in range(lag, n - lag):
        a, c = by[t - lag], by[t + lag]
        if np.isnan(a) or np.isnan(c):
            continue
        d = c - a
        sig[t] = 1 if d > dead else (-1 if d < -dead else 0)
    pieces = [r for r in _runs(sig) if r[2] - r[1] + 1 >= min_len]
    merged = []
    for p in pieces:
        if merged and merged[-1][0] == p[0] and p[1] - merged[-1][2] <= max_gap:
            merged[-1] = (p[0], merged[-1][1], p[2])
        else:
            merged.append(p)
    return merged


def _fit(b: np.ndarray, frames: np.ndarray):
    frames = frames[~np.isnan(b[frames, 0])]
    if len(frames) < 3:
        return None
    deg = 2 if len(frames) >= 5 else 1
    return np.polyfit(frames, b[frames, 0], deg), np.polyfit(frames, b[frames, 1], deg)


def _meet(b: np.ndarray, inc: tuple, out: tuple) -> int:
    """Frame where the incoming and outgoing flights meet (the contact estimate)."""
    fin = _fit(b, np.arange(max(inc[1], inc[2] - 7), inc[2] + 1))
    fout = _fit(b, np.arange(out[1], min(out[2], out[1] + 7) + 1))
    lo, hi = inc[2], out[1]
    if fin is None or fout is None or hi <= lo:
        return (lo + hi) // 2
    ts = np.arange(lo, hi + 1)
    d = np.hypot(np.polyval(fin[0], ts) - np.polyval(fout[0], ts), np.polyval(fin[1], ts) - np.polyval(fout[1], ts))
    return int(ts[np.argmin(d)])


def detect_hits(tr: SegmentTracks, b: np.ndarray) -> list[dict]:
    n = len(b)
    if n < 20:
        return []
    v = np.full((n, 2), np.nan)
    v[1:-1] = b[2:] - b[:-2]
    dv = np.full(n, np.nan)
    dv[2:-2] = np.linalg.norm(v[4:] - v[:-4], axis=1)

    flights = _flights(b)
    cands = []
    for inc, out in zip(flights[:-1], flights[1:]):
        if inc[0] == out[0] or out[1] - inc[2] > 45:
            continue
        side = "near" if inc[0] == 1 else "far"
        box_track = tr.players.get(side)
        f_meet = _meet(b, inc, out)
        # When the ball is tracked through contact, the sharpest velocity change marks it
        # (for serves this lands on the racket contact rather than the top of the toss).
        best, best_dv = f_meet, 0.0
        for f in range(max(inc[2] - 4, 0), min(out[1] + 18, n - 1) + 1):
            if not np.isnan(dv[f]) and dv[f] > max(best_dv, 8):
                best, best_dv = f, dv[f]
        if best_dv == 0.0 or abs(best - f_meet) > 20:
            best = f_meet
        pos = b[best] if not np.isnan(b[best, 0]) else _estimate_pos(b, inc, out, best)
        box = None if box_track is None else box_track[best]
        if box is not None and not np.isnan(box).any() and not _reach(pos, box, margin_x=1.2, above=1.4, below=0.35):
            continue
        cands.append({"frame": int(best), "side": side, "dv": float(best_dv), "pos": pos,
                      "gap_frames": int(out[1] - inc[2]), "player_seen": box is not None and not np.isnan(box).any(),
                      "toss": _is_toss(b, inc, box)})

    # A flight that starts at a player with no detected hit just before it was struck by that
    # player while the ball was untracked (typically a serve whose toss was not seen).
    for fl in flights:
        if fl[2] - fl[1] < 6:
            continue
        side = "far" if fl[0] == 1 else "near"
        if any(fl[1] - 25 <= c["frame"] <= fl[1] + 5 for c in cands):
            continue
        box_track = tr.players.get(side)
        f0 = fl[1]
        box = None if box_track is None else box_track[f0]
        if box is None or np.isnan(box).any() or not _reach(b[f0], box, margin_x=1.2, above=1.6, below=0.2):
            continue
        cands.append({"frame": int(f0), "side": side, "dv": float(np.nan_to_num(dv[f0])), "pos": b[f0],
                      "gap_frames": 0, "player_seen": True, "toss": False, "inferred": True})

    cands.sort(key=lambda c: c["frame"])
    hits = []
    for c in cands:
        if hits and c["frame"] - hits[-1]["frame"] < 12:
            if c["dv"] > hits[-1]["dv"]:
                hits[-1] = c
            continue
        if hits and c["side"] == hits[-1]["side"] and c["frame"] - hits[-1]["frame"] < 45:
            if c["dv"] > hits[-1]["dv"]:
                hits[-1] = c
            continue
        hits.append(c)
    return hits


def _is_toss(b: np.ndarray, inc: tuple, box) -> bool:
    """Did the incoming flight start at the hitter's own body (a service toss), not across the net?"""
    if box is None or np.isnan(box).any() or inc[2] - inc[1] < 5:
        return False
    x, y = b[inc[1]]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return bool(x1 - 1.0 * w <= x <= x2 + 1.0 * w and y1 - 1.8 * h <= y <= y2 - 0.2 * h)


def _estimate_pos(b, inc, out, f):
    fin = _fit(b, np.arange(max(inc[1], inc[2] - 7), inc[2] + 1))
    if fin is None:
        return np.array([np.nan, np.nan])
    return np.array([np.polyval(fin[0], f), np.polyval(fin[1], f)])


def kink_bounce(b: np.ndarray, f0: int, f1: int, min_pts: int = 8) -> int | None:
    """Bounce as the breakpoint of a two-segment linear fit to the ball's image track in (f0, f1)."""
    frames = np.arange(f0 + 3, min(f1 - 2, len(b)))
    frames = frames[~np.isnan(b[frames, 1])] if len(frames) else frames
    if len(frames) < min_pts:
        return None
    y, x = b[frames, 1], b[frames, 0]
    best, best_sse, best_turn = None, np.inf, 0.0
    for k in range(3, len(frames) - 3):
        sse, slopes = 0.0, []
        for seg in (slice(0, k + 1), slice(k, len(frames))):
            t = frames[seg]
            for v in (y[seg], x[seg]):
                coef = np.polyfit(t, v, 1)
                sse += float(((np.polyval(coef, t) - v) ** 2).sum())
            slopes.append(np.polyfit(t, y[seg], 1)[0])
        if sse < best_sse:
            best, best_sse, best_turn = int(frames[k]), sse, abs(slopes[1] - slopes[0])
    return best if best_turn > 2.5 else None


def _player_state(tr: SegmentTracks, side: str, f: int):
    box = tr.players.get(side, np.full((tr.n, 4), np.nan))[f]
    calib = tr.calibs[f]
    court = None
    if calib is not None and not np.isnan(box).any():
        court = calib.to_court_m(np.array([[(box[0] + box[2]) / 2, box[3]]]))[0]
    return box, court


def annotate_hits(tr: SegmentTracks, b: np.ndarray, hits: list[dict], bounces: list[int]) -> list[dict]:
    """Add court positions, serve flags, bounce-after-hit, and ball speed estimates."""
    out = []
    for k, h in enumerate(hits):
        f = h["frame"]
        box, court = _player_state(tr, h["side"], f)
        calib = tr.calibs[f]
        rec = {k: v for k, v in h.items() if k != "pos"}
        pos = h.get("pos", b[f])
        rec["t"] = tr.t0 + f / FPS
        rec["ball_px_x"], rec["ball_px_y"] = pos
        rec["hitter_box"] = box
        rec["hitter_x_m"], rec["hitter_y_m"] = (np.nan, np.nan) if court is None else court
        h_px = box[3] - box[1] if not np.isnan(box).any() else np.nan
        # Serve contacts are refined to the sharpest velocity change, which can land a frame or
        # two after contact; also accept the highest ball point in the preceding 6 frames.
        prior = b[max(f - 6, 0):f + 1, 1]
        top = np.nanmin(np.append(prior, pos[1])) if len(prior) else pos[1]
        rec["ball_above_head"] = bool(not np.isnan(h_px) and top < box[1] + 0.12 * h_px)
        next_f = hits[k + 1]["frame"] if k + 1 < len(hits) else tr.n
        prev_f = hits[k - 1]["frame"] if k > 0 else -1
        before = [x for x in bounces if prev_f < x < f]
        hitter_sign = 1 if h["side"] == "near" else -1
        after = []
        if calib is not None:
            for x in bounces:
                if f + 3 < x < next_f and not np.isnan(b[x, 0]):
                    bxy = calib.to_court_m(b[x][None])[0]
                    if np.sign(bxy[1]) == -hitter_sign:
                        after.append((x, bxy))
                        break
        rec["bounce_source"] = "catboost" if after else None
        if not after and calib is not None:
            kb = kink_bounce(b, f, min(next_f, f + 60))
            if kb is not None:
                bxy = calib.to_court_m(b[kb][None])[0]
                if np.sign(bxy[1]) == -hitter_sign and abs(bxy[0]) < 8 and abs(bxy[1]) < HALF_LENGTH + 4:
                    after.append((kb, bxy))
                    rec["bounce_source"] = "kink"
        rec["bounce_frame"] = after[0][0] if after else None
        if after:
            x, (bx, byy) = after[0]
            rec["bounce_x_m"], rec["bounce_y_m"] = bx, byy
            rec["bounce_t"] = tr.t0 + x / FPS
            if court is not None:
                dist = float(np.hypot(bx - court[0], byy - court[1]))
                speed = dist / ((x - f) / FPS) * 3.6
                rec["avg_speed_kmh"] = speed if speed < 260 else np.nan
        rec["bounced_before_hit_own_side"] = any(
            calib is not None and np.sign(calib.to_court_m(b[x][None])[0][1]) == (1 if h["side"] == "near" else -1)
            for x in before) if before else False
        out.append(rec)

    for k, rec in enumerate(out):
        gap_before = rec["t"] - out[k - 1]["t"] if k > 0 else np.inf
        behind_baseline = abs(rec["hitter_y_m"]) > HALF_LENGTH - 1.5 if not np.isnan(rec["hitter_y_m"]) else False
        # A serve is struck overhead from behind the baseline after a toss (the incoming flight
        # starts at the server), and not ~1 s after an opponent's shot.
        paused = gap_before > 2.0 or (out[k - 1]["side"] == rec["side"] and gap_before > 0.8)
        by_rule = bool(behind_baseline and rec["ball_above_head"] and (rec.get("toss") or (paused and k == 0)))
        by_detector = rec.get("serve_score") is not None
        rec["is_serve"] = by_rule or by_detector
        rec["serve_source"] = ("rule+detector" if by_rule and by_detector else "rule" if by_rule
                               else "detector" if by_detector else None)
    return out


def serve_in(rec: dict) -> bool | None:
    """Did a serve land in the diagonally opposite service box?

    At serve speed the ball moves ~1.4 m between 30 fps frames, so the bounce frame's position
    carries ~0.7 m of depth error; tolerances reflect that.
    """
    if "bounce_x_m" not in rec or np.isnan(rec.get("hitter_x_m", np.nan)) or np.isnan(rec.get("bounce_x_m", np.nan)):
        return None
    bx, byy = rec["bounce_x_m"], rec["bounce_y_m"]
    server_sign = np.sign(rec["hitter_y_m"])
    correct_half = byy * server_sign < 0 and abs(byy) <= SERVICE_LINE + 0.8
    cross = np.sign(bx) != np.sign(rec["hitter_x_m"]) or abs(bx) < 0.4
    return bool(correct_half and cross and abs(bx) <= SINGLES_HALF_WIDTH + 0.4)


def hitter_crop(frame: np.ndarray, box: np.ndarray, out: int = 256, scale: float = 1.7):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(y2 - y1, x2 - x1) * scale
    cy -= 0.1 * (y2 - y1)
    M = np.array([[out / side, 0, out / 2 - cx * out / side], [0, out / side, out / 2 - cy * out / side]])
    crop = cv2.warpAffine(frame, M, (out, out), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    return crop, M


def stroke_features(rec: dict, kps: np.ndarray | None, ball_xy: np.ndarray) -> dict:
    """Egocentric pose/ball features for a right-hander's frame of reference.

    Positive lateral values are on a right-hander's forehand side; `strokes.py` flips the
    sign for left-handers once players are identified.
    """
    box = rec["hitter_box"]
    feats = {}
    if box is None or np.isnan(box).any():
        return feats
    h = box[3] - box[1]
    ego = 1 if rec["side"] == "near" else -1  # near player's back faces the camera
    cx = (box[0] + box[2]) / 2
    if kps is not None and not np.isnan(kps).all():
        hips = kps[[COCO["l_hip"], COCO["r_hip"]]]
        if (hips[:, 2] > 0.3).all():
            cx = hips[:, 0].mean()
        wr = kps[[COCO["l_wr"], COCO["r_wr"]]]
        sh = kps[[COCO["l_sh"], COCO["r_sh"]]]
        if (wr[:, 2] > 0.2).all():
            lat = (wr[:, 0] - cx) * ego / h
            feats["l_wrist_lat"] = float(lat[0])
            feats["r_wrist_lat"] = float(lat[1])
            feats["wrist_gap"] = float(np.hypot(*(wr[0, :2] - wr[1, :2])) / h)
            if (sh[:, 2] > 0.2).all():
                feats["wrist_above_sh"] = float((sh[:, 1].mean() - wr[:, 1].min()) / h)
    if not np.isnan(ball_xy).any():
        feats["ball_lat"] = float((ball_xy[0] - cx) * ego / h)
        feats["ball_height"] = float((box[1] - ball_xy[1]) / h)
    feats["court_abs_y"] = float(abs(rec["hitter_y_m"])) if not np.isnan(rec["hitter_y_m"]) else np.nan
    return feats
