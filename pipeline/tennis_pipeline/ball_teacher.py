"""Automatic ball labels from trajectory context, for fine-tuning TrackNet without hand labels.

TrackNet sees three frames. The teacher sees about eight on each side: robust quadratic fits to
the pretrained model's detections before and after a frame (never across a hit or bounce)
predict where the ball must be, and a three-frame difference blob near that prediction confirms
the ball is actually visible there. That gives labels of three kinds, all inside rallies:

    fill  the model found nothing; a moving blob sits where the trajectory says the ball is
    fix   the model picked something else; a moving blob sits on the trajectory instead
    easy  the model's detection agrees with the trajectory (keeps fine-tuning from drifting)

Frames without image evidence near the prediction, or whose two sides disagree, are skipped.
Labels go to `eval/ball_labels.csv` with labeler `teacher-v1`; frames are written where
`balllabels --action export` would put them, so `balltrain` uses them unchanged.
"""
import cv2
import numpy as np
import pandas as pd

from . import video
from .ball_labels import _rally_windows, frame_paths, load_labels, save_labels
from .paths import match_dir
from .process import load_tracks
from .track import FPS

LABELER = "teacher-v1"
SIDE_FRAMES = 8
FIT_TOL_PX = 3.5
AGREE_PX = 6.0
BLOB_RADIUS_PX = 8.0
FIX_MIN_PX = 12.0


def robust_fit(f: np.ndarray, xy: np.ndarray, tol: float = FIT_TOL_PX, min_pts: int = 4):
    """Quadratic (linear below 5 points) x(f), y(f) fit, dropping the worst point until all fit."""
    keep = np.ones(len(f), bool)
    while keep.sum() >= min_pts:
        deg = 2 if keep.sum() >= 5 else 1
        cx = np.polyfit(f[keep], xy[keep, 0], deg)
        cy = np.polyfit(f[keep], xy[keep, 1], deg)
        res = np.hypot(np.polyval(cx, f) - xy[:, 0], np.polyval(cy, f) - xy[:, 1])
        worst = int(np.argmax(np.where(keep, res, -1.0)))
        if res[worst] <= tol:
            return cx, cy, keep
        keep[worst] = False
    return None


def predict(raw: np.ndarray, t: int, events: np.ndarray, max_extrap: int = 3):
    """Ball position at frame t from detections on each side, or None.

    Returns (xy, n_sides): both sides must agree within AGREE_PX; a single side needs five inliers
    and must reach within two frames of t.
    """
    n = len(raw)
    prev_e = events[events < t].max() if (events < t).any() else -1
    next_e = events[events > t].min() if (events > t).any() else n
    preds = []
    for lo, hi, one_side_reach in ((max(t - SIDE_FRAMES, prev_e, 0), t - 1, 2),
                                   (t + 1, min(t + SIDE_FRAMES, next_e, n - 1), 2)):
        f = np.arange(lo, hi + 1)
        f = f[~np.isnan(raw[f, 0])] if len(f) else f
        if len(f) < 4:
            continue
        fit = robust_fit(f.astype(float), raw[f])
        if fit is None:
            continue
        cx, cy, keep = fit
        reach = int(np.abs(f[keep] - t).min())
        if reach > max_extrap:
            continue
        preds.append((np.array([np.polyval(cx, t), np.polyval(cy, t)]), int(keep.sum()), reach <= one_side_reach))
    if len(preds) == 2:
        if np.hypot(*(preds[0][0] - preds[1][0])) > AGREE_PX:
            return None
        return (preds[0][0] + preds[1][0]) / 2, 2
    if len(preds) == 1 and preds[0][1] >= 5 and preds[0][2]:
        return preds[0][0], 1
    return None


def diff_blob(gray: np.ndarray, t: int, p: np.ndarray, radius: float = BLOB_RADIUS_PX, win: int = 24,
              thr: int = 15, max_area: int = 120, max_side: int = 24):
    """Moving blob at frame t nearest p: min(|I_t - I_t-1|, |I_t - I_t+1|) isolates where a moving
    object is at t. Returns (centroid, area) or None."""
    h, w = gray.shape[1:]
    x, y = int(round(p[0])), int(round(p[1]))
    x0, x1, y0, y1 = max(x - win, 0), min(x + win, w), max(y - win, 0), min(y + win, h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    cur = gray[t, y0:y1, x0:x1].astype(np.int16)
    d = np.minimum(np.abs(cur - gray[t - 1, y0:y1, x0:x1]), np.abs(cur - gray[t + 1, y0:y1, x0:x1]))
    n, _, stats, cents = cv2.connectedComponentsWithStats((d > thr).astype(np.uint8))
    best = None
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not 3 <= area <= max_area or max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) > max_side:
            continue
        c = cents[i] + (x0, y0)
        dist = float(np.hypot(*(c - p)))
        if dist <= radius and (best is None or dist < best[2]):
            best = (c, area, dist)
    return None if best is None else (best[0], best[1])


def _on_player(players: dict, t: int, p: np.ndarray, grow: float = 0.15) -> bool:
    for box in players.values():
        b = box[t]
        if np.isnan(b).any():
            continue
        gx, gy = grow * (b[2] - b[0]), grow * (b[3] - b[1])
        if b[0] - gx <= p[0] <= b[2] + gx and b[1] - gy <= p[1] <= b[3] + gy:
            return True
    return False


def label_chunk(raw: np.ndarray, gray: np.ndarray, events: np.ndarray, frames: np.ndarray,
                players: dict | None = None, passes: int = 4) -> list[dict]:
    """Teacher labels for `frames` of one chunk.

    Misses come in runs longer than the fits reach, so labeling propagates: each pass adds the
    previous pass's fills as detections, growing into a gap from both ends, one to three frames
    per pass, with a moving blob required at every step. No fills or fixes on a player, where a
    moving racket, hand or shoe can pass for the ball.
    """
    players = players or {}
    frames = [t for t in frames if 2 <= t <= len(raw) - 2]
    track = raw.copy()
    out = {}
    for _ in range(passes):
        added = {}
        for t in frames:
            if t in out:
                continue
            pred = predict(track, t, events)
            if pred is None:
                continue
            p, sides = pred
            det = raw[t] if not np.isnan(raw[t, 0]) else None
            if det is not None and np.hypot(*(det - p)) <= 3.0:
                out[t] = {"frame": int(t), "x": float(det[0]), "y": float(det[1]), "kind": "easy", "visibility": 1,
                          "sides": sides, "det": det}
                continue
            if det is not None and (np.hypot(*(det - p)) < FIX_MIN_PX or sides < 2):
                continue  # near miss of the trajectory, or one-sided: ambiguous either way
            if _on_player(players, t, p):
                continue
            blob = diff_blob(gray, t, p)
            if blob is None:
                continue
            c, area = blob
            added[t] = {"frame": int(t), "x": float(c[0]), "y": float(c[1]),
                        "kind": "fix" if det is not None else "fill", "visibility": 1 if area <= 60 else 2,
                        "sides": sides, "det": det}
        if not added:
            break
        for t, r in added.items():
            out[t] = r
            track[t] = (r["x"], r["y"])
    return [out[t] for t in sorted(out)]


def _spaced(rows: list[dict], spacing: int) -> list[dict]:
    kept, last = [], -10 ** 9
    for r in sorted(rows, key=lambda r: r["frame"]):
        if r["frame"] - last >= spacing:
            kept.append(r)
            last = r["frame"]
    return kept


def auto_label(video_ids: list[str], video_path_fn, holdout: list[str] = (), spacing: int = 3,
               easy_ratio: float = 1.0, seed: int = 0, dry_run: bool = False) -> dict:
    """Label in-rally frames of each match with the teacher and append them to the label file."""
    rng = np.random.default_rng(seed)
    labels = load_labels()
    have = set(labels.key)
    new, report = [], {}
    for vid in video_ids:
        out_dir = match_dir(vid)
        hits = pd.read_parquet(out_dir / "hits_raw.parquet")
        ball = pd.read_parquet(out_dir / "ball.parquet")
        windows = _rally_windows(hits)
        bounces = ball[ball.is_bounce.fillna(False)].groupby("chunk_id").frame.apply(np.array).to_dict()
        hit_frames = hits.groupby("chunk_id").frame.apply(np.array).to_dict()
        counts = {"fill": 0, "fix": 0, "easy": 0, "in_rally_frames": 0, "missed_in_rally": 0}
        rows = []
        for chunk, spans in windows.items():
            tr = load_tracks(out_dir / "tracks" / f"{chunk}.npz")
            if tr.court_ok < 0.5:
                continue
            cand = np.unique(np.concatenate([np.arange(max(a, 2), min(b, tr.n - 2) + 1) for a, b in spans]))
            if not len(cand):
                continue
            counts["in_rally_frames"] += len(cand)
            counts["missed_in_rally"] += int(np.isnan(tr.ball[cand, 0]).sum())
            clip = video.read_clip(video_path_fn(vid), tr.t0, tr.n / FPS)[:tr.n]
            if len(clip) < tr.n:
                continue
            gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in clip])
            events = np.unique(np.concatenate([hit_frames.get(chunk, np.zeros(0, int)),
                                               bounces.get(chunk, np.zeros(0, int))])).astype(int)
            got = label_chunk(tr.ball, gray, events, cand, tr.players)
            hard = _spaced([r for r in got if r["kind"] != "easy"], spacing)
            easy = _spaced([r for r in got if r["kind"] == "easy"], spacing)
            k = min(len(easy), int(round(max(len(hard), 1) * easy_ratio)))
            easy = [easy[i] for i in sorted(rng.choice(len(easy), k, replace=False))] if k else []
            for r in hard + easy:
                t = round(tr.t0 + r["frame"] / FPS, 4)
                key = f"{vid}@{t:.4f}"
                if key in have:
                    continue
                have.add(key)
                counts[r["kind"]] += 1
                if not dry_run:
                    paths = frame_paths(vid, t)
                    paths[0].parent.mkdir(parents=True, exist_ok=True)
                    for p, img in zip(paths, (clip[r["frame"]], clip[r["frame"] - 1], clip[r["frame"] - 2])):
                        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                det = r["det"]
                rows.append({"key": key, "video_id": vid, "t": t, "chunk_id": chunk, "frame": r["frame"],
                             "bucket": r["kind"], "split": "val" if vid in holdout else "train",
                             "pre_x": np.nan if det is None else float(det[0]),
                             "pre_y": np.nan if det is None else float(det[1]),
                             "pre_source": "tracknet" if det is not None else "none",
                             "x": r["x"], "y": r["y"], "visibility": r["visibility"], "status": "done",
                             "labeler": LABELER})
        new += rows
        report[vid] = counts
        print(vid, counts, flush=True)
    if new and not dry_run:
        add = pd.DataFrame(new)
        labels = pd.concat([labels, add], ignore_index=True) if len(labels) else add
        save_labels(labels)
    return report
