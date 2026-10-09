"""Hitter crops around contacts, decoded at exact presentation times.

For each contact the hitter's box is interpolated from the track at several offsets around the
contact and a square crop (with margin for the racket) is cut from the full-resolution frame.
The same crops feed pose estimation and the Decisions API contact sheets.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from uso.paths import match_dir
from uso.video import frames_at, video_path

OFFSETS = (-0.30, -0.15, -0.05, 0.0, 0.08, 0.20)


def _box_at(trk: pd.DataFrame, seg: int, half: str, t: float):
    g = trk[(trk.seg == seg) & (trk.half == half)]
    if g.empty:
        return None
    ts = g.t.to_numpy()
    if t < ts[0] - 0.6 or t > ts[-1] + 0.6:
        return None
    return np.array([np.interp(t, ts, g[c].to_numpy()) for c in ("x1", "y1", "x2", "y2")])


def square(box: np.ndarray, W: int, H: int, scale: float = 1.7, min_side: int = 64):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = max(min_side, scale * max(box[3] - box[1], 1.2 * (box[2] - box[0])))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2 - 0.05 * side))
    return x0, y0, int(round(side))


def cut(frame: np.ndarray, x0: int, y0: int, side: int, out: int) -> np.ndarray:
    H, W = frame.shape[:2]
    pad = side
    padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    c = padded[y0 + pad: y0 + pad + side, x0 + pad: x0 + pad + side]
    return cv2.resize(c, (out, out), interpolation=cv2.INTER_AREA if side > out else cv2.INTER_CUBIC)


def hit_crops(video_id: str, hits: pd.DataFrame, trk: pd.DataFrame, offsets=OFFSETS, out_size: int = 192,
              context_w: int = 384) -> dict:
    """hits: columns hit_id, seg, t, half. Returns {hit_id: {'crops': [img per offset],
    'context': full frame at contact with the hitter boxed, 'box': box at contact}}."""
    want = []
    for h in hits.itertuples():
        for o in offsets:
            want.append(h.t + o)
    frames = frames_at(video_path(video_id), want)
    out = {}
    for h in hits.itertuples():
        crops, ctx, box0 = [], None, None
        for o in offsets:
            ft, f = frames[h.t + o]
            box = _box_at(trk, h.seg, h.half, ft)
            if box is None:
                crops.append(None)
                continue
            Hh, W = f.shape[:2]
            if o == 0.0:
                box0 = box
                im = f.copy()
                cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), max(2, W // 400))
                s = context_w / W
                ctx = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            x0, y0, side = square(box, W, Hh)
            crops.append(cut(f, x0, y0, side, out_size))
        out[h.hit_id] = dict(crops=crops, context=ctx, box=box0)
    return out


def contact_sheet(entry: dict, cols: int = 3) -> np.ndarray | None:
    crops = [c for c in entry["crops"] if c is not None]
    if len(crops) < 3:
        return None
    while len(crops) % cols:
        crops.append(np.zeros_like(crops[0]))
    rows = [np.hstack(crops[i:i + cols]) for i in range(0, len(crops), cols)]
    grid = np.vstack(rows)
    for k in range(len(entry["crops"])):  # frame index labels
        r, c = divmod(k, cols)
        s = crops[0].shape[0]
        cv2.putText(grid, str(k + 1), (c * s + 4, r * s + 16), 0, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return grid


def save_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def crops_dir(video_id: str):
    d = match_dir(video_id) / "crops"
    d.mkdir(exist_ok=True)
    return d
