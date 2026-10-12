"""Keypoints for the two tracked players, from upscaled crops (YOLO11m-pose).

Runs on the same fixed-rate samples as the people pass, so every track row gets 17 COCO keypoints
in full-resolution pixels (x, y, confidence).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uso.crops import cut, square
from uso.paths import WEIGHTS, match_dir
from uso.video import Meta, ff_frames, video_path

CROP = 256


class PoseModel:
    def __init__(self, weights: str = "yolo11m-pose.pt"):
        from ultralytics import YOLO

        self.model = YOLO(str(WEIGHTS / weights))

    def __call__(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        """17x3 keypoints in crop pixels for the most central confident person per crop."""
        res = self.model.predict(crops, imgsz=CROP, conf=0.1, verbose=False, device="mps")
        out = []
        for r in res:
            if r.boxes is None or len(r.boxes) == 0:
                out.append(np.full((17, 3), np.nan, np.float32))
                continue
            b = r.boxes.xyxy.cpu().numpy()
            c = r.boxes.conf.cpu().numpy()
            ctr = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
            d = np.linalg.norm(ctr - CROP / 2, axis=1) / CROP
            i = int(np.argmax(c - d))
            k = r.keypoints.data[i].cpu().numpy().astype(np.float32)
            out.append(k)
        return out


def run(video_id: str, every: float = 0.25, force: bool = False, model: PoseModel | None = None,
        batch: int = 32) -> pd.DataFrame:
    out = match_dir(video_id) / "pose.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    trk = pd.read_parquet(match_dir(video_id) / "tracks.parquet")
    meta = Meta.probe(video_path(video_id))
    model = model or PoseModel()
    by_t = {round(t / every): g for t, g in trk.groupby("t")}
    rows, buf = [], []

    def flush():
        kps = model([c for *_, c in buf])
        for (idx, x0, y0, side, _), k in zip(buf, kps):
            k = k.copy()
            s = side / CROP
            k[:, 0] = k[:, 0] * s + x0
            k[:, 1] = k[:, 1] * s + y0
            rows.append((idx, k.ravel().tolist()))
        buf.clear()

    for t, f in ff_frames(video_path(video_id), fps=1.0 / every, size=(meta.width, meta.height)):
        g = by_t.get(round(t / every))
        if g is None:
            continue
        for idx, r in g.iterrows():
            box = np.array([r.x1, r.y1, r.x2, r.y2])
            x0, y0, side = square(box, meta.width, meta.height, scale=1.5)
            buf.append((idx, x0, y0, side, cut(f, x0, y0, side, CROP)))
        if len(buf) >= batch:
            flush()
    if buf:
        flush()
    kp = pd.DataFrame(rows, columns=["idx", "kps"]).set_index("idx")
    res = trk.join(kp, how="left")
    res.to_parquet(out)
    return res
