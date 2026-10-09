"""Person detection on main-view frames, in two views per frame.

The near player is large and is found in the full frame at detector resolution. The far player is
often 40-70 px tall, too small for a full-frame pass, so the far half of the court (located with the
court homography) is cropped and upscaled before a second pass. Boxes from both views are mapped back
to full-resolution pixels and to court meters.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd

from uso import scene as scene_mod
from uso.paths import WEIGHTS, match_dir
from uso.video import Meta, ff_frames, video_path

FAR_REGION_M = np.array([(-7.5, 0.5), (7.5, 0.5), (-7.5, 20.0), (7.5, 20.0)], np.float32)
NEAR_REGION_M = np.array([(-8.0, -0.5), (8.0, -0.5), (-8.0, -19.0), (8.0, -19.0)], np.float32)


class BoxDetector:
    """COCO person boxes. backend 'rfdetr' (RF-DETR, Small by default) or 'yolo' (YOLO11m)."""

    BATCH = 8
    fixed_batch = False

    def __init__(self, backend: str = "rfdetr", conf: float = 0.3, size: str = "small"):
        self.backend = backend
        self.conf = conf
        if backend == "rfdetr":
            cwd = os.getcwd()
            os.makedirs(WEIGHTS / "rfdetr", exist_ok=True)
            os.chdir(WEIGHTS / "rfdetr")
            try:
                import rfdetr

                cls = {"small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium, "nano": rfdetr.RFDETRNano}[size]
                self.model = cls()
                # TorchScript + fp16: ~30% faster on MPS with identical detections (checked 2026-10-09);
                # the traced graph is fixed to batches of BATCH images, so calls are padded to it.
                try:
                    import torch

                    self.model.inference(compile=True, batch_size=self.BATCH, dtype=torch.float16)
                    self.fixed_batch = True
                except Exception as e:  # fall back to eager fp32
                    print(f"[people] inference optimisation unavailable: {type(e).__name__}: {e}")
            finally:
                os.chdir(cwd)
            # predictions use COCO category ids (person = 1) although class_names is a 0-based list
            self.person_ids = {1}
        else:
            from ultralytics import YOLO

            self.model = YOLO(str(WEIGHTS / "yolo11m.pt"))

    def __call__(self, images_bgr: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        if self.backend == "rfdetr" and self.fixed_batch and len(images_bgr) != self.BATCH:
            out = []
            for i in range(0, len(images_bgr), self.BATCH):
                chunk = images_bgr[i:i + self.BATCH]
                pad = self.BATCH - len(chunk)
                res = self._predict(chunk + [chunk[-1]] * pad)
                out.extend(res[:len(chunk)])
            return out
        return self._predict(images_bgr)

    def _predict(self, images_bgr: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        if self.backend == "rfdetr":
            preds = self.model.predict([im[:, :, ::-1].copy() for im in images_bgr], threshold=self.conf,
                                       include_source_image=False)
            if not isinstance(preds, list):
                preds = [preds]
            out = []
            for p in preds:
                keep = np.isin(p.class_id, list(self.person_ids)) if len(p) else np.zeros(0, bool)
                out.append((np.asarray(p.xyxy, float)[keep], np.asarray(p.confidence, float)[keep]))
            return out
        res = self.model.predict(images_bgr, imgsz=640, conf=self.conf, classes=[0], verbose=False, device="mps")
        return [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()) for r in res]


def _region_box(H: np.ndarray, region_m: np.ndarray, w: int, h: int, top_pad: float, side_pad: float):
    p = cv2.perspectiveTransform(region_m.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    x0, x1 = p[:, 0].min() - side_pad, p[:, 0].max() + side_pad
    y0, y1 = p[:, 1].min() - top_pad, p[:, 1].max() + 0.25 * top_pad
    x0, y0 = int(max(0, x0)), int(max(0, y0))
    x1, y1 = int(min(w, x1)), int(min(h, y1))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    return x0, y0, x1, y1


def _to_court(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), np.linalg.inv(H)).reshape(-1, 2)


def detect(video_id: str, every: float = 0.25, backend: str = "rfdetr", force: bool = False,
           det: BoxDetector | None = None, batch: int = 8, far_target_w: int = 1280) -> pd.DataFrame:
    """Every person box on main-view frames with ground point in court meters. Cached."""
    out = match_dir(video_id) / f"people_{backend}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    path = video_path(video_id)
    meta = Meta.probe(path)
    sc = pd.read_parquet(match_dir(video_id) / "scene.parquet")
    segs = scene_mod.segments(sc)
    det = det or BoxDetector(backend)
    W, Hh = meta.width, meta.height
    rows = []
    buf: list[tuple[int, float, np.ndarray, np.ndarray]] = []

    def flush():
        images, meta_rows = [], []
        for si, t, f, H in buf:
            # near view: whole frame (detector resizes internally)
            images.append(f)
            meta_rows.append((si, t, H, "near", 0, 0, 1.0))
            fb = _region_box(H, FAR_REGION_M, W, Hh, top_pad=0.12 * Hh, side_pad=0.03 * W)
            if fb is not None:
                x0, y0, x1, y1 = fb
                crop = f[y0:y1, x0:x1]
                s = min(3.0, max(1.0, far_target_w / crop.shape[1]))
                images.append(cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else crop)
                meta_rows.append((si, t, H, "far", x0, y0, s))
        res = det(images)
        for (si, t, H, view, x0, y0, s), (boxes, conf) in zip(meta_rows, res):
            if len(boxes) == 0:
                continue
            b = boxes / s + np.array([x0, y0, x0, y0])
            gp = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], 1)
            cxy = _to_court(H, gp)
            for i in range(len(b)):
                rows.append(dict(seg=si, t=t, view=view, conf=float(conf[i]), x1=b[i, 0], y1=b[i, 1], x2=b[i, 2],
                                 y2=b[i, 3], cx=float(cxy[i, 0]), cy=float(cxy[i, 1])))
        buf.clear()

    starts, ends = segs.start.to_numpy(), segs.end.to_numpy()
    for t, f in ff_frames(path, fps=1.0 / every, size=(W, Hh)):
        k = int(np.searchsorted(starts, t, side="right")) - 1
        if k < 0 or t > ends[k]:
            continue
        H = scene_mod.homography_at(sc, t)
        if H is None:
            continue
        buf.append((k, t, f.copy(), H))
        if len(buf) == batch:
            flush()
    if buf:
        flush()
    df = pd.DataFrame(rows)
    # the near view also sees far people (too small to trust) and vice versa: keep each view's half
    if not df.empty:
        keep = ((df.view == "near") & (df.cy < 1.0)) | ((df.view == "far") & (df.cy >= -1.0))
        df = df[keep].reset_index(drop=True)
    df.to_parquet(out)
    return df
