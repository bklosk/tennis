"""Stage 2: per-segment court calibration, ball tracking (TrackNet), and player detection."""
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import tracknet
from .court import Calibration, CourtDetector
from .paths import WEIGHTS

FPS = 30.0
BALL_CROP_X = (64, 576)  # in 640-wide TrackNet space; keeps the court, drops the stands


@dataclass
class SegmentTracks:
    t0: float
    n: int
    calibs: list  # per-frame Calibration or None
    ball: np.ndarray  # (n, 2) image px, NaN if missing
    players: dict = field(default_factory=dict)  # side -> (n, 4) bbox, NaN if missing
    player_kps: dict = field(default_factory=dict)  # side -> (n, 17, 3)
    court_ok: float = 0.0
    ball_weights: str = ""


class BallTracker:
    """TrackNet ball detector on MPS/CUDA (PyTorch) or the Neural Engine (Core ML).

    The Core ML backend's process memory grows with repeated predictions (~15 GB over a few
    minutes of footage), so it is only suitable for short runs. It is a conversion of the
    pretrained weights only.
    """

    def __init__(self, dev: torch.device, backend: str = "mps", weights: str | None = None):
        self.dev = dev
        self.backend = backend
        if backend == "coreml":
            import coremltools as ct

            if weights or tracknet.ball_weights() != tracknet.PRETRAINED_BALL:
                raise ValueError("the Core ML backend only has the pretrained ball weights; use --ball-backend mps")
            path = WEIGHTS / "tracknet_512.mlpackage"
            self.model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
            self.tag = f"coreml:{tracknet.weights_tag(tracknet.PRETRAINED_BALL)}"
        else:
            path = tracknet.ball_weights(weights)
            self.dtype = torch.float32 if dev.type == "cpu" else torch.float16
            self.model = tracknet.load("ball", dev, path).to(self.dtype).to(memory_format=torch.channels_last)
            self.tag = tracknet.weights_tag(path)
            self.compiled = False
            if dev.type == "cuda" and os.environ.get("TENNIS_COMPILE", "1") != "0":
                # +37% TrackNet throughput on an L40S for a ~10 s one-time compile. Batches are
                # padded to a fixed size in _heat so it compiles once.
                self.eager = self.model
                self.model = torch.compile(self.model)
                self.compiled = True

    batch_override: int | None = None

    @property
    def batch(self) -> int:
        return self.batch_override or (16 if self.dev.type == "cuda" else 8)

    def _small(self, frames: np.ndarray) -> torch.Tensor:
        """TrackNet input frames (n, 3, 360, 512) on the device, scaled to [0, 1].

        Cropping at full resolution and 2x average pooling on the device reproduces
        cv2.resize(INTER_LINEAR) to 640x360 (an exact 2x downscale averages each 2x2 block),
        without a per-frame CPU resize.
        """
        x0, x1 = BALL_CROP_X
        n = len(frames)
        if frames.shape[1:3] != (720, 1280):
            small = np.stack([cv2.resize(f, (640, 360))[:, x0:x1] for f in frames])
            return torch.from_numpy(small).to(self.dev).permute(0, 3, 1, 2).to(self.dtype) / 255
        out = torch.empty((n, 3, 360, x1 - x0), dtype=self.dtype, device=self.dev)
        for s in range(0, n, 64):
            part = torch.from_numpy(np.ascontiguousarray(frames[s:s + 64, :, 2 * x0:2 * x1])).to(self.dev)
            # floor(x + 0.5): cv2 rounds halves up, torch.round rounds them to even.
            pooled = torch.floor(F.avg_pool2d(part.permute(0, 3, 1, 2).float(), 2) + 0.5)
            out[s:s + len(part)] = (pooled / 255).to(self.dtype)
        return out

    def _heat(self, frames: np.ndarray):
        n = len(frames)
        if self.backend == "coreml":
            x0, x1 = BALL_CROP_X
            small = np.stack([cv2.resize(f, (640, 360))[:, x0:x1] for f in frames])
            Fs = small.astype(np.float32).transpose(0, 3, 1, 2) / 255
            for s in range(2, n):
                x = np.concatenate([Fs[s], Fs[s - 1], Fs[s - 2]], 0)[None]
                yield s, next(iter(self.model.predict({"x": x}).values()))[0].astype(np.uint8)
            return
        with torch.no_grad():
            X = self._small(frames)
            for s in range(2, n, self.batch):
                m = min(self.batch, n - s)
                # Pad the last batch with repeats: fixed shapes avoid recompiles and cuDNN re-tuning.
                idx = torch.arange(s, s + self.batch, device=self.dev).clamp(max=n - 1)
                inp = torch.cat([X[idx], X[idx - 1], X[idx - 2]], 1).contiguous(memory_format=torch.channels_last)
                heat = self._forward(inp).argmax(1).to(torch.uint8)[:m].cpu().numpy()
                for k, hm in enumerate(heat):
                    yield s + k, hm

    def _forward(self, inp: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "compiled", False):
            return self.model(inp)
        try:
            return self.model(inp)
        except Exception as e:  # compiler/toolchain problems must not stop tracking
            print(f"torch.compile failed ({type(e).__name__}); using eager TrackNet", flush=True)
            self.model, self.compiled = self.eager, False
            return self.model(inp)

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        n = len(frames)
        out = np.full((n, 2), np.nan)
        if n < 3:
            return out
        x0 = BALL_CROP_X[0]
        prev = None
        for f, hm in self._heat(frames):
            xy = self._pick(hm, prev)
            if xy is not None:
                out[f] = ((xy[0] + x0) * 2, xy[1] * 2)
            prev = None if xy is None else ((xy[0] + x0) * 2, xy[1] * 2)
        return out

    @staticmethod
    def _pick(hm: np.ndarray, prev, max_dist: float = 80.0):
        mask = (hm > 127).astype(np.uint8)
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            return None
        order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
        cands = [cents[i] for i in order if 2 <= stats[i, cv2.CC_STAT_AREA] <= 400]
        if not cands:
            return None
        if prev is not None:
            for c in cands:
                if np.hypot(c[0] * 2 + BALL_CROP_X[0] * 2 - prev[0], c[1] * 2 - prev[1]) < max_dist:
                    return c
        return cands[0]


class PlayerDetector:
    """Person detection on the full frame (near player) plus an upscaled far-court crop (far player).

    A plain detector finds the ~55 px far player far more reliably than pose models; pose is run
    later only on hitter crops at contact frames.
    """

    def __init__(self, model_name: str = "yolo11s.pt", pose_name: str = "yolo11s-pose.pt"):
        from ultralytics import YOLO

        WEIGHTS.mkdir(parents=True, exist_ok=True)
        self.detector = YOLO(str(WEIGHTS / model_name))
        self.model = YOLO(str(WEIGHTS / pose_name))
        self.device = tracknet.yolo_device()
        self.precision = {"quantize": 16} if self.device == "0" else {}

    def _run(self, imgs, imgsz):
        res = self.detector.predict(imgs, imgsz=imgsz, device=self.device, conf=0.15, classes=[0], verbose=False,
                                    **self.precision)
        return [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), None) for r in res]

    def pose(self, crops: list[np.ndarray], imgsz: int = 256):
        return self.model.predict(crops, imgsz=imgsz, device=self.device, conf=0.2, classes=[0], verbose=False,
                                  **self.precision)

    def detect(self, frames: list[np.ndarray], far_rect: tuple[int, int, int, int] | None, up: float = 1.5,
               near_imgsz: int = 640):
        # The near player is 110-150 px tall at 720p, so half resolution is ample; the ~55 px far
        # player gets the upscaled court crop below.
        near = self._run(frames, near_imgsz)
        if far_rect is None:
            return near, [None] * len(frames)
        x0, y0, x1, y1 = far_rect
        crops = [cv2.resize(f[y0:y1, x0:x1], None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) for f in frames]
        imgsz = int(np.ceil(max(crops[0].shape[:2]) / 32) * 32)
        far = []
        for boxes, confs, kps in self._run(crops, imgsz):
            boxes = boxes / up + np.array([x0, y0, x0, y0])
            if kps is not None:
                kps = kps.copy()
                kps[..., 0] = kps[..., 0] / up + x0
                kps[..., 1] = kps[..., 1] / up + y0
            far.append((boxes, confs, kps))
        return near, far


def far_court_rect(calib: Calibration, pad: int = 30) -> tuple[int, int, int, int]:
    corners = calib.to_img(np.array([[-8.5, -19.0], [8.5, -19.0], [-8.5, 1.0], [8.5, 1.0]]))
    x0, y0 = np.floor(corners.min(0)).astype(int) - pad
    x1, y1 = np.ceil(corners.max(0)).astype(int) + pad
    # Leave headroom above the far players for raised arms and serves.
    y0 -= 80
    return max(x0, 0), max(y0, 0), min(x1, 1280), min(y1, 720)


def select_players(near_dets, far_dets, calib: Calibration | None, prev: dict):
    """Pick the near- and far-side player from person detections using court position."""
    chosen = {}
    if calib is None:
        return chosen
    for side, sign, dets in (("far", -1, far_dets), ("near", 1, near_dets)):
        if dets is None or len(dets[0]) == 0:
            continue
        boxes, confs, kps = dets
        feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], 1)
        court = calib.to_court_m(feet)
        heights = boxes[:, 3] - boxes[:, 1]
        y = court[:, 1] * sign
        ok = (y > 0.8) & (y < 19) & (np.abs(court[:, 0]) < 7.5)
        cand = np.where(ok)[0]
        if len(cand) == 0:
            continue
        if side in prev:
            px, py = prev[side]
            d = np.hypot(court[cand, 0] - px, court[cand, 1] - py)
            near = cand[d < 2.5]
            if len(near):
                cand = near
        # Prefer tall, confident, central detections (ball kids sit at the corners).
        score = heights[cand] * confs[cand] / (1 + 0.15 * np.maximum(np.abs(court[cand, 0]) - 4.5, 0))
        i = cand[int(np.argmax(score))]
        chosen[side] = (boxes[i], None if kps is None else kps[i], court[i])
    return chosen


def track_segment(frames: np.ndarray, t0: float, court_det: CourtDetector, ball: BallTracker,
                  players: PlayerDetector, calib_every: int = 60, player_every: int = 4,
                  player_batch: int = 32) -> SegmentTracks:
    n = len(frames)
    probe_idx = list(range(0, n, calib_every)) or [0]
    probe_cal = court_det.calibrate([frames[i] for i in probe_idx])
    good = [(i, c) for i, c in zip(probe_idx, probe_cal) if c is not None and c.reproj_px < 6]
    tracks = SegmentTracks(t0=t0, n=n, calibs=[None] * n, ball=np.full((n, 2), np.nan),
                           court_ok=len(good) / len(probe_idx))
    if tracks.court_ok < 0.5:
        return tracks
    anchors = np.array([i for i, _ in good])
    for f in range(n):
        tracks.calibs[f] = good[int(np.argmin(np.abs(anchors - f)))][1]

    tracks.ball = ball(frames)
    tracks.ball_weights = ball.tag

    for side in ("near", "far"):
        tracks.players[side] = np.full((n, 4), np.nan)
        tracks.player_kps[side] = np.full((n, 17, 3), np.nan)
    sample = list(range(0, n, player_every))
    prev = {}
    far_rect = far_court_rect(good[len(good) // 2][1])
    for chunk in range(0, len(sample), player_batch):
        idx = sample[chunk:chunk + player_batch]
        near_dets, far_dets = players.detect([frames[i] for i in idx], far_rect)
        for f, nd, fd in zip(idx, near_dets, far_dets):
            sel = select_players(nd, fd, tracks.calibs[f], prev)
            for side, (box, kp, court_xy) in sel.items():
                tracks.players[side][f] = box
                if kp is not None:
                    tracks.player_kps[side][f] = kp
                prev[side] = court_xy
    for side in ("near", "far"):
        tracks.players[side] = interp_nan(tracks.players[side], max_gap=player_every * 3)
    return tracks


def interp_nan(arr: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN rows across gaps of at most `max_gap` frames."""
    arr = arr.copy()
    valid = ~np.isnan(arr.reshape(len(arr), -1)).any(1)
    idx = np.where(valid)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap:
            w = (np.arange(a + 1, b) - a) / (b - a)
            arr[a + 1:b] = arr[a] * (1 - w[:, None]) + arr[b] * w[:, None]
    return arr
