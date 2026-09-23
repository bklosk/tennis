"""Stage 2: per-segment court calibration, ball tracking (TrackNet), and player detection."""
import os
from concurrent.futures import ThreadPoolExecutor
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
# Player detection runs here, overlapping TrackNet on the main thread.
_PLAYER_POOL = ThreadPoolExecutor(max_workers=1)


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
    """TrackNet ball detector: PyTorch on CUDA/MPS/CPU, or the Apple Neural Engine via Core ML.

    `auto` picks the Neural Engine on Apple silicon (39 fps on an M3 Pro vs 18 fps on its GPU),
    which leaves the GPU to player detection. Giving the GPU a share of the ball frames as well
    was slower end to end (25-28 vs 32 fps). Both backends run the same weights, so they share
    the weights tag and cached tracks are not re-run when the backend changes.
    """

    def __init__(self, dev: torch.device, backend: str = "auto", weights: str | None = None):
        self.dev = dev
        if backend == "auto":
            backend = "ane" if tracknet.coreml_available() else "torch"
        self.backend = backend
        path = tracknet.ball_weights(weights)
        self.tag = tracknet.weights_tag(path)
        if backend == "ane":
            self.model = tracknet.coreml_ball(path, 360, BALL_CROP_X[1] - BALL_CROP_X[0])
        else:
            self.dtype = torch.float32 if dev.type == "cpu" else torch.float16
            # channels_last is kept for CUDA; on MPS it is 11% slower than contiguous.
            fmt = torch.channels_last if dev.type == "cuda" else torch.contiguous_format
            self.fmt = fmt
            self.model = tracknet.load("ball", dev, path).to(self.dtype).to(memory_format=fmt)
            self.compiled = False
            if dev.type == "cuda" and os.environ.get("TENNIS_COMPILE", "1") != "0":
                # +37% TrackNet throughput on an L40S for a ~10 s one-time compile. Batches are
                # padded to a fixed size in _masks so it compiles once.
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
        without a per-frame CPU resize. Bit-exact with OpenCV on x86; OpenCV on ARM rounds
        each axis separately and can be one level higher.
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

    def _masks(self, frames: np.ndarray):
        """Yield (frame index, uint8 mask of heatmap pixels above 127) from frame 2 on."""
        n = len(frames)
        if self.backend == "ane":
            x0, x1 = BALL_CROP_X
            small = [cv2.resize(f, (640, 360))[None, :, x0:x1].astype(np.float16) for f in frames[:2]]
            for s in range(2, n):
                small.append(cv2.resize(frames[s], (640, 360))[None, :, x0:x1].astype(np.float16))
                margin = self.model.predict({"cur": small[2], "prev1": small[1], "prev2": small[0]})["margin"]
                small.pop(0)
                yield s, (margin[0] > 0).astype(np.uint8)
            return
        with torch.no_grad():
            X = self._small(frames)
            for s in range(2, n, self.batch):
                m = min(self.batch, n - s)
                # Pad the last batch with repeats: fixed shapes avoid recompiles and cuDNN re-tuning.
                idx = torch.arange(s, s + self.batch, device=self.dev).clamp(max=n - 1)
                inp = torch.cat([X[idx], X[idx - 1], X[idx - 2]], 1).contiguous(memory_format=self.fmt)
                masks = (self._forward(inp).argmax(1) > 127).to(torch.uint8)[:m].cpu().numpy()
                for k, mask in enumerate(masks):
                    yield s + k, mask

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
        for f, mask in self._masks(frames):
            xy = self._pick(mask, prev)
            if xy is not None:
                out[f] = ((xy[0] + x0) * 2, xy[1] * 2)
            prev = None if xy is None else ((xy[0] + x0) * 2, xy[1] * 2)
        return out

    @staticmethod
    def _pick(mask: np.ndarray, prev, max_dist: float = 80.0):
        """Ball blob in a uint8 mask of heatmap pixels above 127 (crop space, 640x360 scale)."""
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
        """Pose per crop. Batches are padded to a power of two: the number of hits varies per chunk,
        and every new batch size otherwise triggers cuDNN re-tuning."""
        n = len(crops)
        if n == 0:
            return []
        padded = 1 << (n - 1).bit_length()
        res = self.model.predict(list(crops) + [crops[-1]] * (padded - n), imgsz=imgsz, device=self.device,
                                 conf=0.2, classes=[0], verbose=False, **self.precision)
        return res[:n]

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

    far_rect = far_court_rect(good[len(good) // 2][1])
    args = (frames, tracks.calibs, players, far_rect, player_every, player_batch)
    if ball.backend == "torch" and ball.dev.type == "mps":
        # MPS trips a Metal assertion when two threads encode GPU commands at once.
        tracks.players, tracks.player_kps = _track_players(*args)
        tracks.ball = ball(frames)
    else:
        # TrackNet on the Neural Engine or CUDA; players (GPU inference plus CPU pre/post-
        # processing) overlap it. On an L40S they took about as long per chunk as TrackNet.
        people = _PLAYER_POOL.submit(_track_players, *args)
        tracks.ball = ball(frames)
        tracks.players, tracks.player_kps = people.result()
    tracks.ball_weights = ball.tag
    return tracks


def _track_players(frames, calibs, players, far_rect, player_every, player_batch):
    n = len(frames)
    boxes = {s: np.full((n, 4), np.nan) for s in ("near", "far")}
    kps = {s: np.full((n, 17, 3), np.nan) for s in ("near", "far")}
    sample = list(range(0, n, player_every))
    prev = {}
    for chunk in range(0, len(sample), player_batch):
        idx = sample[chunk:chunk + player_batch]
        near_dets, far_dets = players.detect([frames[i] for i in idx], far_rect)
        for f, nd, fd in zip(idx, near_dets, far_dets):
            sel = select_players(nd, fd, calibs[f], prev)
            for side, (box, kp, court_xy) in sel.items():
                boxes[side][f] = box
                if kp is not None:
                    kps[side][f] = kp
                prev[side] = court_xy
    return {s: interp_nan(b, max_gap=player_every * 3) for s, b in boxes.items()}, kps


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
