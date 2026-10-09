"""Court keypoints, homography and registration quality.

Court frame, in meters: origin on the ground at the centre of the net. x runs across the court,
positive to the right as seen from the camera (near) end; y runs along it, positive toward the far
end. The near baseline is y = -11.885 and the far baseline y = +11.885; singles sidelines are at
x = ±4.115 and doubles sidelines at x = ±5.485.

Keypoints come from the pretrained 14-point court network of yastrebksv/TennisCourtDetector
(weights in .cache/weights/court.pt; upstream states no license, used here non-commercially).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn

from uso.paths import WEIGHTS

HALF_LEN = 11.885
SERVICE = 6.40
SINGLES = 4.115
DOUBLES = 5.485

# Keypoint order of the court network.
KPS_M = np.array(
    [
        (-DOUBLES, HALF_LEN), (DOUBLES, HALF_LEN), (-DOUBLES, -HALF_LEN), (DOUBLES, -HALF_LEN),
        (-SINGLES, HALF_LEN), (-SINGLES, -HALF_LEN), (SINGLES, HALF_LEN), (SINGLES, -HALF_LEN),
        (-SINGLES, SERVICE), (SINGLES, SERVICE), (-SINGLES, -SERVICE), (SINGLES, -SERVICE),
        (0.0, SERVICE), (0.0, -SERVICE),
    ],
    dtype=np.float32,
)

# Ground-plane court lines as segments in meters (the net is above the ground and is excluded).
LINES_M = np.array(
    [
        ((-DOUBLES, HALF_LEN), (DOUBLES, HALF_LEN)),
        ((-DOUBLES, -HALF_LEN), (DOUBLES, -HALF_LEN)),
        ((-DOUBLES, -HALF_LEN), (-DOUBLES, HALF_LEN)),
        ((DOUBLES, -HALF_LEN), (DOUBLES, HALF_LEN)),
        ((-SINGLES, -HALF_LEN), (-SINGLES, HALF_LEN)),
        ((SINGLES, -HALF_LEN), (SINGLES, HALF_LEN)),
        ((-SINGLES, SERVICE), (SINGLES, SERVICE)),
        ((-SINGLES, -SERVICE), (SINGLES, -SERVICE)),
        ((0.0, -SERVICE), (0.0, SERVICE)),
    ],
    dtype=np.float32,
)


class _Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(cout))

    def forward(self, x):
        return self.block(x)


class CourtNet(nn.Module):
    """TrackNet-style encoder/decoder with 15 heatmaps (14 keypoints + court centre)."""

    def __init__(self, out_channels: int = 15):
        super().__init__()
        chans = [(3, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 256), (256, 256), (256, 512),
                 (512, 512), (512, 512), (512, 256), (256, 256), (256, 256), (256, 128), (128, 128),
                 (128, 64), (64, 64)]
        for i, (a, b) in enumerate(chans, start=1):
            setattr(self, f"conv{i}", _Block(a, b))
        self.conv18 = _Block(64, out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        c = lambda i, v: getattr(self, f"conv{i}")(v)  # noqa: E731
        x = c(2, c(1, x)); x = self.pool(x)
        x = c(4, c(3, x)); x = self.pool(x)
        x = c(7, c(6, c(5, x))); x = self.pool(x)
        x = c(10, c(9, c(8, x))); x = self.up(x)
        x = c(13, c(12, c(11, x))); x = self.up(x)
        x = c(15, c(14, x)); x = self.up(x)
        x = c(17, c(16, x))
        return self.conv18(x)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CourtDetector:
    W, H = 640, 360

    def __init__(self, device: str | None = None, half: bool = False):
        self.device = device or default_device()
        net = CourtNet(15)
        state = torch.load(WEIGHTS / "court.pt", map_location="cpu")
        net.load_state_dict(state)
        # fp16 is only ~10% faster on MPS and breaks on real frames (one spurious peak, nothing else), so fp32
        self.dtype = torch.float16 if half and self.device != "cpu" else torch.float32
        self.net = net.eval().to(self.device, dtype=self.dtype)

    def _prep(self, frame: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        """Letterbox a BGR frame into 640x360 keeping aspect ratio. Returns (img, scale, dx, dy)."""
        h, w = frame.shape[:2]
        s = min(self.W / w, self.H / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        img = np.zeros((self.H, self.W, 3), np.uint8)
        dx, dy = (self.W - nw) // 2, (self.H - nh) // 2
        img[dy:dy + nh, dx:dx + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        return img, s, dx, dy

    @torch.inference_mode()
    def keypoints(self, frames: list[np.ndarray], batch: int = 8) -> list[tuple[np.ndarray, np.ndarray]]:
        """For each BGR frame: (14x2 keypoints in frame pixels, NaN where missing; 14 peak scores)."""
        out = []
        for i in range(0, len(frames), batch):
            chunk = frames[i:i + batch]
            preps = [self._prep(f) for f in chunk]
            x = torch.from_numpy(np.stack([p[0] for p in preps])).to(self.device)
            x = x.permute(0, 3, 1, 2).to(self.dtype) / 255.0
            heat = torch.sigmoid(self.net(x))[:, :14].float().cpu().numpy()
            for hm, (_, s, dx, dy) in zip(heat, preps):
                out.append(_peaks(hm, s, dx, dy))
        return out


def _peaks(hm: np.ndarray, s: float, dx: float, dy: float, thresh: float = 0.67):
    """Blob centres. The network was trained on radius-55 Gaussians, so peaks are broad saturated
    plateaus: argmax lands on the plateau's first raster row, so take the blob's centroid instead."""
    k = hm.shape[0]
    pts = np.full((k, 2), np.nan, np.float32)
    scores = hm.reshape(k, -1).max(1)
    for j in range(k):
        if scores[j] < thresh:
            continue
        mask = (hm[j] >= thresh).astype(np.uint8)
        n, lab = cv2.connectedComponents(mask, connectivity=8)
        y, x = np.unravel_index(int(hm[j].argmax()), hm[j].shape)
        blob = lab == lab[y, x]
        yy, xx = np.nonzero(blob)
        w = hm[j][yy, xx]
        cx, cy = float((xx * w).sum() / w.sum()), float((yy * w).sum() / w.sum())
        pts[j] = ((cx - dx) / s, (cy - dy) / s)
    return pts, scores


@dataclass
class Registration:
    H: np.ndarray | None  # court meters -> image pixels
    n_kps: int
    n_inliers: int
    reproj_px: float
    line_score: float  # fraction of projected line samples that sit on bright line pixels

    @property
    def ok(self) -> bool:
        return self.H is not None and self.n_inliers >= 6 and self.line_score >= 0.5

    def to_court(self, xy_px: np.ndarray) -> np.ndarray:
        Hinv = np.linalg.inv(self.H)
        pts = cv2.perspectiveTransform(np.asarray(xy_px, np.float64).reshape(-1, 1, 2), Hinv)
        return pts.reshape(-1, 2)

    def to_image(self, xy_m: np.ndarray) -> np.ndarray:
        pts = cv2.perspectiveTransform(np.asarray(xy_m, np.float64).reshape(-1, 1, 2), self.H)
        return pts.reshape(-1, 2)


def fit_homography(kps: np.ndarray, frame_w: int) -> tuple[np.ndarray | None, int, float]:
    valid = ~np.isnan(kps[:, 0])
    if valid.sum() < 4:
        return None, 0, np.inf
    thr = max(3.0, 0.006 * frame_w)
    H, mask = cv2.findHomography(KPS_M[valid], kps[valid], cv2.RANSAC, thr)
    if H is None:
        return None, 0, np.inf
    inl = mask.ravel().astype(bool)
    proj = cv2.perspectiveTransform(KPS_M[valid][inl].reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    err = float(np.median(np.linalg.norm(proj - kps[valid][inl], axis=1))) if inl.any() else np.inf
    return H, int(inl.sum()), err


def line_mask(frame: np.ndarray) -> np.ndarray:
    """Thin bright structures (court lines): white top-hat on the grey image."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = max(5, int(round(frame.shape[1] / 160)) | 1)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return (th > 25) & (g > 120)


def line_score(H: np.ndarray, mask: np.ndarray, n: int = 40) -> float:
    h, w = mask.shape
    t = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    hits = tot = 0
    dil = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    for a, b in LINES_M:
        seg = a[None] * (1 - t) + b[None] * t
        p = cv2.perspectiveTransform(seg.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
        inside = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
        p = p[inside].astype(int)
        tot += len(p)
        hits += int(dil[p[:, 1], p[:, 0]].sum())
    return hits / tot if tot else 0.0


def register(frame: np.ndarray, kps: np.ndarray) -> Registration:
    H, n_in, err = fit_homography(kps, frame.shape[1])
    n_kps = int((~np.isnan(kps[:, 0])).sum())
    if H is None:
        return Registration(None, n_kps, 0, np.inf, 0.0)
    return Registration(H, n_kps, n_in, err, line_score(H, line_mask(frame)))


def draw_court(frame: np.ndarray, H: np.ndarray, color=(0, 255, 255)) -> np.ndarray:
    out = frame.copy()
    for a, b in LINES_M:
        p = cv2.perspectiveTransform(np.array([a, b], np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)
        cv2.line(out, tuple(int(v) for v in p[0]), tuple(int(v) for v in p[1]), color, 1, cv2.LINE_AA)
    return out
