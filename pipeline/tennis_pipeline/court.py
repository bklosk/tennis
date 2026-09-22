"""Court keypoints, homography, and conversion to court meters.

Court frame: origin at the center of the net, x to the right as seen from the main
camera, y toward the camera (near baseline at y=+11.885 m, far baseline at y=-11.885 m).
"""
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from . import tracknet

# Reference keypoints from yastrebksv/TennisCourtDetector (units ~1 cm).
REF_KPS = np.array([
    (286, 561), (1379, 561), (286, 2935), (1379, 2935),
    (423, 561), (423, 2935), (1242, 561), (1242, 2935),
    (423, 1110), (1242, 1110), (423, 2386), (1242, 2386),
    (832, 1110), (832, 2386),
], np.float32)
REF_CENTER = (832.5, 1748.0)
REF_PER_M = (1093 / 10.97, 2374 / 23.77)
NO_REFINE = {8, 9, 12}

HALF_LENGTH = 11.885
SERVICE_LINE = 6.40
SINGLES_HALF_WIDTH = 4.115
DOUBLES_HALF_WIDTH = 5.485


def ref_to_m(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64)
    return np.stack([(pts[..., 0] - REF_CENTER[0]) / REF_PER_M[0],
                     (pts[..., 1] - REF_CENTER[1]) / REF_PER_M[1]], -1)


def m_to_ref(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64)
    return np.stack([pts[..., 0] * REF_PER_M[0] + REF_CENTER[0],
                     pts[..., 1] * REF_PER_M[1] + REF_CENTER[1]], -1)


@dataclass
class Calibration:
    court_to_img: np.ndarray  # 3x3, reference units -> image pixels (1280x720)
    img_to_court: np.ndarray
    reproj_px: float
    n_points: int

    def to_court_m(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, np.float64).reshape(-1, 1, 2)
        ref = cv2.perspectiveTransform(xy, self.img_to_court).reshape(-1, 2)
        return ref_to_m(ref)

    def to_img(self, xy_m: np.ndarray) -> np.ndarray:
        ref = m_to_ref(np.asarray(xy_m, np.float64)).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(ref, self.court_to_img).reshape(-1, 2)


def _intersect(l1, l2):
    x1, y1, x2, y2 = map(float, l1)
    x3, y3, x4, y4 = map(float, l2)
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    a, b = x1 * y2 - y1 * x2, x3 * y4 - y3 * x4
    return (a * (x3 - x4) - (x1 - x2) * b) / den, (a * (y3 - y4) - (y1 - y2) * b) / den


def _merge_lines(lines):
    lines = sorted(lines.tolist(), key=lambda l: l[0])
    used = [False] * len(lines)
    merged = []
    for i, line in enumerate(lines):
        if used[i]:
            continue
        line = np.array(line, float)
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            other = np.array(lines[j], float)
            if np.hypot(*(line[:2] - other[:2])) < 20 and np.hypot(*(line[2:] - other[2:])) < 20:
                line = (line + other) / 2
                used[j] = True
        merged.append(line)
    return merged


def _refine(img: np.ndarray, x: float, y: float, crop: int = 40) -> tuple[float, float]:
    h, w = img.shape[:2]
    x0, x1 = max(int(x) - crop, 0), min(w, int(x) + crop)
    y0, y1 = max(int(y) - crop, 0), min(h, int(y) + crop)
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        return x, y
    gray = cv2.threshold(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), 155, 255, cv2.THRESH_BINARY)[1]
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=30)
    if lines is None or len(lines) < 2:
        return x, y
    lines = _merge_lines(lines.reshape(-1, 4))
    if len(lines) != 2:
        return x, y
    p = _intersect(lines[0], lines[1])
    if p is None or not (0 < p[0] < patch.shape[1] and 0 < p[1] < patch.shape[0]):
        return x, y
    return x0 + p[0], y0 + p[1]


class CourtDetector:
    def __init__(self, dev: torch.device | None = None):
        self.dev = dev or tracknet.device()
        self.dtype = torch.float32 if self.dev.type == "cpu" else torch.float16
        self.model = tracknet.load("court", self.dev).to(self.dtype).to(memory_format=torch.channels_last)

    @torch.no_grad()
    def keypoints(self, frames_bgr: list[np.ndarray]) -> list[list]:
        """Detect 14 court keypoints (1280x720 pixel coords or None) per frame."""
        n = len(frames_bgr)
        small = np.stack([cv2.resize(f, (640, 360)) for f in frames_bgr])
        padded = 1 << (n - 1).bit_length()  # power-of-two batches limit cuDNN re-tuning to a few shapes
        if padded > n:
            small = np.concatenate([small, np.repeat(small[-1:], padded - n, 0)])
        inp = torch.from_numpy(small).to(self.dev).permute(0, 3, 1, 2).to(self.dtype) / 255
        heat = torch.sigmoid(self.model(inp.contiguous(memory_format=torch.channels_last)).float())[:n, :14]
        present = (heat.amax((2, 3)) >= 0.67).cpu().numpy()
        maps_present = heat[torch.from_numpy(present).to(heat.device)].cpu().numpy()  # only copy detected maps
        results, j = [], 0
        for frame, flags in zip(frames_bgr, present):
            pts = []
            for k, found in enumerate(flags):
                if not found:
                    pts.append(None)
                    continue
                hm = maps_present[j]
                j += 1
                mask = (hm > 0.67).astype(np.uint8)
                _, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
                peak = np.unravel_index(hm.argmax(), hm.shape)
                cx, cy = cents[labels[peak]]
                x, y = cx * 2, cy * 2
                if k not in NO_REFINE:
                    x, y = _refine(frame, x, y)
                pts.append((float(x), float(y)))
            results.append(pts)
        return results

    def calibrate(self, frames_bgr: list[np.ndarray]) -> list[Calibration | None]:
        return [fit_homography(p) for p in self.keypoints(frames_bgr)]


def fit_homography(points: list) -> Calibration | None:
    idx = [i for i, p in enumerate(points) if p is not None]
    if len(idx) < 6:
        return None
    src = REF_KPS[idx]
    dst = np.array([points[i] for i in idx], np.float32)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 8.0)
    if H is None or inliers.sum() < 6:
        return None
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - dst, axis=1)[inliers.ravel() > 0]
    # Reject degenerate fits: the full court must project to a plausible on-screen quad.
    corners = cv2.perspectiveTransform(REF_KPS[[0, 1, 3, 2]].reshape(-1, 1, 2), H).reshape(-1, 2)
    area = cv2.contourArea(corners.astype(np.float32))
    if not (40_000 < area < 700_000) or corners[0, 1] > corners[2, 1]:
        return None
    return Calibration(H, np.linalg.inv(H), float(np.median(err)), int(inliers.sum()))
