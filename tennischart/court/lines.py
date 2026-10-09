"""Court-line pixels and straight lines (numpy Hough transform)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def luminance(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _shift(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.empty_like(a)
    h, w = a.shape
    xs = slice(max(dx, 0), w + min(dx, 0))
    xd = slice(max(-dx, 0), w + min(-dx, 0))
    ys = slice(max(dy, 0), h + min(dy, 0))
    yd = slice(max(-dy, 0), h + min(-dy, 0))
    out[:] = a
    out[yd, xd] = a[ys, xs]
    return out


def dilate(mask: np.ndarray, r: int = 1) -> np.ndarray:
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx or dy:
                out |= _shift(mask, dx, dy)
    return out


def line_mask(rgb: np.ndarray, tau: int | None = None, diff: float = 15.0, max_sat: float = 80.0,
              exclude: list[tuple[int, int, int, int]] | None = None) -> np.ndarray:
    """Thin bright low-saturation structures: candidate court-line pixels."""
    y = luminance(rgb)
    h, w = y.shape
    tau = tau or max(2, int(round(w / 320)))
    l = y - _shift(y, tau, 0)
    r = y - _shift(y, -tau, 0)
    u = y - _shift(y, 0, tau)
    d = y - _shift(y, 0, -tau)
    thin = ((l > diff) & (r > diff)) | ((u > diff) & (d > diff))
    rgbf = rgb.astype(np.int16)
    sat = rgbf.max(axis=2) - rgbf.min(axis=2)
    min_lum = max(95.0, float(np.median(y)) + 20.0)
    m = thin & (sat < max_sat) & (y > min_lum)
    for x0, y0, x1, y1 in exclude or []:
        m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
    return m


@dataclass
class Line:
    a: float
    b: float
    c: float           # a x + b y + c = 0, (a, b) unit normal
    votes: int
    t0: float = 0.0    # extent along the direction (-b, a)
    t1: float = 0.0

    @property
    def h(self) -> np.ndarray:
        return np.array([self.a, self.b, self.c])

    @property
    def angle(self) -> float:
        """Direction angle in degrees in [0, 180): 0 = horizontal."""
        return float(np.degrees(np.arctan2(self.a, -self.b)) % 180.0)

    def y_at(self, x: float) -> float:
        return -(self.a * x + self.c) / self.b if abs(self.b) > 1e-9 else np.nan

    def x_at(self, y: float) -> float:
        return -(self.b * y + self.c) / self.a if abs(self.a) > 1e-9 else np.nan

    def distance(self, pts: np.ndarray) -> np.ndarray:
        return np.abs(pts @ np.array([self.a, self.b]) + self.c)


def hough_lines(mask: np.ndarray, max_lines: int = 30, n_theta: int = 360, min_votes: int | None = None,
                nms_theta_deg: float = 2.5, nms_rho: float = 10.0, max_points: int = 80000,
                seed: int = 0) -> list[Line]:
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return []
    if len(xs) > max_points:
        idx = np.random.default_rng(seed).choice(len(xs), max_points, replace=False)
        xs, ys = xs[idx], ys[idx]
    h, w = mask.shape
    diag = int(np.ceil(np.hypot(h, w)))
    thetas = np.linspace(0, np.pi, n_theta, endpoint=False)
    cos, sin = np.cos(thetas), np.sin(thetas)
    n_rho = 2 * diag + 1
    acc = np.zeros(n_theta * n_rho, np.int64)
    xf, yf = xs.astype(np.float32), ys.astype(np.float32)
    step = 20000
    for i in range(0, len(xf), step):
        rho = np.rint(np.outer(xf[i:i + step], cos) + np.outer(yf[i:i + step], sin)).astype(np.int64) + diag
        flat = (np.arange(n_theta)[None, :] * n_rho + rho).ravel()
        acc += np.bincount(flat, minlength=n_theta * n_rho)
    acc = acc.reshape(n_theta, n_rho).astype(np.float64)
    min_votes = min_votes or max(15, int(0.04 * max(h, w)))
    lines: list[Line] = []
    dt = max(1, int(round(nms_theta_deg / 180 * n_theta)))
    dr = int(nms_rho)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    for _ in range(max_lines):
        k = int(np.argmax(acc))
        ti, ri = divmod(k, n_rho)
        votes = acc[ti, ri]
        if votes < min_votes:
            break
        # suppress neighbourhood (wrapping theta: theta ~ 0 and ~ 180 are the same line flipped)
        for tt in range(ti - dt, ti + dt + 1):
            if 0 <= tt < n_theta:
                acc[tt, max(0, ri - dr):ri + dr + 1] = 0
            else:
                t2 = tt % n_theta
                r2 = 2 * diag - ri
                acc[t2, max(0, r2 - dr):r2 + dr + 1] = 0
        th = thetas[ti]
        rho = ri - diag
        ln = Line(float(np.cos(th)), float(np.sin(th)), -float(rho), int(votes))
        ln = refine_line(ln, pts) or ln
        if not any(_same_line(ln, other) for other in lines):
            lines.append(ln)
    return lines


def _same_line(a: Line, b: Line, max_angle: float = 2.0, max_dist: float = 5.0) -> bool:
    da = abs(a.angle - b.angle)
    if min(da, 180 - da) > max_angle:
        return False
    tm = 0.5 * (a.t0 + a.t1)
    mid = np.array([-a.c * a.a, -a.c * a.b]) + tm * np.array([-a.b, a.a])
    return float(b.distance(mid[None, :])[0]) < max_dist


def refine_line(ln: Line, pts: np.ndarray, tol: float = 2.0) -> Line | None:
    d = ln.distance(pts)
    sel = pts[d <= tol]
    if len(sel) < 8:
        return None
    c = sel.mean(axis=0)
    _, _, vt = np.linalg.svd(sel - c, full_matrices=False)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]])
    a, b = normal
    cc = -normal @ c
    t = sel @ np.array([-b, a])
    return Line(float(a), float(b), float(cc), int(len(sel)), float(t.min()), float(t.max()))
