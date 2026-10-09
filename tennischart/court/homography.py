"""Homography estimation (normalised DLT, RANSAC) and projection, numpy only."""

from __future__ import annotations

import numpy as np


def _normalizer(p: np.ndarray) -> np.ndarray:
    c = p.mean(axis=0)
    d = np.sqrt(((p - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / d if d > 0 else 1.0
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


def fit_homography(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """H with dst ~ H @ src (both (N, 2), N >= 4)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if len(src) < 4:
        raise ValueError("need at least 4 correspondences")
    ts, td = _normalizer(src), _normalizer(dst)
    s = apply(ts, src)
    d = apply(td, dst)
    n = len(s)
    a = np.zeros((2 * n, 9))
    x, y = s[:, 0], s[:, 1]
    u, v = d[:, 0], d[:, 1]
    a[0::2, 0:3] = np.stack([-x, -y, -np.ones(n)], 1)
    a[0::2, 6:9] = np.stack([u * x, u * y, u], 1)
    a[1::2, 3:6] = np.stack([-x, -y, -np.ones(n)], 1)
    a[1::2, 6:9] = np.stack([v * x, v * y, v], 1)
    if weights is not None:
        w = np.repeat(np.sqrt(np.asarray(weights, float)), 2)
        a = a * w[:, None]
    _, _, vt = np.linalg.svd(a)
    hn = vt[-1].reshape(3, 3)
    h = np.linalg.inv(td) @ hn @ ts
    return h / h[2, 2]


def apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, float))
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    w = ph[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return ph[:, :2] / w


def ransac_homography(src: np.ndarray, dst: np.ndarray, thresh: float = 4.0, iters: int = 500,
                      seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Robust H and inlier mask; falls back to a plain fit for <= 5 points."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    if n <= 5:
        h = fit_homography(src, dst)
        return h, np.ones(n, bool)
    rng = np.random.default_rng(seed)
    best, best_in = None, np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, 4, replace=False)
        try:
            h = fit_homography(src[idx], dst[idx])
        except (np.linalg.LinAlgError, ValueError):
            continue
        err = np.linalg.norm(apply(h, src) - dst, axis=1)
        inl = err < thresh
        if inl.sum() > best_in.sum():
            best, best_in = h, inl
    if best is None or best_in.sum() < 4:
        return fit_homography(src, dst), np.ones(n, bool)
    return fit_homography(src[best_in], dst[best_in]), best_in


def homographies_from_4(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Batch exact homographies from 4-point sets: src, dst (B, 4, 2) -> (B, 3, 3)."""
    b = src.shape[0]
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    a = np.zeros((b, 8, 8))
    rhs = np.zeros((b, 8))
    a[:, 0::2, 0] = x
    a[:, 0::2, 1] = y
    a[:, 0::2, 2] = 1
    a[:, 0::2, 6] = -u * x
    a[:, 0::2, 7] = -u * y
    a[:, 1::2, 3] = x
    a[:, 1::2, 4] = y
    a[:, 1::2, 5] = 1
    a[:, 1::2, 6] = -v * x
    a[:, 1::2, 7] = -v * y
    rhs[:, 0::2] = u
    rhs[:, 1::2] = v
    ok = np.abs(np.linalg.det(a)) > 1e-9
    h = np.full((b, 9), np.nan)
    if ok.any():
        sol = np.linalg.solve(a[ok], rhs[ok][..., None])[..., 0]
        h[ok, :8] = sol
        h[ok, 8] = 1.0
    return h.reshape(b, 3, 3)


def apply_batch(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(B, 3, 3) x (N, 2) -> (B, N, 2)."""
    ph = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    out = np.einsum("bij,nj->bni", h, ph)
    w = out[..., 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return out[..., :2] / w


def line_intersection(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """Intersection of homogeneous lines (a, b, c) with ax + by + c = 0."""
    p = np.cross(l1, l2)
    return p[:2] / p[2] if abs(p[2]) > 1e-12 else np.array([np.nan, np.nan])
