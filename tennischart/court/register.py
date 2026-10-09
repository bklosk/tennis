"""Court registration: automatic detection, refinement against a reference, quality scores.

`detect_court` searches 4-line hypotheses (two lines across the court, two along it) over all
consistent model assignments, scores every candidate homography by how many projected model-line
samples land on line pixels, then refines the winner with every line it can explain.
`refine_court` tracks small pans and zooms from a previous homography. A human click (see
`clicker.py`) can always replace the automatic reference.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from . import model as cm
from .homography import apply, apply_batch, fit_homography, homographies_from_4, line_intersection
from .lines import Line, dilate, hough_lines, line_mask

_SAMPLES, _SAMPLE_IDS = cm.line_samples(0.25)


@dataclass
class CourtFit:
    H: np.ndarray                     # court metres -> image pixels
    quality: float                    # fraction of model-line samples on line pixels
    visible: float                    # fraction of model-line samples inside the image
    lines_used: int = 0
    rms: float = float("nan")
    method: str = "auto"
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.quality >= 0.45 and self.visible >= 0.5

    def to_dict(self) -> dict:
        return {"H": self.H.tolist(), "quality": round(self.quality, 4), "visible": round(self.visible, 4),
                "lines_used": self.lines_used, "rms": None if np.isnan(self.rms) else round(self.rms, 3),
                "method": self.method}

    @classmethod
    def from_dict(cls, d: dict) -> "CourtFit":
        return cls(H=np.array(d["H"], float), quality=d.get("quality", 0.0), visible=d.get("visible", 0.0),
                   lines_used=d.get("lines_used", 0), rms=d.get("rms") or float("nan"),
                   method=d.get("method", "auto"))


def image_to_court(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return apply(np.linalg.inv(H), pts)


def score(H: np.ndarray, mask_d: np.ndarray) -> tuple[float, float]:
    """(fraction of samples on dilated line mask, fraction of samples inside the image)."""
    h, w = mask_d.shape
    uv = apply(H, _SAMPLES)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if not inside.any():
        return 0.0, 0.0
    ui = uv[inside].astype(int)
    hits = mask_d[ui[:, 1], ui[:, 0]]
    # count each image pixel once, so a court collapsed onto one line cannot score well
    uniq = len(np.unique(ui[hits, 1] * w + ui[hits, 0])) if hits.any() else 0
    return float(uniq / len(_SAMPLES)), float(inside.mean())


def _batch_scores(Hs: np.ndarray, mask_d: np.ndarray, samples: np.ndarray) -> np.ndarray:
    h, w = mask_d.shape
    out = np.zeros(len(Hs))
    for i in range(0, len(Hs), 2000):
        uv = apply_batch(Hs[i:i + 2000], samples)
        u, v = uv[..., 0], uv[..., 1]
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h) & np.isfinite(u) & np.isfinite(v)
        ui = np.where(inside, u, 0).astype(int)
        vi = np.where(inside, v, 0).astype(int)
        hit = mask_d[vi, ui] & inside
        key = np.where(hit, vi * w + ui, -1)
        key.sort(axis=1)
        uniq = ((np.diff(key, axis=1) != 0) & (key[:, 1:] >= 0)).sum(axis=1) + (key[:, 0] >= 0)
        out[i:i + 2000] = uniq / samples.shape[0]
    return out


def _plausible(Hs: np.ndarray, w: int, h: int) -> np.ndarray:
    """Reject mirrored, degenerate, behind-camera and absurdly sized courts."""
    corners = np.array([[-cm.HALF_DOUBLES, -cm.HALF_LENGTH], [cm.HALF_DOUBLES, -cm.HALF_LENGTH],
                        [cm.HALF_DOUBLES, cm.HALF_LENGTH], [-cm.HALF_DOUBLES, cm.HALF_LENGTH]])
    ph = np.einsum("bij,nj->bni", Hs, np.hstack([corners, np.ones((4, 1))]))
    wz = ph[..., 2]
    ok = np.isfinite(wz).all(axis=1) & ((wz > 0).all(axis=1) | (wz < 0).all(axis=1))
    uv = ph[..., :2] / np.where(np.abs(wz[..., None]) < 1e-12, 1e-12, wz[..., None])
    nl, nr, fr, fl = uv[:, 0], uv[:, 1], uv[:, 2], uv[:, 3]
    # near baseline below the far baseline, left on the left, near wider than far
    ok &= (nl[:, 1] > fl[:, 1]) & (nr[:, 1] > fr[:, 1])
    ok &= (nr[:, 0] > nl[:, 0]) & (fr[:, 0] > fl[:, 0])
    near_w = nr[:, 0] - nl[:, 0]
    far_w = fr[:, 0] - fl[:, 0]
    ok &= (far_w >= 0.15 * near_w) & (far_w <= 1.02 * near_w)
    ok &= ((nl[:, 1] - fl[:, 1]) > 0.10 * h) & ((nr[:, 1] - fr[:, 1]) > 0.10 * h)
    # convex with one orientation
    quad = np.stack([nl, nr, fr, fl], axis=1)
    e = np.roll(quad, -1, axis=1) - quad
    cross = e[..., 0] * np.roll(e, -1, axis=1)[..., 1] - e[..., 1] * np.roll(e, -1, axis=1)[..., 0]
    ok &= (cross > 0).all(axis=1) | (cross < 0).all(axis=1)
    x, y = uv[..., 0], uv[..., 1]
    area = 0.5 * np.abs((x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1))
    ok &= (area > 0.04 * w * h) & (area < 1.5 * w * h)
    cx, cy = uv.mean(axis=1).T
    ok &= (cx > 0) & (cx < w) & (cy > 0) & (cy < h)
    return ok & np.isfinite(uv).all(axis=(1, 2))


def _split(lines: list[Line]) -> tuple[list[Line], list[Line]]:
    across, along = [], []
    for ln in lines:
        a = ln.angle
        if a < 12 or a > 168:
            across.append(ln)
        elif 25 < a < 155:
            along.append(ln)
    return across, along


def detect_court(rgb: np.ndarray, max_across: int = 9, max_along: int = 8,
                 exclude: list[tuple[int, int, int, int]] | None = None) -> CourtFit | None:
    h, w = rgb.shape[:2]
    mask = line_mask(rgb, exclude=exclude)
    mask_d = dilate(mask, 1)
    lines = hough_lines(mask)
    across, along = _split(lines)
    across = sorted(across, key=lambda l: -l.votes)[:max_across]
    along = sorted(along, key=lambda l: -l.votes)[:max_along]
    if len(across) < 2 or len(along) < 2:
        return None
    across.sort(key=lambda l: l.y_at(w / 2))                    # top to bottom
    yref = np.nanmean([l.y_at(w / 2) for l in across])
    along.sort(key=lambda l: l.x_at(yref))                      # left to right
    model_y = sorted(cm.ACROSS.values(), reverse=True)          # far (top) to near (bottom)
    model_x = sorted(cm.ALONG.values())
    src, dst = [], []
    for (i, j) in itertools.combinations(range(len(across)), 2):
        top, bot = across[i], across[j]
        for (k, m) in itertools.combinations(range(len(along)), 2):
            lft, rgt = along[k], along[m]
            pts = [line_intersection(top.h, lft.h), line_intersection(top.h, rgt.h),
                   line_intersection(bot.h, lft.h), line_intersection(bot.h, rgt.h)]
            if any(not np.isfinite(p).all() for p in pts):
                continue
            for ya, yb in itertools.combinations(model_y, 2):
                for xa, xb in itertools.combinations(model_x, 2):
                    src.append([[xa, ya], [xb, ya], [xa, yb], [xb, yb]])
                    dst.append(pts)
    if not src:
        return None
    Hs = homographies_from_4(np.array(src, float), np.array(dst, float))
    good = np.isfinite(Hs).all(axis=(1, 2))
    Hs = Hs[good]
    if not len(Hs):
        return None
    Hs = Hs[_plausible(Hs, w, h)]
    if not len(Hs):
        return None
    coarse = _SAMPLES[::4]
    sc = _batch_scores(Hs, mask_d, coarse)
    top = np.argsort(-sc)[:25]
    best = None
    for idx in top:
        fit = refine_court(rgb, Hs[idx], mask=mask, lines=lines, tol=8.0)
        if fit is None:
            q, v = score(Hs[idx], mask_d)
            fit = CourtFit(Hs[idx], q, v, method="auto")
        if best is None or fit.quality > best.quality:
            best = fit
    if best is not None:
        best.method = "auto"
    return best


def assign_lines(H: np.ndarray, lines: list[Line], shape: tuple[int, int], tol: float = 8.0,
                 max_angle: float = 4.0) -> dict[str, Line]:
    """Match detected lines to model lines projected through H."""
    h, w = shape
    out: dict[str, tuple[float, Line]] = {}
    model = {**{k: ("across", v) for k, v in cm.ACROSS.items()}, **{k: ("along", v) for k, v in cm.ALONG.items()}}
    for name, (kind, val) in model.items():
        a, b = cm.LINES[name]
        s = np.linspace(0, 1, 40)[:, None]
        seg = np.array(a) + s * (np.array(b) - np.array(a))
        uv = apply(H, seg)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if inside.sum() < 5:
            continue
        uv = uv[inside]
        d = uv[-1] - uv[0]
        ang = float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)
        for ln in lines:
            da = abs(ang - ln.angle)
            da = min(da, 180 - da)
            if da > max_angle:
                continue
            dist = float(np.mean(ln.distance(uv)))
            if dist < tol and (name not in out or dist < out[name][0]):
                out[name] = (dist, ln)
    return {k: v[1] for k, v in out.items()}


def fit_from_lines(assigned: dict[str, Line]) -> tuple[np.ndarray, float, int] | None:
    acr = [(cm.ACROSS[k], ln) for k, ln in assigned.items() if k in cm.ACROSS]
    alo = [(cm.ALONG[k], ln) for k, ln in assigned.items() if k in cm.ALONG]
    if len(acr) < 2 or len(alo) < 2:
        return None
    src, dst = [], []
    for y, la in acr:
        for x, lb in alo:
            p = line_intersection(la.h, lb.h)
            if np.isfinite(p).all():
                src.append((x, y))
                dst.append(p)
    if len(src) < 4:
        return None
    src, dst = np.array(src), np.array(dst)
    H = fit_homography(src, dst)
    rms = float(np.sqrt(np.mean(np.sum((apply(H, src) - dst) ** 2, axis=1))))
    return H, rms, len(acr) + len(alo)


def refine_court(rgb: np.ndarray | None, H0: np.ndarray, mask: np.ndarray | None = None,
                 lines: list[Line] | None = None, tol: float = 12.0, iters: int = 3) -> CourtFit | None:
    """Re-estimate H from lines near the projection of H0 (tracks pans/zooms)."""
    if mask is None:
        assert rgb is not None
        mask = line_mask(rgb)
    if lines is None:
        lines = hough_lines(mask)
    mask_d = dilate(mask, 1)
    H = H0
    rms, used = float("nan"), 0
    ok = False
    for it in range(iters):
        assigned = assign_lines(H, lines, mask.shape, tol=tol if it == 0 else min(tol, 5.0))
        res = fit_from_lines(assigned)
        if res is None:
            break
        Hn, rms, used = res
        if not np.isfinite(Hn).all():
            break
        H, ok = Hn, True
    if not ok:
        return None
    q, v = score(H, mask_d)
    q0, v0 = score(H0, mask_d)
    if q < q0 - 0.02:   # refinement made things clearly worse
        return CourtFit(H0, q0, v0, 0, float("nan"), "refined")
    return CourtFit(H, q, v, used, rms, "refined")


def track_court(rgb: np.ndarray, H_prev: np.ndarray, ref_quality: float = 0.8) -> CourtFit:
    """Register a frame given the previous homography: refine, widen, then full search."""
    mask = line_mask(rgb)
    lines = hough_lines(mask)
    mask_d = dilate(mask, 1)
    q0, v0 = score(H_prev, mask_d)
    cands = [CourtFit(H_prev, q0, v0, 0, float("nan"), "previous")]
    good = 0.9 * ref_quality
    for tol in (12.0, 25.0, 45.0):
        fit = refine_court(None, H_prev, mask=mask, lines=lines, tol=tol)
        if fit is not None:
            cands.append(fit)
        best = max(cands, key=lambda f: f.quality)
        if best.quality >= good and best.lines_used >= 5:
            return best
    full = detect_court(rgb)
    if full is not None:
        cands.append(full)
    return max(cands, key=lambda f: f.quality)


def court_quality(rgb: np.ndarray, H: np.ndarray) -> tuple[float, float]:
    return score(H, dilate(line_mask(rgb), 1))
