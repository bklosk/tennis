"""Synthetic broadcast generator for testing the pipeline end to end without real footage.

A pinhole camera sits high behind the near baseline. The court is rendered through its ground
homography (lines have a physical width, so far lines come out thin), with a net, a crowd band
and a score bug as distractors. Players are flat-shaded figures standing on the ground plane
with coloured joint markers (used by the `marker` pose backend). Each scripted contact gets a
broadband click on the audio track; points are separated by cuts to a crowd shot.

Nothing here is meant to look like television, only to exercise the geometry, timing and
decoding logic with known ground truth.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import wave
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from .court import model as cm
from .court.homography import apply

# ----------------------------------------------------------------------------- camera


@dataclass
class Camera:
    width: int = 960
    height: int = 540
    fov_deg: float = 46.0
    pos: tuple[float, float, float] = (0.0, -28.0, 11.5)
    look_at: tuple[float, float, float] = (0.0, 0.5, 0.0)

    def __post_init__(self) -> None:
        f = 0.5 * self.width / math.tan(math.radians(self.fov_deg) / 2)
        self.K = np.array([[f, 0, self.width / 2], [0, f, self.height / 2], [0, 0, 1.0]])
        c = np.array(self.pos, float)
        fwd = np.array(self.look_at, float) - c
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        self.R = np.stack([right, down, fwd])        # world -> camera
        self.t = -self.R @ c

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        pc = X @ self.R.T + self.t
        uvw = pc @ self.K.T
        return uvw[:, :2] / uvw[:, 2:3]

    @property
    def H(self) -> np.ndarray:
        """Court ground plane (x, y) -> pixels."""
        h = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.t])
        return h / h[2, 2]


# ----------------------------------------------------------------------------- court

COLORS = {
    "inner": (38, 84, 150),      # blue hard court
    "outer": (45, 110, 75),      # green surround
    "wall": (25, 45, 35),
    "line": (238, 238, 238),
    "crowd": (60, 55, 60),
}


def render_background(cam: Camera, ss: int = 2, colors: dict | None = None, seed: int = 0) -> np.ndarray:
    col = {**COLORS, **(colors or {})}
    w, h = cam.width * ss, cam.height * ss
    us, vs = np.meshgrid((np.arange(w) + 0.5) / ss, (np.arange(h) + 0.5) / ss)
    uv = np.stack([us.ravel(), vs.ravel()], 1)
    hinv = np.linalg.inv(cam.H)
    ph = np.hstack([uv, np.ones((len(uv), 1))]) @ hinv.T
    ground = ph[:, 2] > 0
    xy = ph[:, :2] / np.where(np.abs(ph[:, 2:3]) < 1e-12, 1e-12, ph[:, 2:3])
    img = np.empty((len(uv), 3), np.float32)
    rng = np.random.default_rng(seed)
    crowd = np.array(col["crowd"], np.float32) + rng.normal(0, 18, (len(uv), 3)).astype(np.float32)
    img[:] = crowd
    x, y = xy[:, 0], xy[:, 1]
    surround = ground & (np.abs(x) < cm.HALF_DOUBLES + 6.0) & (y > -cm.HALF_LENGTH - 7.5) & (y < cm.HALF_LENGTH + 6.5)
    wall = ground & ~surround & (np.abs(x) < 30) & (y < cm.HALF_LENGTH + 9.0)
    img[wall] = col["wall"]
    img[surround] = col["outer"]
    inner = (np.abs(x) <= cm.HALF_DOUBLES) & (np.abs(y) <= cm.HALF_LENGTH) & ground
    img[inner] = col["inner"]
    # lines with physical widths
    on_line = np.zeros(len(uv), bool)
    for name, (a, b) in cm.LINES.items():
        width = 0.08 if "baseline" in name else 0.05
        a, b = np.array(a), np.array(b)
        ab = b - a
        tt = np.clip(((xy - a) @ ab) / (ab @ ab), 0, 1)
        d = np.linalg.norm(xy - (a + tt[:, None] * ab), axis=1)
        on_line |= ground & (d <= width / 2)
    # centre marks
    for yy in (cm.HALF_LENGTH, -cm.HALF_LENGTH):
        on_line |= ground & (np.abs(x) <= 0.025) & (np.abs(y - yy) <= 0.10) & (np.abs(y) <= cm.HALF_LENGTH)
    img[on_line] = col["line"]
    img = img.reshape(h, w, 3).reshape(cam.height, ss, cam.width, ss, 3).mean(axis=(1, 3))
    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out, "RGBA")
    # net: posts 0.914 m outside the doubles lines
    xp = cm.HALF_DOUBLES + 0.914
    quad = cam.project([(-xp, 0, 0), (xp, 0, 0), (xp, 0, 1.07), (0, 0, 0.914), (-xp, 0, 1.07)])
    draw.polygon([tuple(p) for p in quad], fill=(20, 20, 20, 120))
    tape = cam.project([(-xp, 0, 1.07), (0, 0, 0.914), (xp, 0, 1.07)])
    draw.line([tuple(p) for p in tape], fill=(230, 230, 230, 255), width=max(1, cam.width // 480))
    # score bug
    bw, bh = int(cam.width * 0.22), int(cam.height * 0.08)
    draw.rectangle([20, 20, 20 + bw, 20 + bh], fill=(15, 15, 40, 235))
    draw.text((28, 24), "PLAYER A   6 3 15", fill=(255, 255, 255, 255))
    draw.text((28, 24 + bh // 2), "PLAYER B   4 5 30", fill=(255, 255, 255, 255))
    return np.asarray(out)


def render_crowd(cam: Camera, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, (cam.height // 8 + 1, cam.width // 8 + 1, 3)).astype(np.uint8)
    img = Image.fromarray(base).resize((cam.width, cam.height), Image.NEAREST)
    return np.asarray(img)
