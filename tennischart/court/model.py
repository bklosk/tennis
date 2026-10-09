"""ITF court geometry in metres.

Frame: origin on the ground at the centre of the net. x runs across the court and is positive to
the right as seen from the camera end; y runs along the court and is positive toward the far
baseline. The near (camera) baseline is y = -11.885, the far baseline y = +11.885.
"left"/"right" in landmark names are as seen from the camera.
"""

from __future__ import annotations

import numpy as np

HALF_LENGTH = 11.885
HALF_SINGLES = 4.115
HALF_DOUBLES = 5.485
SERVICE = 6.40
NET_HEIGHT_CENTER = 0.914

LANDMARKS: dict[str, tuple[float, float]] = {
    "far_left_doubles": (-HALF_DOUBLES, HALF_LENGTH),
    "far_left_singles": (-HALF_SINGLES, HALF_LENGTH),
    "far_center_mark": (0.0, HALF_LENGTH),
    "far_right_singles": (HALF_SINGLES, HALF_LENGTH),
    "far_right_doubles": (HALF_DOUBLES, HALF_LENGTH),
    "far_left_service": (-HALF_SINGLES, SERVICE),
    "far_T": (0.0, SERVICE),
    "far_right_service": (HALF_SINGLES, SERVICE),
    "near_left_service": (-HALF_SINGLES, -SERVICE),
    "near_T": (0.0, -SERVICE),
    "near_right_service": (HALF_SINGLES, -SERVICE),
    "near_left_doubles": (-HALF_DOUBLES, -HALF_LENGTH),
    "near_left_singles": (-HALF_SINGLES, -HALF_LENGTH),
    "near_center_mark": (0.0, -HALF_LENGTH),
    "near_right_singles": (HALF_SINGLES, -HALF_LENGTH),
    "near_right_doubles": (HALF_DOUBLES, -HALF_LENGTH),
}

# Ground lines as segments ((x0, y0), (x1, y1)). Net and posts are not on the ground plane.
LINES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "far_baseline": ((-HALF_DOUBLES, HALF_LENGTH), (HALF_DOUBLES, HALF_LENGTH)),
    "near_baseline": ((-HALF_DOUBLES, -HALF_LENGTH), (HALF_DOUBLES, -HALF_LENGTH)),
    "far_service": ((-HALF_SINGLES, SERVICE), (HALF_SINGLES, SERVICE)),
    "near_service": ((-HALF_SINGLES, -SERVICE), (HALF_SINGLES, -SERVICE)),
    "left_doubles": ((-HALF_DOUBLES, -HALF_LENGTH), (-HALF_DOUBLES, HALF_LENGTH)),
    "right_doubles": ((HALF_DOUBLES, -HALF_LENGTH), (HALF_DOUBLES, HALF_LENGTH)),
    "left_singles": ((-HALF_SINGLES, -HALF_LENGTH), (-HALF_SINGLES, HALF_LENGTH)),
    "right_singles": ((HALF_SINGLES, -HALF_LENGTH), (HALF_SINGLES, HALF_LENGTH)),
    "center_service": ((0.0, -SERVICE), (0.0, SERVICE)),
}

# Lines of constant y (across the court) and constant x (along it), for registration.
ACROSS = {"far_baseline": HALF_LENGTH, "far_service": SERVICE, "near_service": -SERVICE,
          "near_baseline": -HALF_LENGTH}
ALONG = {"left_doubles": -HALF_DOUBLES, "left_singles": -HALF_SINGLES, "center_service": 0.0,
         "right_singles": HALF_SINGLES, "right_doubles": HALF_DOUBLES}


def line_samples(step: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Points along every ground line (N, 2) and the index of the line each belongs to."""
    pts, ids = [], []
    for k, (a, b) in enumerate(LINES.values()):
        a, b = np.array(a), np.array(b)
        n = max(2, int(np.linalg.norm(b - a) / step) + 1)
        s = np.linspace(0, 1, n)[:, None]
        pts.append(a + s * (b - a))
        ids.append(np.full(n, k))
    return np.vstack(pts), np.concatenate(ids)


def in_play_area(xy: np.ndarray, margin_x: float = 4.0, back: float = 8.0) -> np.ndarray:
    """Points inside the extended playing area around the court (where players can be)."""
    xy = np.atleast_2d(xy)
    return (np.abs(xy[:, 0]) <= HALF_DOUBLES + margin_x) & (np.abs(xy[:, 1]) <= HALF_LENGTH + back)
