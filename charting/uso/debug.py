"""Contact sheets for eyeballing stage outputs (a minimal point browser)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from uso import scene as scene_mod
from uso.court import draw_court
from uso.paths import match_dir
from uso.video import frames_at, video_path


def sheet(video_id: str, times: list[float], out: Path, trk: pd.DataFrame | None = None,
          people: pd.DataFrame | None = None, labels: list[str] | None = None, cols: int = 4,
          tile_w: int = 640) -> Path:
    sc = pd.read_parquet(match_dir(video_id) / "scene.parquet")
    got = frames_at(video_path(video_id), times)
    tiles = []
    for i, t in enumerate(times):
        ft, f = got[t]
        im = f.copy()
        H = scene_mod.homography_at(sc, ft)
        if H is not None:
            im = draw_court(im, H)
        if people is not None and not people.empty:
            p = people[(people.t - ft).abs() < 0.13]
            for r in p.itertuples():
                cv2.rectangle(im, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (160, 160, 160), 1)
        if trk is not None and not trk.empty:
            near = trk[(trk.t - ft).abs() < 0.13]
            for r in near.itertuples():
                c = (0, 255, 0) if r.half == "near" else (0, 0, 255)
                cv2.rectangle(im, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), c, 2)
                cv2.putText(im, f"{r.cx:.1f},{r.cy:.1f}", (int(r.x1), int(r.y1) - 4), 0, 0.6, c, 2)
        txt = f"{ft:.2f}s" + (f" {labels[i]}" if labels else "")
        cv2.putText(im, txt, (10, 30), 0, 1.0, (0, 255, 255), 2)
        s = tile_w / im.shape[1]
        tiles.append(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    return out


def onset_sheet(video_id: str, t0: float, t1: float, out: Path, z_min: float = 8.0, size: int = 160) -> Path:
    """For each audio onset in [t0, t1] with z >= z_min: near and far player crops side by side,
    labelled with time and z. Shows which onsets line up with a swing."""
    from uso.crops import cut, square

    d = match_dir(video_id)
    on = pd.read_parquet(d / "audio_onsets.parquet")
    trk = pd.read_parquet(d / "tracks.parquet")
    on = on[(on.t >= t0) & (on.t <= t1) & (on.z >= z_min)]
    got = frames_at(video_path(video_id), on.t.tolist())
    tiles = []
    for r in on.itertuples():
        ft, f = got[r.t]
        pair = []
        for half in ("near", "far"):
            g = trk[trk.half == half]
            g = g.iloc[(g.t - ft).abs().argsort()[:1]]
            if g.empty or abs(g.t.iloc[0] - ft) > 0.5:
                pair.append(np.zeros((size, size, 3), np.uint8))
                continue
            b = g[["x1", "y1", "x2", "y2"]].iloc[0].to_numpy()
            x0, y0, s = square(b, f.shape[1], f.shape[0], scale=1.6)
            pair.append(cut(f, x0, y0, s, size))
        tile = np.hstack(pair)
        cv2.putText(tile, f"{r.t - t0:.2f}s z{r.z:.0f}", (4, 16), 0, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    return out
