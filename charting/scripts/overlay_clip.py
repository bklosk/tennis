"""Render a short overlay clip for spot-checking the model.

    uv run python scripts/overlay_clip.py VIDEO_ID [--point MCP_PT | --start SECONDS] [--seconds 10]

Draws the registered court lines, both tracked players with their court coordinates (meters),
a flash on the hitter at every detected contact with the predicted stroke next to MCP's letter
(green = agrees, red = disagrees), the running rally count against MCP's, and a top-down court map
of player positions and contact points. The original broadcast audio is kept so contacts can be
checked by ear. Writes outputs/clips/VIDEO_ID_*.mp4 (gitignored: clips are broadcast footage and
must not be redistributed).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from uso import scene, truth
from uso.court import LINES_M, draw_court
from uso.paths import OUTPUTS, match_dir
from uso.video import iter_frames, video_path

W_OUT = 1280
NEAR_C, FAR_C, HIT_C = (80, 220, 80), (80, 80, 240), (0, 230, 255)
LETTER = {"F": "FH", "B": "BH"}


def pick_point(shots: pd.DataFrame, pts: pd.DataFrame, seconds: float) -> int:
    """An aligned point whose detected rally matches MCP, 5-9 contacts, that fits in the clip."""
    ok = pts[(pts.rally_length_pred == pts.mcp_rally_length) & pts.rally_length_pred.between(5, 9)]
    for r in ok.itertuples():
        c = shots[(shots.video_point == r.video_point) & (shots.kind.isin(["serve", "rally"]))]
        if len(c) and c.t_video_s.max() - c.t_video_s.min() <= seconds - 2.0:
            return int(r.video_point)
    return int(pts.iloc[len(pts) // 2].video_point)


def court_map(h: int = 360, margin: float = 3.0):
    """Top-down court canvas and a meters -> pixel mapping (far end at the top)."""
    length = 2 * 11.885 + 2 * margin
    scale = h / length
    w = int((10.97 + 2 * margin) * scale)
    img = np.full((h, w, 3), (90, 60, 30), np.uint8)

    def px(x, y):
        return int(round((x + 5.485 + margin) * scale)), int(round((11.885 + margin - y) * scale))

    for a, b in LINES_M:
        cv2.line(img, px(*a), px(*b), (235, 235, 235), 1, cv2.LINE_AA)
    cv2.line(img, px(-5.485, 0), px(5.485, 0), (200, 200, 200), 2)
    return img, px


def label(img, text, org, color, scale=0.6, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def stroke_text(r) -> str:
    if r.kind == "serve":
        return "serve"
    side = LETTER.get(r.stroke_side, "?")
    fam = str(r.stroke_family) if isinstance(r.stroke_family, str) else "?"
    p = r.stroke_p_forehand if r.stroke_side == "F" else 1 - r.stroke_p_forehand
    return f"{side} {fam} ({p:.2f})" if np.isfinite(p) else f"{side} {fam}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("--point", type=int, help="MCP point number")
    ap.add_argument("--start", type=float, help="clip start in video seconds")
    ap.add_argument("--seconds", type=float, default=10.0)
    a = ap.parse_args()
    vid = a.video_id
    d = match_dir(vid)
    shots = pd.read_csv(d / "shots.csv")
    pts = pd.read_csv(d / "points.csv")
    trk = pd.read_parquet(d / "tracks.parquet")
    sc = pd.read_parquet(d / "scene.parquet")
    m = truth.match_row(vid)
    mcp = truth.mcp_points(vid).set_index("Pt")

    if a.start is not None:
        t0 = a.start
        vp_id = None
    else:
        if a.point is not None:
            vp_id = int(pts[pts.mcp_point == a.point].video_point.iloc[0])
        else:
            vp_id = pick_point(shots, pts, a.seconds)
        c = shots[shots.video_point == vp_id]
        t0 = float(c.t_video_s.min()) - 1.5
    t1 = t0 + a.seconds
    win = shots[(shots.t_video_s >= t0) & (shots.t_video_s <= t1) & shots.kind.isin(["serve", "rally"])].sort_values("t_video_s")

    out_dir = OUTPUTS / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"pt{int(mcp_pt)}" if (vp_id is not None and pd.notna(mcp_pt := pts.set_index("video_point").at[vp_id, "mcp_point"])) else f"t{int(t0)}"
    silent = out_dir / f"{vid}_{tag}.silent.mp4"
    final = out_dir / f"{vid}_{tag}.mp4"

    cmap, px = court_map()
    writer, size, fps = None, None, 30.0
    trail: list[tuple] = []
    next_frame_t = t0
    for t, f in iter_frames(video_path(vid), start=t0, end=t1):
        if t + 1e-6 < next_frame_t:  # render at 30 fps whatever the source rate
            continue
        next_frame_t += 1.0 / fps
        H = scene.homography_at(sc, t)
        live = H is not None and sc.iloc[(sc.t - t).abs().argmin()]["main"]
        img = draw_court(f, H) if live else f.copy()
        # contacts: flash the hitter for 0.35 s, keep the latest label on screen
        done = win[win.t_video_s <= t]
        cur = done.iloc[-1] if len(done) else None
        flash = cur.hitter_end if (cur is not None and t - cur.t_video_s <= 0.35 and live) else None
        # tracked players (only meaningful on live views)
        seg_rows = trk[(trk.t - t).abs() <= 0.5]
        for half, col in (("near", NEAR_C), ("far", FAR_C)):
            g = seg_rows[seg_rows.half == half]
            if not live or g.empty:
                continue
            g = trk[(trk.half == half) & (trk.seg == g.seg.iloc[0])]
            ts = g.t.to_numpy()
            box = [np.interp(t, ts, g[k].to_numpy()) for k in ("x1", "y1", "x2", "y2")]
            cx, cy = np.interp(t, ts, g.cx_s.to_numpy()), np.interp(t, ts, g.cy_s.to_numpy())
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), col, 2)
            if half == flash:
                cv2.rectangle(img, (int(box[0]) - 8, int(box[1]) - 8), (int(box[2]) + 8, int(box[3]) + 8), HIT_C, 5)
            label(img, f"{half} ({cx:+.1f}, {cy:+.1f}) m", (int(box[0]), int(box[1]) - 8), col, 0.7)
            cv2.circle(cmap, px(cx, cy), 1, col, -1)
        for r in done.itertuples():
            if np.isfinite(r.hitter_x_m) and (r.t_video_s, r.hitter_end) not in trail:
                trail.append((r.t_video_s, r.hitter_end))
                cv2.circle(cmap, px(r.hitter_x_m, r.hitter_y_m), 5, HIT_C, -1)
                cv2.putText(cmap, str(int(r.shot_no)) if r.shot_no > 0 else "s", px(r.hitter_x_m + 0.4, r.hitter_y_m - 0.4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        # resize, then HUD (constant size regardless of source resolution)
        s = W_OUT / img.shape[1]
        img = cv2.resize(img, (W_OUT, int(round(img.shape[0] * s / 2)) * 2), interpolation=cv2.INTER_AREA)
        hud = img
        # translucent panels behind the text block and the court map, for legibility
        panel = hud.copy()
        cv2.rectangle(panel, (0, 0), (900, 150), (20, 20, 20), -1)
        mh, mw = cmap.shape[:2]
        cv2.rectangle(panel, (hud.shape[1] - mw - 20, 0), (hud.shape[1], mh + 60), (20, 20, 20), -1)
        hud = cv2.addWeighted(panel, 0.55, hud, 0.45, 0)
        title = f"{m.year} US Open {m['round']}  {m.player1} v {m.player2}   video {t:7.2f}s"
        label(hud, title, (14, 30), (255, 255, 255), 0.65)
        pt = None
        if cur is not None and pd.notna(cur.mcp_point):
            pt = int(cur.mcp_point)
        elif vp_id is not None:
            v = pts.set_index("video_point").at[vp_id, "mcp_point"]
            pt = int(v) if pd.notna(v) else None
        if pt is not None and pt in mcp.index:
            mrow = mcp.loc[pt]
            pred_len = pts[pts.mcp_point == pt].rally_length_pred
            pred_len = int(pred_len.iloc[0]) if len(pred_len) else -1
            n_so_far = int(done[(done.mcp_point == pt) & (done.shot_no >= 1)].shot_no.max()) if len(done[(done.mcp_point == pt) & (done.shot_no >= 1)]) else 0
            label(hud, f"MCP point {pt}: {mrow['first']}{(' | ' + mrow['second']) if isinstance(mrow['second'], str) and mrow['second'] else ''}",
                  (14, 58), (220, 220, 220), 0.55)
            label(hud, f"contacts so far {n_so_far}   predicted rally {pred_len}   MCP rally {int(mrow.rally)}",
                  (14, 84), HIT_C, 0.6)
        if cur is not None:
            txt = f"shot {int(cur.shot_no) if cur.shot_no > 0 else 'fault'} {cur.hitter_end} {cur.hitter if isinstance(cur.hitter, str) else ''}: {stroke_text(cur)}"
            ok = None
            if cur.kind == "rally" and isinstance(cur.mcp_side, str):
                ok = cur.mcp_side == cur.stroke_side
                txt += f"   MCP '{cur.mcp_letter}'"
            col = HIT_C if ok is None else ((80, 230, 80) if ok else (60, 60, 255))
            label(hud, txt, (14, 112), col, 0.6)
            if np.isfinite(cur.hitter_x_m):
                label(hud, f"at contact: hitter ({cur.hitter_x_m:+.1f}, {cur.hitter_y_m:+.1f}) m, "
                           f"opponent ({cur.opponent_x_m:+.1f}, {cur.opponent_y_m:+.1f}) m", (14, 138), (220, 220, 220), 0.55)
        if not live:
            label(hud, "not a live view (no tracking)", (14, hud.shape[0] - 20), (180, 180, 180), 0.6)
        # court map in the top-right corner
        mh, mw = cmap.shape[:2]
        hud[10:10 + mh, hud.shape[1] - mw - 10:hud.shape[1] - 10] = cmap
        label(hud, "top-down, far end up", (hud.shape[1] - mw - 10, 10 + mh + 18), (220, 220, 220), 0.45, 1)
        label(hud, "near", (hud.shape[1] - mw - 10, 10 + mh + 40), NEAR_C, 0.45, 1)
        label(hud, "far", (hud.shape[1] - mw + 30, 10 + mh + 40), FAR_C, 0.45, 1)
        label(hud, "contact", (hud.shape[1] - mw + 60, 10 + mh + 40), HIT_C, 0.45, 1)
        if writer is None:
            size = (hud.shape[1], hud.shape[0])
            writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        writer.write(hud)
    writer.release()
    # H.264 for QuickTime, with the broadcast audio for the same window
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-ss", f"{t0:.3f}", "-t", f"{a.seconds:.3f}",
                    "-i", str(video_path(vid)), "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-c:a", "aac", "-b:a", "128k", "-shortest", str(final)], check=True)
    silent.unlink(missing_ok=True)
    print(final)


if __name__ == "__main__":
    main()
