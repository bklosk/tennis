"""Stage 7: analysis-ready exports, court maps, QA contact sheets and overlay clips."""
import json
import subprocess

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import video  # noqa: E402
from .cli import video_path  # noqa: E402
from .court import DOUBLES_HALF_WIDTH, HALF_LENGTH, SERVICE_LINE, SINGLES_HALF_WIDTH  # noqa: E402
from .paths import OUTPUTS, match_dir  # noqa: E402
from .process import load_tracks  # noqa: E402

SHOT_COLUMNS = [
    "video_id", "point_number", "vp", "shot_num", "t", "player", "side", "hand", "stroke", "is_serve",
    "is_volley", "serve_in", "hitter_x_m", "hitter_y_m", "bounce_x_m", "bounce_y_m", "bounce_t",
    "avg_speed_kmh", "stroke_source", "stroke_vlm", "stroke_model", "stroke_rule", "audio_confirmed",
    "chunk_id", "frame", "hit_id",
]


def export(video_id: str) -> pd.DataFrame:
    out_dir = match_dir(video_id)
    shots = pd.read_parquet(out_dir / "shots_strokes.parquet").sort_values("t")
    for col in ("bounce_x_m", "bounce_y_m", "bounce_t"):  # absent when no shot's bounce was found
        if col not in shots:
            shots[col] = np.nan
    shots["shot_num"] = shots.groupby("vp").cumcount() + 1
    shots["video_id"] = video_id
    # Average ground speed from contact (audio-snapped time) to the first bounce.
    dist = np.hypot(shots.bounce_x_m - shots.hitter_x_m, shots.bounce_y_m - shots.hitter_y_m)
    dt = shots.bounce_t - shots.t
    speed = dist / dt * 3.6
    shots["avg_speed_kmh"] = speed.where((dt > 0.15) & (speed < 260))
    cols = [c for c in SHOT_COLUMNS if c in shots]
    shots[cols].to_csv(out_dir / "shots_all.csv", index=False)
    # Analysis table: only shots inside points matched to the official record (named players).
    shots[shots.point_number.notna()][cols].to_csv(out_dir / "shots.csv", index=False)

    abbrev = {"serve": "S", "forehand": "FH", "backhand": "BH", "overhead": "OH", "unknown": "?"}
    seq = shots.groupby("vp").apply(
        lambda g: " > ".join(f"{abbrev.get(r.stroke, '?')}{'-V' if r.is_volley else ''}:"
                             f"{r.player.split()[-1] if isinstance(r.player, str) else r.side}"
                             for r in g.itertuples()), include_groups=False)
    points = pd.read_csv(out_dir / "points.csv")
    points["shot_sequence"] = points.vp.map(seq)
    points.to_csv(out_dir / "points.csv", index=False)

    ball = pd.read_parquet(out_dir / "ball.parquet")
    spans = points[["vp", "point_number", "t_start", "t_end"]].dropna(subset=["point_number"])
    ball["point_number"] = np.nan
    for r in spans.itertuples():
        sel = (ball.t >= r.t_start - 1.0) & (ball.t <= r.t_end + 1.5)
        ball.loc[sel, "point_number"] = r.point_number
    ball.to_csv(out_dir / "ball_trajectory.csv", index=False)

    players = pd.read_parquet(out_dir / "players.parquet")
    players["point_number"] = np.nan
    players["player"] = None
    for r in points.dropna(subset=["point_number"]).itertuples():
        sel = (players.t >= r.t_start - 1.0) & (players.t <= r.t_end + 1.5)
        players.loc[sel, "point_number"] = r.point_number
        players.loc[sel & (players.side == "near"), "player"] = r.near_player
        players.loc[sel & (players.side == "far"), "player"] = r.far_player
    players.to_csv(out_dir / "player_positions.csv", index=False)
    return shots


def draw_court(ax):
    L, W, w = HALF_LENGTH, DOUBLES_HALF_WIDTH, SINGLES_HALF_WIDTH
    ax.add_patch(plt.Rectangle((-W, -L), 2 * W, 2 * L, fill=True, color="#3b6ea5", alpha=0.25, lw=0))
    lines = [((-W, -L), (W, -L)), ((-W, L), (W, L)), ((-W, -L), (-W, L)), ((W, -L), (W, L)),
             ((-w, -L), (-w, L)), ((w, -L), (w, L)), ((-w, -SERVICE_LINE), (w, -SERVICE_LINE)),
             ((-w, SERVICE_LINE), (w, SERVICE_LINE)), ((0, -SERVICE_LINE), (0, SERVICE_LINE)),
             ((-W - 0.9, 0), (W + 0.9, 0))]
    for (x0, y0), (x1, y1) in lines:
        ax.plot([x0, x1], [y0, y1], color="black", lw=1)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-17, 17)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def court_maps(video_id: str, shots: pd.DataFrame, points: pd.DataFrame):
    """Serve bounce map and groundstroke bounce map, both normalised so the hitter is at the bottom."""
    s = shots.copy()
    # Near hitters stand at y > 0; matplotlib draws positive y upward, so negate to put them at the bottom.
    flip = np.where(s.side == "far", 1.0, -1.0)
    s["bx"], s["by"] = s.bounce_x_m * flip, s.bounce_y_m * flip
    s["hx"], s["hy"] = s.hitter_x_m * flip, s.hitter_y_m * flip
    players = [p for p in s.player.dropna().unique()]
    fig, axes = plt.subplots(2, max(len(players), 1), figsize=(4.2 * max(len(players), 1), 11))
    axes = np.atleast_2d(axes).reshape(2, -1)
    for k, name in enumerate(players):
        ps = s[s.player == name]
        ax = axes[0, k]
        draw_court(ax)
        sv = ps[ps.is_serve & ps.bx.notna()]
        ax.scatter(sv.bx, sv.by, c=np.where(sv.serve_in.fillna(False).astype(bool), "tab:green", "tab:red"), s=14)
        ax.set_title(f"{name}\nserve bounces (green=in)", fontsize=9)
        ax = axes[1, k]
        draw_court(ax)
        for stroke, color in (("forehand", "tab:orange"), ("backhand", "tab:purple"), ("overhead", "black")):
            g = ps[(ps.stroke == stroke) & ps.bx.notna()]
            ax.scatter(g.bx, g.by, c=color, s=10, label=f"{stroke} ({len(g)})")
        hh = ps[~ps.is_serve & ps.hy.notna()]
        ax.scatter(hh.hx, hh.hy, c="gray", s=4, alpha=0.4, label="contact position")
        ax.legend(fontsize=6, loc="lower left")
        ax.set_title(f"{name}\nshot bounces by stroke", fontsize=9)
    fig.suptitle(f"{video_id}: hitter always at the bottom of the court", fontsize=10)
    fig.tight_layout()
    fig.savefig(match_dir(video_id) / "court_maps.png", dpi=90)
    plt.close(fig)


def serve_speed_plot(video_ids: list[str]):
    rows = []
    for vid in video_ids:
        shots = pd.read_csv(match_dir(vid) / "shots.csv")
        pts = pd.read_csv(match_dir(vid) / "points.csv")
        sv = shots[shots.is_serve & shots.avg_speed_kmh.notna()].copy()
        last = sv.sort_values("t").groupby("vp").tail(1)  # the serve that started the rally
        m = last.merge(pts[["vp", "serve_speed_kmh"]], on="vp")
        m = m[m.serve_speed_kmh > 0]
        m["video_id"] = vid
        rows.append(m)
    d = pd.concat(rows) if rows else pd.DataFrame()
    if d.empty:
        return {}
    ratio = float(np.median(d.serve_speed_kmh / d.avg_speed_kmh))
    d["video_est_kmh"] = d.avg_speed_kmh * ratio
    err = (d.video_est_kmh - d.serve_speed_kmh).abs()
    fig, ax = plt.subplots(figsize=(5, 5))
    for vid, g in d.groupby("video_id"):
        ax.scatter(g.serve_speed_kmh, g.video_est_kmh, s=10, label=vid)
    lim = [80, 230]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("Official serve speed (km/h)")
    ax.set_ylabel("Video estimate (km/h), calibrated")
    ax.set_title(f"Serve speed: video vs radar (MAE {err.mean():.1f} km/h)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "serve_speed_calibration.png", dpi=90)
    plt.close(fig)
    return {"n": int(len(d)), "calibration_ratio": ratio, "mae_kmh": float(err.mean()),
            "corr": float(np.corrcoef(d.serve_speed_kmh, d.avg_speed_kmh)[0, 1])}


def hit_sheet(video_id: str, shots: pd.DataFrame, n: int = 12, seed: int = 0):
    """Contact sheet of hitter crop strips with the predicted labels, for eyeballing accuracy."""
    s = shots[shots.crop_path.notna()] if "crop_path" in shots else shots.iloc[0:0]
    s = s.sample(min(n, len(s)), random_state=seed)
    tiles = []
    for r in s.itertuples():
        img = cv2.imread(r.crop_path)
        if img is None:
            continue
        label = f"{r.player or r.side} {r.stroke}{' volley' if r.is_volley else ''}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(img, label[:60], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(img)
    if not tiles:
        return
    rows = [np.concatenate(tiles[i:i + 2], 1) if len(tiles[i:i + 2]) == 2
            else np.concatenate([tiles[i], np.zeros_like(tiles[i])], 1) for i in range(0, len(tiles), 2)]
    cv2.imwrite(str(match_dir(video_id) / "hit_sheet.jpg"), np.concatenate(rows, 0))


def overlay_clip(video_id: str, vp: int, out_name: str | None = None):
    """Render one point with ball trail, players, hits, and a mini court map."""
    out_dir = match_dir(video_id)
    points = pd.read_csv(out_dir / "points.csv")
    shots = pd.read_csv(out_dir / "shots.csv")
    p = points[points.vp == vp].iloc[0]
    ps = shots[shots.vp == vp]
    chunk_id = ps.chunk_id.iloc[0]
    tr = load_tracks(out_dir / "tracks" / f"{chunk_id}.npz")
    ball = pd.read_parquet(out_dir / "ball.parquet")
    ball = ball[ball.chunk_id == chunk_id].set_index("frame")
    t0, t1 = p.t_start - 1.0, p.t_end + 1.5
    frames = video.read_clip(video_path(video_id), t0, t1 - t0)
    out_path = out_dir / (out_name or f"point_{int(p.point_number) if not pd.isna(p.point_number) else vp}.mp4")
    h, w = frames.shape[1:3]
    proc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                             "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out_path)],
                            stdin=subprocess.PIPE)
    mini_scale = 7.0
    for k, frame in enumerate(frames):
        img = frame.copy()
        f = int(round((t0 + k / 30 - tr.t0) * 30))
        if 0 <= f < tr.n:
            for side, color in (("near", (0, 200, 0)), ("far", (200, 0, 200))):
                box = tr.players.get(side, np.full((tr.n, 4), np.nan))[f]
                if not np.isnan(box).any():
                    cv2.rectangle(img, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), color, 2)
            for j in range(max(0, f - 8), f + 1):
                if j in ball.index and not np.isnan(ball.at[j, "ball_x_px"]):
                    cv2.circle(img, (int(ball.at[j, "ball_x_px"]), int(ball.at[j, "ball_y_px"])), 3 if j < f else 6,
                               (0, 255, 255), -1)
        recent = ps[(ps.frame <= f) & (ps.frame > f - 25)]
        for r in recent.itertuples():
            label = f"{r.player if isinstance(r.player, str) else r.side}: {r.stroke}"
            cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        mini = np.full((int(34 * mini_scale), int(16 * mini_scale), 3), (60, 110, 40), np.uint8)

        def to_mini(x, y):
            return int((x + 8) * mini_scale), int((y + 17) * mini_scale)

        cv2.rectangle(mini, to_mini(-DOUBLES_HALF_WIDTH, -HALF_LENGTH), to_mini(DOUBLES_HALF_WIDTH, HALF_LENGTH), (255, 255, 255), 1)
        cv2.line(mini, to_mini(-6.4, 0), to_mini(6.4, 0), (255, 255, 255), 1)
        for yy in (-SERVICE_LINE, SERVICE_LINE):
            cv2.line(mini, to_mini(-SINGLES_HALF_WIDTH, yy), to_mini(SINGLES_HALF_WIDTH, yy), (255, 255, 255), 1)
        for r in ps[(ps.frame <= f) & ps.bounce_x_m.notna()].itertuples():
            cv2.circle(mini, to_mini(r.bounce_x_m, r.bounce_y_m), 4, (0, 255, 255), -1)
        if 0 <= f < tr.n and tr.calibs[f] is not None:
            for side, color in (("near", (0, 200, 0)), ("far", (200, 0, 200))):
                box = tr.players.get(side, np.full((tr.n, 4), np.nan))[f]
                if not np.isnan(box).any():
                    cx, cy = tr.calibs[f].to_court_m(np.array([[(box[0] + box[2]) / 2, box[3]]]))[0]
                    cv2.circle(mini, to_mini(cx, cy), 6, color, -1)
        img[20:20 + mini.shape[0], w - 20 - mini.shape[1]:w - 20] = mini
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()
    return out_path


def match_report(video_id: str, clip: bool = True) -> dict:
    """Exports, court maps, contact sheet and (if the video is available) the longest-rally clip."""
    shots = export(video_id)
    points = pd.read_csv(match_dir(video_id) / "points.csv")
    court_maps(video_id, shots, points)
    hit_sheet(video_id, shots)
    aligned = points.dropna(subset=["point_number"])
    long_rally = aligned.sort_values("n_shots", ascending=False).head(1)
    if clip and len(long_rally) and video_path(video_id).exists():
        overlay_clip(video_id, int(long_rally.vp.iloc[0]))
    summary = json.loads((match_dir(video_id) / "align_summary.json").read_text())
    summary["shots"] = int(len(shots))
    summary["stroke_counts"] = shots.stroke.value_counts().to_dict()
    return summary


def run(video_ids: list[str]):
    summary = {vid: match_report(vid) for vid in video_ids}
    summary["serve_speed"] = serve_speed_plot(video_ids)
    (OUTPUTS / "pilot_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
