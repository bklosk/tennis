"""Ball-position labels for fine-tuning TrackNet on US Open footage.

    sample  pick ~2,000 frames across processed matches, weighted toward the pretrained model's
            failure modes, and pre-label them from the cached tracks
    export  decode each labeled frame plus the two frames before it (the TrackNet input)
    review  click through the pre-labels: accept, correct, or mark the ball not visible

Labels live in `eval/ball_labels.csv` (versioned, like the stroke gold set); frames are
regenerated from the videos into `.cache/ball_frames/`. Coordinates are 1280x720 pixels.

Buckets (default share of each match's quota):
    miss    in a rally, pretrained TrackNet found no ball            35%
    serve   -0.5 s .. +0.7 s around a detected serve                 20%
    bounce  within 3 frames of a detected bounce                     15%
    far     in a rally with the ball over the far half               15%
    random  any main-camera frame (includes between-point negatives) 15%
Matches that have scene segments but no tracks yet contribute `untracked` random frames with
no pre-label, which is how older eras get into the set before they are processed.
"""
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import video
from .paths import CACHE, match_dir
from .track import FPS

LABELS_PATH = Path(__file__).resolve().parents[1] / "eval" / "ball_labels.csv"
FRAMES_DIR = CACHE / "ball_frames"
FRACTIONS = {"miss": 0.35, "serve": 0.20, "bounce": 0.15, "far": 0.15, "random": 0.15}
COLUMNS = ["key", "video_id", "t", "chunk_id", "frame", "bucket", "split", "pre_x", "pre_y", "pre_source",
           "x", "y", "visibility", "status", "labeler"]
# TrackNet dataset convention: 0 not in frame / not visible, 1 clearly visible, 2 visible but
# blurred or hard to see, 3 occluded (position estimated).
VISIBILITY = {0: "none", 1: "visible", 2: "blurred", 3: "occluded"}


def load_labels(path: Path | None = None) -> pd.DataFrame:
    path = path or LABELS_PATH
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"key": str, "video_id": str, "chunk_id": str})


def save_labels(df: pd.DataFrame, path: Path | None = None):
    path = path or LABELS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df[COLUMNS].to_csv(tmp, index=False)
    tmp.replace(path)


def frame_paths(video_id: str, t: float) -> list[Path]:
    """Current frame, then 1 and 2 frames earlier (TrackNet channel order)."""
    stem = f"{t:010.4f}"
    return [FRAMES_DIR / video_id / f"{stem}_{k}.jpg" for k in range(3)]


def _split(video_id: str, segment: str, val_frac: float, holdout: set) -> str:
    if video_id in holdout:
        return "val"
    h = zlib.crc32(f"{video_id}/{segment}".encode()) % 1000
    return "val" if h < val_frac * 1000 else "train"


def _rally_windows(hits: pd.DataFrame) -> dict:
    """chunk_id -> list of (first_frame, last_frame) spans in which the ball is in play."""
    spans = {}
    for chunk, g in hits.sort_values(["chunk_id", "frame"]).groupby("chunk_id"):
        cur = None
        for h in g.itertuples():
            if cur and h.frame - cur[1] <= 3.0 * FPS:
                cur[1] = h.frame
            else:
                if cur:
                    spans.setdefault(chunk, []).append((cur[0] - 10, cur[1] + 30))
                cur = [h.frame, h.frame]
        if cur:
            spans.setdefault(chunk, []).append((cur[0] - 10, cur[1] + 30))
    return spans


def _fit_prelabel(chunk_ball: pd.DataFrame, frame: int, reach: int = 10):
    """Quadratic fit through raw detections on both sides of a miss (no hit in between assumed)."""
    near = chunk_ball[(chunk_ball.frame - frame).abs().between(1, reach) & chunk_ball.raw_detected]
    before, after = near[near.frame < frame], near[near.frame > frame]
    if len(before) < 3 or len(after) < 3:
        return None
    fx = np.polyfit(near.frame, near.ball_x_px, 2)
    fy = np.polyfit(near.frame, near.ball_y_px, 2)
    x, y = float(np.polyval(fx, frame)), float(np.polyval(fy, frame))
    if not (0 <= x < 1280 and 0 <= y < 720):
        return None
    return x, y


def bucket_masks(ball: pd.DataFrame, hits: pd.DataFrame | None) -> dict[str, np.ndarray]:
    n = len(ball)
    ok = (ball.frame >= 2).to_numpy()
    masks = {"random": ok.copy()}
    in_rally = np.zeros(n, bool)
    serve = np.zeros(n, bool)
    bounce = ball.is_bounce.fillna(False).to_numpy(bool).copy()
    if hits is not None and len(hits):
        spans = _rally_windows(hits)
        frames, chunks = ball.frame.to_numpy(), ball.chunk_id.to_numpy()
        for chunk, sp in spans.items():
            sel = chunks == chunk
            for a, b in sp:
                in_rally |= sel & (frames >= a) & (frames <= b)
        for h in hits[hits.is_serve.fillna(False).astype(bool)].itertuples():
            serve |= (chunks == h.chunk_id) & (frames >= h.frame - 15) & (frames <= h.frame + 20)
    if bounce.any():
        idx = np.where(bounce)[0]
        grown = np.zeros(n, bool)
        for d in range(-3, 4):
            j = np.clip(idx + d, 0, n - 1)
            same = ball.chunk_id.to_numpy()[j] == ball.chunk_id.to_numpy()[idx]
            grown[j[same]] = True
        bounce = grown
    far = in_rally & (ball.ball_ground_y_m.to_numpy() < 0 if "ball_ground_y_m" in ball else False)
    masks["miss"] = ok & in_rally & ~ball.raw_detected.to_numpy(bool)
    masks["serve"] = ok & serve
    masks["bounce"] = ok & bounce
    masks["far"] = ok & np.asarray(far, bool)
    return masks


def _draw(cands: np.ndarray, k: int, taken: dict, chunk_of, frame_of, rng, spacing: int) -> list[int]:
    out = []
    for i in rng.permutation(cands):
        if len(out) >= k:
            break
        c, f = chunk_of[i], frame_of[i]
        if any(abs(f - g) < spacing for g in taken.get(c, ())):
            continue
        taken.setdefault(c, []).append(f)
        out.append(int(i))
    return out


def sample_match(video_id: str, quota: int, rng, fractions: dict = FRACTIONS, spacing: int = 6,
                 val_frac: float = 0.15, holdout: set = frozenset(), taken_t: set = frozenset()) -> list[dict]:
    out_dir = match_dir(video_id)
    ball_path = out_dir / "ball.parquet"
    if not ball_path.exists():
        return _sample_untracked(video_id, quota, rng, val_frac, holdout, taken_t)
    ball = pd.read_parquet(ball_path).reset_index(drop=True)
    hits_path = out_dir / "hits_raw.parquet"
    hits = pd.read_parquet(hits_path) if hits_path.exists() else None
    masks = bucket_masks(ball, hits)
    chunk_of, frame_of = ball.chunk_id.to_numpy(), ball.frame.to_numpy()
    taken = {}
    for t in taken_t:
        hit = ball.index[(ball.t - t).abs() < 0.5 / FPS]
        for i in hit:
            taken.setdefault(chunk_of[i], []).append(frame_of[i])
    picked = []
    for name in ("miss", "serve", "bounce", "far", "random"):
        # Shortfalls and rounding roll forward; the last bucket takes whatever quota remains.
        k = quota - len(picked) if name == "random" else int(round(quota * fractions.get(name, 0)))
        got = _draw(np.where(masks[name])[0], k, taken, chunk_of, frame_of, rng, spacing)
        picked += [(i, name) for i in got]
    rows = []
    by_chunk = {c: g for c, g in ball.groupby("chunk_id")}
    for i, name in picked:
        r = ball.iloc[i]
        if not np.isnan(r.ball_x_px):
            pre, src = (float(r.ball_x_px), float(r.ball_y_px)), "tracknet" if r.raw_detected else "interp"
        else:
            pre = _fit_prelabel(by_chunk[r.chunk_id], int(r.frame))
            src = "fit" if pre else "none"
        rows.append({"key": f"{video_id}@{r.t:.4f}", "video_id": video_id, "t": round(float(r.t), 4),
                     "chunk_id": r.chunk_id, "frame": int(r.frame), "bucket": name,
                     "split": _split(video_id, r.chunk_id.split("_")[0], val_frac, holdout),
                     "pre_x": pre[0] if pre else np.nan, "pre_y": pre[1] if pre else np.nan, "pre_source": src,
                     "x": np.nan, "y": np.nan, "visibility": np.nan, "status": "pending", "labeler": ""})
    return rows


def _sample_untracked(video_id, quota, rng, val_frac, holdout, taken_t) -> list[dict]:
    seg_path = match_dir(video_id) / "segments.csv"
    if not seg_path.exists():
        print(f"{video_id}: no segments.csv; run the scenes stage first")
        return []
    segs = pd.read_csv(seg_path)
    w = segs.duration.to_numpy() / segs.duration.sum()
    rows, seen = [], set(taken_t)
    for _ in range(quota * 5):
        if len(rows) >= quota:
            break
        s = segs.iloc[rng.choice(len(segs), p=w)]
        t = round(float(rng.uniform(s.start + 0.2, s.end - 0.2)), 4)
        if any(abs(t - u) < 0.2 for u in seen):
            continue
        seen.add(t)
        rows.append({"key": f"{video_id}@{t:.4f}", "video_id": video_id, "t": t, "chunk_id": "", "frame": -1,
                     "bucket": "untracked", "split": _split(video_id, f"{int(s.segment_id):04d}", val_frac, holdout),
                     "pre_x": np.nan, "pre_y": np.nan, "pre_source": "none", "x": np.nan, "y": np.nan,
                     "visibility": np.nan, "status": "pending", "labeler": ""})
    return rows


def sample(video_ids: list[str], n: int = 2000, seed: int = 0, holdout: list[str] | None = None,
           val_frac: float = 0.15) -> pd.DataFrame:
    """Top the label file up to `n` rows, spread evenly over `video_ids`. Existing rows are kept."""
    labels = load_labels()
    need = n - len(labels)
    if need <= 0:
        return labels
    rng = np.random.default_rng(seed + len(labels))
    new, active = [], list(video_ids)
    # Matches that run out of frames hand their remaining quota to the others.
    for _ in range(4):
        remaining = need - len(new)
        if remaining <= 0 or not active:
            break
        per = int(np.ceil(remaining / len(active)))
        still = []
        for vid in active:
            taken = set(labels.loc[labels.video_id == vid, "t"].astype(float)) | {r["t"] for r in new if r["video_id"] == vid}
            got = sample_match(vid, per, rng, val_frac=val_frac, holdout=set(holdout or ()), taken_t=taken)
            new += got
            if len(got) >= per:
                still.append(vid)
        active = still
    new_df = pd.DataFrame(new, columns=COLUMNS).drop_duplicates("key")
    new_df = new_df[~new_df.key.isin(labels.key)].head(need)
    labels = pd.concat([labels, new_df], ignore_index=True) if len(labels) else new_df
    save_labels(labels)
    return labels


def export(video_path_fn, video_ids: list[str] | None = None) -> int:
    """Decode the 3-frame input for every labeled row whose frames are not cached yet."""
    labels = load_labels()
    if video_ids:
        labels = labels[labels.video_id.isin(video_ids)]
    n = 0
    for vid, g in labels.groupby("video_id"):
        path = video_path_fn(vid)
        if not path.exists():
            print(f"{vid}: video not found at {path}; skipped {len(g)} frames")
            continue
        for r in g.itertuples():
            paths = frame_paths(vid, r.t)
            if all(p.exists() for p in paths):
                continue
            clip = video.read_clip(path, max(r.t - 2 / FPS, 0), 3.5 / FPS, fps=FPS)[:3]
            if len(clip) < 3:
                continue
            paths[0].parent.mkdir(parents=True, exist_ok=True)
            for p, img in zip(paths, (clip[2], clip[1], clip[0])):
                cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n += 1
    return n


def summary(labels: pd.DataFrame | None = None) -> dict:
    labels = load_labels() if labels is None else labels
    done = labels[labels.status == "done"]
    return {
        "rows": int(len(labels)), "labeled": int(len(done)),
        "by_bucket": labels.bucket.value_counts().to_dict(),
        "by_split": labels.split.value_counts().to_dict(),
        "by_video": labels.video_id.value_counts().to_dict(),
        "visibility": done.visibility.map(lambda v: VISIBILITY.get(int(v), "?")).value_counts().to_dict(),
        "prelabel_error_px_median": float(np.nanmedian(np.hypot(done.pre_x - done.x, done.pre_y - done.y)))
        if len(done) and done.pre_x.notna().any() and done.x.notna().any() else None,
    }


class Reviewer:
    """Matplotlib labeling UI (works with the macOS backend; no OpenCV GUI needed).

    left click     ball here, clearly visible          -> next
    right click    ball here, blurred / hard to see     -> next
    space / a      accept the pre-label (red circle)    -> next
    o              occluded: keep pre-label/click position, visibility 3 -> next
    n              no ball in frame                     -> next
    backspace / u  previous frame
    f              flicker to the previous frame (motion makes the ball easy to find)
    d              toggle frame-difference view
    q              save and quit
    """

    def __init__(self, labeler: str, redo: bool = False):
        import matplotlib.pyplot as plt

        for k in plt.rcParams:
            if k.startswith("keymap."):
                plt.rcParams[k] = []
        self.plt = plt
        self.labeler = labeler
        self.df = load_labels()
        todo = self.df.status != "done" if not redo else self.df.status.notna()
        exported = self.df.apply(lambda r: frame_paths(r.video_id, r.t)[0].exists(), axis=1)
        self.queue = list(self.df.index[todo & exported])
        self.pos = 0
        self.mode = "frame"
        self.unsaved = 0
        self.fig, (self.ax, self.zoom) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [3, 1]})
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def run(self):
        if not self.queue:
            print("nothing to review (run sample + export first)")
            return
        self.show()
        self.plt.show()
        save_labels(self.df)

    def _imgs(self, i):
        r = self.df.loc[i]
        return [cv2.imread(str(p))[:, :, ::-1] for p in frame_paths(r.video_id, r.t)]

    def show(self):
        i = self.queue[self.pos]
        r = self.df.loc[i]
        cur, prev1, _ = self._imgs(i)
        img = {"frame": cur, "prev": prev1,
               "diff": np.clip(np.abs(cur.astype(int) - prev1.astype(int)) * 4, 0, 255).astype(np.uint8)}[self.mode]
        self.ax.clear()
        self.ax.imshow(img)
        cx, cy = (r.x, r.y) if not pd.isna(r.x) else (r.pre_x, r.pre_y)
        if not pd.isna(r.pre_x):
            self.ax.add_patch(self.plt.Circle((r.pre_x, r.pre_y), 9, fill=False, color="red", lw=1.2))
        if not pd.isna(r.x):
            self.ax.plot(r.x, r.y, "+", color="cyan", ms=14)
        self.ax.set_title(f"{self.pos + 1}/{len(self.queue)}  {r.key}  [{r.bucket}, pre={r.pre_source}, "
                          f"view={self.mode}]  status={r.status}", fontsize=9)
        self.ax.set_axis_off()
        self.zoom.clear()
        if pd.isna(cx):
            cx, cy = 640, 360
        x0, y0 = int(np.clip(cx - 48, 0, 1280 - 96)), int(np.clip(cy - 48, 0, 720 - 96))
        self.zoom.imshow(img[y0:y0 + 96, x0:x0 + 96], extent=(x0, x0 + 96, y0 + 96, y0), interpolation="nearest")
        if not pd.isna(r.pre_x):
            self.zoom.add_patch(self.plt.Circle((r.pre_x, r.pre_y), 3, fill=False, color="red", lw=1))
        self.zoom.set_title("zoom (click here works too)", fontsize=8)
        self.fig.canvas.draw_idle()

    def _set(self, x, y, vis):
        i = self.queue[self.pos]
        self.df.loc[i, ["x", "y", "visibility", "status", "labeler"]] = [x, y, vis, "done", self.labeler]
        self.unsaved += 1
        if self.unsaved >= 10:
            save_labels(self.df)
            self.unsaved = 0
        self.pos = min(self.pos + 1, len(self.queue) - 1)
        self.mode = "frame"
        self.show()

    def on_click(self, ev):
        if ev.inaxes not in (self.ax, self.zoom) or ev.xdata is None:
            return
        self._set(float(ev.xdata), float(ev.ydata), 2 if ev.button == 3 else 1)

    def on_key(self, ev):
        r = self.df.loc[self.queue[self.pos]]
        if ev.key in (" ", "a") and not pd.isna(r.pre_x):
            self._set(float(r.pre_x), float(r.pre_y), 1)
        elif ev.key == "o":
            x, y = (r.x, r.y) if not pd.isna(r.x) else (r.pre_x, r.pre_y)
            self._set(x, y, 3)
        elif ev.key == "n":
            self._set(np.nan, np.nan, 0)
        elif ev.key in ("backspace", "u"):
            self.pos = max(self.pos - 1, 0)
            self.show()
        elif ev.key == "f":
            self.mode = "prev" if self.mode != "prev" else "frame"
            self.show()
        elif ev.key == "d":
            self.mode = "diff" if self.mode != "diff" else "frame"
            self.show()
        elif ev.key == "q":
            save_labels(self.df)
            self.plt.close(self.fig)
