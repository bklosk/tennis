"""Per-match stages: track (GPU, slow) -> events (CPU, seconds) -> crops (hitter crops + pose).

Tracking results are cached per chunk so hit/bounce logic can be iterated on without
re-running the models. While a chunk's frames are still in memory, tracking also runs that
chunk's events and extracts hitter crops, so the crops stage only has to decode hits that a
later events run adds.
"""
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import events, serve, tracknet, video
from .court import Calibration, CourtDetector
from .paths import CACHE, match_dir
from .track import FPS, BallTracker, PlayerDetector, SegmentTracks, track_segment

CHUNK_S = 30.0
CROP_DIR = CACHE / "hit_crops"


def _tracks_dir(video_id: str) -> Path:
    path = match_dir(video_id) / "tracks"
    path.mkdir(exist_ok=True)
    return path


def save_tracks(path: Path, tr: SegmentTracks):
    anchors, mats = [], []
    for f, c in enumerate(tr.calibs):
        if c is not None and (not mats or c.court_to_img is not mats[-1]):
            anchors.append(f)
            mats.append(c.court_to_img)
    np.savez_compressed(
        path, t0=tr.t0, n=tr.n, court_ok=tr.court_ok, ball=tr.ball, ball_weights=tr.ball_weights,
        anchors=np.array(anchors, int), mats=np.array(mats) if mats else np.zeros((0, 3, 3)),
        **{f"box_{s}": v for s, v in tr.players.items()},
        **{f"kps_{s}": v for s, v in tr.player_kps.items()},
    )


def load_tracks(path: Path) -> SegmentTracks:
    z = np.load(path)
    n = int(z["n"])
    calibs = [None] * n
    anchors, mats = z["anchors"], z["mats"]
    for k, a in enumerate(anchors):
        end = anchors[k + 1] if k + 1 < len(anchors) else n
        cal = Calibration(mats[k], np.linalg.inv(mats[k]), 0.0, 0)
        for f in range(a, end):
            calibs[f] = cal
    if len(anchors):
        for f in range(anchors[0]):
            calibs[f] = calibs[anchors[0]]
    tr = SegmentTracks(t0=float(z["t0"]), n=n, calibs=calibs, ball=z["ball"], court_ok=float(z["court_ok"]),
                       ball_weights=str(z["ball_weights"]) if "ball_weights" in z else "")
    for s in ("near", "far"):
        if f"box_{s}" in z:
            tr.players[s] = z[f"box_{s}"]
            tr.player_kps[s] = z[f"kps_{s}"]
    return tr


def _legacy_ball_tag(tr: SegmentTracks) -> str:
    # Chunks cached before weights were tagged were all tracked with the pretrained model.
    if tr.ball_weights or not tracknet.PRETRAINED_BALL.exists():
        return tr.ball_weights
    return tracknet.weights_tag(tracknet.PRETRAINED_BALL)


@dataclass
class EventContext:
    """Per-match inputs for event extraction, loaded once."""
    prob: np.ndarray
    onsets: tuple | None
    bounce_model: events.BounceModel
    serve_detector: bool = True
    serve_params: "serve.ServeParams | None" = None

    @classmethod
    def load(cls, video_id: str, video_path: Path | None, serve_detector: bool = True,
             serve_params: "serve.ServeParams | None" = None) -> "EventContext":
        from . import audio
        from .scenes import scene_prob

        onsets = None
        if video_path is not None or (match_dir(video_id) / "audio_onsets.npz").exists():
            onsets = audio.onsets(video_id, video_path)
        return cls(scene_prob(video_id), onsets, events.BounceModel(), serve_detector, serve_params)


def chunk_events(tr: SegmentTracks, chunk_id: str, ctx: EventContext):
    """Bounces, hits and serve candidates for one tracked chunk.

    Masks the chunk's padded frames where the broadcast has already cut away from the main
    camera (modifies `tr`). Returns (clean ball track, bounce frames, hit records, serve records).
    """
    from .scenes import SAMPLE_FPS

    idx = np.clip(np.round((tr.t0 + np.arange(tr.n) / FPS) * SAMPLE_FPS).astype(int), 0, len(ctx.prob) - 1)
    off = ctx.prob[idx] < 0.5
    tr.ball = tr.ball.copy()
    tr.ball[off] = np.nan
    for side in tr.players:
        tr.players[side] = tr.players[side].copy()
        tr.players[side][off] = np.nan
    tr.calibs = [None if o else c for c, o in zip(tr.calibs, off)]
    b = events.clean_ball(tr.ball)
    bounces = ctx.bounce_model.predict(b)
    raw = events.detect_hits(tr, b)
    serve_rows = []
    if ctx.serve_detector:
        cands = serve.detect(tr, b, raw, ctx.onsets, ctx.serve_params)
        serve_rows = [{**c, "chunk_id": chunk_id} for c in cands]
        raw = serve.merge(raw, cands, b, tr, ctx.serve_params)
    hits = events.annotate_hits(tr, b, raw, bounces)
    hit_rows = []
    for h in hits:
        rec = {k: v for k, v in h.items()}
        box = rec.pop("hitter_box")
        rec["box"] = None if box is None or np.isnan(box).any() else [float(v) for v in box]
        rec.update({"hit_id": f"{chunk_id}_{h['frame']}", "chunk_id": chunk_id,
                    "serve_in": events.serve_in(h) if h["is_serve"] else None})
        hit_rows.append(rec)
    return b, bounces, hit_rows, serve_rows


def snap_to_audio(hits: pd.DataFrame, onsets) -> pd.DataFrame:
    if onsets is None or not len(hits):
        return hits
    from . import audio

    hits = hits.copy()
    on_t, on_s = onsets
    hits["t_audio"], hits["audio_strength"] = audio.snap(hits.t.to_numpy(), on_t, on_s)
    hits["audio_confirmed"] = hits.t_audio.notna()
    hits["t_visual"] = hits.t
    hits["t"] = hits.t_audio.fillna(hits.t)
    return hits


def track_match(video_id: str, video_path: Path, limit_segments: int | None = None,
                ball_backend: str = "auto", ball_weights: str | None = None, inline_crops: bool = True) -> dict:
    """Track every main-camera chunk; chunks cached with other ball weights get the ball re-run only.

    Decoding of the next chunk overlaps GPU work on the current one. With `inline_crops`, each new
    chunk's hitter crops and pose features are taken from the frames already in memory.
    """
    out_dir = match_dir(video_id)
    tdir = _tracks_dir(video_id)
    segs = pd.read_csv(out_dir / "segments.csv")
    if limit_segments:
        segs = segs.head(limit_segments)
    dev = tracknet.device()
    ball = BallTracker(dev, ball_backend, ball_weights)
    court_det = players = None

    jobs = []
    for seg in segs.itertuples():
        t = seg.start
        while t < seg.end - 0.5:
            dur = min(CHUNK_S, seg.end - t)
            path = tdir / f"{seg.segment_id:04d}_{int(round(t * 10)):06d}.npz"
            if not path.exists():
                jobs.append((path, t, dur, None))
            else:
                tr = load_tracks(path)
                if tr.court_ok >= 0.5 and _legacy_ball_tag(tr) != ball.tag:
                    jobs.append((path, t, dur, tr))
            t += dur

    ctx = crops = None
    if inline_crops and any(j[3] is None for j in jobs):
        ctx = EventContext.load(video_id, video_path)
        crops = CropWriter(video_id)
    timing = {"decode_wait": 0.0, "track": 0.0, "inline_crops": 0.0}
    t_start, n_frames, n_retracked = time.time(), 0, 0
    wait_from = time.time()
    loader = lambda job: video.read_clip(video_path, job[1], job[2], fps=FPS)  # noqa: E731
    for (path, t, dur, cached), frames in tqdm(video.prefetch(jobs, loader), total=len(jobs),
                                               desc=f"track {video_id}"):
        timing["decode_wait"] += time.time() - wait_from
        n_frames += len(frames)
        t0 = time.time()
        if cached is None:
            if court_det is None:
                court_det, players = CourtDetector(dev), PlayerDetector()
            tr = track_segment(frames, t, court_det, ball, players)
            save_tracks(path, tr)
            timing["track"] += time.time() - t0
            if crops is not None and tr.court_ok >= 0.5:
                t1 = time.time()
                _, _, hits, _ = chunk_events(tr, path.stem, ctx)
                crops.add(snap_to_audio(pd.DataFrame(hits), ctx.onsets), frames, t, players)
                timing["inline_crops"] += time.time() - t1
        else:
            n_retracked += 1
            cached.ball, cached.ball_weights = ball(frames), ball.tag
            save_tracks(path, cached)
            timing["track"] += time.time() - t0
        del frames
        gc.collect()
        tracknet.empty_cache()
        wait_from = time.time()
    if crops is not None:
        crops.save()
    elapsed = time.time() - t_start
    stats = {"video_id": video_id, "frames": n_frames, "seconds": round(elapsed, 1),
             "fps": round(n_frames / max(elapsed, 1e-9), 2), "ball_weights": ball.tag,
             "device": dev.type, "hwaccel": " ".join(video.hwaccel()) or "none",
             "chunks_ball_retracked": n_retracked, **{f"{k}_s": round(v, 1) for k, v in timing.items()}}
    if n_frames:  # keep the last real throughput measurement when everything was cached
        (out_dir / "track_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def events_match(video_id: str, video_path: Path | None = None, serve_detector: bool = True,
                 serve_params: "serve.ServeParams | None" = None) -> pd.DataFrame:
    """Recompute bounces/hits from cached tracks; writes ball/players/hits/serve-candidate tables."""
    out_dir = match_dir(video_id)
    ctx = EventContext.load(video_id, video_path, serve_detector, serve_params)
    ball_rows, player_rows, hit_rows, serve_rows = [], [], [], []
    for path in sorted(_tracks_dir(video_id).glob("*.npz")):
        tr = load_tracks(path)
        if tr.court_ok < 0.5:
            continue
        b, bounces, hits, serves = chunk_events(tr, path.stem, ctx)
        _collect_frames(tr, b, bounces, path.stem, ball_rows, player_rows)
        hit_rows += hits
        serve_rows += serves
    pd.DataFrame(ball_rows).to_parquet(out_dir / "ball.parquet", index=False)
    pd.DataFrame(player_rows).to_parquet(out_dir / "players.parquet", index=False)
    pd.DataFrame(serve_rows).to_parquet(out_dir / "serve_candidates.parquet", index=False)
    hits = snap_to_audio(pd.DataFrame(hit_rows), ctx.onsets)
    hits.to_parquet(out_dir / "hits_raw.parquet", index=False)
    return hits


def _collect_frames(tr, b, bounces, chunk_id, ball_rows, player_rows):
    bounce_set = set(bounces)
    for f in range(tr.n):
        calib = tr.calibs[f]
        rec = {"chunk_id": chunk_id, "frame": f, "t": tr.t0 + f / FPS,
               "ball_x_px": b[f, 0], "ball_y_px": b[f, 1], "raw_detected": not np.isnan(tr.ball[f, 0]),
               "is_bounce": f in bounce_set}
        if calib is not None and not np.isnan(b[f, 0]):
            gx, gy = calib.to_court_m(b[f][None])[0]
            rec["ball_ground_x_m"], rec["ball_ground_y_m"] = gx, gy
        ball_rows.append(rec)
        if f % 4 == 0:
            for side, boxes in tr.players.items():
                box = boxes[f]
                if np.isnan(box).any() or calib is None:
                    continue
                cx, cy = calib.to_court_m(np.array([[(box[0] + box[2]) / 2, box[3]]]))[0]
                player_rows.append({"chunk_id": chunk_id, "frame": f, "t": tr.t0 + f / FPS, "side": side,
                                    "x_m": cx, "y_m": cy, "box_x1": box[0], "box_y1": box[1],
                                    "box_x2": box[2], "box_y2": box[3]})


class CropWriter:
    """Hitter crop strips (contact frame and +/-4 frames) plus pose features, keyed by hit_id."""

    def __init__(self, video_id: str):
        self.dir = CROP_DIR / video_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = match_dir(video_id) / "hit_features.parquet"
        existing = pd.read_parquet(self.path) if self.path.exists() else pd.DataFrame(columns=["hit_id"])
        self.rows = {r["hit_id"]: r for r in existing.to_dict("records")}

    def done(self, hit_id: str) -> bool:
        return hit_id in self.rows

    def add(self, hits: pd.DataFrame, frames: np.ndarray, t0: float, players: PlayerDetector):
        """Crop every hit in `hits` from `frames`, which start at time `t0` (30 fps)."""
        todo, crops, mats = [], [], []
        for h in hits.itertuples():
            if self.done(h.hit_id):
                continue
            mid = int(round((h.t - t0) * FPS))
            if h.box is None or not 4 <= mid < len(frames) - 4:
                self.rows[h.hit_id] = {"hit_id": h.hit_id}
                continue
            box = np.array(h.box)
            crop, M = events.hitter_crop(frames[mid], box)
            strip = np.concatenate([events.hitter_crop(frames[i], box)[0] for i in (mid - 4, mid, mid + 4)], 1)
            path = self.dir / f"{h.hit_id}.jpg"
            cv2.imwrite(str(path), strip)
            todo.append((h, box, path))
            crops.append(crop)
            mats.append(M)
        if not todo:
            return
        results = players.pose(crops)
        for (h, box, path), r, M in zip(todo, results, mats):
            rec = {"hit_id": h.hit_id, "crop_path": str(path)}
            hit = {"hitter_box": box, "side": h.side, "hitter_y_m": h.hitter_y_m}
            rec.update(events.stroke_features(hit, _keypoints(r, M), np.array([h.ball_px_x, h.ball_px_y])))
            self.rows[h.hit_id] = rec

    def save(self, keep: set[str] | None = None) -> pd.DataFrame:
        rows = [r for k, r in self.rows.items() if keep is None or k in keep]
        feats = pd.DataFrame(rows)
        feats.to_parquet(self.path, index=False)
        return feats


def crops_match(video_id: str, video_path: Path, players: PlayerDetector | None = None) -> pd.DataFrame:
    """Hitter crops + pose for hits not already cropped during tracking.

    Hits are grouped by chunk and each chunk's hit span is decoded once, instead of one
    seek-and-decode per hit.
    """
    out_dir = match_dir(video_id)
    hits = pd.read_parquet(out_dir / "hits_raw.parquet")
    writer = CropWriter(video_id)
    todo = hits[~hits.hit_id.map(writer.done)]
    if len(todo):
        players = players or PlayerDetector()
        for _, grp in tqdm(todo.groupby("chunk_id"), desc=f"crops {video_id}"):
            t0 = max(grp.t.min() - 6 / FPS, 0)
            frames = video.read_clip(video_path, t0, grp.t.max() + 6 / FPS - t0, fps=FPS)
            writer.add(grp, frames, t0, players)
    return writer.save(keep=set(hits.hit_id))


def _keypoints(result, M: np.ndarray):
    """Pose keypoints of the person nearest the crop centre, mapped back to frame pixels."""
    if result.keypoints is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    i = int(np.argmin(np.linalg.norm((boxes[:, :2] + boxes[:, 2:]) / 2 - np.array([128, 128]), axis=1)))
    kp = result.keypoints.data.cpu().numpy()[i].copy()
    kp[:, 0] = (kp[:, 0] - M[0, 2]) / M[0, 0]
    kp[:, 1] = (kp[:, 1] - M[1, 2]) / M[1, 1]
    return kp
