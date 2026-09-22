"""Per-match stages: track (GPU, slow) -> events (CPU, seconds) -> crops (short decodes per hit).

Tracking results are cached per chunk so hit/bounce logic can be iterated on without
re-running the models.
"""
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
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


def track_match(video_id: str, video_path: Path, limit_segments: int | None = None,
                ball_backend: str = "mps", ball_weights: str | None = None) -> dict:
    """Track every main-camera chunk; chunks cached with other ball weights get the ball re-run only."""
    out_dir = match_dir(video_id)
    tdir = _tracks_dir(video_id)
    segs = pd.read_csv(out_dir / "segments.csv")
    if limit_segments:
        segs = segs.head(limit_segments)
    dev = tracknet.device()
    ball = BallTracker(dev, ball_backend, ball_weights)
    court_det = players = None
    t_start, n_frames, n_retracked = time.time(), 0, 0
    for seg in tqdm(segs.itertuples(), total=len(segs), desc=f"track {video_id}"):
        t = seg.start
        while t < seg.end - 0.5:
            dur = min(CHUNK_S, seg.end - t)
            path = tdir / f"{seg.segment_id:04d}_{int(round(t * 10)):06d}.npz"
            if not path.exists():
                if court_det is None:
                    court_det, players = CourtDetector(dev), PlayerDetector()
                frames = video.read_clip(video_path, t, dur, fps=FPS)
                n_frames += len(frames)
                save_tracks(path, track_segment(frames, t, court_det, ball, players))
            else:
                tr = load_tracks(path)
                if tr.court_ok < 0.5 or _legacy_ball_tag(tr) == ball.tag:
                    t += dur
                    continue
                frames = video.read_clip(video_path, t, dur, fps=FPS)
                n_frames += len(frames)
                n_retracked += 1
                tr.ball, tr.ball_weights = ball(frames), ball.tag
                save_tracks(path, tr)
            del frames
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            t += dur
    elapsed = time.time() - t_start
    stats = {"video_id": video_id, "frames": n_frames, "seconds": round(elapsed, 1),
             "fps": round(n_frames / max(elapsed, 1e-9), 2), "ball_weights": ball.tag,
             "chunks_ball_retracked": n_retracked}
    if n_frames:  # keep the last real throughput measurement when everything was cached
        (out_dir / "track_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def events_match(video_id: str, video_path: Path | None = None, serve_detector: bool = True,
                 serve_params: "serve.ServeParams | None" = None) -> pd.DataFrame:
    """Recompute bounces/hits from cached tracks; writes ball/players/hits/serve-candidate tables."""
    from . import audio
    from .scenes import SAMPLE_FPS, scene_prob

    out_dir = match_dir(video_id)
    bounce_model = events.BounceModel()
    prob = scene_prob(video_id)
    onsets = None
    if video_path is not None or (out_dir / "audio_onsets.npz").exists():
        onsets = audio.onsets(video_id, video_path)
    ball_rows, player_rows, hit_rows, serve_rows = [], [], [], []
    for path in sorted(_tracks_dir(video_id).glob("*.npz")):
        tr = load_tracks(path)
        if tr.court_ok < 0.5:
            continue
        # Drop padded frames where the broadcast has already cut away from the main camera.
        idx = np.clip(np.round((tr.t0 + np.arange(tr.n) / FPS) * SAMPLE_FPS).astype(int), 0, len(prob) - 1)
        off = prob[idx] < 0.5
        tr.ball = tr.ball.copy()
        tr.ball[off] = np.nan
        for side in tr.players:
            tr.players[side] = tr.players[side].copy()
            tr.players[side][off] = np.nan
        tr.calibs = [None if o else c for c, o in zip(tr.calibs, off)]
        chunk_id = path.stem
        b = events.clean_ball(tr.ball)
        bounces = bounce_model.predict(b)
        raw = events.detect_hits(tr, b)
        if serve_detector:
            cands = serve.detect(tr, b, raw, onsets, serve_params)
            serve_rows += [{**c, "chunk_id": chunk_id} for c in cands]
            raw = serve.merge(raw, cands, b, tr, serve_params)
        hits = events.annotate_hits(tr, b, raw, bounces)
        _collect_frames(tr, b, bounces, chunk_id, ball_rows, player_rows)
        for h in hits:
            rec = {k: v for k, v in h.items()}
            box = rec.pop("hitter_box")
            rec["box"] = None if box is None or np.isnan(box).any() else [float(v) for v in box]
            rec.update({"hit_id": f"{chunk_id}_{h['frame']}", "chunk_id": chunk_id,
                        "serve_in": events.serve_in(h) if h["is_serve"] else None})
            hit_rows.append(rec)
    pd.DataFrame(ball_rows).to_parquet(out_dir / "ball.parquet", index=False)
    pd.DataFrame(player_rows).to_parquet(out_dir / "players.parquet", index=False)
    pd.DataFrame(serve_rows).to_parquet(out_dir / "serve_candidates.parquet", index=False)
    hits = pd.DataFrame(hit_rows)
    if onsets is not None and len(hits):
        on_t, on_s = onsets
        hits["t_audio"], hits["audio_strength"] = audio.snap(hits.t.to_numpy(), on_t, on_s)
        hits["audio_confirmed"] = hits.t_audio.notna()
        hits["t_visual"] = hits.t
        hits["t"] = hits.t_audio.fillna(hits.t)
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


def crops_match(video_id: str, video_path: Path, players: PlayerDetector | None = None) -> pd.DataFrame:
    """Decode a short window around each hit: hitter crop strip (for the VLM) + pose features."""
    out_dir = match_dir(video_id)
    hits = pd.read_parquet(out_dir / "hits_raw.parquet")
    players = players or PlayerDetector()
    crop_dir = CROP_DIR / video_id
    crop_dir.mkdir(parents=True, exist_ok=True)
    feats_path = out_dir / "hit_features.parquet"
    cached = pd.read_parquet(feats_path) if feats_path.exists() else pd.DataFrame(columns=["hit_id"])
    cached = cached[cached.hit_id.isin(hits.hit_id)]
    rows = cached.to_dict("records")
    todo = hits[~hits.hit_id.isin(cached.hit_id)]
    for h in tqdm(todo.itertuples(), total=len(todo), desc=f"crops {video_id}"):
        rec = {"hit_id": h.hit_id}
        if h.box is None:
            rows.append(rec)
            continue
        box = np.array(h.box)
        clip = video.read_clip(video_path, max(h.t - 5 / FPS, 0), 11 / FPS, fps=FPS)
        if len(clip) < 9:
            rows.append(rec)
            continue
        mid = min(5, len(clip) - 1)
        crop, M = events.hitter_crop(clip[mid], box)
        strip = np.concatenate([events.hitter_crop(clip[i], box)[0] for i in (max(mid - 4, 0), mid, min(mid + 4, len(clip) - 1))], 1)
        path = crop_dir / f"{h.hit_id}.jpg"
        cv2.imwrite(str(path), strip)
        rec["crop_path"] = str(path)
        kp = _pose_on_crop(players, crop, M)
        hit = {"hitter_box": box, "side": h.side, "hitter_y_m": h.hitter_y_m}
        rec.update(events.stroke_features(hit, kp, np.array([h.ball_px_x, h.ball_px_y])))
        rows.append(rec)
    feats = pd.DataFrame(rows)
    feats.to_parquet(feats_path, index=False)
    return feats


def _pose_on_crop(players: PlayerDetector, crop: np.ndarray, M: np.ndarray):
    r = players.model.predict([crop], imgsz=256, device="mps", conf=0.2, classes=[0], verbose=False)[0]
    if r.keypoints is None or len(r.boxes) == 0:
        return None
    boxes = r.boxes.xyxy.cpu().numpy()
    i = int(np.argmin(np.linalg.norm((boxes[:, :2] + boxes[:, 2:]) / 2 - np.array([128, 128]), axis=1)))
    kp = r.keypoints.data.cpu().numpy()[i].copy()
    kp[:, 0] = (kp[:, 0] - M[0, 2]) / M[0, 0]
    kp[:, 1] = (kp[:, 1] - M[1, 2]) / M[1, 1]
    return kp
