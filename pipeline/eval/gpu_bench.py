"""Component and end-to-end tracking throughput on this machine.

    uv run python -m eval.gpu_bench BENCH_VIDEO.mp4 --out bench.json

The benchmark footage need not be a broadcast: a fixed broadcast-style court calibration stands
in for the court model's output (the model still runs, so its cost is counted), which lets the
full per-chunk path run on any 720p tennis video. Results are written after every step.
"""
import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from tennis_pipeline import tracknet, video
from tennis_pipeline.court import Calibration, CourtDetector, m_to_ref
from tennis_pipeline.process import CropWriter
from tennis_pipeline.track import BallTracker, PlayerDetector, far_court_rect, track_segment

# Doubles-court corners (meters) -> typical main-camera pixel positions in a 1280x720 frame.
BROADCAST_CORNERS = {(-5.485, -11.885): (446, 172), (5.485, -11.885): (834, 172),
                     (-5.485, 11.885): (256, 541), (5.485, 11.885): (1025, 541)}


def broadcast_calib() -> Calibration:
    src = m_to_ref(np.array(list(BROADCAST_CORNERS))).astype(np.float32)
    dst = np.array(list(BROADCAST_CORNERS.values()), np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return Calibration(H, np.linalg.inv(H), 0.0, 14)


class FixedCourt:
    """Runs the real court model (so its cost counts) but returns a fixed calibration."""

    def __init__(self, real: CourtDetector):
        self.real, self.cal = real, broadcast_calib()

    def calibrate(self, frames):
        self.real.calibrate(frames)
        return [self.cal] * len(frames)


class GpuSampler:
    def __init__(self):
        self.samples, self._stop = [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            try:
                u, m = out.strip().split(",")
                self.samples.append((float(u), float(m)))
            except ValueError:
                pass
            time.sleep(0.5)

    def __enter__(self):
        if torch.cuda.is_available():
            self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()

    def summary(self) -> dict:
        if not self.samples:
            return {}
        u = np.array(self.samples)
        return {"gpu_util_mean": round(float(u[:, 0].mean()), 1), "gpu_mem_max_mb": float(u[:, 1].max())}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=Path("gpu_bench.json"))
    ap.add_argument("--chunks", type=int, default=6, help="30 s chunks for the end-to-end runs")
    ap.add_argument("--seconds", type=float, default=30.0, help="chunk length for component timings")
    ap.add_argument("--batches", default="8,16,32", help="TrackNet batch sizes to time")
    args = ap.parse_args()

    res: dict = {"device": tracknet.device().type, "hwaccel": " ".join(video.hwaccel()) or "none",
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 "torch": torch.__version__}

    def save(key, value):
        res[key] = value
        args.out.write_text(json.dumps(res, indent=2))
        print(key, json.dumps(value), flush=True)

    dur = video.probe(args.video)["duration"]
    import os as _os
    import sys as _sys
    _os.environ["TENNIS_HWACCEL"] = "videotoolbox" if _sys.platform == "darwin" else "cuda"
    video.hwaccel.cache_clear()
    hw = video.hwaccel()
    _os.environ["TENNIS_HWACCEL"] = "none"
    video.hwaccel.cache_clear()
    for name, accel in (("cpu", ()), ("hw", hw)):
        if name == "hw" and not accel:
            continue
        t = time.time()
        n = sum(1 for _ in video.iter_frames(args.video, 30, (1280, 720), start=60, duration=30, accel=accel))
        save(f"decode_chunk_{name}", {"frames": n, "seconds": round(time.time() - t, 2),
                                      "fps": round(n / (time.time() - t), 1)})
        t = time.time()
        n = sum(1 for _ in video.iter_frames(args.video, 2, (640, 360), accel=accel))
        save(f"decode_scene_2fps_{name}", {"video_s": round(dur, 1), "seconds": round(time.time() - t, 2),
                                           "x_realtime": round(dur / (time.time() - t), 1)})
        if name == "cpu":
            t = time.time()
            n = sum(1 for _ in video.iter_frames(args.video, 2, (640, 360), accel=accel, keyframes_only=True))
            save("decode_scene_2fps_keyframes", {"seconds": round(time.time() - t, 2),
                                                 "x_realtime": round(dur / (time.time() - t), 1)})

    frames = video.read_clip(args.video, 60, args.seconds)
    dev = tracknet.device()
    import os
    os.environ["TENNIS_COMPILE"] = "0"
    ball = BallTracker(dev, "torch")
    ball(frames[:64])
    t = time.time()
    np.stack([cv2.resize(f, (640, 360))[:, 64:576] for f in frames])
    save("cpu_resize_900_frames_s", round(time.time() - t, 2))
    for b in (int(x) for x in args.batches.split(",")):
        ball.batch_override = b
        ball(frames[:64])
        sync()
        t = time.time()
        ball(frames)
        sync()
        save(f"tracknet_batch{b}", {"fps": round(len(frames) / (time.time() - t), 1)})
    ball.batch_override = None
    os.environ["TENNIS_COMPILE"] = "1"
    t = time.time()
    ball = BallTracker(dev)
    ball(frames[:64])
    sync()
    compile_s = time.time() - t
    t = time.time()
    ball(frames)
    sync()
    save("tracknet_default", {"compiled": getattr(ball, "compiled", False), "warmup_s": round(compile_s, 1),
                              "fps": round(len(frames) / (time.time() - t), 1)})

    players = PlayerDetector()
    rect = far_court_rect(broadcast_calib())
    idx = list(range(0, len(frames), 4))
    for near_imgsz, bs in ((960, 16), (640, 32)):
        players.detect([frames[i] for i in idx[:bs]], rect, near_imgsz=near_imgsz)
        sync()
        t = time.time()
        for k in range(0, len(idx), bs):
            players.detect([frames[i] for i in idx[k:k + bs]], rect, near_imgsz=near_imgsz)
        sync()
        save(f"players_detect_near{near_imgsz}_batch{bs}",
             {"sampled_frames": len(idx), "ms_per_sampled_frame": round((time.time() - t) / len(idx) * 1000, 1)})

    court = CourtDetector(dev)
    probes = [frames[i] for i in range(0, len(frames), 60)]
    court.calibrate(probes)  # warm up at the real batch shape
    sync()
    t = time.time()
    for _ in range(3):
        court.calibrate(probes)
    sync()
    save("court_calibrate", {"probes": len(probes), "ms_per_probe": round((time.time() - t) / (3 * len(probes)) * 1000, 1)})

    crops = [cv2.resize(frames[i][200:520, 500:820], (256, 256)) for i in range(0, len(frames), max(len(frames) // 64, 1))][:64]
    players.pose(crops[:4])
    sync()
    t = time.time()
    players.pose(crops)
    sync()
    save("pose_on_crops", {"crops": len(crops), "ms_per_crop": round((time.time() - t) / len(crops) * 1000, 1)})

    from tennis_pipeline.scenes import Embedder
    emb = Embedder()
    small = [cv2.resize(f, (640, 360)) for f in frames[:256]]
    emb(small[:8])
    t = time.time()
    for k in range(0, len(small), 64):
        emb(small[k:k + 64])
    save("scene_embed", {"fps": round(len(small) / (time.time() - t), 1)})

    fixed = FixedCourt(court)
    starts = [args.seconds * k for k in range(args.chunks) if args.seconds * (k + 1) <= dur]
    loader = lambda s: video.read_clip(args.video, s, args.seconds)  # noqa: E731
    for mode in ("sequential", "prefetch"):
        with GpuSampler() as gs:
            t, n = time.time(), 0
            items = ((s, loader(s)) for s in starts) if mode == "sequential" else video.prefetch(starts, loader)
            for s, fr in items:
                track_segment(fr, s, fixed, ball, players)
                n += len(fr)
                tracknet.empty_cache()
            save(f"track_end_to_end_{mode}", {"frames": n, "seconds": round(time.time() - t, 1),
                                              "fps": round(n / (time.time() - t), 1), **gs.summary()})

    # Hitter crops: from in-memory frames (inline) vs one seek-and-decode per hit (old crops stage).
    hits = pd.DataFrame([{"hit_id": f"bench_{i}", "t": 60 + 0.5 + i * (args.seconds - 1) / 5, "box": [560.0, 380.0, 640.0, 560.0],
                          "side": "near", "hitter_y_m": 12.0, "ball_px_x": 600.0, "ball_px_y": 420.0}
                         for i in range(5)])
    writer = CropWriter("_bench")
    writer.rows = {}
    t = time.time()
    writer.add(hits, frames, 60.0, players)
    save("crops_inline_s_per_hit", round((time.time() - t) / len(hits), 3))
    t = time.time()
    for h in hits.itertuples():
        video.read_clip(args.video, h.t - 5 / 30, 11 / 30)
    save("crops_old_decode_s_per_hit", round((time.time() - t) / len(hits), 3))

    print("done", flush=True)


if __name__ == "__main__":
    main()
