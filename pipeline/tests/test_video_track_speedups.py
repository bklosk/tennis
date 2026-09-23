import subprocess
import time

import cv2
import numpy as np
import pandas as pd
import torch

from tennis_pipeline import process, tracknet, video
from tennis_pipeline.track import BALL_CROP_X, BallTracker


def test_device_downscale_matches_cv2_resize_exactly():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (5, 720, 1280, 3), dtype=np.uint8)
    bt = BallTracker.__new__(BallTracker)
    bt.dev, bt.dtype = torch.device("cpu"), torch.float32
    got = (bt._small(frames) * 255).round().permute(0, 2, 3, 1).numpy()
    x0, x1 = BALL_CROP_X
    ref = np.stack([cv2.resize(f, (640, 360))[:, x0:x1] for f in frames]).astype(np.float32)
    assert np.array_equal(got, ref)


def test_prefetch_preserves_order_and_overlaps_loading():
    def load(i):
        time.sleep(0.05)
        return i * 10

    t = time.time()
    out = []
    for item, res in video.prefetch(range(6), load):
        time.sleep(0.05)  # "GPU work" on the previous item overlaps the next load
        out.append((item, res))
    assert out == [(i, i * 10) for i in range(6)]
    assert time.time() - t < 6 * 0.1 * 0.8


def test_hwaccel_can_be_disabled(monkeypatch):
    video.hwaccel.cache_clear()
    monkeypatch.setenv("TENNIS_HWACCEL", "none")
    assert video.hwaccel() == ()
    video.hwaccel.cache_clear()


def _synthetic_video(path, seconds=3):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=1280x720:r=60:d={seconds}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)], check=True)


class FakePose:
    class R:
        keypoints = None
        boxes = []

    def pose(self, crops):
        return [self.R() for _ in crops]


def test_inline_and_batched_crops_match(tmp_path, monkeypatch):
    monkeypatch.setattr(process, "CROP_DIR", tmp_path / "crops")
    monkeypatch.setattr(process, "match_dir", lambda vid: (tmp_path / vid).mkdir(exist_ok=True) or tmp_path / vid)
    src = tmp_path / "v.mp4"
    _synthetic_video(src)
    hits = pd.DataFrame([{"hit_id": f"h{i}", "chunk_id": "0000_000000", "t": 0.8 + i * 0.6,
                          "box": [560.0, 300.0, 640.0, 480.0], "side": "near", "hitter_y_m": 12.0,
                          "ball_px_x": 600.0, "ball_px_y": 350.0} for i in range(3)])
    frames = video.read_clip(src, 0.0, 3.0)
    inline = process.CropWriter("a")
    inline.add(hits, frames, 0.0, FakePose())
    hits.to_parquet(tmp_path / "hits.parquet")
    (tmp_path / "b").mkdir()
    hits.to_parquet(tmp_path / "b" / "hits_raw.parquet")
    batched = process.crops_match("b", src, players=FakePose())
    assert set(batched.hit_id) == set(hits.hit_id)
    for h in hits.hit_id:
        a = cv2.imread(inline.rows[h]["crop_path"]).astype(int)
        b = cv2.imread(batched.set_index("hit_id").loc[h, "crop_path"]).astype(int)
        assert np.abs(a - b).mean() < 2.0  # same frames; only JPEG noise differs


def test_empty_cache_is_safe_without_gpu():
    tracknet.empty_cache()
