import subprocess

import numpy as np
import pandas as pd
import pytest
import torch

from tennis_pipeline import ball_labels, paths, tracknet, tracknet_train
from tennis_pipeline.tracknet import BallTrackerNet

FPS = 30


def ball_xy(t):
    """Ball centre drawn into the synthetic video (8x8 white box at x=100+100t, y=300+30t)."""
    return 100 + 100 * t + 3.5, 300 + 30 * t + 3.5


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("vid") / "synthetic.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x2a6e3c:s=1280x720:r=30:d=10",
                    "-f", "lavfi", "-i", "color=c=white:s=8x8:r=30:d=10",
                    "-filter_complex", "[0][1]overlay=x='100+100*t':y='300+30*t'",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", str(path)], check=True)
    return path


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(ball_labels, "LABELS_PATH", tmp_path / "ball_labels.csv")
    monkeypatch.setattr(ball_labels, "FRAMES_DIR", tmp_path / "frames")
    rng = np.random.default_rng(0)
    n = 290
    t = np.arange(n) / FPS
    x, y = ball_xy(t)
    detected = rng.random(n) > 0.3
    ball = pd.DataFrame({"chunk_id": "0000_000000", "frame": np.arange(n), "t": t,
                         "ball_x_px": np.where(detected, x, np.nan), "ball_y_px": np.where(detected, y, np.nan),
                         "raw_detected": detected, "is_bounce": np.arange(n) == 60,
                         "ball_ground_y_m": np.where(np.arange(n) < 55, -5.0, 5.0)})
    d = paths.match_dir("vidA")
    ball.to_parquet(d / "ball.parquet", index=False)
    pd.DataFrame({"chunk_id": "0000_000000", "frame": [20, 50, 80], "t": [20 / FPS, 50 / FPS, 80 / FPS],
                  "side": ["near", "far", "near"], "is_serve": [True, False, False]}
                 ).to_parquet(d / "hits_raw.parquet", index=False)
    pd.DataFrame({"segment_id": [0], "start": [0.2], "end": [1.2], "duration": [1.0]}
                 ).to_csv(paths.match_dir("vidB") / "segments.csv", index=False)
    return tmp_path


def test_sample_buckets_topup_and_export(workspace, video):
    labels = ball_labels.sample(["vidA", "vidB"], n=30, holdout=["vidB"])
    assert len(labels) == 30 and labels.key.is_unique
    a = labels[labels.video_id == "vidA"]
    assert set(a.bucket) >= {"miss", "serve", "bounce"}
    assert (labels[labels.video_id == "vidB"].split == "val").all()
    assert (labels[labels.video_id == "vidB"].bucket == "untracked").all()
    frames = a.sort_values("frame").frame.to_numpy()
    assert np.diff(frames).min() >= 6
    miss = a[a.bucket == "miss"]
    assert set(miss.pre_source) <= {"interp", "fit", "none", "tracknet"}

    more = ball_labels.sample(["vidA", "vidB"], n=40, holdout=["vidB"])
    assert len(more) == 40 and set(labels.key) <= set(more.key)

    n = ball_labels.export(lambda vid: video)
    assert n == 40
    r = more.iloc[0]
    assert all(p.exists() for p in ball_labels.frame_paths(r.video_id, r.t))


def label_all(visible_every=4):
    df = ball_labels.load_labels()
    x, y = ball_xy(df.t.to_numpy(float))
    df["x"], df["y"] = x, y
    df["visibility"] = 1
    df.loc[df.index[::visible_every], ["x", "y", "visibility"]] = [np.nan, np.nan, 0]
    df["status"] = "done"
    ball_labels.save_labels(df)
    return df


def test_dataset_target_follows_ball_under_flip(workspace, video):
    ball_labels.sample(["vidA"], n=12)
    ball_labels.export(lambda vid: video)
    df = label_all(visible_every=100)
    row = df[df.visibility == 1].iloc[[0]]
    for augment in (False, True):
        ds = tracknet_train.BallFrames(row, augment=augment, seed=3)
        for _ in range(4):
            inp, tgt, _, vis = ds[0]
            assert inp.shape == (9, 360, 512) and tgt.shape == (360, 512)
            if not vis:
                continue
            ty, tx = np.unravel_index(int(tgt.argmax()), tgt.shape)
            brightness = inp[:3].mean(0).numpy()
            by, bx = np.unravel_index(int(brightness[max(ty - 6, 0):ty + 7, max(tx - 6, 0):tx + 7].argmax()),
                                      brightness[max(ty - 6, 0):ty + 7, max(tx - 6, 0):tx + 7].shape)
            assert tgt.max() > 200  # sub-pixel centres peak just under 255
            assert brightness[max(ty - 6, 0) + by, max(tx - 6, 0) + bx] > 0.8  # white ball under the peak


def test_heat_target_encoding():
    tgt = tracknet_train.heat_target(640.0, 360.0, True)
    y, x = np.unravel_index(int(tgt.argmax()), tgt.shape)
    assert (x, y) == (640 // 2 - 64, 180)
    assert 5 <= (tgt > 127).sum() <= 20  # blob size the tracker's picker accepts
    assert tracknet_train.heat_target(640.0, 360.0, False).max() == 0


def test_one_epoch_fine_tune_writes_weights_and_report(workspace, video, monkeypatch):
    ball_labels.sample(["vidA", "vidB"], n=10, holdout=["vidB"])
    ball_labels.export(lambda vid: video)
    label_all()
    init = workspace / "tracknet.pt"
    torch.manual_seed(0)
    torch.save(BallTrackerNet(9, 256).state_dict(), init)
    monkeypatch.setattr(tracknet, "PRETRAINED_BALL", init)
    monkeypatch.setattr(tracknet, "FINETUNED_BALL", workspace / "tracknet_ft.pt")
    monkeypatch.setattr(tracknet, "device", lambda: torch.device("cpu"))
    rep = tracknet_train.train(epochs=1, batch=2, workers=0, min_train=2, min_val=1)
    assert (workspace / "tracknet_ft.pt").exists()
    assert (workspace / "tracknet_ft_report.json").exists()
    assert "f1@5px" in rep["val_finetuned"]["all"] and rep["history"][0]["train_loss"] > 0
    assert tracknet.weights_tag(workspace / "tracknet_ft.pt").startswith("tracknet_ft:")
