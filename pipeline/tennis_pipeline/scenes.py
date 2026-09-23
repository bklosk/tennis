"""Stage 1: find main-camera segments (drop crowd shots, close-ups, replays from other angles).

Frames are sampled at 2 fps and embedded with an ImageNet MobileNetV3. A small labeled sample
(labeled by the local VLM) trains a logistic-regression classifier that is applied to every
sampled frame, then smoothed into time segments.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from scipy.ndimage import median_filter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from . import tracknet, video
from .paths import CACHE, match_dir

SAMPLE_FPS = 2.0
LABEL_DIR = CACHE / "scene_labels"
CLASSIFIER_PATH = CACHE / "scene_classifier.npz"


class Embedder:
    def __init__(self):
        self.dev = tracknet.device()
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        net = torchvision.models.mobilenet_v3_small(weights=weights)
        net.classifier = torch.nn.Identity()
        self.net = net.eval().to(self.dev)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        rgb = np.stack([cv2.resize(f, (224, 224))[:, :, ::-1] for f in frames_bgr])
        x = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).float().to(self.dev) / 255
        return self.net((x - self.mean) / self.std).cpu().numpy()


def embed_video(video_id: str, video_path: Path, n_label_samples: int = 160, seed: int = 0,
                keyframes_only: bool = False) -> np.ndarray:
    """Embed every 2 fps frame; save a random subset as JPEGs for labeling.

    `keyframes_only` decodes ~6x faster but samples the scene once per keyframe interval.
    """
    cache = match_dir(video_id) / "scene_embeddings.npz"
    if cache.exists():
        return np.load(cache)["emb"]
    info = video.probe(video_path)
    n_total = int(info["duration"] * SAMPLE_FPS)
    rng = np.random.default_rng(seed)
    label_idx = set(rng.choice(n_total, size=min(n_label_samples, n_total), replace=False).tolist())
    label_dir = LABEL_DIR / video_id
    label_dir.mkdir(parents=True, exist_ok=True)

    embedder = Embedder()
    embs, batch = [], []
    for i, frame in enumerate(video.iter_frames(video_path, SAMPLE_FPS, (640, 360), keyframes_only=keyframes_only)):
        if i in label_idx:
            cv2.imwrite(str(label_dir / f"{i:06d}.jpg"), frame)
        batch.append(frame)
        if len(batch) == 64:
            embs.append(embedder(batch))
            batch = []
    if batch:
        embs.append(embedder(batch))
    emb = np.concatenate(embs)
    np.savez_compressed(cache, t=np.arange(len(emb)) / SAMPLE_FPS, emb=emb)
    return emb


def vlm_label(video_ids: list[str]) -> pd.DataFrame:
    """Label the saved sample frames with the VLM (cached per image)."""
    from .vlm import SCENE_PROMPT, VLM

    labels_path = LABEL_DIR / "labels.jsonl"
    done = {}
    if labels_path.exists():
        for line in labels_path.read_text().splitlines():
            rec = json.loads(line)
            done[(rec["video_id"], rec["idx"])] = rec
    model = VLM()
    with labels_path.open("a") as fh:
        for vid in video_ids:
            for img in sorted((LABEL_DIR / vid).glob("*.jpg")):
                key = (vid, int(img.stem))
                if key in done:
                    continue
                rec = {"video_id": vid, "idx": int(img.stem),
                       "main_camera": model.ask_yes_no(str(img), SCENE_PROMPT)}
                done[key] = rec
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
    return pd.DataFrame([r for r in done.values() if r["video_id"] in video_ids])


def court_label(video_ids: list[str]) -> pd.DataFrame:
    """Label the saved sample frames by whether the court-keypoint model calibrates them.

    The main game camera is the only broadcast view that shows the whole court from behind a
    baseline, which is what the court model needs. Used where the MLX VLM is unavailable (Linux).
    """
    from .court import CourtDetector

    labels_path = LABEL_DIR / "court_labels.jsonl"
    done = {}
    if labels_path.exists():
        for line in labels_path.read_text().splitlines():
            rec = json.loads(line)
            done[(rec["video_id"], rec["idx"])] = rec
    det = CourtDetector()
    with labels_path.open("a") as fh:
        for vid in video_ids:
            imgs = [p for p in sorted((LABEL_DIR / vid).glob("*.jpg")) if (vid, int(p.stem)) not in done]
            for k in range(0, len(imgs), 16):
                batch = imgs[k:k + 16]
                frames = [cv2.resize(cv2.imread(str(p)), (1280, 720)) for p in batch]
                for p, cal in zip(batch, det.calibrate(frames)):
                    rec = {"video_id": vid, "idx": int(p.stem), "labeler": "court",
                           "main_camera": bool(cal is not None and cal.reproj_px < 6)}
                    done[(vid, rec["idx"])] = rec
                    fh.write(json.dumps(rec) + "\n")
    return pd.DataFrame([r for r in done.values() if r["video_id"] in video_ids])


def _vlm_available() -> bool:
    try:
        import mlx_vlm  # noqa: F401
    except ImportError:
        return False
    return True


def train_classifier(video_ids: list[str], labeler: str = "auto") -> dict:
    if labeler == "auto":
        labeler = "vlm" if _vlm_available() else "court"
    labels = (vlm_label(video_ids) if labeler == "vlm" else court_label(video_ids)).dropna(subset=["main_camera"])
    X, y = [], []
    for vid, grp in labels.groupby("video_id"):
        emb = np.load(match_dir(vid) / "scene_embeddings.npz")["emb"]
        X.append(emb[grp["idx"].to_numpy()])
        y.append(grp["main_camera"].astype(int).to_numpy())
    X, y = np.concatenate(X), np.concatenate(y)
    clf = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")
    cv_pred = cross_val_predict(clf, X, y, cv=5)
    clf.fit(X, y)
    np.savez(CLASSIFIER_PATH, coef=clf.coef_, intercept=clf.intercept_)
    return {
        "labeler": labeler,
        "n_labels": int(len(y)),
        "positive_rate": float(y.mean()),
        "cv_accuracy_vs_labels": float((cv_pred == y).mean()),
    }


def scene_prob(video_id: str) -> np.ndarray:
    """Main-camera probability for each 2 fps sample (index = t * SAMPLE_FPS).

    Uses the cached probabilities from `save_scene_prob` when the embeddings are not present
    (batch runs upload the probabilities, not the much larger embeddings).
    """
    emb_path = match_dir(video_id) / "scene_embeddings.npz"
    if not emb_path.exists() and (match_dir(video_id) / "scene_prob.npz").exists():
        return np.load(match_dir(video_id) / "scene_prob.npz")["prob"]
    z = np.load(emb_path)
    c = np.load(CLASSIFIER_PATH)
    logits = z["emb"] @ c["coef"].ravel() + c["intercept"][0]
    return 1 / (1 + np.exp(-logits))


def save_scene_prob(video_id: str) -> np.ndarray:
    prob = scene_prob(video_id)
    np.savez_compressed(match_dir(video_id) / "scene_prob.npz", prob=prob.astype(np.float32))
    return prob


def predict_segments(video_id: str, min_len: float = 2.5, pad: float = 0.75,
                     merge_gap: float = 1.5) -> pd.DataFrame:
    prob = scene_prob(video_id)
    keep = median_filter((prob > 0.5).astype(np.uint8), size=5).astype(bool)
    dt = 1 / SAMPLE_FPS
    segs, start = [], None
    for i, k in enumerate(np.append(keep, False)):
        if k and start is None:
            start = i
        elif not k and start is not None:
            segs.append([start * dt - pad, i * dt + pad])
            start = None
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] < merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    df = pd.DataFrame(merged, columns=["start", "end"])
    df["start"] = df["start"].clip(lower=0)
    df = df[df["end"] - df["start"] >= min_len].reset_index(drop=True)
    df["duration"] = df["end"] - df["start"]
    df.insert(0, "segment_id", range(len(df)))
    df.to_csv(match_dir(video_id) / "segments.csv", index=False)
    return df
