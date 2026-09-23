"""Fine-tune the TrackNet ball model on the labeled US Open frames in `eval/ball_labels.csv`.

The input matches tracking exactly: current frame and the two before it at 640x360, cropped to
x in [64, 576). The target is the yastrebksv encoding (a Gaussian blob with variance 10 px^2 at
1280x720, quantised to 0..255 and trained as 256-way per-pixel classification), so the
pretrained output head is reused unchanged and `BallTracker` needs no code changes.

The held-out split is scored with the tracker's own blob picker, for the pretrained and the
fine-tuned weights, overall and per sampling bucket (miss, serve, bounce, far, random).
"""
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from . import ball_labels, tracknet
from .track import BALL_CROP_X, BallTracker

W_FULL, H = 640, 360
X0, X1 = BALL_CROP_X
SIGMA2 = 10.0 / 4  # variance 10 at 1280x720 is 2.5 at 640x360


def heat_target(x: float, y: float, visible: bool) -> np.ndarray:
    """(H, X1-X0) class map in crop space; x, y in 1280x720 pixels."""
    tgt = np.zeros((H, X1 - X0), np.int64)
    if not visible or np.isnan(x):
        return tgt
    cx, cy = x / 2 - X0, y / 2
    yy, xx = np.mgrid[0:H, 0:X1 - X0]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * SIGMA2))
    return (g * 255).astype(np.int64)


class BallFrames(Dataset):
    def __init__(self, rows: pd.DataFrame, augment: bool = False, seed: int = 0):
        self.rows = rows.reset_index(drop=True)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        imgs = [cv2.resize(cv2.imread(str(p)), (W_FULL, H)) for p in ball_labels.frame_paths(r.video_id, r.t)]
        vis = int(r.visibility)
        x, y = (float(r.x), float(r.y)) if vis > 0 else (np.nan, np.nan)
        if self.augment:
            imgs, x, y = self._augment(imgs, x, y)
        crop = np.concatenate([im[:, X0:X1] for im in imgs], 2)  # current, -1, -2 (BGR, as tracked)
        inp = torch.from_numpy(crop.transpose(2, 0, 1).astype(np.float32) / 255)
        visible = vis > 0 and not np.isnan(x) and X0 <= x / 2 < X1 and 0 <= y / 2 < H
        tgt = torch.from_numpy(heat_target(x, y, visible))
        return inp, tgt, torch.tensor([x if visible else -1.0, y if visible else -1.0]), torch.tensor(int(visible))

    def _augment(self, imgs, x, y):
        rng = self.rng
        if rng.random() < 0.5:
            imgs = [im[:, ::-1].copy() for im in imgs]
            x = 2 * (W_FULL - 1) - x if not np.isnan(x) else x  # column i -> 639 - i at 640 wide
        dx, dy = rng.integers(-24, 25), rng.integers(-16, 17)
        if dx or dy:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            imgs = [cv2.warpAffine(im, M, (W_FULL, H), borderMode=cv2.BORDER_REPLICATE) for im in imgs]
            if not np.isnan(x):
                x, y = x + 2 * dx, y + 2 * dy
        a, b = rng.uniform(0.8, 1.2), rng.uniform(-20, 20)
        imgs = [np.clip(im.astype(np.float32) * a + b, 0, 255).astype(np.uint8) for im in imgs]
        return imgs, x, y


def labeled_rows(split: str | None = None) -> pd.DataFrame:
    df = ball_labels.load_labels()
    df = df[(df.status == "done") & df.visibility.notna()]
    if split:
        df = df[df.split == split]
    have = df.apply(lambda r: all(p.exists() for p in ball_labels.frame_paths(r.video_id, r.t)), axis=1)
    return df[have] if len(df) else df


def _freeze_bn(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


@torch.no_grad()
def evaluate(model: nn.Module, rows: pd.DataFrame, dev: torch.device, tols=(5.0, 10.0), batch: int = 8) -> dict:
    model.eval()
    ds = BallFrames(rows)
    preds = []
    for s in range(0, len(ds), batch):
        items = [ds[i] for i in range(s, min(s + batch, len(ds)))]
        masks = (model(torch.stack([it[0] for it in items]).to(dev)).argmax(1) > 127).to(torch.uint8).cpu().numpy()
        for mask in masks:
            xy = BallTracker._pick(mask, None)
            preds.append((np.nan, np.nan) if xy is None else ((xy[0] + X0) * 2, xy[1] * 2))
    res = rows[["bucket", "visibility", "x", "y"]].copy().reset_index(drop=True)
    res["px"], res["py"] = [p[0] for p in preds], [p[1] for p in preds]
    res["err"] = np.hypot(res.px - res.x, res.py - res.y)
    visible = res.visibility > 0
    found = res.px.notna()

    def score(d):
        out = {"n": int(len(d)), "visible": int((d.visibility > 0).sum())}
        for tol in tols:
            tp = int(((d.visibility > 0) & (d.err < tol)).sum())
            wrong = int((d.px.notna() & ~((d.visibility > 0) & (d.err < tol))).sum())
            fn = int(((d.visibility > 0) & ~(d.err < tol)).sum())
            p = tp / max(tp + wrong, 1)
            r = tp / max(tp + fn, 1)
            out[f"precision@{tol:g}px"] = round(p, 4)
            out[f"recall@{tol:g}px"] = round(r, 4)
            out[f"f1@{tol:g}px"] = round(2 * p * r / max(p + r, 1e-9), 4)
        hits = d[(d.visibility > 0) & (d.err < tols[-1])]
        out["median_err_px"] = round(float(hits.err.median()), 2) if len(hits) else None
        return out

    report = {"all": score(res), "detected_when_visible": round(float((found & visible).sum() / max(visible.sum(), 1)), 4)}
    report["by_bucket"] = {b: score(g) for b, g in res.groupby("bucket")}
    return report


def train(epochs: int = 30, batch: int = 4, lr: float = 1e-4, freeze_bn: bool = True, init: str | None = None,
          out: str | None = None, workers: int = 2, seed: int = 0, min_train: int = 50, min_val: int = 20) -> dict:
    torch.manual_seed(seed)
    dev = tracknet.device()
    train_rows, val_rows = labeled_rows("train"), labeled_rows("val")
    if len(train_rows) < min_train or len(val_rows) < min_val:
        raise SystemExit(f"need labeled frames: {len(train_rows)} train / {len(val_rows)} val "
                         "(run balllabels sample/export/review first)")
    init_path = tracknet.ball_weights(init) if init else tracknet.PRETRAINED_BALL
    out_path = tracknet.FINETUNED_BALL if out is None else Path(out)
    model = tracknet.load("ball", dev, init_path)
    baseline = evaluate(model, val_rows, dev)
    loader = DataLoader(BallFrames(train_rows, augment=True, seed=seed), batch_size=batch, shuffle=True,
                        num_workers=workers, drop_last=True, persistent_workers=workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / steps, 1))))
    loss_fn = nn.CrossEntropyLoss()
    best, best_f1, history = None, -1.0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        if freeze_bn:
            _freeze_bn(model)
        total = 0.0
        for inp, tgt, _, _ in loader:
            loss = loss_fn(model(inp.to(dev)), tgt.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        val = evaluate(model, val_rows, dev)
        f1 = val["all"]["f1@5px"]
        history.append({"epoch": ep + 1, "train_loss": round(total / max(len(loader), 1), 5), "val_f1@5px": f1,
                        "val_recall@5px": val["all"]["recall@5px"], "minutes": round((time.time() - t0) / 60, 1)})
        print(json.dumps(history[-1]))
        if f1 > best_f1:
            best_f1, best = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best, out_path)
    model.load_state_dict(best)
    report = {"init": str(init_path), "weights": str(out_path), "train_frames": int(len(train_rows)),
              "val_frames": int(len(val_rows)), "epochs": epochs, "history": history,
              "val_pretrained": baseline, "val_finetuned": evaluate(model, val_rows, dev)}
    out_path.with_name(out_path.stem + "_report.json").write_text(json.dumps(report, indent=2))
    return report


def compare(weights: list[str]) -> dict:
    """Score any set of ball weights on the held-out labeled frames."""
    dev = tracknet.device()
    rows = labeled_rows("val")
    return {w: evaluate(tracknet.load("ball", dev, w), rows, dev) for w in weights}
