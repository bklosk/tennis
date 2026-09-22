"""Stage 6: stroke type (serve / forehand / backhand / overhead) and volley flags.

The production label is a geometric rule: which side of the hitter's body the ball is on at
contact, in the player's own frame (far players face the camera; left-handers are mirrored).
On hand-checked gold labels (eval/stroke_gold.csv) it beats both Qwen3-VL-8B labels and a
classifier trained on them, so VLM labeling is kept only as an opt-in experiment.

Hitter crops are mirrored into a canonical view in which a forehand always has the racket on
the image's right, which is what the VLM experiment and the gold review look at.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from .court import SERVICE_LINE
from .paths import CACHE, match_dir

LABELS_PATH = CACHE / "stroke_labels.jsonl"
CANON_DIR = CACHE / "stroke_canonical"
RACKET_PROMPT = (
    "This image shows a tennis player at the moment of hitting the ball. Relative to the "
    "player's torso, is the racket on the LEFT side or the RIGHT side of the image? "
    "Answer with one word: left or right."
)
POSE_FEATURES = ["ball_lat", "dom_wrist_lat", "nondom_wrist_lat", "wrist_gap", "ball_height", "is_near"]


def _egocentric(shots: pd.DataFrame) -> pd.DataFrame:
    s = shots.copy()
    sign = np.where(s.hand == "L", -1.0, 1.0)
    for col in ("ball_lat", "l_wrist_lat", "r_wrist_lat", "wrist_gap", "ball_height", "wrist_above_sh"):
        if col not in s:
            s[col] = np.nan
    s["ball_lat"] = s.ball_lat * sign
    lefty = s.hand == "L"
    s["dom_wrist_lat"] = np.where(lefty, -s.l_wrist_lat, s.r_wrist_lat)
    s["nondom_wrist_lat"] = np.where(lefty, -s.r_wrist_lat, s.l_wrist_lat)
    s["is_near"] = (s.side == "near").astype(float)
    return s


def rule_stroke(r) -> str:
    if r.is_serve:
        return "serve"
    if r.ball_above_head and (r.wrist_above_sh if not pd.isna(r.wrist_above_sh) else 0) > 0.15:
        return "overhead"
    lat = r.ball_lat if not pd.isna(r.ball_lat) else r.dom_wrist_lat
    if pd.isna(lat):
        return "unknown"
    return "forehand" if lat > 0 else "backhand"


def canonical_crop(r) -> str | None:
    """Middle (contact) frame of the strip, mirrored so the forehand side is image-right."""
    if not isinstance(r.crop_path, str):
        return None
    mirror = (r.side == "far") != (r.hand == "L")
    out = CANON_DIR / f"{r.video_id}_{r.hit_id}_{'m' if mirror else 'n'}.jpg"
    if not out.exists():
        strip = cv2.imread(r.crop_path)
        if strip is None:
            return None
        w = strip.shape[1] // 3
        mid = cv2.resize(strip[:, w:2 * w], (512, 512), interpolation=cv2.INTER_CUBIC)
        if mirror:
            mid = mid[:, ::-1]
        CANON_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), mid)
    return str(out)


def vlm_label(shots: pd.DataFrame, per_match: int = 150, seed: int = 0) -> pd.DataFrame:
    from .vlm import VLM

    done = {}
    if LABELS_PATH.exists():
        for line in LABELS_PATH.read_text().splitlines():
            rec = json.loads(line)
            done[rec["key"]] = rec
    pool = shots[shots.canon_path.notna() & (shots.stroke_rule.isin(["forehand", "backhand"]))]
    parts = [g.sample(min(len(g), per_match // 2), random_state=seed) for _, g in pool.groupby(["video_id", "side"])]
    sample = pd.concat(parts) if parts else pool.iloc[0:0]
    model = VLM()
    with LABELS_PATH.open("a") as fh:
        for r in sample.itertuples():
            key = f"{r.video_id}_{r.hit_id}"
            if key in done:
                continue
            ans = model.ask(r.canon_path, RACKET_PROMPT, max_tokens=4).lower().strip(". ")
            stroke = {"right": "forehand", "left": "backhand"}.get(ans)
            rec = {"key": key, "racket_side": ans, "stroke": stroke}
            done[key] = rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    labels = pd.DataFrame(list(done.values()))
    return labels


def vlm_experiment(shots: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Qwen3-VL labels on a sample + a crop-embedding classifier trained on them."""
    from .scenes import Embedder

    labels = vlm_label(shots)
    shots = shots.merge(labels[["key", "stroke"]].rename(columns={"stroke": "stroke_vlm"}), on="key", how="left")
    has_img = shots.canon_path.notna()
    emb = np.zeros((len(shots), 576), np.float32)
    embedder = Embedder()
    idx = np.where(has_img)[0]
    for k in range(0, len(idx), 64):
        batch = idx[k:k + 64]
        emb[batch] = embedder([cv2.imread(shots.canon_path.iloc[i]) for i in batch])
    X = np.concatenate([emb, shots[POSE_FEATURES].fillna(0).to_numpy()], 1)
    groundstroke = shots.stroke_rule.isin(["forehand", "backhand"]) & has_img
    train = groundstroke & shots.stroke_vlm.isin(["forehand", "backhand"])
    y = (shots.loc[train, "stroke_vlm"] == "forehand").astype(int).to_numpy()
    report = {"n_vlm_labels": int(train.sum()),
              "rule_vs_vlm_agreement": float((shots.loc[train, "stroke_rule"] == shots.loc[train, "stroke_vlm"]).mean())}
    shots["stroke_model"] = None
    if train.sum() >= 40 and len(set(y)) == 2:
        clf = LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")
        report["model_cv_agreement_vs_vlm"] = float((cross_val_predict(clf, X[train.to_numpy()], y, cv=5) == y).mean())
        clf.fit(X[train.to_numpy()], y)
        pred = clf.predict(X[groundstroke.to_numpy()])
        shots.loc[groundstroke, "stroke_model"] = np.where(pred == 1, "forehand", "backhand")
    return shots, report


def run(video_ids: list[str], vlm: bool = False) -> dict:
    frames = []
    for vid in video_ids:
        s = pd.read_parquet(match_dir(vid) / "shots_aligned.parquet")
        s["video_id"] = vid
        frames.append(s)
    shots = _egocentric(pd.concat(frames, ignore_index=True))
    shots["stroke_rule"] = [rule_stroke(r) for r in shots.itertuples()]
    shots["canon_path"] = [canonical_crop(r) if isinstance(r.hand, str) else None for r in shots.itertuples()]
    shots["key"] = shots.video_id + "_" + shots.hit_id
    report = {}
    if vlm:
        shots, report = vlm_experiment(shots)
    shots["stroke"] = shots.stroke_rule
    shots["stroke_source"] = "rule"

    gold_path = Path(__file__).resolve().parents[1] / "eval" / "stroke_gold.csv"
    if gold_path.exists():
        gold = pd.read_csv(gold_path).merge(shots, on=["video_id", "hit_id"], suffixes=("_gold", ""))
        report["gold_n"] = int(len(gold))
        report["rule_accuracy_vs_gold"] = float((gold.stroke_gold == gold.stroke).mean()) if len(gold) else None
        report["rule_accuracy_by_side"] = gold.groupby("side").apply(
            lambda d: float((d.stroke_gold == d.stroke).mean()), include_groups=False).to_dict()
        for col in ("stroke_vlm", "stroke_model"):
            if col in gold:
                g = gold[gold[col].notna()]
                report[f"{col}_accuracy_vs_gold"] = float((g.stroke_gold == g[col]).mean()) if len(g) else None

    near_net = shots.hitter_y_m.abs() < SERVICE_LINE + 1.0
    shots["is_volley"] = (~shots.is_serve) & near_net & ~shots.bounced_before_hit_own_side.fillna(False).astype(bool)

    for vid, grp in shots.groupby("video_id"):
        grp.to_parquet(match_dir(vid) / "shots_strokes.parquet", index=False)
    report["stroke_counts"] = shots.stroke.value_counts().to_dict()
    report["volleys"] = int(shots.is_volley.sum())
    (CACHE / "stroke_report.json").write_text(json.dumps(report, indent=2))
    return report
