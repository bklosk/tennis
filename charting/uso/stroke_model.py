"""Stroke side and family classifiers fused from local pose features and Decisions API answers.

Trained on dev matches with MCP labels (paired shots), evaluated on held-out matches. Both models
are gradient-boosted trees; probabilities are calibrated with isotonic regression on
leave-one-match-out predictions so they can be thresholded honestly.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from uso import strokes
from uso.paths import OUTPUTS, match_dir

FAM_CLASSES = ["groundstroke", "slice", "volley", "overhead", "lob", "drop_shot"]


def match_table(video_id: str) -> pd.DataFrame:
    """Rally contacts with local features, API answers and MCP labels (when paired)."""
    d = match_dir(video_id)
    sh = pd.read_parquet(d / "shots.parquet")
    sh = sh[sh.kind == "rally"].copy()
    f = pd.read_parquet(d / "stroke_feats.parquet")
    out = sh.merge(f.drop(columns=["dist_net"], errors="ignore"), on="hit_id", how="left")
    for name in ("api_strokes.parquet", "api_imgside.parquet"):
        p = d / name
        if p.exists():
            out = out.merge(pd.read_parquet(p), on="hit_id", how="left")
    out["video_id"] = video_id
    # context: time since the previous contact in the point, hitter speed, opponent depth
    out = out.sort_values(["pid", "t"])
    out["dt_prev"] = out.groupby("pid").t.diff()
    out["opp_dnet"] = out.oy.abs()
    out["hit_speed"] = np.hypot(out.hvx, out.hvy)
    out["mcp_fam"] = out.mcp_family.map(strokes.mcp_family) if "mcp_family" in out else None
    return out


def feature_cols(df: pd.DataFrame) -> list[str]:
    skip = {"hit_id", "pid", "t", "seg", "kind", "serve_no", "hitter_end", "in_play", "shot_no", "hx", "hy", "ox", "oy",
            "hvx", "hvy", "mcp_pt", "hitter", "hitter_name", "hitter_hand", "mcp_letter", "mcp_side", "mcp_family",
            "mcp_hitter", "point_count_exact", "half", "video_id", "mcp_fam", "api_tokens"}
    return [c for c in df.columns if c not in skip and df[c].dtype.kind in "fiub"]


class StrokeModel:
    def __init__(self):
        self.side = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                                                   l2_regularization=1.0, random_state=0)
        self.fam = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                                                  l2_regularization=1.0, class_weight="balanced", random_state=0)
        self.cols: list[str] = []
        self.calib: IsotonicRegression | None = None

    def fit(self, df: pd.DataFrame, calib_scores: np.ndarray | None = None, calib_y: np.ndarray | None = None):
        self.cols = feature_cols(df)
        s = df[df.mcp_side.isin(["F", "B"])]
        self.side.fit(s[self.cols], (s.mcp_side == "F").astype(int))
        f = df[df.mcp_fam.isin(FAM_CLASSES)]
        self.fam.fit(f[self.cols], f.mcp_fam)
        if calib_scores is not None:
            self.calib = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(calib_scores, calib_y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.reindex(columns=self.cols)
        p = self.side.predict_proba(X)[:, 1]
        if self.calib is not None:
            p = self.calib.predict(p)
        fam_p = self.fam.predict_proba(X)
        out = pd.DataFrame({"hit_id": df.hit_id.to_numpy(), "p_forehand": p})
        for k, c in enumerate(self.fam.classes_):
            out[f"p_{c}"] = fam_p[:, k]
        out["family"] = np.asarray(self.fam.classes_)[fam_p.argmax(1)]
        out["side"] = np.where(p >= 0.5, "F", "B")
        return out


def loo(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-match-out predictions (for model selection and calibration)."""
    preds = []
    for v in df.video_id.unique():
        tr, te = df[df.video_id != v], df[df.video_id == v]
        m = StrokeModel().fit(tr)
        preds.append(m.predict(te).assign(video_id=v))
    return pd.concat(preds, ignore_index=True)


def save(m: StrokeModel, name: str = "stroke_model.pkl"):
    with open(OUTPUTS / name, "wb") as f:
        pickle.dump(m, f)


def load(name: str = "stroke_model.pkl") -> StrokeModel:
    with open(OUTPUTS / name, "rb") as f:
        return pickle.load(f)
