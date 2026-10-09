"""Learned rally-contact model, trained from MCP-aligned points.

For an aligned point MCP gives the number of contacts but not their times. Inside the point's
window a constrained decoder picks exactly that many alternating contacts from the audio
candidates (using the current scores); those become positives and the rest negatives. A gradient
boosted classifier then scores p(contact by hitter h | features of h and the opponent), and the
loop can be repeated with the new scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from uso import rally

AUDIO = ["z", "flux", "bg", "rise_b0", "rise_b1", "rise_b2", "rise_b3", "decay_b2", "lvl_b1", "lvl_b2", "sharp",
         "dt_prev", "dt_next"]
PLAYER = ["y", "x", "spd_before", "spd_after", "wrist_ext_at", "arm_up_at", "arm_up_pre", "both_up_at",
          "wrists_apart_at", "torso_px_at"]


API_ACTIONS = ("serve", "hit", "bounce", "none")


def hitter_frame(c: pd.DataFrame, hitter: str) -> pd.DataFrame:
    """Features for 'this onset is a contact by `hitter`' ('near'/'far'), hitter-relative: audio,
    Decisions-API action probabilities for hitter and opponent, positions, speeds and pose cues."""
    h, o = ("n", "f") if hitter == "near" else ("f", "n")
    hw, ow = ("near", "far") if hitter == "near" else ("far", "near")
    X = c[[a for a in AUDIO if a in c]].copy()
    for who, tag in ((hw, "h"), (ow, "o")):
        for act in API_ACTIONS:
            col = f"{who}_{act}"
            X[f"{tag}_api_{act}"] = c[col] if col in c else np.nan
    for p, tag in ((h, "h"), (o, "o")):
        for f in PLAYER:
            col = f"{p}_{f}"
            X[f"{tag}_{f}"] = c[col] if col in c else np.nan
    X["h_y"] = X["h_y"].abs()
    X["o_y"] = X["o_y"].abs()
    X["h_x"] = X["h_x"].abs()
    X["o_x"] = X["o_x"].abs()
    X["is_far"] = 1.0 if hitter == "far" else 0.0
    return X


def decode_fixed(t: np.ndarray, s_near: np.ndarray, s_far: np.ndarray, t_serve: float, first: str, k: int,
                 max_gap: float = 3.2) -> list[int] | None:
    """Exactly k alternating contacts after the serve (k >= 1), maximising score + interval prior."""
    n = len(t)
    if k <= 0:
        return []
    if n < k:
        return None
    NEG = -1e18
    best = np.full((n, k), NEG)
    back = -np.ones((n, k), int)
    for i in range(n):
        for j in range(min(k, i + 1)):
            hitter_near = (first == "near") == (j % 2 == 0)
            s_i = s_near[i] if hitter_near else s_far[i]
            if j == 0:
                dt = t[i] - t_serve
                if 0.3 <= dt <= max_gap:
                    best[i, 0] = s_i + rally.interval_logprior(np.array([dt]))[0]
                continue
            for p in range(i):
                if best[p, j - 1] <= NEG / 2:
                    continue
                dt = t[i] - t[p]
                if dt < 0.3 or dt > max_gap:
                    continue
                v = best[p, j - 1] + s_i + rally.interval_logprior(np.array([dt]))[0]
                if v > best[i, j]:
                    best[i, j], back[i, j] = v, p
    i = int(np.argmax(best[:, k - 1]))
    if best[i, k - 1] <= NEG / 2 or not np.isfinite(best[i, k - 1]):
        return None
    path = [i]
    for j in range(k - 1, 0, -1):
        i = int(back[i, j])
        path.append(i)
    return path[::-1]


def make_labels(c: pd.DataFrame, pairs: pd.DataFrame, video_pts: pd.DataFrame, contacts: pd.DataFrame,
                scores: pd.DataFrame) -> pd.DataFrame:
    """Rows: (onset index, hitter, label) from aligned points with the right server."""
    c = c.drop(columns=[x for x in ("hit_n", "hit_f") if x in c]).join(scores[["hit_n", "hit_f"]])
    serves = contacts[contacts.kind == "serve"]
    starts = video_pts.sort_values("t_start").t_start.to_numpy()
    rows = []
    for r in pairs.itertuples():
        if not r.server_ok or pd.isna(r.m_rally):
            continue
        k = int(r.m_rally) - 1
        sv = serves[(serves.pid == r.v_pid) & serves.in_play]
        if sv.empty or k < 0:
            continue
        t0 = float(sv.t.iloc[0])
        nxt = starts[starts > t0 + 0.5]
        t1 = min(float(nxt[0]) - 0.5 if len(nxt) else np.inf, t0 + 3.2 * (k + 1) + 2.0)
        w = c[(c.seg == r.v_seg) & (c.t > t0 + 0.05) & (c.t < t1)]
        first = "far" if r.v_server_end == "near" else "near"
        idx = decode_fixed(w.t.to_numpy(), w.hit_n.to_numpy(), w.hit_f.to_numpy(), t0, first, k) if k > 0 else []
        if idx is None:
            continue
        chosen = {w.index[i]: ("near" if ((first == "near") == (j % 2 == 0)) else "far") for j, i in enumerate(idx)}
        for oi in w.index:
            for hitter in ("near", "far"):
                rows.append(dict(onset=oi, hitter=hitter, y=int(chosen.get(oi) == hitter), point=r.v_pid))
    return pd.DataFrame(rows)


class HitModel:
    def __init__(self):
        self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                  l2_regularization=1.0, random_state=0)
        self.cols: list[str] | None = None

    def fit(self, frames: list[tuple[pd.DataFrame, pd.DataFrame]]):
        """frames: list of (candidates, labels) per match."""
        Xs, ys = [], []
        for c, lab in frames:
            for hitter in ("near", "far"):
                lh = lab[lab.hitter == hitter]
                X = hitter_frame(c.loc[lh.onset], hitter)
                Xs.append(X)
                ys.append(lh.y.to_numpy())
        X = pd.concat(Xs, ignore_index=True)
        y = np.concatenate(ys)
        self.cols = list(X.columns)
        self.clf.fit(X, y)
        self.prior = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
        return self

    def scores(self, c: pd.DataFrame, bias: float = 0.0) -> pd.DataFrame:
        """Log likelihood ratio (posterior log-odds minus the training base rate's) plus a bias.
        The decoder supplies the structure (alternation, timing), so the per-onset score should be
        evidence, not posterior: with ~3% positives a posterior would veto almost everything."""
        out = {}
        base = np.log(self.prior / (1 - self.prior))
        for hitter, key in (("near", "hit_n"), ("far", "hit_f")):
            X = hitter_frame(c, hitter)[self.cols]
            p = np.clip(self.clf.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
            out[key] = np.log(p / (1 - p)) - base + bias
        return pd.DataFrame(out, index=c.index)
