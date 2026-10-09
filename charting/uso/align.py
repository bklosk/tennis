"""Global alignment of detected video points to charted (MCP) points.

Both sequences are in match order. Matching costs use what the video can tell us about a point:
which end served (from the scoring state machine plus the unknown starting ends), the serve side
(deuce/ad), the number of serve attempts, and the rally length. Gaps are allowed on both sides:
broadcasts miss points and the detector produces spurious ones (replays, warm-up hits).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _match_score(v: dict, m: dict) -> float:
    s = 0.0
    s += 2.0 if v["server_end"] == m["server_end"] else -3.0
    if v.get("side") and m.get("side"):
        s += 1.0 if v["side"] == m["side"] else -1.0
    if v.get("n_serves") and m.get("n_serves"):
        s += 0.6 if v["n_serves"] == m["n_serves"] else -0.4
    rv, rm = v.get("rally"), m.get("rally")
    if rv is not None and rm is not None and rm > 0:
        d = abs(rv - rm)
        s += 1.2 - 0.6 * min(d, 4) - 0.1 * max(0, d - 4)
    return s


def needleman_wunsch(video: list[dict], mcp: list[dict], gap_v: float = -1.2, gap_m: float = -0.8):
    """Return list of (i_video | None, j_mcp | None) pairs and the total score."""
    n, m = len(video), len(mcp)
    S = np.zeros((n + 1, m + 1))
    B = np.zeros((n + 1, m + 1), np.int8)  # 0 diag, 1 up (skip video), 2 left (skip mcp)
    S[1:, 0] = gap_v * np.arange(1, n + 1)
    S[0, 1:] = gap_m * np.arange(1, m + 1)
    B[1:, 0] = 1
    B[0, 1:] = 2
    for i in range(1, n + 1):
        vi = video[i - 1]
        for j in range(1, m + 1):
            d = S[i - 1, j - 1] + _match_score(vi, mcp[j - 1])
            u = S[i - 1, j] + gap_v
            l = S[i, j - 1] + gap_m
            if d >= u and d >= l:
                S[i, j], B[i, j] = d, 0
            elif u >= l:
                S[i, j], B[i, j] = u, 1
            else:
                S[i, j], B[i, j] = l, 2
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        b = B[i, j]
        if i > 0 and j > 0 and b == 0:
            pairs.append((i - 1, j - 1)); i, j = i - 1, j - 1
        elif i > 0 and (j == 0 or b == 1):
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    return pairs[::-1], float(S[n, m])


def align(video_pts: pd.DataFrame, mcp_pts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """video_pts: server_end ('near'/'far'), side, n_serves, rally. mcp_pts: server_end_if_p1_near
    computed for both possible starting layouts via column 'server_end_a' / 'server_end_b'.
    Tries both starting layouts and keeps the better alignment."""
    best = None
    v = video_pts.to_dict("records")
    for layout in ("a", "b"):
        m = mcp_pts.rename(columns={f"server_end_{layout}": "server_end"}).to_dict("records")
        pairs, score = needleman_wunsch(v, m)
        if best is None or score > best[1]:
            best = (pairs, score, layout)
    pairs, score, layout = best
    rows = []
    for i, j in pairs:
        rows.append(dict(vi=i, mj=j))
    al = pd.DataFrame(rows)
    matched = al.dropna()
    info = dict(score=score, layout=layout, n_video=len(video_pts), n_mcp=len(mcp_pts), n_matched=len(matched))
    return al, info
