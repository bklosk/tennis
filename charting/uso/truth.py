"""Charted (MCP) truth for the evaluation matches (and MCP-charted batch matches), in the shape the
aligner needs."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from uso import mcp, scoring
from uso.paths import BATCH_MATCHES, EVAL_MATCHES


@lru_cache(maxsize=1)
def eval_matches() -> pd.DataFrame:
    return pd.read_csv(EVAL_MATCHES, dtype={"year": int})


@lru_cache(maxsize=1)
def batch_matches() -> pd.DataFrame:
    """Archive matches decoded for analytics. role is 'charted' (has an MCP record) or 'uncharted'."""
    if not BATCH_MATCHES.exists():
        return eval_matches().iloc[0:0]
    return pd.read_csv(BATCH_MATCHES, dtype={"year": int, "mcp_match_id": str})


def match_row(video_id: str) -> pd.Series:
    """Evaluation matches first, then MCP-charted batch matches (same columns)."""
    for t in (eval_matches(), batch_matches()):
        r = t[(t.video_id == video_id) & t.mcp_match_id.notna()]
        if len(r):
            return r.iloc[0]
    raise KeyError(f"{video_id} has no MCP record in eval_matches.csv or batch_matches.csv")


@lru_cache(maxsize=32)
def mcp_points(video_id: str) -> pd.DataFrame:
    """One row per MCP point with server, side, end parity, serve attempts and contacts.
    server_end_a assumes Player 1 starts at the near (camera) end; server_end_b the opposite."""
    m = match_row(video_id)
    mid = m.mcp_match_id
    matches = mcp.load_matches()
    pts = mcp.load_points(match_ids=[mid])
    tbl = mcp.points_table(pts, matches)
    best_of = scoring.grand_slam_best_of(m.gender, m["round"])
    first_server = int(pts.Svr.dropna().iloc[0])
    ann = scoring.annotate_points(pts, first_server, best_of, "US Open", int(m.year))
    tbl = tbl.reset_index(drop=True)
    ann = ann.reset_index(drop=True)
    out = pd.DataFrame({
        "Pt": tbl.Pt, "server": tbl.server.astype("Int64"), "side": ann.side,
        "swapped": ann.ends_swapped_before_point, "n_serves": tbl.n_serve_attempts,
        "rally": tbl.n_contacts, "outcome": tbl.outcome, "winner": tbl.PtWinner,
        "set_no": ann.set_no, "game_no": ann.game_no_in_match, "in_tiebreak": ann.in_tiebreak,
        "first": pts["1st"].reset_index(drop=True), "second": pts["2nd"].reset_index(drop=True),
        "parse_ok": tbl.parse_ok,
    })
    p1_near_a = out.swapped == 0
    srv1 = out.server == 1
    out["server_end_a"] = ["near" if a == b else "far" for a, b in zip(srv1, p1_near_a)]
    out["server_end_b"] = ["far" if e == "near" else "near" for e in out.server_end_a]
    return out


def mcp_shots(video_id: str) -> pd.DataFrame:
    m = match_row(video_id)
    matches = mcp.load_matches()
    pts = mcp.load_points(match_ids=[m.mcp_match_id])
    return mcp.shots_table(pts, matches)
