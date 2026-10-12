"""Detect points and rallies for processed matches and score them against MCP.

    uv run python scripts/rally_eval.py VIDEO_ID [VIDEO_ID ...] [--model PATH]
"""

from __future__ import annotations

import argparse
import json
import pickle

import pandas as pd

from uso import points, rally, truth


def eval_match(vid: str, model=None, serve_thr: float = 2.0, verbose: bool = True) -> dict:
    c = rally.features(vid)
    serves = rally.pick_serves(c, thr=serve_thr)
    w = None
    if model is not None:
        sc = model.scores(c)
        c = c.drop(columns=[x for x in ("hit_n", "hit_f") if x in c]).join(sc)
    m = truth.mcp_points(vid)
    vp, contacts = points.assemble(c, serves, w)
    pairs, info = points.evaluate(vp, m)
    info.update(video=vid, n_candidates=len(c), n_serves_detected=len(serves),
                mcp_serve_attempts=int(m.n_serves.sum()))
    if verbose:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}))
    return dict(info=info, c=c, serves=serves, vp=vp, contacts=contacts, pairs=pairs, mcp=m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--model")
    ap.add_argument("--serve-thr", type=float, default=2.0)
    a = ap.parse_args()
    model = pickle.load(open(a.model, "rb")) if a.model else None
    for vid in a.videos:
        eval_match(vid, model, a.serve_thr)


if __name__ == "__main__":
    main()
