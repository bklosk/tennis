"""Self-train the rally-contact classifier on dev matches and report leave-one-match-out rally accuracy.

    uv run python scripts/train_hitmodel.py [--iters 2]

Labels: inside each aligned point (correct server), a constrained decoder picks exactly MCP's
contact count from the candidates using the current scores; picked (onset, hitter) pairs are
positives, everything else in the window negatives. Round 0 uses the API-based scores.
"""

from __future__ import annotations

import argparse
import json
import pickle

import pandas as pd

from uso import hitmodel, points, rally, truth
from uso.paths import OUTPUTS, match_dir


def load(vid, p):
    c = rally.with_api(rally.features(vid), pd.read_parquet(match_dir(vid) / "onset_api.parquet"), p)
    return c


def decode(c, p, scores=None):
    cc = c if scores is None else c.drop(columns=["hit_n", "hit_f"]).join(scores)
    sv = rally.pick_serves_api(cc, p_min=p["serve_p"], quiet=p["quiet"])
    vp, ct = points.assemble(cc, sv, w_prior=p["w_prior"], phantom=p.get("phantom"))
    return cc, vp, ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--bias", type=float, default=-3.0)
    a = ap.parse_args()
    p = {**rally.DECODER, **json.loads((OUTPUTS / "decoder_params.json").read_text())}
    em = truth.eval_matches()
    vids = a.videos or [v for v in em[em.role == "dev"].video_id if (match_dir(v) / "onset_api.parquet").exists()]
    data = {v: load(v, p) for v in vids}
    mcp = {v: truth.mcp_points(v) for v in vids}
    scores = {v: None for v in vids}
    for it in range(a.iters):
        labs = {}
        for v in vids:
            cc, vp, ct = decode(data[v], p, scores[v])
            pairs, info = points.evaluate(vp, mcp[v])
            labs[v] = hitmodel.make_labels(cc, pairs, vp, ct, cc[["hit_n", "hit_f"]])
            print(f"iter {it} {v}: exact {info['rally_exact']:.3f} within1 {info['rally_within1']:.3f} "
                  f"cov {info['mcp_coverage']:.3f} labels {len(labs[v])} pos {int(labs[v].y.sum())}", flush=True)
        # leave-one-match-out: score each match with a model trained on the others
        new = {}
        for v in vids:
            tr = [(data[u], labs[u]) for u in vids if u != v]
            m = hitmodel.HitModel().fit(tr)
            new[v] = m.scores(data[v], bias=a.bias)
        scores = new
    for v in vids:
        cc, vp, ct = decode(data[v], p, scores[v])
        pairs, info = points.evaluate(vp, mcp[v])
        print(f"final LOO {v}: exact {info['rally_exact']:.3f} within1 {info['rally_within1']:.3f} "
              f"bias {info['rally_bias']:.2f} cov {info['mcp_coverage']:.3f}", flush=True)
    full = hitmodel.HitModel().fit([(data[v], labs[v]) for v in vids])
    with open(OUTPUTS / "hitmodel.pkl", "wb") as f:
        pickle.dump(full, f)


if __name__ == "__main__":
    main()
