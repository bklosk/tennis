"""Grid-search rally-decoder parameters on the dev matches (API onset labels must exist).

    uv run python scripts/tune_decoder.py [--labeler luna|jev] [VIDEO_ID ...]

Objective: mean over matches of (exact rally accuracy x MCP coverage x point precision), so a
setting cannot win by dropping hard points or by inventing points the alignment then skips
(precision = aligned points / detected points; an uncharted match has no alignment to hide them).
The luna parameters in outputs/decoder_params.json were tuned before the precision term existed.

luna: one grid. jev: coordinate search, because Jev needs its own serve threshold per court end
(it reads the near player's pre-serve routine as a serve far more often): serve thresholds at the
luna setting, then the decoder grid at the best thresholds, then the thresholds again.
Writes the labeler's decoder parameters (outputs/decoder_params[_jev].json) and grid.
"""

from __future__ import annotations

import argparse
import itertools
import json

import pandas as pd

from uso import labeler as labeler_mod
from uso import points, rally, truth
from uso.paths import OUTPUTS, match_dir

GRID = dict(audio_w=[0.25, 0.5, 0.75], z0=[25.0], bias=[-1.0, -0.5, 0.0, 0.5], unlabeled=[-4.0],
            w_prior=[0.2, 0.4], serve_p=[0.4], quiet=[2.5], phantom=[None, -2.0])
SERVE_GRID = dict(serve_p_near=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8], serve_p_far=[0.3, 0.4, 0.5, 0.6, 0.7])


def score(vid_data, params):
    out = []
    for vid, c0, lab, m in vid_data:
        c = rally.with_api(c0, lab, params)
        sv = rally.serves(c, params)
        vp, _ = points.assemble(c, sv, w_prior=params["w_prior"], phantom=params.get("phantom"))
        if vp.empty:
            out.append(dict(video=vid, exact=0, within1=0, coverage=0, precision=0, video_points=0))
            continue
        pairs, info = points.evaluate(vp, m)
        out.append(dict(video=vid, exact=info["rally_exact"], within1=info["rally_within1"], coverage=info["mcp_coverage"],
                        precision=len(pairs) / len(vp), video_points=len(vp)))
    return pd.DataFrame(out)


def search(data, grid, base, rows, best):
    keys = list(grid)
    for vals in itertools.product(*grid.values()):
        p = {**base, **dict(zip(keys, vals))}
        r = score(data, p)
        obj = float((r.exact * r.coverage * r.precision).mean())
        rows.append({**p, "obj": obj, "exact": r.exact.mean(), "within1": r.within1.mean(),
                     "coverage": r.coverage.mean(), "precision": r.precision.mean()})
        if best[0] is None or obj > best[0]:
            best[0], best[1] = obj, p
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--labeler", default="luna", choices=list(labeler_mod.NAMES))
    a = ap.parse_args()
    em = truth.eval_matches()
    vids = a.videos or [v for v in em[em.role == "dev"].video_id if labeler_mod.labels_path(v, a.labeler).exists()]
    data = []
    for v in vids:
        c = rally.features(v)
        lab = pd.read_parquet(labeler_mod.labels_path(v, a.labeler))
        data.append((v, c, lab, truth.mcp_points(v)))
    print(f"tuning {a.labeler} on {vids}", flush=True)
    rows, best = [], [None, None]
    if a.labeler == "luna":
        search(data, GRID, {}, rows, best)
    else:
        start = {**rally.DECODER, **json.loads((OUTPUTS / "decoder_params.json").read_text()), "learned_w": 0.0}
        search(data, SERVE_GRID, start, rows, best)
        search(data, {k: v for k, v in GRID.items() if k != "serve_p"}, best[1], rows, best)
        search(data, SERVE_GRID, best[1], rows, best)
    res = pd.DataFrame(rows).sort_values("obj", ascending=False)
    print(res.head(10).round(3).to_string(index=False))
    out = {k: v for k, v in best[1].items() if k != "learned_w"}
    labeler_mod.params_path(a.labeler).write_text(json.dumps(out, indent=1))
    res.to_csv(OUTPUTS / f"decoder_grid{'' if a.labeler == 'luna' else '_' + a.labeler}.csv", index=False)


if __name__ == "__main__":
    main()
