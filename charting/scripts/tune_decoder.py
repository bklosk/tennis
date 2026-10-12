"""Grid-search rally-decoder parameters on the dev matches (API onset labels must exist).

    uv run python scripts/tune_decoder.py [VIDEO_ID ...]

Objective: mean over matches of (exact rally accuracy x MCP coverage), so a setting cannot win by
dropping hard points. Writes outputs/decoder_params.json.
"""

from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pandas as pd

from uso import points, rally, truth
from uso.paths import OUTPUTS, match_dir


def score(vid_data, params):
    out = []
    for vid, c0, lab, m in vid_data:
        c = rally.with_api(c0, lab, params)
        sv = rally.pick_serves_api(c, p_min=params["serve_p"], quiet=params["quiet"])
        vp, _ = points.assemble(c, sv, w_prior=params["w_prior"], phantom=params.get("phantom"))
        if vp.empty:
            out.append(dict(video=vid, exact=0, within1=0, coverage=0))
            continue
        _, info = points.evaluate(vp, m)
        out.append(dict(video=vid, exact=info["rally_exact"], within1=info["rally_within1"], coverage=info["mcp_coverage"]))
    return pd.DataFrame(out)


def main():
    em = truth.eval_matches()
    vids = sys.argv[1:] or [v for v in em[em.role == "dev"].video_id if (match_dir(v) / "onset_api.parquet").exists()]
    data = []
    for v in vids:
        c = rally.features(v)
        lab = pd.read_parquet(match_dir(v) / "onset_api.parquet")
        data.append((v, c, lab, truth.mcp_points(v)))
    grid = dict(audio_w=[0.25, 0.5, 0.75], z0=[25.0], bias=[-1.0, -0.5, 0.0, 0.5], unlabeled=[-4.0],
                w_prior=[0.2, 0.4], serve_p=[0.4], quiet=[2.5], phantom=[None, -2.0])
    keys = list(grid)
    best, rows = None, []
    for vals in itertools.product(*grid.values()):
        p = dict(zip(keys, vals))
        r = score(data, p)
        obj = float((r.exact * r.coverage).mean())
        rows.append({**p, "obj": obj, "exact": r.exact.mean(), "within1": r.within1.mean(), "coverage": r.coverage.mean()})
        if best is None or obj > best[0]:
            best = (obj, p)
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()}), flush=True)
    res = pd.DataFrame(rows).sort_values("obj", ascending=False)
    print(res.head(10).round(3).to_string(index=False))
    (OUTPUTS / "decoder_params.json").write_text(json.dumps(best[1], indent=1))
    res.to_csv(OUTPUTS / "decoder_grid.csv", index=False)


if __name__ == "__main__":
    main()
