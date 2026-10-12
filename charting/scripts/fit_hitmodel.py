"""Fit the final rally-contact classifiers on the dev matches.

    uv run python scripts/fit_hitmodel.py

Labels come from the API-only decoder constrained to MCP counts (see uso/hitmodel.py). Saves one
model per dev match trained without it (outputs/hitmodel_loo_VIDEO.pkl, so dev metrics stay
out-of-sample) and one trained on all dev matches (outputs/hitmodel.pkl, used for test matches).
"""

from __future__ import annotations

import json
import pickle

import pandas as pd

from uso import hitmodel, points, rally, truth
from uso.paths import OUTPUTS, match_dir


def main():
    p = {**rally.DECODER, **json.loads((OUTPUTS / "decoder_params.json").read_text())}
    p_api = {**p, "learned_w": 0.0}
    em = truth.eval_matches()
    vids = [v for v in em[em.role == "dev"].video_id if (match_dir(v) / "onset_api.parquet").exists()]
    data, labs = {}, {}
    for v in vids:
        c = rally.with_api(rally.features(v), pd.read_parquet(match_dir(v) / "onset_api.parquet"), p_api)
        sv = rally.pick_serves_api(c, p_min=p["serve_p"], quiet=p["quiet"])
        vp, ct = points.assemble(c, sv, w_prior=p["w_prior"])
        pairs, _ = points.evaluate(vp, truth.mcp_points(v))
        data[v], labs[v] = c, hitmodel.make_labels(c, pairs, vp, ct, c[["hit_n", "hit_f"]])
    for v in vids:
        m = hitmodel.HitModel().fit([(data[u], labs[u]) for u in vids if u != v])
        with open(OUTPUTS / f"hitmodel_loo_{v}.pkl", "wb") as f:
            pickle.dump(m, f)
    m = hitmodel.HitModel().fit([(data[u], labs[u]) for u in vids])
    with open(OUTPUTS / "hitmodel.pkl", "wb") as f:
        pickle.dump(m, f)
    print("fitted", len(vids), "LOO models + full model")


if __name__ == "__main__":
    main()
