"""Fit the final rally-contact classifiers on the dev matches.

    uv run python scripts/fit_hitmodel.py [--labeler luna|jev]

Labels come from the API-only decoder constrained to MCP counts (see uso/hitmodel.py). Saves one
model per dev match trained without it (outputs/hitmodel[_jev]_loo_VIDEO.pkl, so dev metrics stay
out-of-sample) and one trained on all dev matches (outputs/hitmodel[_jev].pkl, used for every other
match). Each labeler's models are trained on its own answers.

For Jev the blend weight and bias (learned_w, learned_bias) are then chosen on dev with the
leave-one-out models, by the same objective as scripts/tune_decoder.py, and written into Jev's
decoder parameters. (luna's 0.3 / -0.5 were set by hand from the same kind of dev comparison.)
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle

import pandas as pd

from uso import hitmodel, points, rally, shots, truth
from uso import labeler as labeler_mod
from uso.paths import match_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeler", default="luna", choices=list(labeler_mod.NAMES))
    a = ap.parse_args()
    p = labeler_mod.params(a.labeler)
    p_api = {**p, "learned_w": 0.0}
    em = truth.eval_matches()
    vids = [v for v in em[em.role == "dev"].video_id if labeler_mod.labels_path(v, a.labeler).exists()]
    data, labs = {}, {}
    for v in vids:
        c = rally.with_api(rally.features(v), pd.read_parquet(labeler_mod.labels_path(v, a.labeler)), p_api)
        sv = rally.serves(c, p)
        vp, ct = points.assemble(c, sv, w_prior=p["w_prior"])
        pairs, _ = points.evaluate(vp, truth.mcp_points(v))
        data[v], labs[v] = c, hitmodel.make_labels(c, pairs, vp, ct, c[["hit_n", "hit_f"]])
    for v in vids:
        m = hitmodel.HitModel().fit([(data[u], labs[u]) for u in vids if u != v])
        with open(labeler_mod.hitmodel_path(a.labeler, v), "wb") as f:
            pickle.dump(m, f)
    m = hitmodel.HitModel().fit([(data[u], labs[u]) for u in vids])
    with open(labeler_mod.hitmodel_path(a.labeler), "wb") as f:
        pickle.dump(m, f)
    print("fitted", len(vids), "LOO models + full model", flush=True)
    if a.labeler == "luna":
        return
    best = None
    for w, b in itertools.product([0.0, 0.3, 0.5, 0.7, 0.85, 1.0], [-2.0, -1.5, -1.0, -0.5, 0.0]):
        q = {**p, "learned_w": w, "learned_bias": b}
        r = []
        for v in vids:
            c = shots.blend_learned(v, data[v], q, a.labeler)  # dev match -> its LOO model
            vp, _ = points.assemble(c, rally.serves(c, q), w_prior=q["w_prior"], phantom=q.get("phantom"))
            pairs, info = points.evaluate(vp, truth.mcp_points(v))
            r.append(info["rally_exact"] * info["mcp_coverage"] * len(pairs) / max(1, len(vp)))
        obj = sum(r) / len(r)
        print(f"learned_w={w} learned_bias={b} obj={obj:.4f}", flush=True)
        if best is None or obj > best[0]:
            best = (obj, w, b)
    out = json.loads(labeler_mod.params_path(a.labeler).read_text())
    out.update(learned_w=best[1], learned_bias=best[2])
    labeler_mod.params_path(a.labeler).write_text(json.dumps(out, indent=1))
    print("chose", out, flush=True)


if __name__ == "__main__":
    main()
