"""Fit the pose-only stroke model on the dev matches (outputs/stroke_model_pose.pkl).

    uv run python scripts/fit_stroke_pose.py

Same data and classifiers as the fused model in scripts/evaluate_all.py, without the API stroke
answers. Matches charted with Jev onset labels use it (uso.labeler.stroke_model_name).
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import f1_score

from uso import labeler as labeler_mod
from uso import stroke_model as sm
from uso import truth
from uso.paths import match_dir


def main():
    em = truth.eval_matches().set_index("video_id")
    done = [v for v in em.index if (match_dir(v) / "shots.parquet").exists() and (match_dir(v) / "stroke_feats.parquet").exists()]
    tables = {v: sm.match_table(v) for v in done}
    dev = pd.concat([tables[v] for v in done if em.at[v, "role"] == "dev"], ignore_index=True)
    model = sm.StrokeModel(pose_only=True).fit(dev)
    sm.save(model, labeler_mod.stroke_model_name("jev"))
    test = pd.concat([tables[v] for v in done if em.at[v, "role"] == "test"], ignore_index=True)
    pr = model.predict(test).set_index("hit_id")
    s = test[test.mcp_side.isin(["F", "B"])]
    f = test[test.mcp_fam.isin(sm.FAM_CLASSES)]
    yp = pr.loc[f.hit_id, "family"].to_numpy()
    print(f"pose-only on {len(done) - len(dev.video_id.unique())} test matches: side acc "
          f"{(pr.loc[s.hit_id, 'side'].to_numpy() == s.mcp_side.to_numpy()).mean():.4f} (n={len(s)}), family macro-F1 "
          f"{f1_score(f.mcp_fam, yp, average='macro'):.4f}; {len(model.cols)} features")


if __name__ == "__main__":
    main()
