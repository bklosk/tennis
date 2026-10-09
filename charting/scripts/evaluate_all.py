"""Final evaluation and exports.

    uv run python scripts/evaluate_all.py

* Trains the stroke model on all dev matches (leave-one-match-out predictions are used for the dev
  rows, so dev numbers are out-of-sample too) and applies it to the test matches.
* Writes charting/reports/evaluation.md and CSV tables under charting/reports/.
* Exports per-match deliverables (outputs/VIDEO_ID/shots.csv, points.csv) and combined
  outputs/all_shots.csv, outputs/all_points.csv.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from uso import evaluate, report, stroke_model as sm, truth
from uso.decisions import DecisionClient
from uso.paths import OUTPUTS, PROJECT, match_dir

REPORTS = PROJECT / "reports"


def fmt(x, nd=3):
    if isinstance(x, float):
        if np.isnan(x):
            return ""
        return str(int(x)) if x.is_integer() and abs(x) >= 1 else f"{x:.{nd}f}"
    return str(x)


def table(df: pd.DataFrame, cols: list[str], nd: int = 3) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    rows = ["| " + " | ".join(fmt(r[c], nd) for c in cols) + " |" for _, r in df.iterrows()]
    return head + "\n".join(rows) + "\n"


def targets_table(t) -> str:
    rows = [
        ("Serve side (deuce/ad)", "agreement", ">= 0.995", t.side_agree),
        ("Serve attempts per point", "agreement", ">= 0.98", t.serves_agree),
        ("Rally length", "exact", ">= 0.85", t.rally_exact),
        ("Rally length", "within one", ">= 0.97", t.rally_within1),
        ("Stroke side, near player", "accuracy", ">= 0.97", t.stroke_side_acc_near),
        ("Stroke side, far player", "accuracy", ">= 0.93", t.stroke_side_acc_far),
        ("Stroke side", "calibration error", "<= 0.05", t.stroke_side_ece),
        ("Stroke family", "macro-F1", ">= 0.75", t.family_macro_f1),
    ]
    out = "| Field | Metric | Target | Test result |\n|---|---|---|---|\n"
    out += "\n".join(f"| {a} | {b} | {c} | {d:.3f} |" for a, b, c, d in rows)
    return out + ("\n\nPositions have no gold set (the guide's 0.5 m / 0.75 m targets need hand labels); they are"
                  " checked against charted facts below.\n")


def main():
    REPORTS.mkdir(exist_ok=True)
    em = truth.eval_matches()
    done = [v for v in em.video_id if (match_dir(v) / "shots.parquet").exists() and (match_dir(v) / "stroke_feats.parquet").exists()]
    dev = [v for v in done if em.set_index("video_id").at[v, "role"] == "dev"]
    test = [v for v in done if em.set_index("video_id").at[v, "role"] == "test"]
    # ---- strokes: LOO on dev, frozen model on test
    tables = {v: sm.match_table(v) for v in done}
    dev_df = pd.concat([tables[v] for v in dev], ignore_index=True)
    preds = [sm.loo(dev_df)]
    model = sm.StrokeModel().fit(dev_df)
    sm.save(model)
    for v in test:
        preds.append(model.predict(tables[v]).assign(video_id=v))
    pred = pd.concat(preds, ignore_index=True)
    allrows = pd.concat([tables[v] for v in done], ignore_index=True).merge(
        pred.drop(columns=["video_id"]), on="hit_id", how="left")
    # ---- per-match summaries
    summ = pd.DataFrame([report.match_summary(v) for v in done])
    side_rows, fam_rows = [], []
    for v in done:
        r = allrows[allrows.video_id == v]
        s = evaluate.side_metrics(r.mcp_side, r.p_forehand)
        sn = evaluate.side_metrics(r[r.hitter_end == "near"].mcp_side, r[r.hitter_end == "near"].p_forehand)
        sf = evaluate.side_metrics(r[r.hitter_end == "far"].mcp_side, r[r.hitter_end == "far"].p_forehand)
        f = evaluate.family_metrics(r.mcp_fam, r.family)
        side_rows.append(dict(video_id=v, side_n=s.get("n"), side_acc=s.get("acc", np.nan), side_acc_near=sn.get("acc", np.nan),
                              side_acc_far=sf.get("acc", np.nan), side_ece=s.get("ece", np.nan)))
        fam_rows.append(dict(video_id=v, fam_n=f["n"], fam_acc=f["acc"], fam_macro_f1=f["macro_f1"],
                             slice_recall=f["recall"].get("slice", np.nan), volley_recall=f["recall"].get("volley", np.nan)))
    summ = summ.merge(pd.DataFrame(side_rows), on="video_id").merge(pd.DataFrame(fam_rows), on="video_id")
    summ.to_csv(REPORTS / "per_match.csv", index=False)
    # ---- pooled metrics by role / era / gender
    pooled = []
    for role in ("dev", "test"):
        vs = [v for v in done if v in (dev if role == "dev" else test)]
        if not vs:
            continue
        pairs = pd.concat([pd.read_parquet(match_dir(v) / "aligned_points.parquet").assign(video_id=v) for v in vs])
        rm = report.rally_metrics(pairs)
        r = allrows[allrows.video_id.isin(vs)]
        s = evaluate.side_metrics(r.mcp_side, r.p_forehand)
        sn = evaluate.side_metrics(r[r.hitter_end == "near"].mcp_side, r[r.hitter_end == "near"].p_forehand)
        sf = evaluate.side_metrics(r[r.hitter_end == "far"].mcp_side, r[r.hitter_end == "far"].p_forehand)
        f = evaluate.family_metrics(r.mcp_fam, r.family)
        sub = summ[summ.video_id.isin(vs)]
        pooled.append(dict(set=role, matches=len(vs), mcp_points=int(sub.mcp_points.sum()), matched=int(sub.matched.sum()),
                           coverage=sub.matched.sum() / sub.mcp_points.sum(), side_agree=float(np.average(sub.side_agree, weights=sub.matched)),
                           serves_agree=float(np.average(sub.serves_agree, weights=sub.matched)),
                           **rm, stroke_side_n=s.get("n"), stroke_side_acc=s.get("acc"), stroke_side_acc_near=sn.get("acc"),
                           stroke_side_acc_far=sf.get("acc"), stroke_side_ece=s.get("ece"), family_acc=f["acc"],
                           family_macro_f1=f["macro_f1"], slice_recall=f["recall"].get("slice"), volley_recall=f["recall"].get("volley"),
                           serve_wide_vs_T_auc=float(sub.serve_wide_vs_T_auc.mean()), shot_dir_auc=float(sub.shot_dir_1_vs_3_auc.mean()),
                           net_shot_auc=float(sub.net_shot_auc.mean()), server_behind_baseline=float(sub.server_behind_baseline.mean())))
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(REPORTS / "pooled.csv", index=False)
    by_era = summ[summ.role == "test"].groupby("era").apply(
        lambda g: pd.Series(dict(matches=len(g), rally_exact=np.average(g.rally_exact, weights=g.points),
                                 rally_within1=np.average(g.rally_within1, weights=g.points), coverage=g.matched.sum() / g.mcp_points.sum(),
                                 side_acc=np.average(g.side_acc, weights=g.side_n))), include_groups=False).reset_index()
    by_gender = summ[summ.role == "test"].groupby("gender").apply(
        lambda g: pd.Series(dict(matches=len(g), rally_exact=np.average(g.rally_exact, weights=g.points),
                                 rally_within1=np.average(g.rally_within1, weights=g.points), coverage=g.matched.sum() / g.mcp_points.sum(),
                                 side_acc=np.average(g.side_acc, weights=g.side_n))), include_groups=False).reset_index()
    # ---- confusion for family on test
    rt = allrows[allrows.video_id.isin(test) & allrows.mcp_fam.notna()]
    conf = pd.crosstab(rt.mcp_fam, rt.family) if len(rt) else pd.DataFrame()
    # ---- exports
    shots_all, points_all = [], []
    for v in done:
        sp = pred[pred.hit_id.isin(tables[v].hit_id)][["hit_id", "p_forehand", "side", "family"] +
                                                      [c for c in pred.columns if c.startswith("p_") and c != "p_forehand"]]
        a, b = report.export(v, sp)
        shots_all.append(a)
        points_all.append(b)
    pd.concat(shots_all).to_csv(OUTPUTS / "all_shots.csv", index=False)
    pd.concat(points_all).to_csv(OUTPUTS / "all_points.csv", index=False)
    spent = DecisionClient.spent()
    tokens = DecisionClient.tokens_spent()
    # ---- markdown
    md = ["# Evaluation against the Match Charting Project\n",
          f"Generated by `scripts/evaluate_all.py`. Dev matches: {len(dev)}; held-out test matches: {len(test)}. "
          f"Decisions API spend to date: ${spent:.2f} ({tokens:,} input tokens).\n",
          "Dev stroke numbers are leave-one-match-out; test numbers come from a model trained on all dev matches. "
          "Rally-decoder parameters were tuned on dev only.\n",
          "## Pooled\n", table(pooled, ["set", "matches", "mcp_points", "matched", "coverage", "rally_exact", "rally_within1",
                                        "rally_bias", "side_agree", "stroke_side_acc", "stroke_side_acc_near",
                                        "stroke_side_acc_far", "stroke_side_ece", "family_acc", "family_macro_f1",
                                        "slice_recall", "volley_recall"]),
          "\n## Against the guide's provisional targets (test set)\n",
          targets_table(pooled[pooled.set == "test"].iloc[0]) if (pooled.set == "test").any() else "",
          "\n### Position checks (mean over matches)\n",
          table(pooled, ["set", "server_behind_baseline", "serve_wide_vs_T_auc", "shot_dir_auc", "net_shot_auc"]),
          "\n## Per match\n",
          table(summ.sort_values(["role", "year"]), ["role", "year", "gender", "round", "match", "mcp_points", "matched",
                                                    "coverage", "rally_exact", "rally_within1", "rally_bias", "side_agree",
                                                    "side_acc", "side_acc_near", "side_acc_far", "fam_acc", "fam_macro_f1"]),
          "\n## Test set by era\n", table(by_era, ["era", "matches", "coverage", "rally_exact", "rally_within1", "side_acc"]),
          "\n## Test set by gender\n", table(by_gender, ["gender", "matches", "coverage", "rally_exact", "rally_within1", "side_acc"]),
          "\n## Stroke family confusion (test; rows = MCP, columns = predicted)\n",
          conf.to_markdown() if len(conf) else "(none)\n",
          "\n## Position checks per match\n",
          table(summ.sort_values(["role", "year"]), ["role", "year", "match", "server_behind_baseline", "serve_wide_vs_T_auc",
                                                    "shot_dir_1_vs_3_auc", "net_shot_auc", "dnet_median_net_shots",
                                                    "dnet_median_other"]),
          ]
    (REPORTS / "evaluation.md").write_text("\n".join(md))
    print(pooled.round(3).to_string(index=False))
    print(json.dumps(dict(spent_usd=round(spent, 3), tokens=tokens)))


if __name__ == "__main__":
    main()
