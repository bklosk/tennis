import numpy as np
import pandas as pd

from tennis_pipeline import paths, strokes


def test_gold_report_breakdowns(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(strokes, "CACHE", tmp_path)
    gold = pd.read_csv(strokes.Path(strokes.__file__).resolve().parents[1] / "eval" / "stroke_gold.csv")
    for vid, g in gold.groupby("video_id"):
        n = len(g)
        lat = np.where(g.stroke == "forehand", 0.5, -0.5)
        pd.DataFrame({
            "hit_id": g.hit_id.values, "side": np.where(np.arange(n) % 2, "near", "far"),
            "hand": "L" if vid == "Fl33UXv6jKI" else "R", "is_serve": False, "ball_above_head": False,
            "ball_lat": np.where(vid == "Fl33UXv6jKI", -lat, lat), "l_wrist_lat": np.nan, "r_wrist_lat": np.nan,
            "wrist_gap": np.nan, "ball_height": 0.5, "wrist_above_sh": np.nan, "crop_path": None,
            "hitter_y_m": 11.0, "bounced_before_hit_own_side": True,
        }).to_parquet(paths.match_dir(vid) / "shots_aligned.parquet", index=False)
    rep = strokes.run(sorted(gold.video_id.unique()))
    assert rep["gold_n"] == len(gold)
    assert rep["rule_accuracy_vs_gold"] == 1.0
    assert rep["rule_accuracy_by_era"]["2013-26"]["n"] == len(gold)
    assert set(rep["rule_accuracy_by_labeler_kind"]) == {"visual-review"}
    assert set(rep["rule_accuracy_by_side"]) == {"near", "far"}
