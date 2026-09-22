import numpy as np
import pandas as pd
import pytest

from tennis_pipeline import gold, paths, process
from tennis_pipeline.court import Calibration
from tennis_pipeline.track import SegmentTracks


def fake_shots(n=400, seed=0):
    rng = np.random.default_rng(seed)
    eras = rng.choice(["2000-06", "2007-12", "2013-26"], n)
    return pd.DataFrame({
        "video_id": [f"v{e}" for e in eras], "hit_id": [f"h{k}" for k in range(n)], "t": 1.0,
        "box": [[0, 0, 10, 10]] * n, "side": rng.choice(["near", "far"], n), "hand": rng.choice(["R", "R", "R", "L"], n),
        "era": eras, "point_number": 1, "is_serve": False,
        "truth": rng.choice(["forehand", "backhand"], n)})


class FakeLabeler:
    """Answers like the real protocol would: mirror-consistent when right, position-biased otherwise."""

    def __init__(self, shots, accuracy=1.0, biased_share=0.2, seed=0):
        self.truth = dict(zip(shots.hit_id, shots.truth))
        self.rng = np.random.default_rng(seed)
        self.accuracy, self.biased_share = accuracy, biased_share

    def label(self, r):
        if self.rng.random() < self.biased_share:
            v = {"side": "left", "side_mirror": "left", "point": None, "point_mirror": None, "two_hands": "no"}
        else:
            stroke = self.truth[r.hit_id]
            if self.rng.random() > self.accuracy:
                stroke = "backhand" if stroke == "forehand" else "forehand"
            dom = "right" if r.hand == "R" else "left"
            player_side = dom if stroke == "forehand" else gold.FLIP[dom]
            img = player_side if r.side == "near" else gold.FLIP[player_side]
            v = {"side": img, "side_mirror": gold.FLIP[img], "point": img, "point_mirror": gold.FLIP[img],
                 "two_hands": "no"}
        stroke, conf, reason = gold.decide(v, r.side, r.hand)
        return {"stroke": stroke, "confidence": conf, "reason": reason, "votes": gold._votes_str(v)}


@pytest.fixture
def gold_env(tmp_path, monkeypatch):
    shots = fake_shots()
    human = shots.head(40)[["video_id", "hit_id"]].assign(stroke=shots.head(40).truth.values,
                                                          labeler="claude-visual-review", sheet="sheet0")
    path = tmp_path / "stroke_gold.csv"
    human.to_csv(path, index=False)
    monkeypatch.setattr(gold, "GOLD_PATH", path)
    monkeypatch.setattr(gold, "candidate_shots", lambda vids: shots.drop(columns=["truth"]))
    monkeypatch.setattr(gold, "contact_sheets", lambda rows: [])
    return shots, path


def test_expand_writes_balanced_labels(gold_env, monkeypatch):
    shots, path = gold_env
    monkeypatch.setattr(gold, "Labeler", lambda model_id, fn: FakeLabeler(shots))
    rep = gold.expand(["x"], None, target=200)
    assert rep["calibration"]["accuracy_when_accepted"] == 1.0
    out = pd.read_csv(path)
    assert len(out) == 200 and rep["written"] == 160
    new = out[out.labeler.str.startswith("qwen3-vl")]
    truth = dict(zip(shots.hit_id, shots.truth))
    assert (new.stroke == new.hit_id.map(truth)).all()
    assert new.side.value_counts(normalize=True)["near"] > 0.5
    assert new.hand.value_counts(normalize=True)["L"] >= 0.25
    assert set(new.era) == {"2000-06", "2007-12", "2013-26"}
    assert rep["reasons"].get("no_mirror_consistent_vote", 0) > 0


def test_expand_refuses_when_protocol_disagrees_with_human_gold(gold_env, monkeypatch):
    shots, path = gold_env
    monkeypatch.setattr(gold, "Labeler", lambda model_id, fn: FakeLabeler(shots, accuracy=0.6))
    rep = gold.expand(["x"], None, target=200)
    assert rep["written"] == 0 and "stopped" in rep
    assert len(pd.read_csv(path)) == 40


def test_retrack_ball_only_when_weights_change(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS", tmp_path / "outputs")
    vid = "v"
    out = paths.match_dir(vid)
    pd.DataFrame({"segment_id": [0], "start": [10.0], "end": [11.0], "duration": [1.0]}).to_csv(out / "segments.csv",
                                                                                              index=False)
    n = 30
    H = np.eye(3)
    old = SegmentTracks(t0=10.0, n=n, calibs=[Calibration(H, H, 0.0, 4)] * n, ball=np.zeros((n, 2)),
                        players={"near": np.ones((n, 4))}, player_kps={"near": np.zeros((n, 17, 3))},
                        court_ok=1.0, ball_weights="tracknet:old")
    path = process._tracks_dir(vid) / "0000_000100.npz"
    process.save_tracks(path, old)

    class FakeBall:
        tag = "tracknet_ft:new"

        def __init__(self, *a, **k):
            pass

        def __call__(self, frames):
            return np.full((len(frames), 2), 7.0)

    def no_models(*a, **k):
        raise AssertionError("court/player models should not load for a ball-only re-track")

    monkeypatch.setattr(process, "BallTracker", FakeBall)
    monkeypatch.setattr(process, "CourtDetector", no_models)
    monkeypatch.setattr(process, "PlayerDetector", no_models)
    monkeypatch.setattr(process.video, "read_clip", lambda *a, **k: np.zeros((n, 4, 4, 3), np.uint8))
    stats = process.track_match(vid, tmp_path / "missing.mp4")
    assert stats["chunks_ball_retracked"] == 1
    tr = process.load_tracks(path)
    assert tr.ball_weights == "tracknet_ft:new" and (tr.ball == 7.0).all()
    assert (tr.players["near"] == 1).all()
    assert process.track_match(vid, tmp_path / "missing.mp4")["chunks_ball_retracked"] == 0
