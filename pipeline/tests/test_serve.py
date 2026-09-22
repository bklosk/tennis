import cv2
import numpy as np

from tennis_pipeline import events, serve
from tennis_pipeline.court import REF_KPS, Calibration, m_to_ref
from tennis_pipeline.track import FPS, SegmentTracks

N = 300
CONTACT = 185


def calib() -> Calibration:
    src = REF_KPS[[0, 1, 2, 3]].astype(np.float32)  # doubles court corners, far then near
    dst = np.float32([[430, 170], [850, 170], [190, 640], [1090, 640]])
    H = cv2.getPerspectiveTransform(src, dst)
    return Calibration(H, np.linalg.inv(H), 0.0, 4)


def player_box(c: Calibration, x_m: float, y_m: float, h: float) -> np.ndarray:
    fx, fy = c.to_img(np.array([[x_m, y_m]]))[0]
    return np.array([fx - 0.2 * h, fy - h, fx + 0.2 * h, fy])


def make_tracks(server_y=12.4, extend=True, toss=True):
    c = calib()
    tr = SegmentTracks(t0=100.0, n=N, calibs=[c] * N, ball=np.full((N, 2), np.nan))
    near = np.array([player_box(c, 1.2, server_y, 150) for _ in range(N)])
    far = np.array([player_box(c, -1.5, -12.6, 60) for _ in range(N)])
    if extend:  # tossing arm, then hitting arm and racket above the head
        for f in range(150, 196):
            k = 1 + 0.4 * (f - 150) / (CONTACT - 150) if f <= CONTACT else 1.1
            near[f, 1] = near[f, 3] - 150 * k
    tr.players = {"near": near, "far": far}
    b = np.full((N, 2), np.nan)
    if toss:
        top = near[150, 1]
        for f in range(160, CONTACT):
            b[f] = (near[f, 0] + 20, top - 40 * np.sin(np.pi * (f - 160) / 30))
    return tr, b


def onsets_at(*frames):
    t = np.array([100.0 + f / FPS for f in frames])
    return t, np.ones(len(t))


def test_detects_serve_with_ball_lost_after_contact():
    tr, b = make_tracks()
    cands = serve.detect(tr, b, [], onsets_at(CONTACT))
    acc = [c for c in cands if c["accepted"]]
    assert len(acc) == 1
    assert acc[0]["side"] == "near" and abs(acc[0]["frame"] - CONTACT) <= 2
    assert acc[0]["toss"] == 1.0 and acc[0]["audio"] == 1.0


def test_merge_inserts_serve_and_drops_pre_serve_bounce_hits():
    tr, b = make_tracks()
    ritual = {"frame": 130, "side": "near", "dv": 10.0, "pos": np.array([500.0, 500.0]), "gap_frames": 2,
              "player_seen": True, "toss": False}
    ret = {"frame": CONTACT + 30, "side": "far", "dv": 20.0, "pos": np.array([600.0, 200.0]), "gap_frames": 2,
           "player_seen": True, "toss": False}
    prev_point = {**ret, "frame": 20}  # last shot of the previous point, 5.5 s earlier
    raw = [prev_point, ritual, ret]
    cands = serve.detect(tr, b, raw, onsets_at(CONTACT))
    merged = serve.merge(raw, cands, b, tr)
    assert [h["frame"] for h in merged] == [20, CONTACT, CONTACT + 30]
    hits = events.annotate_hits(tr, b, merged, [])
    assert hits[1]["is_serve"] and hits[1]["serve_source"] == "detector"
    assert not hits[2]["is_serve"]


def test_rejects_stance_inside_the_court():
    tr, b = make_tracks(server_y=5.0)
    cands = serve.detect(tr, b, [], onsets_at(CONTACT))
    assert not any(c["accepted"] for c in cands)
    assert any(c["reject"] == "not_behind_baseline" for c in cands)


def test_rejects_rally_shot_after_opponent_hit():
    tr, b = make_tracks()
    opp = {"frame": CONTACT - 35, "side": "far", "dv": 20.0, "pos": np.array([600.0, 200.0]), "gap_frames": 2,
           "player_seen": True, "toss": False}
    cands = serve.detect(tr, b, [opp], onsets_at(CONTACT))
    assert not any(c["accepted"] for c in cands)
    assert any(c["reject"] == "opponent_hit_recently" for c in cands)


def test_requires_ball_or_audio_evidence():
    tr, b = make_tracks(toss=False)
    cands = serve.detect(tr, b, [], None)
    assert not any(c["accepted"] for c in cands)
    assert any(c["reject"] == "no_ball_or_audio_evidence" for c in cands)


def test_no_extension_no_candidate():
    tr, b = make_tracks(extend=False)
    assert serve.detect(tr, b, [], onsets_at(CONTACT)) == []


def test_court_geometry_sanity():
    c = calib()
    x, y = c.to_court_m(c.to_img(np.array([[1.0, 12.0]])))[0]
    assert abs(x - 1.0) < 1e-6 and abs(y - 12.0) < 1e-6
    assert m_to_ref(np.array([0.0, 0.0])).shape == (2,)
