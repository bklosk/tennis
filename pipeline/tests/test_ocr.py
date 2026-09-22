import cv2
import numpy as np
import pandas as pd
import pytest

from tennis_pipeline import ocr, score
from tennis_pipeline.ocr import Word
from tennis_pipeline.score import Rules, State

P1, P2 = "Andre Agassi", "Pete Sampras"


def words(*rows):
    out = []
    for k, row in enumerate(rows):
        x = 0
        for text in row:
            out.append(Word(text, x, 30 * k, x + 20 * len(text), 30 * k + 20, 0.99))
            x += 20 * len(text) + 15
    return out


def test_parse_speed_units_and_split_boxes():
    assert ocr.parse_speed(words(["SERVE SPEED"], ["128", "MPH"])) == pytest.approx(206.0, abs=0.1)
    assert ocr.parse_speed(words(["206 KM/H"])) == 206.0
    assert ocr.parse_speed(words(["12 MPH"])) is None
    assert ocr.parse_speed(words(["AGASSI 6 3"])) is None


def test_parse_score_with_points_column():
    p = ocr.parse_score(words(["AGASSI", "6 3 40"], ["SAMPRAS", "4 6 15"]), P1, P2)
    assert p == {"games1": [6, 3], "games2": [4, 6], "pts1": "40", "pts2": "15", "has_points": True}
    # Rows can come in either order, with initials and three-letter codes.
    p = ocr.parse_score(words(["P. SAMPRAS 2 AD"], ["A. AGASSI 3 40"]), P1, P2)
    assert (p["games1"], p["pts1"], p["pts2"]) == ([3], "40", "AD")
    p = ocr.parse_score(words(["AGA", "5"], ["SAM", "4"]), P1, P2, points_column=False)
    assert p["games1"] == [5] and p["pts1"] is None


def test_parse_score_rejects_other_names_and_ragged_rows():
    assert ocr.parse_score(words(["FEDERER 6 3"], ["NADAL 4 6"]), P1, P2) is None
    assert ocr.parse_score(words(["AGASSI 6 3"], ["SAMPRAS 4"]), P1, P2) is None


def test_attach_speeds_second_serve():
    pts = pd.DataFrame({"video_t_hi": [30.0, 60.0, 90.0]})
    out = ocr.attach_speeds(pts, [(10.0, 190.0), (40.0, 180.0), (47.0, 150.0)])
    assert out.Speed_KMH.tolist()[:2] == [190.0, 150.0]
    assert out.ServeNumber.tolist()[:2] == [1, 2]
    assert np.isnan(out.Speed_KMH.iloc[2])


def render(state: State, speed_mph: int | None, rules: Rules) -> np.ndarray:
    img = np.full((720, 1280, 3), (40, 110, 60), np.uint8)
    cv2.rectangle(img, (500, 250), (780, 520), (200, 200, 200), 3)  # some court lines
    d1, d2 = state.display(rules)
    cv2.rectangle(img, (40, 40), (420, 120), (20, 20, 20), -1)
    for k, (name, games, pts) in enumerate((("AGASSI", state.games1, d1), ("SAMPRAS", state.games2, d2))):
        y = 72 + 36 * k
        cv2.putText(img, name, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, " ".join(map(str, games)), (220, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, pts, (350, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    if speed_mph:
        cv2.rectangle(img, (1000, 620), (1240, 680), (20, 20, 20), -1)
        cv2.putText(img, f"{speed_mph} MPH", (1015, 662), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


def test_end_to_end_on_rendered_graphics():
    rules = Rules(5, "tb7")
    rng = np.random.default_rng(1)
    s, t, frames, truth = State(), 0.0, [], []
    for _ in range(14):
        w = int(rng.integers(1, 3))
        mph = int(rng.integers(95, 135))
        # Serve at t, speed graphic 2-5 s later, score updates at t+12.
        for k in range(15):
            frames.append((t + k, render(s, mph if 2 <= k <= 5 else None, rules)))
        truth.append((w, mph))
        s = score.win_point(s, w, rules)
        t += 15
    frames.append((t, render(s, None, rules)))
    frames.append((t + 1, render(s, None, rules)))

    engine = ocr.RapidEngine()
    regions = ocr.discover_regions([f for _, f in frames[::6]], engine, P1, P2)
    assert regions["score"] and regions["speed"]
    reads = ocr.read_regions(iter(frames), regions, engine, P1, P2)
    points, stats = ocr.build_points(reads, {"player1": P1, "player2": P2, "video_id": "synthetic"}, rules)
    assert stats["points_column"]
    assert len(points) == len(truth)
    assert points.PointWinner.tolist() == [w for w, _ in truth]
    got = points.Speed_KMH.to_numpy()
    want = np.array([mph * ocr.MPH_TO_KMH for _, mph in truth])
    assert np.nanmean(np.abs(got - want) < 1.0) >= 0.9
